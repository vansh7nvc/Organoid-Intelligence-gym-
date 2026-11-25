"""
Parallel Multi-Seed Benchmark Campaign Runner for OrganoidEnv (Stage 4).

Executes the full empirical evaluation grid cleanly across parallel worker processes:
  - C0 Full Model: 10 seeds x 500 episodes
  - C1-C6 Orthogonal Ablations: 6 conditions x 5 seeds x 500 episodes
  - B1-B5 Algorithmic Baselines (Direct D3QN, MLP-DQN, SimpleSNN-LIF, PPO-SB3, Random):
        5 conditions x 5 seeds x 500 episodes

Also captures post-training mechanistic telemetry:
  - 8.0s Stage-4 spike raster & step ATP/firing traces (for Elephant branching/avalanche)
  - 4-direction motor cluster DSI probe matrix on trained synaptic weights
  - Synaptic weight spectra (excitatory/inhibitory + pruning fraction)
  - Zero-shot evaluation across 5 novel out-of-distribution (OOD) obstacle topologies

Supports atomic per-seed persistence (.tmp -> os.replace) and automatic skip-completed resume.
"""

import os
import sys
import csv
import json
import time
import shutil
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


SEEDS_10 = [42, 123, 456, 789, 1024, 2024, 3033, 4044, 5055, 6066]
SEEDS_5 = [42, 123, 456, 789, 1024]

NOVEL_TOPOLOGIES = {
    "T0_nominal": [
        {"type": "circle", "pos": [0.5, 0.5], "radius": 0.15},
        {"type": "box", "bounds": [0.2, 0.3, 0.4, 0.8]},
    ],
    "T1_dual_circle": [
        {"type": "circle", "pos": [0.40, 0.60], "radius": 0.12},
        {"type": "circle", "pos": [0.60, 0.40], "radius": 0.12},
    ],
    "T2_slanted_walls": [
        {"type": "box", "bounds": [0.35, 0.43, 0.25, 0.72]},
        {"type": "box", "bounds": [0.57, 0.65, 0.35, 0.88]},
    ],
    "T3_central_maze": [
        {"type": "circle", "pos": [0.45, 0.55], "radius": 0.16},
        {"type": "box", "bounds": [0.58, 0.80, 0.46, 0.54]},
    ],
    "T4_tight_corridor": [
        {"type": "box", "bounds": [0.25, 0.58, 0.56, 0.65]},
        {"type": "box", "bounds": [0.38, 0.72, 0.32, 0.41]},
    ],
}

CONDITIONS = {
    # Nominal Full Model (10 seeds)
    "c0_full_model": {
        "kind": "organoid",
        "env_kwargs": {},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": True,
    },
    # Orthogonal Ablations C1-C6 (10 seeds each)
    "c1_frozen_stdp": {
        "kind": "organoid",
        "env_kwargs": {"use_plasticity": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c2_random_snn": {
        "kind": "organoid",
        "env_kwargs": {"random_snn": True},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c3_no_metabolism": {
        "kind": "organoid",
        "env_kwargs": {"use_metabolism": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c4_no_gar": {
        "kind": "organoid",
        "env_kwargs": {"use_stabilizer": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c5_no_pruning": {
        "kind": "organoid",
        "env_kwargs": {"use_pruning": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c6_single_trace": {
        "kind": "organoid",
        "env_kwargs": {"use_dual_trace": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    # Matched Single-Trace Ablations C6b, C6c (10 seeds each)
    "c6b_single_trace_fast": {
        "kind": "organoid",
        "env_kwargs": {"use_dual_trace": False, "tau_trace": 100.0, "trace_weight": 1.0},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    "c6c_single_trace_slow": {
        "kind": "organoid",
        "env_kwargs": {"use_dual_trace": False, "tau_trace": 2500.0, "trace_weight": 0.5},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    # GAR-Only Ablation C7 (10 seeds)
    "c7_gar_only": {
        "kind": "organoid",
        "env_kwargs": {"use_stabilizer": True, "use_metabolism": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    # Algorithmic Baselines B1-B5 (10 seeds each, B1/B2/B4 action-granularity matched)
    "b1_direct_d3qn": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
    "b2_mlp_dqn": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True},
        "agent_type": "mlp_dqn",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
    "b3_simple_snn_lif": {
        "kind": "simple_snn",
        "env_kwargs": {},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": True,
    },
    "b4_ppo_sb3": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True},
        "agent_type": "ppo",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
    "b5_random": {
        "kind": "organoid",
        "env_kwargs": {},
        "agent_type": "random",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": True,
    },
    # Unregulated (GAR-off + Metabolism-off) Training Condition C8 (10 seeds)
    "c8_no_gar_no_met": {
        "kind": "organoid",
        "env_kwargs": {"use_stabilizer": False, "use_metabolism": False},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": True,
        "eval_ood": False,
    },
    # Unconfounded Step<=0.05 Baselines (10 seeds each)
    "b1_step005_d3qn": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True, "step_size": 0.05},
        "agent_type": "d3qn",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
    "b2_step005_mlp_dqn": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True, "step_size": 0.05},
        "agent_type": "mlp_dqn",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
    "b4_step005_ppo": {
        "kind": "skeleton",
        "env_kwargs": {"obs_mode": "full", "matched_granularity": True, "step_size": 0.05},
        "agent_type": "ppo",
        "seeds": SEEDS_10,
        "has_snn": False,
        "eval_ood": True,
    },
}


CSV_FIELDS = [
    "condition", "seed", "episode", "stage", "return", "length",
    "goal_reached", "success", "final_dist", "mean_rate_hz", "peak_rate_hz",
    "total_spikes", "mean_atp", "mean_weight", "weight_std",
    "active_syn_frac", "seizure_clamps", "coma_kicks", "wall_sec",
]


def get_curriculum_stage(episode: int, total_episodes: int = 500) -> int:
    """Curriculum schedule matching 16_multiseed_training.py (scaled cleanly if total_episodes != 500)."""
    frac = episode / max(1, total_episodes)
    if frac < 0.20:
        return 1
    elif frac < 0.40:
        return 2
    elif frac < 0.70:
        return 3
    return 4


def apply_inhibitory_homeostasis(env):
    """Maintain E/I synaptic balance without overriding frozen or un-stabilized ablations."""
    if not (getattr(env, "use_stabilizer", True) and getattr(env, "use_plasticity", True) and not getattr(env, "random_snn", False)):
        return
    if len(getattr(env, "synapses", [])) < 2:
        return
    ee_weights = [float(np.mean(np.array(S.w))) for S in env.synapses[:-1]]
    mean_ee_w = float(np.mean(ee_weights))
    target_ie_w = mean_ee_w * 1.5
    S_inh = env.synapses[-1]
    current_ie_w = np.array(S_inh.w, dtype=np.float64)
    updated_ie_w = current_ie_w + 0.1 * (target_ie_w - current_ie_w)
    S_inh.w = np.clip(updated_ie_w, 0.0, 100.0)


def compute_dsi(firing_rates_4) -> float:
    rates = np.asarray(firing_rates_4, dtype=np.float64)
    dsi_vert = abs(rates[0] - rates[1]) / (rates[0] + rates[1] + 1e-9)
    dsi_horiz = abs(rates[2] - rates[3]) / (rates[2] + rates[3] + 1e-9)
    return float(max(dsi_vert, dsi_horiz))


def _atomic_write_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp_{os.getpid()}"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def _atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _atomic_write_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp_{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _build_env(spec: dict, seed: int):
    kind = spec["kind"]
    kwargs = dict(spec.get("env_kwargs", {}))
    if kind == "organoid":
        kwargs["seed"] = seed
        from organoid_rl.environment.core import OrganoidEnv
        return OrganoidEnv(**kwargs)
    elif kind == "simple_snn":
        kwargs["seed"] = seed
        from organoid_rl.environment.baseline_envs import SimpleSNNEnv
        return SimpleSNNEnv(**kwargs)
    elif kind == "skeleton":
        from organoid_rl.environment.baseline_envs import SkeletonEnv
        return SkeletonEnv(**kwargs)
    raise ValueError(f"Unknown env kind: {kind}")


def _set_eval_mode(agent, agent_type: str, eval_on: bool):
    if agent_type == "d3qn":
        if eval_on:
            agent.online_net.eval()
        else:
            agent.online_net.train()
    elif agent_type == "mlp_dqn":
        if eval_on:
            agent._saved_eps = agent.epsilon
            agent.epsilon = 0.0
        else:
            agent.epsilon = getattr(agent, "_saved_eps", 0.05)


def _select_action(agent, agent_type: str, obs):
    if agent_type == "ppo":
        action, _ = agent.predict(obs, deterministic=True)
        return int(action)
    return int(agent.choose_action(obs))


def _apply_topology(env, topo_defs):
    parsed = []
    for item in topo_defs:
        d = dict(item)
        if "pos" in d:
            d["pos"] = np.array(d["pos"], dtype=np.float64)
        parsed.append(d)
    env.obstacles = parsed
    if hasattr(env, "_cached_prox"):
        env._cached_prox = env._compute_obstacle_proximity()


def _collect_snn_telemetry(env, agent, agent_type: str, cond_name: str, seed: int):
    import brian2 as b2

    has_snn = hasattr(env, "neurons") and hasattr(env, "synapses")
    if not has_snn:
        return {}, {}

    # 1. Run 8.0s Stage-4 Criticality & Spike Raster Evaluation Episode
    env.set_curriculum_stage(4)
    opts = {"preserve_weights": True}
    obs, _ = env.reset(seed=seed + 50000, options=opts)
    _set_eval_mode(agent, agent_type, True)

    step_rates = []
    step_atp = []
    step_sdm_sparsity = []
    prev_spikes = int(len(env.spike_mon.t))
    dt_step_sec = float(env.step_duration / b2.second)

    for _ in range(80):
        action = _select_action(agent, agent_type, obs)
        next_obs, reward, done, trunc, info = env.step(action)
        cur_spikes = int(len(env.spike_mon.t))
        delta_spikes = cur_spikes - prev_spikes
        prev_spikes = cur_spikes

        step_rates.append(float(delta_spikes / (env.n_neurons * dt_step_sec)))
        step_atp.append(float(np.mean(env.neurons.E)))
        # Measure SDM sparse expansion occupancy in [0..255]
        sdm_spikes = len(set(int(x) for x in env._cached_recent_spikes_i if x < env.n_sdm)) if hasattr(env, "_cached_recent_spikes_i") else 0
        step_sdm_sparsity.append(float(sdm_spikes / max(1, env.n_sdm)))
        obs = next_obs

    spike_t_ms = np.array(env.spike_mon.t / b2.ms, dtype=np.float32)
    spike_i = np.array(env.spike_mon.i, dtype=np.int16)

    # 2. Extract synaptic weight spectrum
    w_exc_list = [np.array(S.w, dtype=np.float32) for S in env.synapses[:-1]]
    w_exc = np.concatenate(w_exc_list) if w_exc_list else np.array([], dtype=np.float32)
    w_inh = np.array(env.synapses[-1].w, dtype=np.float32) if env.synapses else np.array([], dtype=np.float32)

    # 3. 4-Direction Motor DSI Probe on trained connectome
    dsi_matrix = np.zeros((4, 4), dtype=np.float64)
    motor_slices = [env.motor_up, env.motor_down, env.motor_left, env.motor_right]
    for action_dir in range(4):
        env.set_curriculum_stage(4)
        env.reset(seed=seed + 60000 + action_dir, options={"preserve_weights": True})
        start_spikes = int(len(env.spike_mon.t))
        for _ in range(10):
            env.step(action_dir)
        new_i = np.array(env.spike_mon.i[start_spikes:], dtype=np.int32)
        probe_sec = 10.0 * dt_step_sec
        for m_idx, (lo, hi) in enumerate(motor_slices):
            cnt = int(np.sum((new_i >= lo) & (new_i < hi)))
            dsi_matrix[action_dir, m_idx] = cnt / (25.0 * probe_sec)

    # Preferred vs null per driven direction
    dsi_per_dir = [
        compute_dsi(dsi_matrix[d]) for d in range(4)
    ]
    dsi_overall = float(np.mean(dsi_per_dir))

    _set_eval_mode(agent, agent_type, False)

    arrays = {
        "spike_t_ms": spike_t_ms,
        "spike_i": spike_i,
        "step_rates_hz": np.asarray(step_rates, dtype=np.float32),
        "step_atp": np.asarray(step_atp, dtype=np.float32),
        "step_sdm_sparsity": np.asarray(step_sdm_sparsity, dtype=np.float32),
        "w_exc": w_exc,
        "w_inh": w_inh,
        "dsi_matrix": dsi_matrix,
    }
    summary = {
        "condition": cond_name,
        "seed": seed,
        "eval_mean_rate_hz": float(np.mean(step_rates)) if step_rates else 0.0,
        "eval_final_atp": float(step_atp[-1]) if step_atp else 1.0,
        "mean_sdm_sparsity": float(np.mean(step_sdm_sparsity)) if step_sdm_sparsity else 0.0,
        "w_exc_mean": float(np.mean(w_exc)) if len(w_exc) else 0.0,
        "w_exc_zero_frac": float(np.mean(w_exc < 0.1)) if len(w_exc) else 0.0,
        "w_inh_mean": float(np.mean(w_inh)) if len(w_inh) else 0.0,
        "dsi_matrix": dsi_matrix.tolist(),
        "dsi_per_dir": dsi_per_dir,
        "dsi_overall": dsi_overall,
    }
    return arrays, summary


def _evaluate_ood_topologies(env, agent, agent_type: str, seed: int, n_eval: int = 10):
    ood_summary = {}
    _set_eval_mode(agent, agent_type, True)
    has_snn = hasattr(env, "neurons")

    for topo_name, topo_defs in NOVEL_TOPOLOGIES.items():
        returns = []
        successes = []
        dists = []
        for ep_idx in range(n_eval):
            env.set_curriculum_stage(4)
            opts = {"preserve_weights": True} if has_snn else None
            obs, _ = env.reset(seed=seed + 70000 + ep_idx, options=opts)
            _apply_topology(env, topo_defs)
            obs = env._get_obs()
            ep_ret = 0.0
            goal = 0
            final_d = 1.0
            for _ in range(80):
                action = _select_action(agent, agent_type, obs)
                next_obs, r, done, trunc, info = env.step(action)
                ep_ret += float(r)
                final_d = float(info.get("distance", 1.0))
                obs = next_obs
                if done:
                    goal = 1
                    break
            returns.append(ep_ret)
            successes.append(goal)
            dists.append(final_d)
        ood_summary[topo_name] = {
            "mean_return": float(np.mean(returns)),
            "goal_rate": float(np.mean(successes)),
            "mean_final_dist": float(np.mean(dists)),
        }

    _set_eval_mode(agent, agent_type, False)
    return ood_summary


def _run_ppo_job(cond_name: str, spec: dict, seed: int, total_episodes: int, steps_per_episode: int,
                 out_csv: str, out_json: str):
    import torch
    torch.set_num_threads(1)
    import gymnasium as gym
    from stable_baselines3.common.callbacks import BaseCallback
    from organoid_rl.agents.sb3_baselines import create_ppo_agent
    from organoid_rl.utils.seeding import set_seed

    set_seed(seed)
    base_env = _build_env(spec, seed)

    rows = []

    class _CurriculumEpisodeWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.ep_idx = 0
            self.step_in_ep = 0
            self.ep_ret = 0.0
            self.ep_t0 = time.perf_counter()

        def reset(self, **kwargs):
            stage = get_curriculum_stage(min(self.ep_idx, total_episodes - 1), total_episodes)
            self.env.set_curriculum_stage(stage)
            self.step_in_ep = 0
            self.ep_ret = 0.0
            self.ep_t0 = time.perf_counter()
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            self.step_in_ep += 1
            self.ep_ret += float(r)
            if self.step_in_ep >= steps_per_episode and not term:
                trunc = True
            if term or trunc:
                if len(rows) < total_episodes:
                    stage = get_curriculum_stage(self.ep_idx, total_episodes)
                    rows.append({
                        "condition": cond_name,
                        "seed": seed,
                        "episode": self.ep_idx,
                        "stage": stage,
                        "return": round(self.ep_ret, 6),
                        "length": self.step_in_ep,
                        "goal_reached": 1 if term else 0,
                        "success": 1 if (term or self.ep_ret > 30.0) else 0,
                        "final_dist": round(float(info.get("distance", 1.0)), 6),
                        "mean_rate_hz": 0.0,
                        "peak_rate_hz": 0.0,
                        "total_spikes": 0,
                        "mean_atp": 1.0,
                        "mean_weight": 0.0,
                        "weight_std": 0.0,
                        "active_syn_frac": 0.0,
                        "seizure_clamps": 0,
                        "coma_kicks": 0,
                        "wall_sec": round(time.perf_counter() - self.ep_t0, 4),
                    })
                self.ep_idx += 1
            return obs, r, term, trunc, info

    class _StopAtEpisodesCallback(BaseCallback):
        def _on_step(self) -> bool:
            return len(rows) < total_episodes

    wrapped_env = _CurriculumEpisodeWrapper(base_env)
    model = create_ppo_agent(wrapped_env, learning_rate=3e-4, batch_size=64, n_steps=512, seed=seed)
    model.learn(total_timesteps=total_episodes * steps_per_episode + 512, callback=_StopAtEpisodesCallback())

    rows = rows[:total_episodes]
    ood_summary = _evaluate_ood_topologies(base_env, model, "ppo", seed) if spec.get("eval_ood", False) else {}

    _atomic_write_csv(out_csv, rows)
    _atomic_write_json(out_json, {"condition": cond_name, "seed": seed, "ood_topologies": ood_summary})
    return cond_name, seed, len(rows), float(np.mean([r["return"] for r in rows[-min(50, len(rows)):]]))


def _run_single_job(task_args):
    cond_name, seed, total_episodes, steps_per_episode, out_dir = task_args

    # Isolate worker temp cache outside OneDrive
    worker_tmp = os.path.join(tempfile.gettempdir(), f"organoid_worker_{os.getpid()}")
    os.makedirs(worker_tmp, exist_ok=True)
    os.environ["TMP"] = worker_tmp
    os.environ["TEMP"] = worker_tmp
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    spec = CONDITIONS[cond_name]
    out_csv = os.path.join(out_dir, "runs", f"{cond_name}_seed_{seed}.csv")
    out_npz = os.path.join(out_dir, "telemetry", f"{cond_name}_seed_{seed}.npz")
    out_json = os.path.join(out_dir, "telemetry", f"{cond_name}_seed_{seed}.json")

    if os.path.exists(out_csv) and os.path.exists(out_json):
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                existing_rows = sum(1 for _ in f) - 1
            if existing_rows == total_episodes and (not spec["has_snn"] or os.path.exists(out_npz)):
                return cond_name, seed, existing_rows, None  # Skipped completed
        except Exception:
            pass

    if spec["agent_type"] == "ppo":
        return _run_ppo_job(cond_name, spec, seed, total_episodes, steps_per_episode, out_csv, out_json)

    import torch
    torch.set_num_threads(1)

    if spec["has_snn"]:
        import brian2 as b2
        b2.start_scope()

    from organoid_rl.utils.seeding import set_seed
    set_seed(seed)

    env = _build_env(spec, seed)
    agent_type = spec["agent_type"]
    if agent_type == "d3qn":
        from organoid_rl.agents.dqn_agent import DQNAgent
        agent = DQNAgent(
            obs_dim=env.obs_dim,
            n_actions=env.action_space.n,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            n_step=3,
            batch_size=64,
            buffer_size=50000,
            her_k=4,
        )
    elif agent_type == "mlp_dqn":
        from organoid_rl.agents.baseline_agent import MLPBaselineAgent
        agent = MLPBaselineAgent(obs_dim=env.obs_dim, n_actions=env.action_space.n, lr=3e-4, gamma=0.99)
    elif agent_type == "random":
        from organoid_rl.agents.baseline_agent import RandomAgent
        agent = RandomAgent(obs_dim=env.obs_dim, n_actions=env.action_space.n, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    has_snn = hasattr(env, "neurons") and hasattr(env, "synapses")
    dt_step_sec = float(env.step_duration / b2.second) if has_snn else 0.1

    rows = []
    for ep in range(total_episodes):
        ep_t0 = time.perf_counter()
        stage = get_curriculum_stage(ep, total_episodes)
        env.set_curriculum_stage(stage)

        opts = {"preserve_weights": True} if (has_snn and ep > 0) else None
        obs, _ = env.reset(seed=seed + ep if not has_snn else None, options=opts)

        total_reward = 0.0
        episode_transitions = []
        step_rates = []
        done = False
        final_dist = 1.0
        prev_spikes = int(len(env.spike_mon.t)) if has_snn else 0

        for s in range(steps_per_episode):
            action = agent.choose_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)

            agent.store_transition(obs, action, float(reward), next_obs, bool(done))
            if agent_type == "d3qn":
                episode_transitions.append((obs.copy(), action, float(reward), next_obs.copy(), bool(done)))

            if s % 4 == 0:
                agent.learn()

            if has_snn:
                cur_spikes = int(len(env.spike_mon.t))
                step_spk = cur_spikes - prev_spikes
                prev_spikes = cur_spikes
                step_rates.append(step_spk / (env.n_neurons * dt_step_sec))

            obs = next_obs
            total_reward += float(reward)
            final_dist = float(info.get("distance", 1.0))

            if done:
                break

        if agent_type == "d3qn":
            agent.flush_n_step_buffer()
            if not done and len(episode_transitions) > 5:
                agent.apply_her(episode_transitions)

        if has_snn:
            apply_inhibitory_homeostasis(env)
            if ep % 10 == 0 and ep > 0:
                env.apply_structural_plasticity()
            w_all = np.concatenate([np.array(S.w, dtype=np.float64) for S in env.synapses])
            mean_w = float(np.mean(w_all))
            std_w = float(np.std(w_all))
            active_frac = float(np.mean(w_all >= 0.1))
            mean_atp = float(np.mean(env.neurons.E))
            counters = getattr(env, "gar_state_counters", {})
            mean_rate = float(np.mean(step_rates)) if step_rates else 0.0
            peak_rate = float(np.max(step_rates)) if step_rates else 0.0
            total_spikes = int(len(env.spike_mon.t))
        else:
            mean_w, std_w, active_frac, mean_atp = 0.0, 0.0, 0.0, 1.0
            counters = {}
            mean_rate, peak_rate, total_spikes = 0.0, 0.0, 0

        rows.append({
            "condition": cond_name,
            "seed": seed,
            "episode": ep,
            "stage": stage,
            "return": round(total_reward, 6),
            "length": s + 1,
            "goal_reached": 1 if done else 0,
            "success": 1 if (done or total_reward > 30.0) else 0,
            "final_dist": round(final_dist, 6),
            "mean_rate_hz": round(mean_rate, 4),
            "peak_rate_hz": round(peak_rate, 4),
            "total_spikes": total_spikes,
            "mean_atp": round(mean_atp, 5),
            "mean_weight": round(mean_w, 5),
            "weight_std": round(std_w, 5),
            "active_syn_frac": round(active_frac, 5),
            "seizure_clamps": int(counters.get("seizure_clamps", 0)),
            "coma_kicks": int(counters.get("coma_kicks", 0)),
            "wall_sec": round(time.perf_counter() - ep_t0, 4),
        })

    # Post-training mechanistic & OOD evaluations
    telem_arrays, telem_summary = _collect_snn_telemetry(env, agent, agent_type, cond_name, seed)
    ood_summary = _evaluate_ood_topologies(env, agent, agent_type, seed) if spec.get("eval_ood", False) else {}
    telem_summary["condition"] = cond_name
    telem_summary["seed"] = seed
    telem_summary["ood_topologies"] = ood_summary

    _atomic_write_csv(out_csv, rows)
    _atomic_write_json(out_json, telem_summary)
    if telem_arrays:
        _atomic_write_npz(out_npz, **telem_arrays)

    last_ret = float(np.mean([r["return"] for r in rows[-min(50, len(rows)):]]))
    return cond_name, seed, len(rows), last_ret


def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-Seed OrganoidEnv Benchmark Runner")
    parser.add_argument("--workers", type=int, default=4, help="Parallel ProcessPoolExecutor workers (default: 4)")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed (default: 500)")
    parser.add_argument("--steps", type=int, default=80, help="Max steps per episode (default: 80)")
    parser.add_argument("--out-dir", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional subset of condition names")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed list override")
    parser.add_argument("--timing-test", action="store_true", help="Run single C0 seed for timing projection")
    args = parser.parse_args()

    if args.timing_test:
        n_eps = min(args.episodes, 50) if args.episodes == 500 else args.episodes
        t0 = time.perf_counter()
        c, s, n_rows, ret = _run_single_job(("c0_full_model", 42, n_eps, args.steps, "results/benchmarks/_timing_probe"))
        dt = time.perf_counter() - t0
        sec_per_ep = dt / max(1, n_rows)
        est_500_sec = sec_per_ep * 500
        est_50_snn_4w_min = (est_500_sec * 50) / (4 * 60.0)
        print(
            f"[TIMING PROBE] {c} seed={s} ({n_rows} eps): {dt:.2f}s "
            f"({sec_per_ep:.3f}s/ep, last_ret={ret:.2f}). "
            f"Proj 500ep={est_500_sec:.1f}s/seed; 50 SNN seeds @ 4w ~ {est_50_snn_4w_min:.1f} min.",
            flush=True,
        )
        return

    cond_names = args.conditions if args.conditions else list(CONDITIONS.keys())
    tasks = []
    for cond_name in cond_names:
        if cond_name not in CONDITIONS:
            raise ValueError(f"Unknown condition '{cond_name}'. Available: {list(CONDITIONS.keys())}")
        seeds = args.seeds if args.seeds else CONDITIONS[cond_name]["seeds"]
        for seed in seeds:
            tasks.append((cond_name, seed, args.episodes, args.steps, args.out_dir))

    print(f"Dispatching {len(tasks)} runs ({args.episodes} eps x {args.steps} steps) across {args.workers} workers...", flush=True)
    t0 = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_single_job, t): t for t in tasks}
        for fut in as_completed(futures):
            cond_name, seed, n_rows, last_ret = fut.result()
            completed += 1
            status = "SKIPPED (cached)" if last_ret is None else f"last50_ret={last_ret:.2f}"
            elapsed = time.perf_counter() - t0
            print(f"[{completed}/{len(tasks)}] {cond_name} seed={seed} ({n_rows} eps) -> {status} [{elapsed:.1f}s elapsed]", flush=True)


if __name__ == "__main__":
    main()
