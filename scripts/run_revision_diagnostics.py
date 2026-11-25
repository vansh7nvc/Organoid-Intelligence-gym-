"""
Revision Diagnostics for OrganoidEnv IEEE TNNLS Manuscript:
1. Alpha Power-Law Goodness-of-Fit (Clauset et al. 2009 KS + LLR vs Exponential)
2. Long-Horizon (30s and 60s) Stability & GAR Trigger Counts
3. Return Decomposition, Step Displacement, Collisions, ||dw||, and GAR Triggers per Condition
"""
import os
import sys
import json
import glob
import numpy as np
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.analyze_benchmarks import compute_avalanche_metrics


def ks_powerlaw_and_llr(sizes: np.ndarray, s_min: int = 2, n_boot: int = 100):
    """Discrete power-law MLE, KS distance, bootstrap p-value, and LLR vs exponential."""
    s = np.asarray([x for x in sizes if x >= s_min], dtype=np.float64)
    if len(s) < 20:
        return {"n": int(len(s)), "alpha": 0.0, "ks": 1.0, "p_gof": 0.0, "llr_vs_exp": 0.0, "p_llr": 1.0}

    n = len(s)
    alpha = 1.0 + n / np.sum(np.log(s / (s_min - 0.5)))

    # Discrete power-law PMF & CDF over [s_min .. max(s)]
    max_s = int(np.max(s))
    vals = np.arange(s_min, max_s + 1, dtype=np.float64)
    # Approximate Hurwitz zeta normalization using tail extension
    weights = vals ** (-alpha)
    tail_approx = ((max_s + 0.5) ** (1.0 - alpha)) / (alpha - 1.0)
    z_norm = np.sum(weights) + tail_approx
    cdf_model = np.cumsum(weights) / z_norm

    uniq, counts = np.unique(s, return_counts=True)
    emp_cdf_at_vals = np.zeros_like(vals)
    idx_map = (uniq.astype(int) - s_min)
    emp_cdf_at_vals[idx_map] = counts / n
    emp_cdf = np.cumsum(emp_cdf_at_vals)
    ks_obs = float(np.max(np.abs(emp_cdf - cdf_model)))

    # LLR vs discrete exponential: P(x) = (1 - exp(-lam)) * exp(-lam * (x - s_min))
    mean_excess = float(np.mean(s - s_min))
    lam = float(np.log(1.0 + 1.0 / max(mean_excess, 1e-9)))
    ll_pl = np.sum(-alpha * np.log(s) - np.log(z_norm))
    ll_exp = np.sum(np.log(1.0 - np.exp(-lam)) - lam * (s - s_min))
    # Per-sample log-likelihood difference
    d_i = (-alpha * np.log(s) - np.log(z_norm)) - (np.log(1.0 - np.exp(-lam)) - lam * (s - s_min))
    llr = float(np.sum(d_i))
    std_d = float(np.std(d_i, ddof=1))
    if std_d > 1e-12:
        z_llr = llr / (np.sqrt(n) * std_d)
        p_llr = float(2.0 * (1.0 - stats.norm.cdf(abs(z_llr))))
    else:
        p_llr = 1.0

    # Synthetic KS bootstrap (fast continuous-rounded discrete sampling)
    rng = np.random.default_rng(42)
    probs = weights / np.sum(weights)
    ks_boot = []
    for _ in range(n_boot):
        sim = rng.choice(vals, size=n, replace=True, p=probs)
        a_b = 1.0 + n / np.sum(np.log(sim / (s_min - 0.5)))
        w_b = vals ** (-a_b)
        z_b = np.sum(w_b) + ((max_s + 0.5) ** (1.0 - a_b)) / max(a_b - 1.0, 1e-6)
        cdf_b = np.cumsum(w_b) / z_b
        u_b, c_b = np.unique(sim, return_counts=True)
        ec_b = np.zeros_like(vals)
        ec_b[(u_b.astype(int) - s_min)] = c_b / n
        ks_boot.append(float(np.max(np.abs(np.cumsum(ec_b) - cdf_b))))
    p_gof = float(np.mean(np.asarray(ks_boot) >= ks_obs))

    return {
        "n": int(n),
        "alpha": float(alpha),
        "ks": ks_obs,
        "p_gof": p_gof,
        "llr_vs_exp": llr,
        "p_llr": p_llr,
    }


def run_alpha_gof():
    telem_dir = "results/benchmarks/telemetry"
    out = {}
    for cond in ["c0_full_model", "c1_frozen_stdp", "c2_random_snn", "c3_no_metabolism", "c4_no_gar", "b3_simple_snn_lif"]:
        sizes = []
        for f in sorted(glob.glob(os.path.join(telem_dir, f"{cond}_seed*.npz"))):
            npz = np.load(f)
            if "spike_t_ms" in npz and "spike_i" in npz:
                av = compute_avalanche_metrics(npz["spike_t_ms"], npz["spike_i"])
                sizes.extend(av["avalanche_sizes"])
        out[cond] = ks_powerlaw_and_llr(np.asarray(sizes), s_min=2, n_boot=100)
    return out


def run_long_horizon_stability():
    import brian2 as b2
    b2.start_scope()
    from organoid_rl.environment.core import OrganoidEnv
    from organoid_rl.environment.stimulator import stabilize_network_activity

    results = []
    configs = [
        (False, False, 50.0, 5.0),
        (False, True, 50.0, 5.0),
        (True, False, 50.0, 5.0),
        (True, True, 50.0, 5.0),
        (True, True, 5.0, 30.0),
    ]
    for use_stab, use_met, i_stim, w_inh in configs:
        b2.start_scope()
        env = OrganoidEnv(use_stabilizer=use_stab, use_metabolism=use_met, use_plasticity=False, seed=42)
        env.reset(seed=42)
        for idx, S in enumerate(env.synapses):
            if getattr(S, "_is_inhibitory", False):
                S.w = float(w_inh)
                S.w0 = float(w_inh)
                if getattr(env, "_fast_runner", None) is not None:
                    env._fast_runner.g_w0[idx] = float(w_inh)

        dt_step_sec = float(env.step_duration / b2.second)
        step_rates = []
        prev_spikes = int(len(env.spike_mon.t))
        # 600 steps = 60 seconds
        for _ in range(600):
            obs = env._get_obs()
            env._stimulate_sdm(obs)
            current_I = np.array(env.neurons.I, dtype=np.float64)
            new_I = np.zeros_like(current_I)
            new_I[current_I > 0.0] = float(i_stim)
            new_I[:300] = np.maximum(new_I[:300], 0.35 * float(i_stim))
            env.neurons.I = new_I
            if env.use_stabilizer:
                stabilize_network_activity(env)
            if getattr(env, "_fast_runner", None) is not None:
                env._fast_runner.run_duration(env.step_duration, poisson_rate_hz=10.0)
            else:
                env.network.run(env.step_duration)
            tot = int(len(env.spike_mon.t))
            step_rates.append((tot - prev_spikes) / (env.n_neurons * dt_step_sec))
            prev_spikes = tot

        counters = dict(getattr(env, "gar_state_counters", {}))
        results.append({
            "use_stabilizer": use_stab,
            "use_metabolism": use_met,
            "i_stim": i_stim,
            "w_inh": w_inh,
            "rate_0_5s_hz": float(np.mean(step_rates[:50])),
            "rate_5_30s_hz": float(np.mean(step_rates[50:300])),
            "rate_30_60s_hz": float(np.mean(step_rates[300:600])),
            "final_atp": float(np.mean(env.neurons.E)),
            "gar_counters": counters,
        })
    return results


def run_episode_decomposition():
    import brian2 as b2
    from organoid_rl.environment.core import OrganoidEnv

    conds = [
        ("c0_full_model", {}),
        ("c1_frozen_stdp", {"use_plasticity": False}),
        ("c3_no_metabolism", {"use_metabolism": False}),
        ("c4_no_gar", {"use_stabilizer": False}),
    ]
    out = {}
    for name, kw in conds:
        b2.start_scope()
        env = OrganoidEnv(seed=42, **kw)
        env.set_curriculum_stage(4)
        rng = np.random.default_rng(42)
        disps = []
        colls = []
        dws = []
        gar_kicks = 0
        gar_clamps = 0
        returns = []
        for ep in range(10):
            env.reset(seed=42 + ep, options={"preserve_weights": ep > 0})
            w_before = np.concatenate([np.array(S.w, copy=True) for S in env.synapses])
            ep_ret = 0.0
            ep_coll = 0
            for _ in range(80):
                prev_pos = env.cursor_pos.copy()
                obs, r, term, trunc, info = env.step(int(rng.integers(0, 8)))
                disps.append(float(np.linalg.norm(env.cursor_pos - prev_pos)))
                if r <= -10.0:
                    ep_coll += 1
                ep_ret += float(r)
                if term or trunc:
                    break
            w_after = np.concatenate([np.array(S.w, copy=True) for S in env.synapses])
            dws.append(float(np.linalg.norm(w_after - w_before)))
            colls.append(ep_coll)
            returns.append(ep_ret)
        c = dict(getattr(env, "gar_state_counters", {}))
        out[name] = {
            "mean_step_disp": float(np.mean(disps)),
            "max_step_disp": float(np.max(disps)),
            "mean_collisions_per_ep": float(np.mean(colls)),
            "mean_dw_l2_per_ep": float(np.mean(dws)),
            "mean_return": float(np.mean(returns)),
            "gar_counters_10ep": c,
        }
    return out


if __name__ == "__main__":
    report = {
        "alpha_gof": run_alpha_gof(),
        "long_horizon_stability_60s": run_long_horizon_stability(),
        "episode_decomposition": run_episode_decomposition(),
    }
    out_path = "results/revision_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
