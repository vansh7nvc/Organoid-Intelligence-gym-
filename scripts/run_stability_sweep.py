"""
2D Stability Phase Space Parameter Sweep for OrganoidEnv.

Sweeps external stimulation amplitude I_stim in [5, 10, 20, 30, 50] pA against
inhibitory synaptic strength w_inh in [5, 10, 20, 30] nS (5 x 4 = 20 grid points),
comparing:
  1. Open-Loop + Metabolism (use_stabilizer=False, use_metabolism=True)
  2. Closed-Loop GAR + Metabolism (use_stabilizer=True, use_metabolism=True)
  3. Unregulated Open-Loop (use_stabilizer=False, use_metabolism=False)

Each mode runs in its own worker with a single persistent OrganoidEnv instance,
calling env.reset(seed=42) between grid points to avoid rebuilding SymPy/Brian2.
Saves JSON & CSV under results/benchmarks/stability_sweep.{json,csv}.
"""

import os
import sys
import json
import csv
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


I_STIM_LEVELS = [5.0, 10.0, 20.0, 30.0, 50.0]
W_INH_LEVELS = [5.0, 10.0, 20.0, 30.0]


def classify_dynamical_regime(mean_firing_rate_hz: float) -> str:
    """
    Classifies dynamical regime per Master Verifier Plan §4.5:
    - Quiescent Death: < 0.5 Hz
    - Homeostatic: 2.0 - 20.0 Hz
    - Epileptiform Hyperactivity: > 100.0 Hz
    - Intermediate/Sub-optimal otherwise
    """
    if mean_firing_rate_hz < 0.5:
        return "quiescent"
    elif 2.0 <= mean_firing_rate_hz <= 20.0:
        return "homeostatic"
    elif mean_firing_rate_hz > 100.0:
        return "epileptiform"
    return "intermediate"


def _run_mode_grid(args):
    use_stabilizer, use_metabolism, n_steps, seed = args
    import brian2 as b2
    b2.start_scope()

    from organoid_rl.environment.core import OrganoidEnv
    from organoid_rl.environment.stimulator import stabilize_network_activity

    env = OrganoidEnv(
        use_stabilizer=use_stabilizer,
        use_metabolism=use_metabolism,
        use_plasticity=False,
        seed=seed,
    )

    dt_step_sec = float(env.step_duration / b2.second)
    mode_results = []

    for i_stim in I_STIM_LEVELS:
        for w_inh in W_INH_LEVELS:
            env.reset(seed=seed)

            for idx, S in enumerate(env.synapses):
                if getattr(S, "_is_inhibitory", False):
                    S.w = float(w_inh)
                    S.w0 = float(w_inh)
                    if getattr(env, "_fast_runner", None) is not None:
                        env._fast_runner.g_w0[idx] = float(w_inh)

            step_rates = []
            atp_means = []
            prev_spikes = int(len(env.spike_mon.t))

            for _ in range(n_steps):
                obs = env._get_obs()
                env._stimulate_sdm(obs)
                current_I = np.array(env.neurons.I, dtype=np.float64)
                stimulated_mask = current_I > 0.0
                new_I = np.zeros_like(current_I)
                new_I[stimulated_mask] = float(i_stim)
                # Background drive on excitatory population (neurons 0..299)
                new_I[:300] = np.maximum(new_I[:300], 0.35 * float(i_stim))
                env.neurons.I = new_I

                if env.use_stabilizer:
                    stabilize_network_activity(env)

                if getattr(env, "_fast_runner", None) is not None:
                    p_rate = 10.0 if getattr(env.bg_noise, "active", True) else 0.0
                    env._fast_runner.run_duration(env.step_duration, poisson_rate_hz=p_rate)
                else:
                    env.network.run(env.step_duration)

                total_spikes = int(len(env.spike_mon.t))
                step_spikes = total_spikes - prev_spikes
                prev_spikes = total_spikes

                step_rate_hz = step_spikes / (env.n_neurons * dt_step_sec)
                step_rates.append(float(step_rate_hz))
                atp_means.append(float(np.mean(env.neurons.E)))

            mean_rate = float(np.mean(step_rates))
            peak_rate = float(np.max(step_rates))
            std_rate = float(np.std(step_rates))
            final_atp = float(atp_means[-1]) if atp_means else 1.0
            counters = dict(
                getattr(env, "gar_state_counters", {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0})
            )

            mode_results.append({
                "i_stim": float(i_stim),
                "w_inh": float(w_inh),
                "use_stabilizer": bool(use_stabilizer),
                "use_metabolism": bool(use_metabolism),
                "mean_rate_hz": mean_rate,
                "peak_rate_hz": peak_rate,
                "std_rate_hz": std_rate,
                "final_mean_atp": final_atp,
                "regime_mean": classify_dynamical_regime(mean_rate),
                "regime_peak": classify_dynamical_regime(peak_rate),
                "seizure_clamps": int(counters.get("seizure_clamps", 0)),
                "coma_kicks": int(counters.get("coma_kicks", 0)),
                "step_rates_hz": step_rates,
            })

    return mode_results


def main():
    os.makedirs("results/benchmarks", exist_ok=True)
    t0 = time.perf_counter()

    modes = [
        (False, False),  # Unregulated Open-Loop (no GAR, no ATP fatigue)
        (False, True),   # Open-Loop + Metabolism
        (True, False),   # GAR-Only (GAR active, no ATP fatigue)
        (True, True),    # Closed-Loop GAR + Metabolism
    ]
    tasks = [(use_stab, use_met, 50, 42) for use_stab, use_met in modes]

    print(f"Running 2D Stability Phase Sweep (60 grid configs across {len(tasks)} parallel workers)...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        for mode_res in executor.map(_run_mode_grid, tasks):
            results.extend(mode_res)

    results.sort(key=lambda r: (r["use_stabilizer"], r["use_metabolism"], r["i_stim"], r["w_inh"]))

    json_path = "results/benchmarks/stability_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "i_stim_levels": I_STIM_LEVELS,
                "w_inh_levels": W_INH_LEVELS,
                "duration_sec": 5.0,
                "results": results,
            },
            f,
            indent=2,
        )

    csv_path = "results/benchmarks/stability_sweep.csv"
    fieldnames = [
        "use_stabilizer", "use_metabolism", "i_stim", "w_inh",
        "mean_rate_hz", "peak_rate_hz", "std_rate_hz", "final_mean_atp",
        "regime_mean", "regime_peak", "seizure_clamps", "coma_kicks",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    elapsed = time.perf_counter() - t0
    print(f"2D Stability Sweep complete in {elapsed:.2f}s -> {json_path}, {csv_path}", flush=True)

    for stab, met in modes:
        sub = [r for r in results if r["use_stabilizer"] == stab and r["use_metabolism"] == met]
        counts = {}
        for r in sub:
            reg = r["regime_mean"]
            counts[reg] = counts.get(reg, 0) + 1
        print(f"  [GAR={stab}, Met={met}] Regime counts (mean rate): {counts}", flush=True)


if __name__ == "__main__":
    main()
