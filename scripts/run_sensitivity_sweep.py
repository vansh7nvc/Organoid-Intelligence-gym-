"""
Biophysical Sensitivity Sweep for OrganoidEnv (Stage 5 / Task B7).

Sweeps across:
  1. Spike Energy Cost Delta_E in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
  2. Recovery Time Constant tau_rec in [500, 750, 1000, 1500, 2000] ms
  3. GAR Seizure Clamp Threshold in [2500, 4000, 5000, 7500, 10000] spikes (50 - 200 Hz)
  4. GAR Coma Kick Threshold in [25, 50, 100] spikes (0.5 - 2.0 Hz)

Evaluates steady-state population firing rates, ATP reserves, empirical vs theoretical
analytic rate upper bounds f_max = 1 / (Delta_E * tau_rec), and homeostatic regime stability.
Saves structured JSON and CSV to results/benchmarks/sensitivity_sweep.{json,csv}.
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


DELTA_E_LEVELS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
TAU_REC_LEVELS = [500.0, 750.0, 1000.0, 1500.0, 2000.0]
GAR_SEIZURE_LEVELS = [2500, 4000, 5000, 7500, 10000]
GAR_COMA_LEVELS = [25, 50, 100]


def classify_regime(rate_hz: float) -> str:
    if rate_hz < 0.5:
        return "quiescent"
    elif 2.0 <= rate_hz <= 20.0:
        return "homeostatic"
    elif rate_hz > 100.0:
        return "epileptiform"
    return "intermediate"


def _eval_single_point(args):
    delta_e, tau_rec, seizure_th, coma_th, n_steps, seed = args
    import brian2 as b2
    b2.start_scope()

    from organoid_rl.environment.core import OrganoidEnv, OrganoidConfig
    from organoid_rl.environment.stimulator import stabilize_network_activity

    cfg = OrganoidConfig(
        spike_cost=float(delta_e),
        tau_recovery_exc=float(tau_rec),
        tau_recovery_inh=float(tau_rec * 0.5),
        gar_seizure_threshold=int(seizure_th),
        gar_coma_threshold=int(coma_th),
        use_stabilizer=True,
        use_metabolism=True,
        use_plasticity=False,
    )
    env = OrganoidEnv(config=cfg, seed=seed)
    env.reset(seed=seed)

    dt_step_sec = float(env.step_duration / b2.second)
    step_rates = []
    atp_means = []
    prev_spikes = int(len(env.spike_mon.t))

    for step in range(n_steps):
        obs = env._get_obs()
        action = step % 8
        obs, reward, term, trunc, info = env.step(action)

        cur_spikes = int(len(env.spike_mon.t))
        step_spk = cur_spikes - prev_spikes
        prev_spikes = cur_spikes

        rate_hz = step_spk / (env.n_neurons * dt_step_sec)
        step_rates.append(float(rate_hz))
        atp_means.append(float(np.mean(env.neurons.E)))

    mean_rate = float(np.mean(step_rates))
    peak_rate = float(np.max(step_rates))
    final_atp = float(atp_means[-1]) if atp_means else 1.0
    counters = dict(getattr(env, "gar_state_counters", {}))

    tau_rec_sec = float(tau_rec) * 0.001
    theoretical_max_hz = float(1.0 / (delta_e * tau_rec_sec))

    return {
        "delta_e": float(delta_e),
        "tau_rec_ms": float(tau_rec),
        "gar_seizure_threshold": int(seizure_th),
        "gar_coma_threshold": int(coma_th),
        "mean_rate_hz": round(mean_rate, 4),
        "peak_rate_hz": round(peak_rate, 4),
        "final_mean_atp": round(final_atp, 5),
        "theoretical_max_hz": round(theoretical_max_hz, 4),
        "bounded_by_theory": bool(mean_rate <= theoretical_max_hz * 1.05),
        "regime": classify_regime(mean_rate),
        "seizure_clamps": int(counters.get("seizure_clamps", 0)),
        "coma_kicks": int(counters.get("coma_kicks", 0)),
    }


def run_sensitivity_sweep(out_dir: str = "results/benchmarks", max_workers: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "sensitivity_sweep.json")
    out_csv = os.path.join(out_dir, "sensitivity_sweep.csv")

    tasks = []

    # Sweep 1: Delta_E x tau_rec grid (nominal GAR)
    for de in DELTA_E_LEVELS:
        for tr in TAU_REC_LEVELS:
            tasks.append((de, tr, 5000, 50, 40, 42))

    # Sweep 2: GAR Seizure Threshold (nominal Delta_E, tau_rec)
    for sth in GAR_SEIZURE_LEVELS:
        if (0.10, 1000.0, sth, 50, 40, 42) not in tasks:
            tasks.append((0.10, 1000.0, sth, 50, 40, 42))

    # Sweep 3: GAR Coma Threshold (nominal Delta_E, tau_rec)
    for cth in GAR_COMA_LEVELS:
        if (0.10, 1000.0, 5000, cth, 40, 42) not in tasks:
            tasks.append((0.10, 1000.0, 5000, cth, 40, 42))

    print(f"Executing biophysical sensitivity sweep ({len(tasks)} configurations across {max_workers} workers)...", flush=True)
    t0 = time.perf_counter()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_eval_single_point, tasks):
            results.append(res)

    results.sort(key=lambda r: (r["delta_e"], r["tau_rec_ms"], r["gar_seizure_threshold"], r["gar_coma_threshold"]))

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Biophysical parameter sensitivity sweep across Delta_E, tau_rec, and GAR thresholds",
        "delta_e_levels": DELTA_E_LEVELS,
        "tau_rec_levels_ms": TAU_REC_LEVELS,
        "gar_seizure_levels": GAR_SEIZURE_LEVELS,
        "gar_coma_levels": GAR_COMA_LEVELS,
        "n_configurations": len(results),
        "results": results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fields = [
        "delta_e", "tau_rec_ms", "gar_seizure_threshold", "gar_coma_threshold",
        "mean_rate_hz", "peak_rate_hz", "final_mean_atp", "theoretical_max_hz",
        "bounded_by_theory", "regime", "seizure_clamps", "coma_kicks",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    elapsed = time.perf_counter() - t0
    print(f"Sensitivity sweep completed in {elapsed:.2f}s -> Saved {out_json} and {out_csv}", flush=True)
    return payload


if __name__ == "__main__":
    run_sensitivity_sweep()
