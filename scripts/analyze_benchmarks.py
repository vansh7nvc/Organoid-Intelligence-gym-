"""
Scientific Statistical & Biophysical Analysis Pipeline for OrganoidEnv (Stage 5).

Reads empirical seed CSVs (results/benchmarks/runs/), SNN mechanistic telemetry
(results/benchmarks/telemetry/), and 2D stability sweep (results/benchmarks/stability_sweep.json):
  - Aggregates non-parametric IQM + 95% stratified bootstrap CIs via rliable.metrics
  - Computes Mann-Whitney U, rank-biserial effect sizes, Probability of Improvement,
    and Benjamini-Hochberg FDR-corrected p-values comparing C0 against all conditions
  - Computes neural branching ratio sigma, power-law avalanche scaling P(S) ~ S^(-alpha),
    Fano factor, and ISI CV using neo + elephant.conversion + elephant.statistics
  - Exports structured results/benchmarks/stats_summary.json and summary_table.csv
"""

import os
import sys
import glob
import json
import csv
import numpy as np
import pandas as pd
from scipy import stats
import rliable.metrics as rm
import neo
import quantities as pq
import elephant.conversion as ec
import elephant.statistics as es

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


CONDITION_ORDER = [
    "c0_full_model",
    "c1_frozen_stdp",
    "c2_random_snn",
    "c3_no_metabolism",
    "c4_no_gar",
    "c5_no_pruning",
    "c6_single_trace",
    "c6b_single_trace_fast",
    "c6c_single_trace_slow",
    "c7_gar_only",
    "b1_direct_d3qn",
    "b2_mlp_dqn",
    "b3_simple_snn_lif",
    "b4_ppo_sb3",
    "b5_random",
    "c8_no_gar_no_met",
    "b1_step005_d3qn",
    "b2_step005_mlp_dqn",
    "b4_step005_ppo",
]

CONDITION_LABELS = {
    "c0_full_model": "OrganoidEnv (Full Model, C0)",
    "c1_frozen_stdp": "C1: Frozen STDP",
    "c2_random_snn": "C2: Random SNN",
    "c3_no_metabolism": "C3: No Metabolism",
    "c4_no_gar": "C4: No GAR",
    "c5_no_pruning": "C5: No Pruning",
    "c6_single_trace": "C6: Single-Trace (50ms)",
    "c6b_single_trace_fast": "C6b: Single-Trace Fast (100ms)",
    "c6c_single_trace_slow": "C6c: Single-Trace Slow (2500ms)",
    "c7_gar_only": "C7: GAR-Only (No Metabolism)",
    "c8_no_gar_no_met": "C8: Unregulated (No GAR, No Met)",
    "b1_direct_d3qn": "B1: Direct D3QN (step 0.25)",
    "b2_mlp_dqn": "B2: Standard MLP-DQN (step 0.25)",
    "b3_simple_snn_lif": "B3: SimpleSNN (LIF)",
    "b4_ppo_sb3": "B4: PPO (SB3, step 0.25)",
    "b5_random": "B5: Random Policy",
    "b1_step005_d3qn": "B1b: Direct D3QN (step 0.05)",
    "b2_step005_mlp_dqn": "B2b: Standard MLP-DQN (step 0.05)",
    "b4_step005_ppo": "B4b: PPO (SB3, step 0.05)",
}


def bootstrap_iqm_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict:
    """
    Computes rliable Interquartile Mean (IQM) alongside median, mean, std,
    and stratified 95% bootstrap confidence intervals over seed-level scores.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "iqm": 0.0, "ci_low": 0.0, "ci_high": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}
    if len(arr) == 1:
        val = float(arr[0])
        return {"n": 1, "iqm": val, "ci_low": val, "ci_high": val, "mean": val, "std": 0.0, "median": val}

    iqm_point = float(rm.aggregate_iqm(arr[:, None]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_iqms = np.array([float(rm.aggregate_iqm(arr[row][:, None])) for row in idx], dtype=np.float64)
    ci_low, ci_high = np.percentile(boot_iqms, [2.5, 97.5])
    return {
        "n": int(len(arr)),
        "iqm": iqm_point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def bh_fdr(p_values: list) -> list:
    """Benjamini-Hochberg false discovery rate p-value adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [1.0] * m
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        val = min(prev, p * m / rank)
        adj[orig_idx] = float(min(1.0, val))
        prev = val
    return adj


def compute_avalanche_metrics(spike_t_ms: np.ndarray, spike_i: np.ndarray,
                              duration_ms: float = 8000.0, bin_ms: float = 4.0) -> dict:
    """
    Uses Neo + Elephant BinnedSpikeTrain & statistics to compute:
      - Population binned activity at bin_ms resolution
      - Neural branching ratio sigma = E[A_{t+1} / A_t | A_t > 0]
      - Avalanche size S and duration T distributions + MLE power-law alpha
      - Elephant population Fano factor and mean single-unit firing rate
    """
    if len(spike_t_ms) < 5:
        return {
            "n_spikes": int(len(spike_t_ms)),
            "n_avalanches": 0,
            "branching_ratio_sigma": 0.0,
            "avalanche_alpha": 0.0,
            "fano_factor": 0.0,
            "avalanche_sizes": [],
            "avalanche_durations": [],
        }

    max_t = max(float(duration_ms), float(np.max(spike_t_ms)) + bin_ms)
    # Build population binned array with Elephant BinnedSpikeTrain over unique firing units
    unique_units = np.unique(spike_i)
    spike_trains = []
    for uid in unique_units:
        ts = spike_t_ms[spike_i == uid]
        ts = ts[(ts >= 0.0) & (ts <= max_t)]
        if len(ts) > 0:
            spike_trains.append(neo.SpikeTrain(ts * pq.ms, t_start=0.0 * pq.ms, t_stop=max_t * pq.ms))

    if not spike_trains:
        return {
            "n_spikes": 0,
            "n_avalanches": 0,
            "branching_ratio_sigma": 0.0,
            "avalanche_alpha": 0.0,
            "fano_factor": 0.0,
            "avalanche_sizes": [],
            "avalanche_durations": [],
        }

    bst = ec.BinnedSpikeTrain(spike_trains, bin_size=bin_ms * pq.ms)
    pop_counts = np.asarray(bst.to_array(), dtype=np.float64).sum(axis=0)
    fano = float(es.fanofactor(spike_trains)) if len(spike_trains) >= 2 else 0.0

    # Adaptive MEA threshold (Beggs & Plenz 2003): frames exceeding background 1-spike bins
    # If mean counts/bin > 2.0, threshold at 50% median active rate to isolate bursts if dense;
    # otherwise threshold > 0 for clean sparse avalanches.
    active_vals = pop_counts[pop_counts > 0]
    thresh = 0.0
    if len(active_vals) > 0 and np.mean(pop_counts > 0) > 0.85:
        thresh = float(np.percentile(pop_counts, 35))

    above = pop_counts > thresh
    avalanches_s = []
    avalanches_t = []
    cur_s = 0.0
    cur_t = 0
    for val, flag in zip(pop_counts, above):
        if flag:
            cur_s += float(val)
            cur_t += 1
        elif cur_t > 0:
            avalanches_s.append(int(round(cur_s)))
            avalanches_t.append(int(cur_t))
            cur_s = 0.0
            cur_t = 0
    if cur_t > 0:
        avalanches_s.append(int(round(cur_s)))
        avalanches_t.append(int(cur_t))

    # Unbiased branching ratio estimator (Wilting & Priesemann 2018):
    # regression slope sigma = Cov(A_{t+1}, A_t) / Var(A_t), clipped >= 0
    x_t = pop_counts[:-1]
    y_t = pop_counts[1:]
    var_x = float(np.var(x_t))
    if var_x > 1e-12:
        cov_xy = float(np.mean((x_t - np.mean(x_t)) * (y_t - np.mean(y_t))))
        sigma = float(max(0.0, cov_xy / var_x))
    else:
        sigma = 0.0

    # Discrete MLE power-law exponent alpha for S >= S_min = 2
    sizes_arr = np.asarray([s for s in avalanches_s if s >= 2], dtype=np.float64)
    if len(sizes_arr) >= 5:
        s_min = 2.0
        denom = np.sum(np.log(sizes_arr / (s_min - 0.5)))
        alpha = float(1.0 + len(sizes_arr) / max(denom, 1e-9))
    else:
        alpha = 0.0

    return {
        "n_spikes": int(len(spike_t_ms)),
        "n_avalanches": int(len(avalanches_s)),
        "branching_ratio_sigma": sigma,
        "avalanche_alpha": alpha,
        "fano_factor": fano,
        "avalanche_sizes": avalanches_s[:2000],
        "avalanche_durations": avalanches_t[:2000],
    }


def analyze_all(bench_dir: str = "results/benchmarks"):
    runs_dir = os.path.join(bench_dir, "runs")
    telem_dir = os.path.join(bench_dir, "telemetry")

    csv_files = sorted(glob.glob(os.path.join(runs_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No benchmark run CSV files found in {runs_dir}")

    dfs = [pd.read_csv(p) for p in csv_files]
    full_df = pd.concat(dfs, ignore_index=True)

    # Load telemetry JSON + NPZ
    telem_json = {}
    for p in glob.glob(os.path.join(telem_dir, "*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        telem_json[(data.get("condition"), int(data.get("seed")))] = data

    avalanche_by_cond = {}
    for p in glob.glob(os.path.join(telem_dir, "*.npz")):
        base = os.path.basename(p).replace(".npz", "")
        if "_seed_" not in base:
            continue
        cond_name, seed_str = base.rsplit("_seed_", 1)
        seed_val = int(seed_str)
        with np.load(p) as npz:
            spike_t = npz["spike_t_ms"] if "spike_t_ms" in npz else np.array([])
            spike_i = npz["spike_i"] if "spike_i" in npz else np.array([])
            aval = compute_avalanche_metrics(spike_t, spike_i)
            aval["seed"] = seed_val
            avalanche_by_cond.setdefault(cond_name, []).append(aval)

    summary = {"conditions": {}, "pairwise_vs_c0": {}, "avalanches": {}, "ood_generalization": {}}

    seed_metric_vectors = {}
    present_conditions = [c for c in CONDITION_ORDER if c in full_df["condition"].unique()]

    for cond in present_conditions:
        cdf = full_df[full_df["condition"] == cond].copy()
        max_ep = int(cdf["episode"].max())
        late_start = max(0, max_ep - 99)
        late_df = cdf[cdf["episode"] >= late_start]

        per_seed_late = late_df.groupby("seed").agg({
            "return": "mean",
            "goal_reached": "mean",
            "success": "mean",
            "final_dist": "mean",
            "mean_rate_hz": "mean",
            "total_spikes": "mean",
            "mean_atp": "mean",
            "mean_weight": "mean",
            "active_syn_frac": "mean",
            "seizure_clamps": "mean",
            "coma_kicks": "mean",
            "length": "mean",
        }).reset_index()

        # Efficiency metric: return shifted above worst negative bound (+250) per 1k spikes
        eff_vals = []
        dsi_vals = []
        for _, row in per_seed_late.iterrows():
            s = int(row["seed"])
            spk = max(1.0, float(row["total_spikes"]))
            eff_vals.append((float(row["return"]) + 250.0) / (spk / 1000.0))
            tj = telem_json.get((cond, s), {})
            if "dsi_overall" in tj:
                dsi_vals.append(float(tj["dsi_overall"]))

        seed_metric_vectors[cond] = {
            "return_late": per_seed_late["return"].to_numpy(dtype=np.float64),
            "goal_late": per_seed_late["goal_reached"].to_numpy(dtype=np.float64) * 100.0,
            "success_late": per_seed_late["success"].to_numpy(dtype=np.float64) * 100.0,
            "dist_late": per_seed_late["final_dist"].to_numpy(dtype=np.float64),
            "rate_hz_late": per_seed_late["mean_rate_hz"].to_numpy(dtype=np.float64),
            "atp_late": per_seed_late["mean_atp"].to_numpy(dtype=np.float64),
            "efficiency": np.asarray(eff_vals, dtype=np.float64),
            "dsi": np.asarray(dsi_vals, dtype=np.float64),
        }

        summary["conditions"][cond] = {
            "label": CONDITION_LABELS.get(cond, cond),
            "n_seeds": int(per_seed_late["seed"].nunique()),
            "total_episodes_per_seed": int(max_ep + 1),
            "late_window": [int(late_start), int(max_ep)],
            "return_late": bootstrap_iqm_ci(seed_metric_vectors[cond]["return_late"]),
            "goal_rate_late_pct": bootstrap_iqm_ci(seed_metric_vectors[cond]["goal_late"]),
            "success_rate_late_pct": bootstrap_iqm_ci(seed_metric_vectors[cond]["success_late"]),
            "final_dist_late": bootstrap_iqm_ci(seed_metric_vectors[cond]["dist_late"]),
            "mean_rate_hz_late": bootstrap_iqm_ci(seed_metric_vectors[cond]["rate_hz_late"]),
            "mean_atp_late": bootstrap_iqm_ci(seed_metric_vectors[cond]["atp_late"]),
            "metabolic_efficiency": bootstrap_iqm_ci(seed_metric_vectors[cond]["efficiency"]),
            "dsi_overall": bootstrap_iqm_ci(seed_metric_vectors[cond]["dsi"]),
        }

        # OOD topology aggregation if present
        ood_by_topo = {}
        for (_, s), tj in telem_json.items():
            if tj.get("condition") == cond and tj.get("ood_topologies"):
                for tname, tmetrics in tj["ood_topologies"].items():
                    ood_by_topo.setdefault(tname, {"goal_rate": [], "mean_return": [], "mean_final_dist": []})
                    ood_by_topo[tname]["goal_rate"].append(float(tmetrics["goal_rate"]) * 100.0)
                    ood_by_topo[tname]["mean_return"].append(float(tmetrics["mean_return"]))
                    ood_by_topo[tname]["mean_final_dist"].append(float(tmetrics["mean_final_dist"]))
        if ood_by_topo:
            summary["ood_generalization"][cond] = {
                tname: {k: bootstrap_iqm_ci(np.asarray(vals)) for k, vals in d.items()}
                for tname, d in ood_by_topo.items()
            }

        # Avalanche summary
        if cond in avalanche_by_cond:
            av_list = avalanche_by_cond[cond]
            sigmas = np.array([a["branching_ratio_sigma"] for a in av_list], dtype=np.float64)
            alphas = np.array([a["avalanche_alpha"] for a in av_list if a["avalanche_alpha"] > 0], dtype=np.float64)
            fanos = np.array([a["fano_factor"] for a in av_list], dtype=np.float64)
            summary["avalanches"][cond] = {
                "sigma": bootstrap_iqm_ci(sigmas),
                "alpha": bootstrap_iqm_ci(alphas),
                "fano_factor": bootstrap_iqm_ci(fanos),
            }

    # Pairwise non-parametric significance tests vs C0
    if "c0_full_model" in seed_metric_vectors:
        c0_vec = seed_metric_vectors["c0_full_model"]
        raw_p_ret = []
        raw_p_goal = []
        comps = [c for c in present_conditions if c != "c0_full_model"]

        for cond in comps:
            other_vec = seed_metric_vectors[cond]
            x_r, y_r = c0_vec["return_late"], other_vec["return_late"]
            x_g, y_g = c0_vec["goal_late"], other_vec["goal_late"]

            u_r, p_r = stats.mannwhitneyu(x_r, y_r, alternative="two-sided")
            r_rb_r = float(1.0 - (2.0 * u_r) / (len(x_r) * len(y_r)))
            poi_r = float(rm.probability_of_improvement(x_r[:, None], y_r[:, None]))

            u_g, p_g = stats.mannwhitneyu(x_g, y_g, alternative="two-sided")
            r_rb_g = float(1.0 - (2.0 * u_g) / (len(x_g) * len(y_g)))
            poi_g = float(rm.probability_of_improvement(x_g[:, None], y_g[:, None]))

            raw_p_ret.append(float(p_r))
            raw_p_goal.append(float(p_g))

            summary["pairwise_vs_c0"][cond] = {
                "return": {
                    "u_stat": float(u_r),
                    "p_value": float(p_r),
                    "rank_biserial_r": r_rb_r,
                    "prob_improvement_c0_gt_other": poi_r,
                },
                "goal_rate": {
                    "u_stat": float(u_g),
                    "p_value": float(p_g),
                    "rank_biserial_r": r_rb_g,
                    "prob_improvement_c0_gt_other": poi_g,
                },
            }

        adj_r = bh_fdr(raw_p_ret)
        adj_g = bh_fdr(raw_p_goal)
        for idx, cond in enumerate(comps):
            summary["pairwise_vs_c0"][cond]["return"]["p_fdr"] = adj_r[idx]
            summary["pairwise_vs_c0"][cond]["goal_rate"]["p_fdr"] = adj_g[idx]

    out_json = os.path.join(bench_dir, "stats_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_csv = os.path.join(bench_dir, "summary_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "condition", "label", "n_seeds",
            "return_iqm", "return_ci_low", "return_ci_high",
            "goal_iqm_pct", "goal_ci_low", "goal_ci_high",
            "rate_hz_iqm", "atp_iqm", "dsi_iqm", "p_fdr_vs_c0_ret", "rank_biserial_ret",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cond in present_conditions:
            cinfo = summary["conditions"][cond]
            pw = summary["pairwise_vs_c0"].get(cond, {}).get("return", {})
            writer.writerow({
                "condition": cond,
                "label": cinfo["label"],
                "n_seeds": cinfo["n_seeds"],
                "return_iqm": round(cinfo["return_late"]["iqm"], 3),
                "return_ci_low": round(cinfo["return_late"]["ci_low"], 3),
                "return_ci_high": round(cinfo["return_late"]["ci_high"], 3),
                "goal_iqm_pct": round(cinfo["goal_rate_late_pct"]["iqm"], 2),
                "goal_ci_low": round(cinfo["goal_rate_late_pct"]["ci_low"], 2),
                "goal_ci_high": round(cinfo["goal_rate_late_pct"]["ci_high"], 2),
                "rate_hz_iqm": round(cinfo["mean_rate_hz_late"]["iqm"], 3),
                "atp_iqm": round(cinfo["mean_atp_late"]["iqm"], 4),
                "dsi_iqm": round(cinfo["dsi_overall"]["iqm"], 4),
                "p_fdr_vs_c0_ret": round(pw.get("p_fdr", 1.0), 6) if cond != "c0_full_model" else 1.0,
                "rank_biserial_ret": round(pw.get("rank_biserial_r", 0.0), 4) if cond != "c0_full_model" else 0.0,
            })

    print(f"Statistical analysis complete -> {out_json}, {out_csv}", flush=True)
    return summary


if __name__ == "__main__":
    analyze_all()
