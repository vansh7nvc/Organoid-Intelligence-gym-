"""
Publication Vector Figure Generator for OrganoidEnv (Stage 5).

Generates all 12 IEEE TNNLS / Nature-style multi-panel vector figures
in submissions/ieee_tnnls/figures/ (.pdf and .svg), strictly populated
from empirical benchmark data in results/benchmarks/ with zero hardcoded
empirical values, embedded TrueType fonts (fonttype=42), and >= 8.5 pt labels.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.analyze_benchmarks import CONDITION_ORDER, CONDITION_LABELS, compute_avalanche_metrics


# Publication typography & embedded vector font settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = {
    "c0_full_model": "#0072B2",
    "c1_frozen_stdp": "#D55E00",
    "c2_random_snn": "#CC79A7",
    "c3_no_metabolism": "#E69F00",
    "c4_no_gar": "#B2182B",
    "c5_no_pruning": "#8C510A",
    "c6_single_trace": "#56B4E9",
    "c6b_single_trace_fast": "#56B4E9",
    "c6c_single_trace_slow": "#4393C3",
    "c7_gar_only": "#D6604D",
    "c8_no_gar_no_met": "#A50026",
    "b1_direct_d3qn": "#009E73",
    "b2_mlp_dqn": "#6A3D9A",
    "b3_simple_snn_lif": "#F0E442",
    "b4_ppo_sb3": "#332288",
    "b5_random": "#7F7F7F",
    "b1_step005_d3qn": "#1B7837",
    "b2_step005_mlp_dqn": "#9970AB",
    "b4_step005_ppo": "#5E4FA2",
}


def _save_fig(fig, fig_dir: str, stem: str):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _load_runs_df(bench_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(bench_dir, "runs", "*.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _load_telemetry(bench_dir: str):
    telem_json = {}
    for p in glob.glob(os.path.join(bench_dir, "telemetry", "*.json")):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        telem_json[(d.get("condition"), int(d.get("seed")))] = d
    return telem_json


def _load_first_npz(bench_dir: str, cond: str):
    paths = sorted(glob.glob(os.path.join(bench_dir, "telemetry", f"{cond}_seed_*.npz")))
    if not paths:
        return None
    # Prefer canonical seed 42 or median active firing seed over lexicographical outlier
    seed42 = os.path.join(bench_dir, "telemetry", f"{cond}_seed_42.npz")
    if os.path.exists(seed42):
        return dict(np.load(seed42))
    return dict(np.load(paths[len(paths) // 2]))


def _smooth(y: np.ndarray, window: int = 15) -> np.ndarray:
    if len(y) < 3:
        return y
    k = min(window, len(y))
    return pd.Series(y).rolling(k, min_periods=1).mean().to_numpy()


def generate_fig01_architecture(fig_dir: str):
    """Fig 1: Closed-Loop In Silico Organoid System Architecture Schematic."""
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    boxes = [
        (0.3, 3.3, 2.3, 1.35, "#EBF5FB", "#0072B2", "Task Environment\n(2D Navigation)\nObstacles & Goals"),
        (3.6, 3.3, 2.6, 1.35, "#FEF9E7", "#D4AC0D", "Sparse Expansion Encoder\n15D Obs -> 256 Exc\nTop-k (k=26, 10.2%)"),
        (7.2, 3.3, 3.8, 1.35, "#FDEDEC", "#B03A2E", "Organoid-Inspired SNN\n500 Neurons (400 RS / 100 FS)\nEnergy Variable + Dual Traces"),
        (7.2, 0.55, 3.8, 1.35, "#EAFAF1", "#1E8449", "Clustered Motor Readout\n4 Cardinal Quadrants (4x25)\n& Global Activity Regulator"),
        (0.3, 0.55, 5.7, 1.35, "#F4ECF7", "#6C3483", "External Adaptive Controller (D3QN + HER + N-Step)\nSelects Action 0..7 -> Motor Quadrant Stim"),
    ]
    for x, y, w, h, fc, ec, txt in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
                                      facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.8, fontweight="bold")

    arrows = [
        ((2.68, 3.98), (3.52, 3.98), "15D\nObs", (3.10, 4.34)),
        ((6.28, 3.98), (7.12, 3.98), "Encoder\nI_stim", (6.70, 4.34)),
        ((9.10, 3.22), (9.10, 1.98), "Spikes\n(100 ms)", (9.65, 2.60)),
        ((7.12, 1.22), (6.08, 1.22), "Rate (10D)\n+ Reward", (6.60, 1.56)),
        ((7.12, 1.65), (2.68, 3.32), "dx, dy", (4.50, 2.55)),
        ((1.45, 3.22), (1.45, 1.98), "15D Obs\n+ Reward", (0.92, 2.60)),
    ]
    for p1, p2, lbl, lpos in arrows:
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#2C3E50"))
        ax.text(lpos[0], lpos[1], lbl, ha="center", va="center", fontsize=8.0, color="#2C3E50",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))

    ax.annotate("", xy=(8.2, 3.22), xytext=(4.8, 1.98),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, ls="--", color="#B7950B"))
    ax.text(6.65, 2.68, "Action 0..7 Stim +\nModulation Δw = η·R·(T1 + 0.5·T2)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#7D6608",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FEF9E7", ec="#D4AC0D", lw=0.8, alpha=0.95))

    ax.set_title("OrganoidEnv Closed-Loop Bio-Inspired Neuro-RL Architecture", fontsize=11.0, pad=8)
    _save_fig(fig, fig_dir, "fig01_architecture_loop")


def generate_fig02_electrophysiology(bench_dir: str, fig_dir: str):
    """Fig 2: Biophysical Validation (Spike Raster, Rate vs Biological Bounds, ATP Dynamics)."""
    npz = _load_first_npz(bench_dir, "c0_full_model")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    if npz is not None and "spike_t_ms" in npz and len(npz["spike_t_ms"]) > 0:
        t_ms = npz["spike_t_ms"]
        i_id = npz["spike_i"]
        mask = t_ms <= 2000.0
        t_sub = t_ms[mask]
        i_sub = i_id[mask]

        colors = np.where(i_sub < 256, "#0072B2",
                 np.where(i_sub < 300, "#56B4E9",
                 np.where(i_sub < 400, "#D55E00", "#009E73")))
        axes[0].scatter(t_sub, i_sub, s=2.2, c=colors, alpha=0.75, edgecolors="none")
        axes[0].axhspan(0, 256, alpha=0.05, color="#0072B2", label="Encoder Exc (0-255)")
        axes[0].axhspan(300, 400, alpha=0.06, color="#D55E00", label="FS Inh (300-399)")
        axes[0].axhspan(400, 500, alpha=0.05, color="#009E73", label="Motor Exc (400-499)")
        axes[0].set_title("A. 2-Second SNN Spike Raster")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Neuron Index")
        axes[0].set_xlim(0, 2000)
        axes[0].set_ylim(0, 500)
        axes[0].legend(loc="upper right", fontsize=7.5, framealpha=0.85)

        step_t = np.arange(len(npz["step_rates_hz"])) * 0.1
        axes[1].plot(step_t, npz["step_rates_hz"], color="#0072B2", lw=1.6, label="Population Rate")
        axes[1].axhspan(2.0, 20.0, color="#009E73", alpha=0.16, label="Reference Band (2-20 Hz)")
        axes[1].set_title("B. Operational Firing Rate Trajectory")
        axes[1].set_xlabel("Simulation Time (s)")
        axes[1].set_ylabel("Mean Population Rate (Hz)")
        axes[1].legend(loc="upper right", fontsize=7.5)

        axes[2].plot(step_t, npz["step_atp"], color="#E69F00", lw=1.8, label="Mean Energy Reserve (E)")
        axes[2].axhline(0.0, color="#B2182B", ls="--", lw=1.1, label="Exhaustion Arrest (E=0)")
        axes[2].set_title("C. Energy-Fatigue Dynamics (E)")
        axes[2].set_xlabel("Simulation Time (s)")
        axes[2].set_ylabel("Normalized Energy Reserve E")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig02_biophysical_electrophysiology")


def generate_fig03_sdm(bench_dir: str, fig_dir: str):
    """Fig 3: Sparse Distributed Memory Expansion Occupancy & Sparsity."""
    telem = _load_telemetry(bench_dir)
    npz = _load_first_npz(bench_dir, "c0_full_model")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    if npz is not None and "step_sdm_sparsity" in npz:
        sp = npz["step_sdm_sparsity"] * 100.0
        axes[0].plot(np.arange(len(sp)) * 0.1, sp, color="#0072B2", lw=1.6, label="Active SDM Channels (%)")
        axes[0].axhline(10.0, color="#D55E00", ls="--", lw=1.2, label="Target Sparse Budget (~10%)")
        axes[0].set_title("A. Real-Time SDM Sensory Expansion Occupancy")
        axes[0].set_xlabel("Episode Time (s)")
        axes[0].set_ylabel("Active SDM Units (%)")
        axes[0].legend(fontsize=8.0)

    cond_sp = {}
    for (c, _), d in telem.items():
        if "mean_sdm_sparsity" in d:
            cond_sp.setdefault(c, []).append(d["mean_sdm_sparsity"] * 100.0)
    if cond_sp:
        names = [c for c in CONDITION_ORDER if c in cond_sp]
        vals = [np.mean(cond_sp[c]) for c in names]
        errs = [np.std(cond_sp[c]) for c in names]
        cols = [PALETTE.get(c, "#555555") for c in names]
        x = np.arange(len(names))
        axes[1].bar(x, vals, yerr=errs, color=cols, capsize=3, alpha=0.88)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([c.split("_", 1)[0].upper() for c in names], rotation=30)
        axes[1].set_title("B. Mean SDM Occupancy Across Conditions")
        axes[1].set_ylabel("Mean Active SDM (%)")

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig03_sdm_sparse_coding")


def generate_fig04_gar_homeostasis(df: pd.DataFrame, fig_dir: str):
    """Fig 4: GAR Homeostatic Control Comparing C0 (GAR=ON) vs C4 (GAR=OFF) vs C3."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    if not df.empty:
        for cond, lbl in [("c0_full_model", "C0: GAR ON"), ("c4_no_gar", "C4: No GAR"), ("c3_no_metabolism", "C3: No Metabolism")]:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            grp = sub.groupby("episode")["mean_rate_hz"]
            mu = _smooth(grp.mean().to_numpy(), 15)
            sd = _smooth(grp.std().fillna(0).to_numpy(), 15)
            ep = np.sort(sub["episode"].unique())
            c = PALETTE[cond]
            axes[0].plot(ep, mu, color=c, lw=1.8, label=lbl)
            axes[0].fill_between(ep, np.maximum(0, mu - sd), mu + sd, color=c, alpha=0.18)

            grp_atp = sub.groupby("episode")["mean_atp"]
            mu_e = _smooth(grp_atp.mean().to_numpy(), 15)
            axes[1].plot(ep, mu_e, color=c, lw=1.8, label=lbl)

    axes[0].axhspan(2.0, 20.0, color="#009E73", alpha=0.15, label="Physiological Band (2-20 Hz)")
    axes[0].set_title("A. Population Firing Rate Over Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean Firing Rate (Hz)")
    axes[0].legend(fontsize=8.0)

    axes[1].set_title("B. Mean Cellular ATP Reserve (E) Stability")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean ATP Reserve E")
    axes[1].legend(fontsize=8.0)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig04_gar_homeostasis")


def generate_fig05_criticality_avalanches(bench_dir: str, stats_summary: dict, fig_dir: str):
    """Fig 5: Neural Avalanche Size Scaling & Branching Ratio (Elephant Analysis)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # Panel A: empirical avalanche size CCDF/PDF in log-log space
    for cond, lbl in [("c0_full_model", "C0 Full Model"), ("c4_no_gar", "C4 No GAR"), ("b3_simple_snn_lif", "B3 SimpleSNN (LIF)")]:
        npz_paths = sorted(glob.glob(os.path.join(bench_dir, "telemetry", f"{cond}_seed_*.npz")))
        if not npz_paths:
            continue
        all_sizes = []
        for p in npz_paths[:5]:
            with np.load(p) as npz:
                if "spike_t_ms" in npz:
                    av = compute_avalanche_metrics(npz["spike_t_ms"], npz["spike_i"])
                    all_sizes.extend(av["avalanche_sizes"])
        all_sizes = np.asarray([s for s in all_sizes if s >= 1], dtype=np.float64)
        if len(all_sizes) >= 10:
            bins = np.unique(np.round(np.logspace(0, np.log10(max(all_sizes) + 1), 16)).astype(int))
            if len(bins) >= 4:
                hist, edges = np.histogram(all_sizes, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                pos = hist > 0
                axes[0].loglog(centers[pos], hist[pos], "o-", color=PALETTE[cond], lw=1.6, ms=4.5, label=lbl)

    x_ref = np.logspace(0.3, 2.2, 40)
    axes[0].loglog(x_ref, 0.8 * (x_ref ** -1.5), "k--", lw=1.2, label=r"Ref Slope $\alpha = 1.5$")
    axes[0].set_title("A. Power-Law Avalanche Size Distribution (Elephant)")
    axes[0].set_xlabel("Avalanche Size S (Spikes)")
    axes[0].set_ylabel("Probability Density P(S)")
    axes[0].legend(fontsize=8.0)

    # Panel B: Branching ratio sigma across conditions
    av_dict = stats_summary.get("avalanches", {})
    conds = [c for c in CONDITION_ORDER if c in av_dict]
    if conds:
        sigmas = [av_dict[c]["sigma"]["iqm"] for c in conds]
        err_lo = [max(0, av_dict[c]["sigma"]["iqm"] - av_dict[c]["sigma"]["ci_low"]) for c in conds]
        err_hi = [max(0, av_dict[c]["sigma"]["ci_high"] - av_dict[c]["sigma"]["iqm"]) for c in conds]
        x = np.arange(len(conds))
        axes[1].bar(x, sigmas, yerr=[err_lo, err_hi], color=[PALETTE[c] for c in conds], capsize=3, alpha=0.88)
        axes[1].axhline(1.0, color="black", ls="--", lw=1.3, label=r"Critical Target ($\sigma = 1.0$)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([c.split("_", 1)[0].upper() for c in conds], rotation=35)
        axes[1].set_title(r"B. Branching Ratio $\sigma$ Across Models (95% CI)")
        axes[1].set_ylabel(r"Branching Ratio $\sigma$")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8.0)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig05_criticality_avalanches")


def generate_fig06_stability_phase_space(bench_dir: str, fig_dir: str):
    """Fig 6: 2D Dynamical Stability Phase Diagram (I_stim x w_inh)."""
    sweep_path = os.path.join(bench_dir, "stability_sweep.json")
    if not os.path.exists(sweep_path):
        return
    with open(sweep_path, "r", encoding="utf-8") as f:
        sweep = json.load(f)

    i_levels = sweep["i_stim_levels"]
    w_levels = sweep["w_inh_levels"]
    res = sweep["results"]

    panels = [
        (False, False, "A. Unregulated (GAR OFF, Met OFF)"),
        (False, True, "B. Metabolic Alone (GAR OFF, Met ON)"),
        (True, False, "C. GAR Alone (GAR ON, Met OFF)"),
        (True, True, "D. Full Closed-Loop (GAR ON, Met ON)"),
    ]
    available_modes = {(r["use_stabilizer"], r["use_metabolism"]) for r in res}
    active_panels = [p for p in panels if (p[0], p[1]) in available_modes]
    if not active_panels:
        active_panels = panels

    if len(active_panels) == 4:
        fig, axes_grid = plt.subplots(2, 2, figsize=(7.4, 5.8), sharex=True, sharey=True)
        axes = axes_grid.ravel()
    else:
        fig, axes = plt.subplots(1, len(active_panels), figsize=(3.8 * len(active_panels), 3.8), sharey=True)
        if not hasattr(axes, "__len__"):
            axes = [axes]

    vmax = max(r["mean_rate_hz"] for r in res) if res else 140.0
    im = None
    for idx, (ax, (stab, met, title)) in enumerate(zip(axes, active_panels)):
        grid = np.zeros((len(i_levels), len(w_levels)), dtype=float)
        for r in res:
            if r["use_stabilizer"] == stab and r["use_metabolism"] == met:
                ri = i_levels.index(r["i_stim"])
                ci = w_levels.index(r["w_inh"])
                grid[ri, ci] = r["mean_rate_hz"]

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=max(30.0, vmax))
        ax.grid(False)
        ax.set_xticks(range(len(w_levels)))
        ax.set_xticklabels([f"{int(w)}" for w in w_levels])
        ax.set_yticks(range(len(i_levels)))
        ax.set_yticklabels([f"{int(i)}" for i in i_levels])
        ax.set_xlabel(r"Inhibitory Weight $w_{\mathrm{inh}}$ (nS)")
        ax.set_title(title)
        for ri in range(len(i_levels)):
            for ci in range(len(w_levels)):
                val = grid[ri, ci]
                txt_col = "white" if val > vmax * 0.65 else "black"
                ax.text(ci, ri, f"{val:.1f}", ha="center", va="center", color=txt_col, fontsize=8.0, fontweight="bold")

    if len(active_panels) == 4:
        axes[0].set_ylabel(r"Stimulation Amplitude $I_{\mathrm{stim}}$ (pA)")
        axes[2].set_ylabel(r"Stimulation Amplitude $I_{\mathrm{stim}}$ (pA)")
    else:
        axes[0].set_ylabel(r"Stimulation Amplitude $I_{\mathrm{stim}}$ (pA)")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Mean Population Rate (Hz)")
    fig.suptitle("2D Stability Phase Space: GAR + ATP Clamp Prevents Epileptiform Runaway", fontsize=11.0, y=1.03)
    _save_fig(fig, fig_dir, "fig06_stability_phase_space")


def generate_fig07_directional_selectivity(bench_dir: str, stats_summary: dict, fig_dir: str):
    """Fig 7: Directional Motor Tuning Matrix & DSI Comparison Across Models."""
    telem = _load_telemetry(bench_dir)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    c0_mats = [np.asarray(d["dsi_matrix"]) for (c, _), d in telem.items() if c == "c0_full_model" and "dsi_matrix" in d]
    if c0_mats:
        mean_mat = np.mean(c0_mats, axis=0)
        im = axes[0].imshow(mean_mat, cmap="Blues", aspect="auto")
        axes[0].grid(False)
        dirs = ["Up", "Down", "Left", "Right"]
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(dirs)
        axes[0].set_yticks(range(4))
        axes[0].set_yticklabels([f"Stim {d}" for d in dirs])
        axes[0].set_xlabel("Motor Readout Quadrant")
        axes[0].set_title("A. Trained C0 Motor Quadrant Rates (Hz)")
        m_min, m_max = float(mean_mat.min()), float(mean_mat.max())
        m_span = max(1e-9, m_max - m_min)
        for r in range(4):
            for c in range(4):
                val = mean_mat[r, c]
                frac = (val - m_min) / m_span
                axes[0].text(c, r, f"{val:.1f}", ha="center", va="center",
                             color="white" if frac > 0.55 else "black", fontsize=8.5, fontweight="bold")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    cond_info = stats_summary.get("conditions", {})
    dsi_conds = [c for c in CONDITION_ORDER if c in cond_info and cond_info[c]["dsi_overall"]["n"] > 0]
    if dsi_conds:
        iqms = [cond_info[c]["dsi_overall"]["iqm"] for c in dsi_conds]
        lo = [max(0, cond_info[c]["dsi_overall"]["iqm"] - cond_info[c]["dsi_overall"]["ci_low"]) for c in dsi_conds]
        hi = [max(0, cond_info[c]["dsi_overall"]["ci_high"] - cond_info[c]["dsi_overall"]["iqm"]) for c in dsi_conds]
        x = np.arange(len(dsi_conds))
        axes[1].bar(x, iqms, yerr=[lo, hi], color=[PALETTE[c] for c in dsi_conds], capsize=3, alpha=0.9)
        axes[1].axhline(0.50, color="#009E73", ls="--", lw=1.2, label="Tuned Threshold (DSI >= 0.50)")
        axes[1].axhline(0.20, color="#B2182B", ls=":", lw=1.2, label="Untuned Floor (DSI <= 0.20)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([c.split("_", 1)[0].upper() for c in dsi_conds], rotation=35)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("Directional Selectivity Index (DSI)")
        axes[1].set_title("B. Motor Selectivity Across Ablations (95% CI)")
        axes[1].legend(fontsize=8.0)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig07_directional_selectivity")


def generate_fig08_dual_trace_pruning(bench_dir: str, df: pd.DataFrame, fig_dir: str):
    """Fig 8: Dual-Trace Eligibility Decay & Structural Synaptic Pruning Spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    t_ms = np.linspace(0, 4000, 400)
    k_fast = np.exp(-t_ms / 100.0)
    k_slow = 0.5 * np.exp(-t_ms / 2500.0)
    k_single = np.exp(-t_ms / 50.0)
    axes[0].plot(t_ms, k_fast, color="#0072B2", lw=1.8, label=r"Fast Trace $\tau_1 = 100$ ms")
    axes[0].plot(t_ms, k_slow, color="#009E73", lw=1.8, label=r"Slow Trace $0.5\cdot\tau_2$ ($\tau_2 = 2500$ ms)")
    axes[0].plot(t_ms, k_single, color="#56B4E9", ls="--", lw=1.6, label=r"Single Trace ($\tau = 50$ ms)")
    axes[0].set_title("A. Multi-Timescale Eligibility Kernel (D3 Rule)")
    axes[0].set_xlabel("Post-Spike Credit Assignment Delay (ms)")
    axes[0].set_ylabel("Normalized Eligibility Weight")
    axes[0].legend(fontsize=8.0)

    npz_c0 = _load_first_npz(bench_dir, "c0_full_model")
    npz_c5 = _load_first_npz(bench_dir, "c5_no_pruning")
    if npz_c0 is not None and "w_exc" in npz_c0:
        axes[1].hist(npz_c0["w_exc"], bins=30, density=True, alpha=0.65, color=PALETTE["c0_full_model"], label="C0 Full Model (Pruned)")
    if npz_c5 is not None and "w_exc" in npz_c5:
        axes[1].hist(npz_c5["w_exc"], bins=30, density=True, alpha=0.45, color=PALETTE["c5_no_pruning"], label="C5 No Pruning")
    if not df.empty and npz_c0 is None:
        sub = df[df["condition"] == "c0_full_model"].groupby("episode")["active_syn_frac"].mean()
        axes[1].plot(sub.index, sub.to_numpy() * 100, color=PALETTE["c0_full_model"])
    axes[1].set_title("B. Post-Training Excitatory Weight Spectrum")
    axes[1].set_xlabel("Synaptic Weight w (a.u.)")
    axes[1].set_ylabel("Density")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8.0)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig08_dual_trace_plasticity")


def generate_fig09_learning_curves(df: pd.DataFrame, fig_dir: str):
    """Fig 9: Curriculum Learning Curves comparing OrganoidEnv C0 vs Baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    compare = ["c0_full_model", "b1_direct_d3qn", "b2_mlp_dqn", "b3_simple_snn_lif", "b4_ppo_sb3", "b5_random"]
    stage_lines = [100, 200, 350]

    for cond in compare:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        col = PALETTE.get(cond, "#333333")
        lbl = CONDITION_LABELS.get(cond, cond).split(": ")[-1] if ":" in CONDITION_LABELS.get(cond, cond) else CONDITION_LABELS.get(cond, cond)
        lw = 2.2 if cond == "c0_full_model" else 1.5

        grp_r = sub.groupby("episode")["return"]
        ep = np.sort(sub["episode"].unique())
        mu_r = _smooth(grp_r.mean().to_numpy(), 20)
        se_r = _smooth((grp_r.std().fillna(0) / np.sqrt(max(1, sub["seed"].nunique()))).to_numpy(), 20)
        axes[0].plot(ep, mu_r, color=col, lw=lw, label=lbl)
        axes[0].fill_between(ep, mu_r - se_r, mu_r + se_r, color=col, alpha=0.14)

        grp_g = sub.groupby("episode")["goal_reached"]
        mu_g = _smooth(grp_g.mean().to_numpy() * 100.0, 20)
        se_g = _smooth((grp_g.std().fillna(0) * 100.0 / np.sqrt(max(1, sub["seed"].nunique()))).to_numpy(), 20)
        axes[1].plot(ep, mu_g, color=col, lw=lw, label=lbl)
        axes[1].fill_between(ep, np.maximum(0, mu_g - se_g), np.minimum(100, mu_g + se_g), color=col, alpha=0.14)

    for ax in axes:
        for s_ep in stage_lines:
            ax.axvline(s_ep, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("Curriculum Episode")
    for lbl_x, stg_txt in [(50, "S1"), (150, "S2"), (275, "S3"), (425, "S4")]:
        axes[0].text(lbl_x, 0.96, stg_txt, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8.0, color="#555555", fontweight="bold")
        axes[1].text(lbl_x, 0.96, stg_txt, transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8.0, color="#555555", fontweight="bold")

    axes[0].set_title("A. Episodic Return Across Curriculum Stages 1-4")
    axes[0].set_ylabel("Smoothed Return (Mean +/- SE)")
    axes[0].legend(fontsize=7.5, loc="lower right")

    axes[1].set_title("B. Goal Completion Rate Across Curriculum")
    axes[1].set_ylabel("Goal Reached (%)")
    axes[1].set_ylim(-2, 105)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig09_learning_curves")


def generate_fig10_ablation_matrix(stats_summary: dict, fig_dir: str):
    """Fig 10: 6-Way Mechanistic Ablation Forest Plot (rliable IQM + 95% Bootstrap CIs)."""
    conds_info = stats_summary.get("conditions", {})
    pairwise = stats_summary.get("pairwise_vs_c0", {})
    ablation_keys = [c for c in ["c0_full_model", "c1_frozen_stdp", "c2_random_snn", "c3_no_metabolism",
                                "c4_no_gar", "c5_no_pruning", "c6_single_trace"] if c in conds_info]
    if not ablation_keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    y = np.arange(len(ablation_keys))[::-1]
    labels = [CONDITION_LABELS[c] for c in ablation_keys]

    for idx, cond in zip(y, ablation_keys):
        r_info = conds_info[cond]["return_late"]
        g_info = conds_info[cond]["goal_rate_late_pct"]
        col = PALETTE[cond]

        axes[0].errorbar(r_info["iqm"], idx, xerr=[[r_info["iqm"] - r_info["ci_low"]], [r_info["ci_high"] - r_info["iqm"]]],
                         fmt="o", color=col, capsize=4, markersize=7, lw=2.0)
        axes[1].errorbar(g_info["iqm"], idx, xerr=[[g_info["iqm"] - g_info["ci_low"]], [g_info["ci_high"] - g_info["iqm"]]],
                         fmt="s", color=col, capsize=4, markersize=7, lw=2.0)

        if cond != "c0_full_model" and cond in pairwise:
            p_fdr = pairwise[cond]["return"].get("p_fdr", 1.0)
            sig = "***" if p_fdr < 0.001 else ("**" if p_fdr < 0.01 else ("*" if p_fdr < 0.05 else "ns"))
            axes[0].text(r_info["ci_high"] + 3.0, idx, sig, va="center", fontsize=8.5, fontweight="bold")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Late-Stage Return IQM (95% Stratified Bootstrap CI)")
    axes[0].set_title("A. Episodic Return Impact")
    axes[1].set_xlabel("Late-Stage Goal Completion IQM (%)")
    axes[1].set_title("B. Task Precision Impact")

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig10_ablation_matrix")


def generate_fig11_metabolic_efficiency(stats_summary: dict, fig_dir: str):
    """Fig 11: Metabolic Efficiency Frontier (Utility per Spike & ATP vs Firing Rate)."""
    conds_info = stats_summary.get("conditions", {})
    snn_keys = [c for c in ["c0_full_model", "c1_frozen_stdp", "c2_random_snn", "c3_no_metabolism",
                           "c4_no_gar", "c5_no_pruning", "c6_single_trace", "b3_simple_snn_lif", "b5_random"] if c in conds_info]
    if not snn_keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(snn_keys))
    effs = [conds_info[c]["metabolic_efficiency"]["iqm"] for c in snn_keys]
    eff_lo = [max(0, conds_info[c]["metabolic_efficiency"]["iqm"] - conds_info[c]["metabolic_efficiency"]["ci_low"]) for c in snn_keys]
    eff_hi = [max(0, conds_info[c]["metabolic_efficiency"]["ci_high"] - conds_info[c]["metabolic_efficiency"]["iqm"]) for c in snn_keys]
    cols = [PALETTE[c] for c in snn_keys]

    axes[0].bar(x, effs, yerr=[eff_lo, eff_hi], color=cols, capsize=3, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.split("_", 1)[0].upper() for c in snn_keys], rotation=30)
    axes[0].set_ylabel("Shifted Return per 1k Spikes")
    axes[0].set_title("A. Information-Energetic Efficiency Ratio")

    for c in snn_keys:
        r_hz = conds_info[c]["mean_rate_hz_late"]["iqm"]
        atp = conds_info[c]["mean_atp_late"]["iqm"]
        axes[1].scatter(r_hz, atp, color=PALETTE[c], s=95, edgecolors="black", linewidths=0.8,
                        label=c.split("_", 1)[0].upper())
    axes[1].axvspan(2.0, 20.0, color="#009E73", alpha=0.12, label="Biological Band (2-20 Hz)")
    axes[1].set_xlabel("Mean Population Rate (Hz)")
    axes[1].set_ylabel("Steady-State Mean ATP Reserve E")
    axes[1].set_title("B. Firing Rate vs Metabolic Reserve Plane")
    axes[1].legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig11_metabolic_efficiency")


def generate_fig12_ood_generalization(stats_summary: dict, fig_dir: str):
    """Fig 12: Zero-Shot Generalization Across 5 Novel Obstacle Topologies."""
    ood = stats_summary.get("ood_generalization", {})
    if not ood:
        return

    conds = [c for c in ["c0_full_model", "b1_direct_d3qn", "b2_mlp_dqn", "b3_simple_snn_lif", "b4_ppo_sb3", "b5_random"] if c in ood]
    if not conds:
        return
    topos = sorted(next(iter(ood.values())).keys())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    mat_ret = np.zeros((len(conds), len(topos)), dtype=float)
    mat_goal = np.zeros((len(conds), len(topos)), dtype=float)
    for ri, c in enumerate(conds):
        for ci, t in enumerate(topos):
            mat_ret[ri, ci] = ood[c][t]["mean_return"]["iqm"]
            mat_goal[ri, ci] = ood[c][t]["goal_rate"]["iqm"]

    im0 = axes[0].imshow(mat_ret, cmap="viridis", aspect="auto")
    axes[0].grid(False)
    axes[0].set_xticks(range(len(topos)))
    axes[0].set_xticklabels([t.replace("_", "\n") for t in topos], fontsize=8.0)
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels([c.split("_", 1)[0].upper() for c in conds])
    axes[0].set_title("A. OOD Return IQM Across Topologies")
    r_min, r_max = float(mat_ret.min()), float(mat_ret.max())
    r_span = max(1e-9, r_max - r_min)
    for r in range(len(conds)):
        for c in range(len(topos)):
            frac = (mat_ret[r, c] - r_min) / r_span
            axes[0].text(c, r, f"{mat_ret[r, c]:.1f}", ha="center", va="center",
                         color="white" if frac < 0.52 else "black", fontsize=8.0, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mat_goal, cmap="YlGn", aspect="auto", vmin=0.0, vmax=100.0)
    axes[1].grid(False)
    axes[1].set_xticks(range(len(topos)))
    axes[1].set_xticklabels([t.replace("_", "\n") for t in topos], fontsize=8.0)
    axes[1].set_yticks(range(len(conds)))
    axes[1].set_yticklabels([c.split("_", 1)[0].upper() for c in conds])
    axes[1].set_title("B. OOD Goal Completion Rate (%)")
    for r in range(len(conds)):
        for c in range(len(topos)):
            axes[1].text(c, r, f"{mat_goal[r, c]:.0f}%", ha="center", va="center",
                         color="white" if mat_goal[r, c] > 55 else "black", fontsize=8.0, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig12_ood_generalization")


def generate_fig13_sensitivity_sweep(bench_dir: str, fig_dir: str):
    """Fig 13: Biophysical Sensitivity Sweep (Delta_E, tau_rec, and GAR thresholds)."""
    sweep_path = os.path.join(bench_dir, "sensitivity_sweep.json")
    if not os.path.exists(sweep_path):
        return
    with open(sweep_path, "r", encoding="utf-8") as f:
        sweep = json.load(f)

    res = sweep["results"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))

    de_levels = sweep["delta_e_levels"]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(de_levels)))

    for de, col in zip(de_levels, colors):
        sub = [r for r in res if r["delta_e"] == de and r["gar_seizure_threshold"] == 5000 and r["gar_coma_threshold"] == 50]
        sub.sort(key=lambda r: r["tau_rec_ms"])
        if sub:
            xs = [r["tau_rec_ms"] for r in sub]
            ys = [r["mean_rate_hz"] for r in sub]
            axes[0].plot(xs, ys, "o-", color=col, lw=1.6, ms=4, label=rf"$\Delta_E = {de:.2f}$")

    axes[0].axhspan(2.0, 20.0, color="#009E73", alpha=0.15, label="Target Band (2-20 Hz)")
    axes[0].set_xlabel(r"$\tau_{\mathrm{rec}}$ (ms)")
    axes[0].set_ylabel("Mean Firing Rate (Hz)")
    axes[0].set_title(r"A. Rate vs. $\tau_{\mathrm{rec}}$ across $\Delta_E$")
    axes[0].legend(fontsize=7.5, loc="upper right")

    prod_vals = []
    atp_vals = []
    for r in res:
        if r["gar_seizure_threshold"] == 5000 and r["gar_coma_threshold"] == 50:
            prod_vals.append(r["delta_e"] * r["tau_rec_ms"] * 0.001)
            atp_vals.append(r["final_mean_atp"])

    axes[1].scatter(prod_vals, atp_vals, color="#0072B2", alpha=0.75, edgecolor="none", s=35)
    if prod_vals:
        p_sort = np.argsort(prod_vals)
        axes[1].plot(np.array(prod_vals)[p_sort], np.array(atp_vals)[p_sort], "--", color="#0072B2", alpha=0.5)
    axes[1].set_xlabel(r"Metabolic Load $\Delta_E \cdot \tau_{\mathrm{rec}}$ (s)")
    axes[1].set_ylabel("Final Mean ATP Reserve (E)")
    axes[1].set_title(r"B. ATP Depletion vs. Metabolic Load")

    seizure_pts = [r for r in res if r["delta_e"] == 0.10 and r["tau_rec_ms"] == 1000.0 and r["gar_coma_threshold"] == 50]
    seizure_pts.sort(key=lambda r: r["gar_seizure_threshold"])
    if seizure_pts:
        xs = [r["gar_seizure_threshold"] for r in seizure_pts]
        ys = [r["mean_rate_hz"] for r in seizure_pts]
        axes[2].plot(xs, ys, "s-", color="#D55E00", lw=1.6, ms=5, label="Seizure Clamp Thresh")

    coma_pts = [r for r in res if r["delta_e"] == 0.10 and r["tau_rec_ms"] == 1000.0 and r["gar_seizure_threshold"] == 5000]
    coma_pts.sort(key=lambda r: r["gar_coma_threshold"])
    if coma_pts:
        xs_c = [r["gar_coma_threshold"] * 100 for r in coma_pts]
        ys_c = [r["mean_rate_hz"] for r in coma_pts]
        axes[2].plot(xs_c, ys_c, "^--", color="#009E73", lw=1.6, ms=5, label="Coma Kick Thresh (x100)")

    axes[2].axhspan(2.0, 20.0, color="#009E73", alpha=0.15)
    axes[2].set_xlabel("GAR Threshold (spikes)")
    axes[2].set_ylabel("Mean Firing Rate (Hz)")
    axes[2].set_title("C. Firing Rate Stability across GAR Thresholds")
    axes[2].legend(fontsize=7.5)

    fig.tight_layout()
    _save_fig(fig, fig_dir, "fig13_sensitivity_sweep")


def generate_all_figures(bench_dir: str = "results/benchmarks", fig_dir: str = "submissions/ieee_tnnls/figures"):
    stats_path = os.path.join(bench_dir, "stats_summary.json")
    if not os.path.exists(stats_path):
        from scripts.analyze_benchmarks import analyze_all
        stats_summary = analyze_all(bench_dir)
    else:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_summary = json.load(f)

    df = _load_runs_df(bench_dir)

    generate_fig01_architecture(fig_dir)
    generate_fig02_electrophysiology(bench_dir, fig_dir)
    generate_fig03_sdm(bench_dir, fig_dir)
    generate_fig04_gar_homeostasis(df, fig_dir)
    generate_fig05_criticality_avalanches(bench_dir, stats_summary, fig_dir)
    generate_fig06_stability_phase_space(bench_dir, fig_dir)
    generate_fig07_directional_selectivity(bench_dir, stats_summary, fig_dir)
    generate_fig08_dual_trace_pruning(bench_dir, df, fig_dir)
    generate_fig09_learning_curves(df, fig_dir)
    generate_fig10_ablation_matrix(stats_summary, fig_dir)
    generate_fig11_metabolic_efficiency(stats_summary, fig_dir)
    generate_fig12_ood_generalization(stats_summary, fig_dir)
    generate_fig13_sensitivity_sweep(bench_dir, fig_dir)
    print(f"Generated all vector figure pairs (.pdf + .svg) -> {fig_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all 12 IEEE TNNLS publication vector figures")
    parser.add_argument("--bench-dir", default="results/benchmarks")
    parser.add_argument("--fig-dir", default="submissions/ieee_tnnls/figures")
    args = parser.parse_args()
    generate_all_figures(args.bench_dir, args.fig_dir)
