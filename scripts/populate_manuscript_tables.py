"""
Populates empirical benchmark tables inside submissions/ieee_tnnls/manuscript.tex
strictly from results/benchmarks/stats_summary.json.
Supports both tagged markers and canonical table environments.
"""

import os
import re
import json
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.analyze_benchmarks import CONDITION_ORDER, CONDITION_LABELS


def format_table(summary: dict, include_markers: bool = False) -> str:
    conds = summary.get("conditions", {})
    pairwise = summary.get("pairwise_vs_c0", {})

    lines = []
    if include_markers:
        lines.append("%%BEGIN:TABLE_RESULTS%%")
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Empirical Late-Stage Performance (Last 100 Episodes): \\texttt{rliable} IQM [95\\% Bootstrap CI], Physiology, and Statistical Comparisons vs. C0}",
        "\\label{tab:results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{Return IQM [95\\% CI]} & \\textbf{Goal Rate (\\%)} & \\textbf{Rate (Hz)} & \\textbf{Mean ATP ($E$)} & \\textbf{DSI} & \\textbf{$p_{\\mathrm{FDR}}$ / $r_{\\mathrm{rb}}$ (vs C0)} \\\\",
        "\\midrule",
    ])

    for cond in CONDITION_ORDER:
        if cond not in conds:
            continue
        c = conds[cond]
        r = c["return_late"]
        g = c["goal_rate_late_pct"]
        hz = c["mean_rate_hz_late"]
        atp = c["mean_atp_late"]
        dsi = c["dsi_overall"]

        r_str = f"${r['iqm']:.1f}\\,[{r['ci_low']:.1f}, {r['ci_high']:.1f}]$"
        g_str = f"${g['iqm']:.1f}\\%$ $[{g['ci_low']:.1f}, {g['ci_high']:.1f}]$"
        hz_str = f"${hz['iqm']:.2f}$" if hz["iqm"] > 0 else "--"
        atp_str = f"${atp['iqm']:.3f}$" if hz["iqm"] > 0 else "--"
        dsi_str = f"${dsi['iqm']:.3f}$" if dsi["n"] > 0 else "--"

        if cond == "c0_full_model":
            stat_str = "Reference"
        else:
            pw = pairwise.get(cond, {}).get("return", {})
            p_val = pw.get("p_fdr", 1.0)
            r_rb = pw.get("rank_biserial_r", 0.0)
            p_fmt = "$<0.001$" if p_val < 0.001 else f"${p_val:.3f}$"
            stat_str = f"{p_fmt} / ${r_rb:+.2f}$"

        label = CONDITION_LABELS.get(cond, cond).replace("_", "\\_")
        lines.append(f"{label} & {r_str} & {g_str} & {hz_str} & {atp_str} & {dsi_str} & {stat_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    if include_markers:
        lines.append("%%END:TABLE_RESULTS%%")
    return "\n".join(lines)


def format_appendices(summary: dict, include_markers: bool = False) -> str:
    av = summary.get("avalanches", {})
    ood = summary.get("ood_generalization", {})

    lines = []
    if include_markers:
        lines.append("%%BEGIN:APPENDIX_TABLES%%")
    lines.extend([
        "\\section{Neural Criticality and Spike Dispersion Summary}",
        "\\label{app:criticality}",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Elephant $4\\text{ ms}$-Binned Criticality Metrics (IQM [95\\% CI]) Across Spiking Conditions}",
        "\\label{tab:criticality}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{Branching $\\sigma_{\\mathrm{MR}}$} & \\textbf{Avalanche $\\alpha$} & \\textbf{Fano Factor} \\\\",
        "\\midrule",
    ])
    for cond in CONDITION_ORDER:
        if cond not in av:
            continue
        c_av = av[cond]
        s = c_av["sigma"]
        a = c_av["alpha"]
        ff = c_av["fano_factor"]
        label = CONDITION_LABELS.get(cond, cond).split(":")[0]
        a_str = f"${a['iqm']:.2f}$ [$ {a['ci_low']:.2f}, {a['ci_high']:.2f} $]" if a["n"] > 0 else "--"
        lines.append(f"{label} & ${s['iqm']:.3f}$ [$ {s['ci_low']:.3f}, {s['ci_high']:.3f} $] & {a_str} & ${ff['iqm']:.2f}$ \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
        "\\section{Zero-Shot Out-of-Distribution Topology Generalization}",
        "\\label{app:ood}",
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{Zero-Shot Goal Completion IQM (\\%) Across Five Unseen Spatial Obstacle Topologies}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{T0 Nominal} & \\textbf{T1 Dual Circle} & \\textbf{T2 Slanted Walls} & \\textbf{T3 Central Maze} & \\textbf{T4 Tight Corridor} \\\\",
        "\\midrule",
    ])
    for cond in ["c0_full_model", "b1_direct_d3qn", "b2_mlp_dqn", "b3_simple_snn_lif", "b4_ppo_sb3", "b5_random"]:
        if cond not in ood:
            continue
        c_ood = ood[cond]
        vals = [f"${c_ood[t]['goal_rate']['iqm']:.1f}\\%$" for t in ["T0_nominal", "T1_dual_circle", "T2_slanted_walls", "T3_central_maze", "T4_tight_corridor"]]
        label = CONDITION_LABELS.get(cond, cond)
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    if include_markers:
        lines.append("%%END:APPENDIX_TABLES%%")
    return "\n".join(lines)


def populate(tex_path: str = "submissions/ieee_tnnls/manuscript.tex",
             stats_path: str = "results/benchmarks/stats_summary.json",
             include_markers: bool = False):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing {stats_path}. Run scripts/analyze_benchmarks.py first.")
    with open(stats_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    new_block = format_table(summary, include_markers=include_markers)
    if "%%BEGIN:TABLE_RESULTS%%" in tex:
        updated, count = re.subn(
            r"%%BEGIN:TABLE_RESULTS%%.*?%%END:TABLE_RESULTS%%",
            lambda _: new_block,
            tex,
            flags=re.DOTALL,
        )
    else:
        updated, count = re.subn(
            r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\caption\{Empirical Late-Stage Performance.*?\\label\{tab:results\}.*?\\end\{table\*\}",
            lambda _: new_block,
            tex,
            flags=re.DOTALL,
        )
    if count != 1:
        raise RuntimeError(f"Could not find unique TABLE_RESULTS block in {tex_path} (found {count})")

    app_block = format_appendices(summary, include_markers=include_markers)
    if "%%BEGIN:APPENDIX_TABLES%%" in updated:
        updated, acount = re.subn(
            r"%%BEGIN:APPENDIX_TABLES%%.*?%%END:APPENDIX_TABLES%%",
            lambda _: app_block,
            updated,
            flags=re.DOTALL,
        )
        if acount != 1:
            raise RuntimeError(f"Could not find unique APPENDIX_TABLES markers in {tex_path}")
    elif "\\section{Neural Criticality and Spike Dispersion Summary}" in updated:
        updated, acount = re.subn(
            r"\\section\{Neural Criticality and Spike Dispersion Summary\}.*?\\label\{app:ood\}.*?\\end\{table\*\}",
            lambda _: app_block,
            updated,
            flags=re.DOTALL,
        )
        if acount != 1:
            raise RuntimeError(f"Could not find unique APPENDIX_TABLES block in {tex_path}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Populated empirical LaTeX tables into {tex_path}", flush=True)


if __name__ == "__main__":
    populate()
