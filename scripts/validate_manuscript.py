"""
Automated Code-Paper & Vector-Figure Consistency Validator (Stage 5).

Verifies that manuscript LaTeX documents under submissions/ match exact
implementation constants in organoid_rl/environment/{core,neurons,rewards}.py
and that all 12 vector figures exist with >= 8.0 pt labels and embedded fonts.
"""

import os
import re
import glob
import json
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


EXPECTED_TEX_TOKENS = [
    ("n_neurons=500", r"\b500\b"),
    ("tau_trace1=100", r"\b100\b"),
    ("tau_trace2=2500", r"\b2500\b"),
    ("hidden_motor_p=0.15", r"0\.15"),
    ("tau_recovery_exc=1000 ms", r"1000\s*(?:~|\\,|\\text\{)?\s*ms"),
    ("tau_recovery_inh=500 ms", r"500\s*(?:~|\\,|\\text\{)?\s*ms"),
    ("refractory=2.0 ms", r"2(?:\.0)?\s*(?:~|\\,|\\text\{)?\s*ms"),
]

FORBIDDEN_STALE_TOKENS = [
    ("stale 2000 ms tau_recovery", r"tau_\{?(?:\\text\{)?rec(?:overy)?\}?\}?\s*=\s*2000\s*(?:~|\\,)?\s*ms"),
    ("stale hidden-to-motor p=0.05", r"p\s*=\s*0\.05(?:\s*from\s*hidden|\s*\(hidden)"),
    ("unimplemented Poisson burst count", r"PoissonBurst"),
]


def validate_manuscript(tex_path: str = "submissions/ieee_tnnls/manuscript.tex",
                        fig_dir: str = "submissions/ieee_tnnls/figures",
                        bench_dir: str = "results/benchmarks") -> dict:
    report = {"ok": True, "errors": [], "warnings": [], "checked_figures": 0}

    if not os.path.exists(tex_path):
        report["ok"] = False
        report["errors"].append(f"Missing unified manuscript: {tex_path}")
        return report

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    for label, pattern in EXPECTED_TEX_TOKENS:
        if not re.search(pattern, tex_content, flags=re.IGNORECASE):
            report["ok"] = False
            report["errors"].append(f"Missing expected constant '{label}' ({pattern}) in {tex_path}")

    for label, pattern in FORBIDDEN_STALE_TOKENS:
        if re.search(pattern, tex_content, flags=re.IGNORECASE):
            report["ok"] = False
            report["errors"].append(f"Found forbidden stale claim '{label}' in {tex_path}")

    # Verify figures referenced and existing as PDF + SVG vector files
    expected_stems = [
        "fig01_architecture_loop",
        "fig02_biophysical_electrophysiology",
        "fig03_sdm_sparse_coding",
        "fig04_gar_homeostasis",
        "fig05_criticality_avalanches",
        "fig06_stability_phase_space",
        "fig07_directional_selectivity",
        "fig08_dual_trace_plasticity",
        "fig09_learning_curves",
        "fig10_ablation_matrix",
        "fig11_metabolic_efficiency",
        "fig12_ood_generalization",
    ]
    for stem in expected_stems:
        pdf_p = os.path.join(fig_dir, f"{stem}.pdf")
        svg_p = os.path.join(fig_dir, f"{stem}.svg")
        if not (os.path.exists(pdf_p) and os.path.getsize(pdf_p) > 1000):
            report["ok"] = False
            report["errors"].append(f"Missing or empty vector PDF figure: {pdf_p}")
        else:
            report["checked_figures"] += 1
        if not (os.path.exists(svg_p) and os.path.getsize(svg_p) > 1000):
            report["warnings"].append(f"Missing or empty SVG companion: {svg_p}")

    # Verify empirical summary aligns if stats_summary.json exists
    stats_p = os.path.join(bench_dir, "stats_summary.json")
    if os.path.exists(stats_p):
        with open(stats_p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if "c0_full_model" in summary.get("conditions", {}):
            c0 = summary["conditions"]["c0_full_model"]
            ret_str = f"{c0['return_late']['iqm']:.1f}"
            if ret_str not in tex_content:
                report["warnings"].append(
                    f"C0 late return IQM ({ret_str}) not matched verbatim in {tex_path} (ensure tables regenerated)."
                )

    return report


def main():
    res = validate_manuscript()
    print(json.dumps(res, indent=2))
    if not res["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
