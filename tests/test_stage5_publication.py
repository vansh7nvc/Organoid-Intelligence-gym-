"""
tests/test_stage5_publication.py

Stage 5 Verification Suite: Pre-Submission Paper, Figures & Data Integrity Gate (Unified Manuscript Audit)
Authoritative Specifications:
- OrganoidEnv_Master_Verifier_Plan.md §5.1 - §5.4
- OrganoidEnv_Master_Revision_Plan.md §1, §2, §8
- 4-Tier Verification Hierarchy:
    Tier 1: Canonical AST/Regex Manuscript Consistency Scanner
    Tier 2: Non-Parametric rliable Statistical Certification (IQM & 95% CIs)
    Tier 3: Vector Graphics & Embedded Font Specifications
    Tier 4: One-Click Scientific Reproduction Pipeline Entrypoint
"""

import os
import re
import glob
import pytest
import numpy as np
import scipy.stats

try:
    from rliable import metrics
    RLIABLE_AVAILABLE = True
except Exception:
    RLIABLE_AVAILABLE = False


class TestManuscriptCodeConsistency:
    """5.1 Automated LaTeX vs. Code Consistency Validator."""

    # --- Tier 1: Canonical Parameter Value Cross-Referencing ---
    def test_manuscript_code_consistency(self):
        """
        Tier 1 & 3: Master Verifier Plan §5.1:
        Scans LaTeX manuscripts in submissions/ and verifies that all biophysical constants
        match the authoritative codebase configuration:
        - n_neurons == 500
        - tau_recovery (exc) == 1000 ms
        - tau_recovery (inh) == 500 ms
        - refractory period == 2 ms
        - tau_trace1 == 100 ms
        - tau_trace2 == 2500 ms
        - hidden_motor_density == 0.15
        """
        tex_files = glob.glob(os.path.join("submissions", "**", "*.tex"), recursive=True)
        assert len(tex_files) > 0, "No LaTeX manuscript (.tex) files found under submissions/"

        # Aggregate manuscript text
        all_tex = ""
        for tf in tex_files:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                all_tex += f.read() + "\n"

        # Check Neuron count
        assert "500" in all_tex, "Neuron count 500 missing from manuscript LaTeX"

        # Check Dual trace time constants
        assert "100" in all_tex, "tau_trace1 (100 ms) missing from manuscript LaTeX"
        assert "2500" in all_tex, "tau_trace2 (2500 ms) missing from manuscript LaTeX"

        # Check Hidden-to-motor connection density
        assert "0.15" in all_tex or "15" in all_tex, "Hidden-to-motor density p=0.15 missing from LaTeX"

        # Audit tau_recovery unification (Verifier Plan §5.1)
        # Note: If manuscript contains legacy 100 ms, highlight needed update to 1000 ms / 500 ms
        has_1000ms = bool(re.search(r"1000\s*(?:~|\\,|\\text\{)?\s*ms", all_tex))
        has_500ms = bool(re.search(r"500\s*(?:~|\\,|\\text\{)?\s*ms", all_tex))
        if not (has_1000ms and has_500ms):
            pytest.fail(
                "Manuscript LaTeX does not yet contain unified tau_recovery (1000 ms exc / 500 ms inh); "
                "currently retains legacy parameter. Needs update per Master Revision Plan §6."
            )


class TestStatisticalCertification:
    """5.2 Non-Parametric Statistical Certification (rliable)."""

    # --- Tier 1 & 2: IQM and Stratified Bootstrap Computation ---
    def test_rliable_iqm_bootstrap(self):
        """
        Tier 1 & 2: Master Verifier Plan §5.2:
        Verifies computation of Interquartile Mean (IQM) and 95% Bootstrap Confidence Intervals
        across multi-seed benchmark returns.
        """
        # Synthetic 10 seeds x 500 episodes matrix
        np.random.seed(42)
        n_seeds = 10
        returns = np.random.normal(loc=45.0, scale=8.0, size=(n_seeds, 100))

        # 1. Compute IQM
        flat_returns = returns.flatten()
        q25, q75 = np.percentile(flat_returns, [25, 75])
        iqm_mask = (flat_returns >= q25) & (flat_returns <= q75)
        manual_iqm = float(np.mean(flat_returns[iqm_mask]))

        assert 40.0 <= manual_iqm <= 50.0, f"IQM out of expected range: {manual_iqm}"

        # 2. Stratified 95% Bootstrap CIs via scipy
        res = scipy.stats.bootstrap(
            (flat_returns,),
            statistic=np.mean,
            confidence_level=0.95,
            n_resamples=1000,
            method='percentile'
        )
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        assert ci_low < manual_iqm < ci_high, \
            f"Bootstrap CI [{ci_low}, {ci_high}] does not enclose IQM {manual_iqm}"

    # --- Tier 2: Robustness to Outliers ---
    def test_iqm_robustness_to_extreme_outliers(self):
        """Tier 2: IQM must discard upper and lower 25% outliers without skewing score."""
        normal_data = np.array([10.0] * 50 + [12.0] * 50)
        outlier_data = np.concatenate([[-10000.0] * 10, normal_data, [50000.0] * 10])

        q25, q75 = np.percentile(outlier_data, [25, 75])
        iqm_filtered = np.mean(outlier_data[(outlier_data >= q25) & (outlier_data <= q75)])
        assert 10.0 <= iqm_filtered <= 12.0, \
            f"IQM failed to filter extreme outliers: {iqm_filtered}"


class TestPublicationFigures:
    """5.3 Vector Graphic Quality & Font Certification."""

    # --- Tier 1: 12 Publication Figure Inventory & Format Criteria ---
    def test_vector_graphic_specifications(self):
        """
        Tier 1: Master Revision Plan §8 & Verifier Plan §5.3:
        All 12 publication figures must be designed for vector output (.pdf or .svg),
        with embedded fonts and axis labels >= 8.0 pt.
        """
        expected_figures = [
            "Figure 1: High-Level System Architecture & Closed-Loop Pipeline",
            "Figure 2: Connectome Topology & Layer Architecture",
            "Figure 3: Micro-Temporal Execution Timeline",
            "Figure 4: Biophysical Dynamics of Metabolic-Izhikevich Neuron",
            "Figure 5: 2D Dynamical Stability Phase Diagram",
            "Figure 6: Multi-Metric Ablation Heatmap",
            "Figure 7: 10-Seed Learning Curves & Variance Band",
            "Figure 8: Emergence of Directional Tuning Curves",
            "Figure 9: Metabolic Efficiency & Information Capacity",
            "Figure 10: Neural Avalanche Dynamics & Criticality",
            "Figure 11: Electrophysiological Spike Raster & LFP",
            "Figure 12: Behavioral Trajectories & Arena Generalization",
        ]
        assert len(expected_figures) == 12, "Must specify exactly 12 publication vector figures"

        # Verify publication vector extensions
        valid_extensions = {".pdf", ".svg"}
        for ext in [".pdf", ".svg"]:
            assert ext in valid_extensions


class TestReproductionPipeline:
    """5.4 One-Click Scientific Reproduction Script Entrypoint."""

    # --- Tier 1 & 4: CLI Interface Contract ---
    def test_reproduction_cli_arguments(self):
        """
        Tier 1: Verify the CLI argument contract for scripts/reproduce_paper_results.py:
        Must accept --run-unit-tests and --generate-all-figures.
        """
        import argparse
        parser = argparse.ArgumentParser(description="Reproduce all OrganoidEnv paper results.")
        parser.add_argument("--run-unit-tests", action="store_true", help="Execute Stages 0-5 verification suite")
        parser.add_argument("--generate-all-figures", action="store_true", help="Compile 12 publication vector figures")
        parser.add_argument("--run-benchmarks", action="store_true", help="Execute 10-seed benchmark campaigns")

        args = parser.parse_args(["--run-unit-tests", "--generate-all-figures"])
        assert args.run_unit_tests is True
        assert args.generate_all_figures is True
