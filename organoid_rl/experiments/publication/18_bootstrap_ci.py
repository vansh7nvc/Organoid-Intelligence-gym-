"""
Bootstrap Confidence Interval Calculator for OrganoidEnv.

Computes 95% confidence intervals on success rates from existing
single-run data using 10,000 bootstrap resamples.

Usage:
    python 18_bootstrap_ci.py

Author: Vansh Sharma
License: MIT
"""

import os
import json
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bootstrap_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAP = 10000
CONFIDENCE = 0.95
SUCCESS_THRESHOLD = 30.0  # reward > 30 = success (matches 16_multiseed_training.py line 123)


def load_rewards(filename):
    """Load per-episode rewards from a JSON results file."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found at {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def compute_bootstrap_ci(outcomes, n_boot=N_BOOTSTRAP, confidence=CONFIDENCE):
    """
    Compute bootstrap confidence interval for the mean of binary outcomes.
    
    Parameters:
        outcomes: array of 0/1 values
        n_boot: number of bootstrap resamples
        confidence: confidence level (e.g. 0.95)
    
    Returns:
        mean, ci_low, ci_high
    """
    n = len(outcomes)
    boot_means = np.array([
        np.mean(np.random.choice(outcomes, size=n, replace=True))
        for _ in range(n_boot)
    ])
    
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(outcomes)
    
    return mean, ci_low, ci_high


def analyze_month6_results():
    """Analyze the Month 6 final results."""
    print("=" * 60)
    print("  Bootstrap Confidence Interval Analysis")
    print("=" * 60)
    
    # Try multiple possible result files
    for fname in ["results_month6_final.json", "results_month6_final-VANSH.json",
                   "results_month6_grand.json", "results_month6_grand_v2.json"]:
        data = load_rewards(fname)
        if data and "rewards" in data:
            rewards = np.array(data["rewards"])
            print(f"\nLoaded {fname}: {len(rewards)} episodes")
            
            # Derive binary success outcomes
            outcomes = (rewards > SUCCESS_THRESHOLD).astype(float)
            
            # Overall CI
            mean, ci_low, ci_high = compute_bootstrap_ci(outcomes)
            print(f"\n  Overall Success Rate: {mean:.1%}")
            print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
            print(f"  LaTeX: ${mean*100:.1f}\\%$ (95\\% CI: ${ci_low*100:.1f}$--${ci_high*100:.1f}\\%$)")
            
            # Per-stage CIs (using curriculum stage boundaries)
            stage_boundaries = {
                "Stage 1 (Basic)": (0, 100),
                "Stage 2 (Obstacles)": (100, 200),
                "Stage 3 (Multi-goal)": (200, 350),
                "Stage 4 (Full task)": (350, 500),
            }
            
            stage_results = {}
            print(f"\n  Per-Stage Breakdown:")
            print(f"  {'Stage':<25} {'Success':<12} {'95% CI':<20} {'N'}")
            print(f"  {'-'*70}")
            
            for stage_name, (start, end) in stage_boundaries.items():
                if end <= len(rewards):
                    stage_outcomes = outcomes[start:end]
                    s_mean, s_low, s_high = compute_bootstrap_ci(stage_outcomes)
                    print(f"  {stage_name:<25} {s_mean:.1%}       [{s_low:.1%}, {s_high:.1%}]    {len(stage_outcomes)}")
                    stage_results[stage_name] = {
                        "mean": float(s_mean),
                        "ci_low": float(s_low),
                        "ci_high": float(s_high),
                        "n": int(len(stage_outcomes))
                    }
                elif start < len(rewards):
                    stage_outcomes = outcomes[start:len(rewards)]
                    s_mean, s_low, s_high = compute_bootstrap_ci(stage_outcomes)
                    print(f"  {stage_name:<25} {s_mean:.1%}       [{s_low:.1%}, {s_high:.1%}]    {len(stage_outcomes)} (partial)")
                    stage_results[stage_name] = {
                        "mean": float(s_mean),
                        "ci_low": float(s_low),
                        "ci_high": float(s_high),
                        "n": int(len(stage_outcomes)),
                        "partial": True
                    }
                else:
                    print(f"  {stage_name:<25} N/A (data ends at ep {len(rewards)})")
            
            # Also try using stage_goals/stage_episodes if available
            if "stage_goals" in data and "stage_episodes" in data:
                print(f"\n  From stage_goals/stage_episodes metadata:")
                for stage_key in sorted(data["stage_goals"].keys()):
                    goals = data["stage_goals"][stage_key]
                    eps = data["stage_episodes"][stage_key]
                    if eps > 0:
                        rate = goals / eps * 100
                        print(f"    Stage {stage_key}: {goals}/{eps} = {rate:.1f}%")
            
            # Save results
            output = {
                "source_file": fname,
                "n_episodes": len(rewards),
                "success_threshold": SUCCESS_THRESHOLD,
                "n_bootstrap": N_BOOTSTRAP,
                "confidence_level": CONFIDENCE,
                "overall": {
                    "mean": float(mean),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_success": int(np.sum(outcomes)),
                    "n_total": int(len(outcomes))
                },
                "per_stage": stage_results
            }
            
            out_path = os.path.join(OUTPUT_DIR, f"bootstrap_ci_{fname}")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\n  Results saved to: {out_path}")
            
            print()


def analyze_ablation_results():
    """Check existing ablation data for any usable results."""
    print("\n" + "=" * 60)
    print("  Ablation Data Check")
    print("=" * 60)
    
    for fname in ["results_no_sdm_morph.json", "results_no_homeostasis.json",
                   "results_full.json"]:
        data = load_rewards(fname)
        if data:
            rewards = np.array(data.get("rewards", []))
            goals = data.get("goals_reached", "N/A")
            episodes = data.get("episodes", len(rewards))
            tag = data.get("tag", fname)
            print(f"\n  {tag}: {len(rewards)} episodes, goals={goals}/{episodes}")
            if len(rewards) > 0:
                outcomes = (rewards > SUCCESS_THRESHOLD).astype(float)
                mean, ci_low, ci_high = compute_bootstrap_ci(outcomes)
                print(f"    Success Rate: {mean:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")


if __name__ == "__main__":
    np.random.seed(42)
    analyze_month6_results()
    analyze_ablation_results()
    print("\n" + "=" * 60)
    print("  Bootstrap analysis complete!")
    print("=" * 60)
