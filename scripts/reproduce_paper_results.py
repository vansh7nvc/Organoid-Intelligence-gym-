"""
One-Click Paper Reproduction Entrypoint for OrganoidEnv (Stage 5).

Supports automated CI / reviewer verification flags required by Master Verifier Plan §5.4:
  --run-unit-tests        Execute complete Stages 0-5 automated test suites
  --run-benchmarks        Run stability phase sweep and parallel multi-seed campaign
  --generate-all-figures  Run statistical analysis + generate all 12 vector figures
"""

import os
import sys
import subprocess
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def run_unit_tests():
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_stage0_platform.py",
        "tests/test_stage1_biophysical.py",
        "tests/test_stage2_checkpoints.py",
        "tests/test_stage3_performance.py",
        "tests/test_stage4_benchmarks.py",
        "tests/test_stage5_publication.py",
        "tests/test_adversarial_biophysical.py",
        "-v",
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_benchmarks(workers: int = 4, episodes: int = 500):
    subprocess.run([sys.executable, "scripts/run_stability_sweep.py"], cwd=ROOT_DIR, check=True)
    subprocess.run(
        [sys.executable, "scripts/run_benchmarks_parallel.py", "--workers", str(workers), "--episodes", str(episodes)],
        cwd=ROOT_DIR,
        check=True,
    )


def generate_all_figures():
    from scripts.analyze_benchmarks import analyze_all
    from scripts.generate_figures import generate_all_figures as _gen_figs

    bench_dir = os.path.join(ROOT_DIR, "results", "benchmarks")
    fig_dir = os.path.join(ROOT_DIR, "submissions", "ieee_tnnls", "figures")
    analyze_all(bench_dir)
    _gen_figs(bench_dir, fig_dir)


def main():
    parser = argparse.ArgumentParser(description="Reproduce all OrganoidEnv verification checks, benchmarks, and publication figures.")
    parser.add_argument("--run-unit-tests", action="store_true", help="Run Stages 0-5 + adversarial pytest suite")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run stability sweep + parallel multi-seed benchmarks")
    parser.add_argument("--generate-all-figures", action="store_true", help="Analyze benchmarks and render all 12 vector figures")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for parallel benchmarks")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed")
    args = parser.parse_args()

    if not (args.run_unit_tests or args.run_benchmarks or args.generate_all_figures):
        parser.print_help()
        return

    if args.run_unit_tests:
        run_unit_tests()
    if args.run_benchmarks:
        run_benchmarks(workers=args.workers, episodes=args.episodes)
    if args.generate_all_figures:
        generate_all_figures()


if __name__ == "__main__":
    main()
