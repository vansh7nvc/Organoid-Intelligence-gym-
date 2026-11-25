# Changelog

All notable changes to the **OrganoidEnv** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-11-11

### Added
- **Native Lazy-Decay C SNN Kernel (`organoid_rl/environment/_snn_kernel.c`, `fast_snn.py`)**: Exact Brian2-validated integration (`dv <= 1.99e-12`, `dw = 0.00e+00`, `dT <= 5.33e-15`) with batched chunk-power decay and canonical LIF mode (`SimpleSNNEnv`).
- **Biophysical Exhaustion Clamp & Orthogonal Ablations**: $K_{\mathrm{ATP}}$-like hyperpolarizing exhaustion recovery in `neurons.py`, cross-episode weight preservation (`reset(options={"preserve_weights": True})`), and single-flag orthogonal toggles (`C1`–`C6`), `SimpleSNNEnv`, and `RandomAgent`.
- **Atomic 12-Key Deterministic Checkpointing (`organoid_rl/utils/checkpointing.py`)**: Bitwise-reproducible mid-episode and mid-training resume across Python/NumPy/PyTorch/Brian2/PER states.
- **Full Stages 0–5 Verification & Empirical Benchmarks**: 87-test automated verification suite, 60-point 2D stability phase sweep, 65-run multi-seed curriculum campaign (`rliable` IQM + bootstrap CIs + Elephant criticality + 5-topology OOD eval), 12 vector `.pdf`/`.svg` figures, and unified IEEE TNNLS manuscript (`submissions/ieee_tnnls/manuscript.tex`).

## [1.0.0] - 2026-08-28

### Added
- **Core Gymnasium SNN Testbed**: Full `OrganoidEnv` implementation supporting 21-dimensional continuous observation and 8-action discrete motor stimulation.
- **Metabolic-Izhikevich Equations**: Added ATP dynamic state variable $E \in [0, 1]$ enforcing sparse, energy-efficient biological dynamics ($\approx 8.5\,\text{Hz}$).
- **Sparse Distributed Memory (SDM)**: High-dimensional 256-neuron expansion layer with 10% sparse activity to resolve continuous spatial coordinates.
- **Global Activity Regulator (GAR)**: Homeostatic seizure suppression (>35 Hz) and hypoactive coma kickstarting (<1.0 Hz).
- **Dopamine-Modulated Dual-Trace Plasticity ($D^3$)**: Millisecond-scale causal STDP trace ($\tau=100\,\text{ms}$) combined with second-scale reward eligibility trace ($\tau=2500\,\text{ms}$).
- **Ablation Flags**: Native modular flags (`use_sdm`, `use_morphology`, `use_dual_trace`, `use_stabilizer`, `use_motor_mapping`) for systematic dynamical ablations.
- **Dueling Double DQN Agent**: Outer optimization agent featuring Prioritized Experience Replay (PER), Noisy Linear Networks, and Hindsight Experience Replay (HER).
- **Baseline Models**: Standard MLP baseline (`MLPBaselineAgent`) and Stable-Baselines3 wrappers (`sb3_baselines.py`).
- **Reproducibility Suite**: Automated 1-click Google Colab notebook (`notebooks/colab_gpu_training.ipynb`), multi-seed publication pipelines, bootstrap confidence interval estimators, and energy estimation scripts.
- **Interactive Examples**: Added `examples/01_quickstart.py`, `examples/02_evaluate_pretrained_agent.py`, and `examples/03_run_ablation_comparison.py`.
- **Packaging & CI/CD**: Standard `pyproject.toml`, `setup.py`, GitHub Actions CI matrix (`.github/workflows/ci.yml`), and JOSS paper compilation check (`.github/workflows/paper.yml`).
- **Journal Submissions**: Formal manuscripts and supplementary material for *Frontiers in Neuroinformatics* and the *Journal of Open Source Software (JOSS)*.
