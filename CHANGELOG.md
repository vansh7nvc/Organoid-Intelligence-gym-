# Changelog

All notable changes to the **OrganoidEnv** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-11-26

### Added
- **Interactive 3D Neuro-RL Workbench (`organoid_rl/platform/`)**: Multi-session FastAPI + WebSocket server (`SessionManager` with LRU/TTL eviction and non-blocking `asyncio.to_thread` stepping) paired with a Three.js 3D 500-neuron connectome viewer, Motion animations, 2D arena canvas, scrolling spike raster, and live single-flag ablation toggles (`scripts/launch_platform.py`).
- **Expanded 18-Condition Benchmark Suite & Diagnostics**: Added unregulated `C8` (`GAR-off, Metabolism-off`), step-size-matched (`step = 0.05`) baselines (`B1b`, `B2b`, `B4b`), 60-second long-horizon stability sweeps, Clauset et al. (2009) power-law goodness-of-fit diagnostics, and 36-configuration parameter sensitivity sweeps (`scripts/run_revision_diagnostics.py`).
- **Containerized Reproducibility & Demo Media**: Updated `Dockerfile` (pre-compiled C99 kernel, port `8000`), added `docker-compose.yml` (`platform` and `verify` profiles), `scripts/verify_reproducibility.py`, and live closed-loop demo recordings (`docs/assets/organoidenv_demo.{gif,mp4}`).

### Changed
- **Repository & Documentation**: Cleaned up root directory, added native Mermaid system architecture diagram and full environment/observation/reward specifications to `README.md`, and updated `CITATION.cff` and `pyproject.toml` to `v1.2.0`.

## [1.1.0] - 2026-11-11

### Added
- **Native Lazy-Decay C SNN Kernel (`organoid_rl/environment/_snn_kernel.c`, `fast_snn.py`)**: Numerical cross-check against Brian2 (`dv <= 1.99e-12`, `dw = 0.00e+00`, `dT <= 5.33e-15`) with batched chunk-power decay and canonical LIF mode (`SimpleSNNEnv`).
- **Biophysical Exhaustion Clamp & Orthogonal Ablations**: $\mathrm{K}_{\mathrm{ATP}}$-like hyperpolarizing exhaustion recovery in `neurons.py`, cross-episode weight preservation (`reset(options={"preserve_weights": True})`), and single-flag orthogonal toggles (`C1`–`C6`), `SimpleSNNEnv`, and `RandomAgent`.
- **Atomic 12-Key Deterministic Checkpointing (`organoid_rl/utils/checkpointing.py`)**: Bitwise-reproducible mid-episode and mid-training resume across Python/NumPy/PyTorch/Brian2/PER states.
- **Full Stages 0–5 Verification Suite**: Automated verification tests, 2D stability phase sweep, and multi-seed benchmark pipelines (`rliable` IQM + bootstrap CIs + 5-topology OOD eval).

## [1.0.0] - 2026-08-28

### Added
- **Core Gymnasium SNN Testbed**: Full `OrganoidEnv` implementation supporting 21-dimensional continuous observation and 8-action discrete motor stimulation.
- **Metabolic-Izhikevich Equations**: Added energy-fatigue state variable $E \in [-0.1, 1.0]$ enforcing sparse, energy-efficient spiking dynamics.
- **Sparse Expansion Encoder**: High-dimensional 256-neuron expansion layer with ~10.2% sparse activation ($k=26$) to resolve continuous spatial coordinates.
- **Global Activity Regulator (GAR)**: Tri-state homeostatic seizure suppression and hypoactive coma kickstarting.
- **Presynaptic Dual-Trace Plasticity**: Fast ($\tau_1 = 100\,\text{ms}$) and slow ($\tau_2 = 2500\,\text{ms}$) eligibility traces modulated by scalar reward.
- **Ablation Flags & Agents**: Native modular flags, Dueling Double DQN (PER + NoisyNet + HER), MLP baseline, and Stable-Baselines3 wrappers.
- **Packaging & CI/CD**: Standard `pyproject.toml`, `setup.py`, and GitHub Actions CI matrix (`.github/workflows/ci.yml`).
