# Project: OrganoidEnv Master Revision & Verification Acceleration

## Architecture
OrganoidEnv is a biophysically grounded reinforcement learning platform embedding biological neural tissue simulations (Izhikevich SNN with metabolic energy constraints, astrocyte glutamate reuptake, synaptic dual-trace STDP, and structural plasticity) into closed-loop sensorimotor environments.

```
       [ Gymnasium Sensorimotor Interface ]
                      │
                      ▼
         [ OrganoidEnv (core.py) ]
          ├── Sensory Encoding (PoissonInput / Current Injection)
          ├── SNN Reservoir (256 SDM + 144 Hidden: 115 RS / 29 FS)
          ├── Motor Readout (100 Motor Neurons, Excitatory RS)
          ├── Metabolic Governor (ATP dynamics, spike cost, recovery)
          ├── Astrocytic Glutamate Reuptake (GAR stabilizer)
          └── Dual-Trace STDP & Structural Pruning
                      │
                      ▼
            [ Reinforcement Learner ]
       (D3QN PER / Baselines: MLP, LIF, PPO)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Python 3.11 Environment Setup | Pin dependencies: brian2==2.9.0, gymnasium==1.2.3, torch, pytest, rliable, elephant | M1 | Verifier §0.2 |
| 2 | Canonical OrganoidConfig | Standalone configuration class encapsulating all biophysical/RL parameters | M1 | Verifier §5.1 |
| 3 | Unified Multi-Engine Seeding | Centralized `set_seed(seed)` setting torch, numpy, random, brian2.seed | M1 | Verifier §0.1 |
| 4 | Parameter Harmonization | Unify `tau_recovery` at 1000 ms exc / 500 ms inh and hidden-motor density p=0.15 | M1 | Verifier R2 |
| 5 | Refractory Period Enforcement | Add `refractory=2*ms` to NeuronGroup definitions in neurons.py and core.py | M1 | Verifier §1.3 |
| 6 | Dale's Principle Realignment | Disentangle Motor group from P_inh; enforce Dale's sign convention | M1 | Verifier §1.4 |
| 7 | Metabolic Governor Invariant | Metabolic clamp holds under continuous 50 pA current; zero firing at E <= 0 | M1 | Verifier §1.2 |
| 8 | Dual-Trace Exponential Decay | Presynaptic traces decay per analytical form tau_fast=20ms, tau_slow=100ms | M1 | Verifier §1.5 |
| 9 | Brian2 Canonical Equivalence | Isolated neuron dynamics match standalone Brian2 reference within tolerance | M1 | Verifier §1.1 |
| 10 | Gymnasium check_env Compliance | Pass gymnasium.utils.env_checker.check_env(OrganoidEnv()) deterministically | M1 | Verifier §0.3 |
| 11 | 12-Component State Serialization | SNN weights, traces, ATP, v, u, mask, GAR, nets, optimizer, RNG states | M2 | Verifier §2.1 |
| 12 | Atomic Checkpoint Replacement | Save to `.tmp` followed by `os.replace` to prevent partial/corrupt writes | M2 | Verifier §0.4 |
| 13 | Resume-Equivalence Verification | 100-step trajectory identical whether run continuously or interrupted at step 50 | M2 | Verifier §2.2 |
| 14 | Profiling & Vectorization | Verify zero Python loops over individual neurons/synapses in step() | M3 | Verifier §3.1 |
| 15 | Sub-10s Throughput Optimization | Optimize codegen/step latency to achieve <10s per 400-step episode | M3 | Verifier §3.2 |
| 16 | Memory Leak Audit | 100-episode run memory growth bounded below 50MB (tracemalloc) | M3 | Verifier §3.3 |
| 17 | Constructor Ablation Toggles | Add `use_plasticity`, `random_snn`, `use_metabolism`, `use_pruning` to OrganoidEnv | M4 | Revision §4 |
| 18 | Canonical LIF Baseline | True Leaky Integrate-and-Fire baseline class (replacing Izhikevich SimpleSNN) | M4 | Verifier §4.2 |
| 19 | Direct D3QN SkeletonEnv Runner | Verified baseline execution on SkeletonEnv with PER/NoisyNet | M4 | Verifier §4.2 |
| 20 | MLP Baseline Runner | 3-layer MLP baseline on SkeletonEnv | M4 | Verifier §4.2 |
| 21 | PPO Baseline Runner | SB3 PPO runner for benchmark comparison | M4 | Verifier §4.2 |
| 22 | Random Baseline Runner | Uniform random action baseline runner | M4 | Verifier §4.2 |
| 23 | 2D Dynamical Stability Sweep | Sweep I_stim x w_inh (20 points, firing rates & spectral radius) | M4 | Verifier §4.3 |
| 24 | Unified Benchmark Campaign Runner | Orchestrator for 10 seeds x 500 eps nominal, 6 ablations (C0-C6), baselines | M4 | Verifier §4.1 |
| 25 | Directional Selectivity Index (DSI) | Polar radar plots of motor neuron directional tuning curves | M5 | Verifier §5.3 |
| 26 | Neural Avalanche Criticality | Power law fit P(S) ~ S^-1.5 and branching ratio sigma ~ 1.0 | M5 | Verifier §5.4 |
| 27 | 12 Publication Vector Figures | Generate all 12 publication-ready PDF/SVG figures per Revision Plan Sec 8 | M5 | Revision §8 |
| 28 | rliable Statistical Certification | Interquartile Mean (IQM), performance profiles, 95% stratified bootstrap CIs | M5 | Verifier §5.2 |
| 29 | Code-Paper Consistency Validator | Automated AST/regex script verifying parameter values between code & manuscript | M5 | Verifier §5.1 |
| 30 | IEEE TNNLS Manuscript Polish | Update LaTeX manuscript to single unified scientific/technological paper | M5 | Revision §1 |
| 31 | E2E Test Suite Pass (Final Milestone) | 100% pass of all Tier 1-4 tests published in TEST_READY.md | M5 | Project Pattern |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| T_E2E | E2E Testing Track | Author opaque-box E2E test suites (Tiers 1-4) & publish TEST_READY.md | none | DONE (TEST_INFRA.md, 53 tests in tests/, TEST_READY.md) |
| M1 | Codebase Alignment & Invariants | Stage 0 determinism, config, parameter harmonization, Stage 1 invariants | none | IN_GATE (Reviewers, Challengers, Auditor) |
| M2 | 12-Component Checkpointing | Stage 2 atomic serialization, get_state_dict/load_state_dict, resume test | M1 | PLANNED |
| M3 | Vectorization & Throughput | Stage 3 zero-loop verification, sub-10s/ep optimization, memory audit | M1 | PLANNED |
| M4 | Compute Campaign & Baselines | Stage 4 ablation toggles, LIF baseline, benchmark runners, stability sweep | M2, M3 | PLANNED |
| M5 | Scientific Deliverables & Manuscript | Stage 5 DSI, criticality, 12 figures, rliable, consistency, E2E test pass | M4, T_E2E | PLANNED |

## Interface Contracts

### OrganoidConfig (`organoid_rl/environment/core.py`)
```python
@dataclass
class OrganoidConfig:
    n_neurons: int = 500
    n_motor: int = 100
    tau_recovery_exc: float = 1000.0  # ms
    tau_recovery_inh: float = 500.0   # ms
    refractory_period: float = 2.0    # ms
    hidden_motor_density: float = 0.15
    use_sdm: bool = True
    use_morphology: bool = True
    use_dual_trace: bool = True
    use_stabilizer: bool = True
    use_motor_mapping: bool = True
    use_plasticity: bool = True
    random_snn: bool = False
    use_metabolism: bool = True
    use_pruning: bool = True
```

### Seeding (`organoid_rl/utils/seeding.py`)
```python
def set_seed(seed: int) -> None:
    """Sets random seeds across random, numpy, torch, and brian2."""
```

### Checkpoint Interface (`OrganoidEnv` & `DQNAgent`)
```python
def get_state_dict(self) -> dict:
    """Returns dictionary of state components."""

def load_state_dict(self, state_dict: dict) -> None:
    """Restores exact state from state dictionary."""

def save_checkpoint(filepath: str, state_dict: dict) -> None:
    """Atomically writes checkpoint using .tmp file and os.replace."""
```

## Code Layout
- `organoid_rl/environment/`: Environment definitions (`core.py`, `neurons.py`, `synapses.py`, `rewards.py`, `baseline_envs.py`)
- `organoid_rl/agents/`: RL agents (`dqn_agent.py`, `baseline_agent.py`, `sb3_baselines.py`, `lif_baseline.py`)
- `organoid_rl/utils/`: Common utilities (`seeding.py`, `checkpointing.py`)
- `organoid_rl/analysis/`: Scientific analysis (`dsi.py`, `criticality.py`, `statistics.py`)
- `tests/`: Verification test suites (`test_stage0_platform.py`, `test_stage1_biophysical.py`, `test_stage2_checkpoints.py`, `test_stage3_performance.py`, `test_stage4_benchmarks.py`, `test_stage5_publication.py`)
- `scripts/`: Benchmark runners & tools (`run_benchmarks.py`, `stability_sweep.py`, `generate_figures.py`, `validate_manuscript.py`, `reproduce_paper_results.py`)
- `submissions/`: LaTeX manuscript (`ieee_tnnls/` or `frontiers_in_neuroinformatics/`)
