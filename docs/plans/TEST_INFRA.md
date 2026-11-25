# TEST_INFRA: OrganoidEnv Verification Infrastructure & Test Methodology

## 1. Executive Summary & Verification Philosophy

OrganoidEnv is a biophysically grounded closed-loop reinforcement learning platform uniting biological cortical network simulations (Metabolic-Izhikevich SNN, astrocytic glutamate regulation, dopamine-modulated dual eligibility traces, and structural pruning) with standard Gymnasium sensorimotor control.

The verification test infrastructure enforces the foundational thesis of the project:
> **"The Science explains the Technology, and the Technology empirically proves the Science."**

To guarantee publication-grade scientific validity, algorithmic correctness, and computational reproducibility across hardware environments, this testing harness applies an authoritative, specification-driven, opaque-box testing methodology across **Stages 0 through 5**, structured according to a **4-Tier Verification Hierarchy**.

---

## 2. 4-Tier Verification Hierarchy

Each verification stage is exercised across four rigorous testing tiers:

```
                  ▲
                 / \
                /   \     Tier 4: Real-World Scenarios & Production Runs
               / Tier\    (End-to-end episodes, resume equivalence, publication pipeline)
              /   4   \
             /─────────\
            /  Tier 3   \  Tier 3: Cross-Feature Interactions & System Invariants
           /─────────────\ (GAR + Metabolism, Plasticity + Dale's Law, Multimodal inputs)
          /    Tier 2     \ Tier 2: Boundary, Numerical & Corner Cases
         /─────────────────\(Zero energy, extreme current, NaN recovery, boundary collisions)
        /      Tier 1       \ Tier 1: Canonical Feature & Contract Verification
       /─────────────────────\(Happy path, parameter defaults, ODE accuracy, API compliance)
```

1. **Tier 1 — Canonical Feature Coverage (Happy Path & API Contracts)**:
   - Validates that every class, function, and equation implements its nominal specification.
   - Verifies Gymnasium API signatures (`reset() -> (obs, info)`, `step() -> (obs, reward, term, trunc, info)`).
   - Validates biophysical default constants (`n_neurons=500`, $\tau_{\text{recovery}}=1000/500\text{ ms}$, $t_{\text{ref}}=2.0\text{ ms}$, $p=0.15$).

2. **Tier 2 — Boundary, Numerical & Corner Cases (Stress & Limits)**:
   - Tests extreme inputs, zero energy reserves ($E_i \le 0$), continuous supra-threshold injection ($I=50\text{ pA}$), zero or negative currents.
   - Evaluates numerical stability under potential floating-point overflow or NaN conditions.
   - Verifies edge handling on physical arena boundaries ($x, y \in [0, 1]$) and obstacle perimeters.

3. **Tier 3 — Cross-Feature Interactions & Multi-System Invariants**:
   - Asserts that coupled dynamics maintain biophysical invariants concurrently:
     - Metabolic governor arrests spiking without corrupting membrane reset states.
     - Global Activity Regulator (GAR) rescues quiescent or epileptiform states without overriding Dale's principle.
     - Dopamine-modulated dual eligibility traces update weights without exceeding bounds $[0.0, 100.0]$ or violating sign polarity.
     - Checkpoint serialization captures all 12 coupled subsystems simultaneously.

4. **Tier 4 — Real-World End-to-End Scenarios & Publication Certification**:
   - Validates bitwise resume-equivalence across interrupted training trajectories.
   - Profiles execution throughput ensuring sub-10s per episode.
   - Cross-references the LaTeX manuscript against authoritative codebase configuration.
   - Certifies statistical integrity using non-parametric estimation (`rliable` IQM and bootstrap CIs).

---

## 3. Test Suite Architecture & File Layout

All automated verification tests are housed in the root `tests/` directory, cleanly segregated by Stage:

```
Organoid Intelligence/
├── tests/
│   ├── __init__.py
│   ├── test_stage0_platform.py       # Stage 0: Determinism, Dependencies & Gymnasium API
│   ├── test_stage1_biophysical.py    # Stage 1: Brian2 Equivalence & Biophysical Invariants
│   ├── test_stage2_checkpoints.py    # Stage 2: 12-Component State Serialization & Resume Equivalence
│   ├── test_stage3_performance.py    # Stage 3: Profiling, Zero-Loop Audit & Memory Leak Tracking
│   ├── test_stage4_benchmarks.py     # Stage 4: Scientific Discoveries, Ablations & Baselines
│   └── test_stage5_publication.py    # Stage 5: LaTeX Consistency, rliable Stats & 12 Figures
├── TEST_INFRA.md                     # This document: Test architecture & methodology
└── TEST_READY.md                     # Test runner commands, coverage & execution status
```

---

## 4. Formal Stage-by-Stage Verification Mapping

| Stage | Module | Target Specification & Pass Rule | Authoritative Source | Primary Test Class / Functions |
|:---:|:---|:---|:---|:---|
| **0** | Platform Integrity | **0.1 Multi-Engine Seeding**: Bitwise reproducibility across Python `random`, `numpy`, `torch` (CPU/CUDA), and `brian2`. Identical runs on seed 42 produce exact matching trajectories. | Verifier Plan §0.1, `PROJECT.md` | `TestMultiEngineSeeding`, `test_bitwise_reproducibility` |
| **0** | Platform Integrity | **0.2 Pinned Dependencies**: Frozen manifests verified; critical libraries (`gymnasium>=0.29`, `brian2>=2.5.4`, `torch>=2.1`) imported with zero conflicts. | Verifier Plan §0.2 | `TestDependencyManifest`, `test_pinned_dependency_imports` |
| **0** | Platform Integrity | **0.3 Gymnasium API**: Passes `check_env(OrganoidEnv())` without errors. Observation space shape `(21,)` in `[-1, 2]`; action space `Discrete(8)`. | Verifier Plan §0.3 | `TestGymnasiumCompliance`, `test_check_env` |
| **0** | Platform Integrity | **0.4 Atomic Checkpoint Serialization**: Atomic writes via `.tmp` staging and `os.replace` preventing file corruption upon interrupted saves. | Verifier Plan §0.4 | `TestAtomicSerialization`, `test_atomic_checkpoint_replace` |
| **1** | Biophysical Invariants | **1.1 Brian2 Numerical Equivalence**: Custom/canonical Euler step matches reference; voltages remain bounded $[-100, 60]\text{ mV}$ with zero NaNs. | Verifier Plan §1.1 | `TestBrian2NumericalEquivalence`, `test_izhikevich_numerical_accuracy` |
| **1** | Biophysical Invariants | **1.2 Metabolic Governor Invariant**: Continuous $I=50\text{ pA}$ depletes ATP reserve $E \le 0.10$. Zero firing occurs while $E \le 0$ despite $v \ge 30\text{ mV}$. Asymptotic recovery with $\tau_{\text{recovery}}$. | Verifier Plan §1.2 | `TestMetabolicGovernor`, `test_metabolic_governor_binding` |
| **1** | Biophysical Invariants | **1.3 Refractory Period Enforcement**: Absolute biological refractory period $t_{\text{ref}}=2.0\text{ ms}$ strictly enforced; $\min(\text{ISI}) \ge 2.0\text{ ms}$. | Verifier Plan §1.3 | `TestRefractoryPeriod`, `test_refractory_enforcement` |
| **1** | Biophysical Invariants | **1.4 Dale's Principle & Polarity**: Strictly non-negative excitatory conductances ($+w$) and inhibitory hyperpolarizations ($-w$). Zero sign reversals. 80% exc / 20% inh. | Verifier Plan §1.4 | `TestDalesPrinciple`, `test_dales_principle_compliance` |
| **1** | Biophysical Invariants | **1.5 Presynaptic Trace Exponential Decay**: Dual traces decay with $\tau_1=100\text{ ms}$ and $\tau_2=2500\text{ ms}$. Residual error vs analytical solution $< 10^{-5}$ across 500 ms. | Verifier Plan §1.5 | `TestPresynapticTraceDecay`, `test_trace_exponential_decay` |
| **2** | Checkpointing | **2.1 12 Mandatory State Components**: Serialization captures all 12 components (weights, fast/slow traces, ATP, v, u, mask, GAR counters, online net, target net, optimizer, RNG states). | Verifier Plan §2.1 | `TestCheckpointComponents`, `test_mandatory_12_components_present` |
| **2** | Checkpointing | **2.2 Resume-Equivalence Verification**: 20-step uninterrupted run produces bitwise identical trajectories and matching SHA-256 state hash compared to interrupted-and-resumed run. | Verifier Plan §2.2 | `TestResumeEquivalence`, `test_resume_equivalence_bitwise` |
| **3** | Performance | **3.1 Episode Latency & Throughput**: Wall-clock latency $< 10.0\text{ s}$ per 400-step episode ($< 25\text{ ms}$ per 100 ms step). | Verifier Plan §3.1 | `TestThroughputLatency`, `test_episode_wall_clock_latency` |
| **3** | Performance | **3.2 Zero Python Neuron/Synapse Loops**: AST inspection verifies exactly 0 Python `for` loops iterating over individual neurons ($N=500$) or synapses in time-critical paths. | Verifier Plan §3.2 | `TestZeroPythonLoops`, `test_ast_zero_neuron_loops` |
| **3** | Performance | **3.3 Memory Leak Audit**: Tracemalloc audit confirms peak heap allocation growth $< 50\text{ MB}$ across repeated simulation episodes. | Verifier Plan §3.3 | `TestMemoryLeakAudit`, `test_heap_memory_leak_bounded` |
| **4** | Scientific Proofs | **4.1 Firing Rate Physiological Range**: Population mean operational firing rate $\bar{f} \in [5.0, 15.0]\text{ Hz}$ (calibrated nominally at $8.5 \pm 2.1\text{ Hz}$). | Verifier Plan §4.1 | `TestPhysiologicalFiringRate`, `test_firing_rate_range` |
| **4** | Scientific Proofs | **4.2 Neural Avalanche & Criticality**: Power-law scaling $P(S) \propto S^{-\alpha}$ ($\alpha \in [1.3, 1.7]$) and branching ratio $\sigma \in [0.9, 1.1]$. | Verifier Plan §4.2 | `TestCriticalityDynamics`, `test_neural_avalanche_criticality` |
| **4** | Scientific Proofs | **4.3 Directional Selectivity Index (DSI)**: Motor readout directional tuning: Plastic $\text{DSI} \ge 0.50$ vs Frozen/Random $\text{DSI} \le 0.20$. | Verifier Plan §4.3 | `TestDirectionalSelectivity`, `test_dsi_tuning_polar_curves` |
| **4** | Scientific Proofs | **4.4 6-Way Ablation Matrix**: Orthogonal gating of C1 (Plasticity), C2 (Random SNN), C3 (Metabolism), C4 (GAR), C5 (Pruning), C6 (Dual Trace). | Verifier Plan §4.4 | `TestAblationOrthogonality`, `test_ablation_matrix_toggles` |
| **4** | Scientific Proofs | **4.5 2D Stability Phase Space**: Phase grid over $I_{\text{stim}} \in [5, 50]\text{ pA} \times w_{\text{inh}} \in [5, 30]\text{ nS}$ resolving seizure, coma, and homeostatic regimes. | Verifier Plan §4.5 | `TestStabilityPhaseSpace`, `test_2d_phase_space_regimes` |
| **5** | Publication Gate | **5.1 LaTeX vs Code Consistency**: Automated AST/regex validator checking all biophysical constants in manuscript against `OrganoidConfig`. | Verifier Plan §5.1 | `TestManuscriptCodeConsistency`, `test_manuscript_code_consistency` |
| **5** | Publication Gate | **5.2 rliable Statistical Certification**: Non-parametric Interquartile Mean (IQM) and 95% stratified bootstrap confidence intervals. | Verifier Plan §5.2 | `TestStatisticalCertification`, `test_rliable_iqm_bootstrap` |
| **5** | Publication Gate | **5.3 12 Vector Publication Figures**: Verification of vector format (`.pdf`/`.svg`), embedded TrueType/Type 1 fonts, and label sizes $\ge 8.0\text{ pt}$. | Verifier Plan §5.3 | `TestPublicationFigures`, `test_vector_graphic_specifications` |
| **5** | Publication Gate | **5.4 Reproduction Script Entrypoint**: CLI reproduction runner (`scripts/reproduce_paper_results.py`) execution test. | Verifier Plan §5.4 | `TestReproductionPipeline`, `test_reproduction_cli_entrypoint` |

---

## 5. Test Isolation & Scope Management

1. **Brian2 Scope Isolation**:
   Brian2 maintains global equation registers and network clocks. Every test that instantiates a Brian2 object must invoke `brian2.start_scope()` in `setup_method` / `teardown_method` or fixture context to prevent cross-test namespace leaks.

2. **Filesystem Isolation**:
   All checkpoint and filesystem tests must execute inside temporary directory sandboxes using Python's `tempfile.TemporaryDirectory()`, ensuring that zero orphan files pollute the workspace.

3. **Deterministic Seed Control**:
   All tests requiring stochastic generation must explicitly seed the environment and random generators (`set_seed(42)`), guaranteeing independent and reproducible results regardless of test runner order.
