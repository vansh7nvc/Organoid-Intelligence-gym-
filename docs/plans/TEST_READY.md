# TEST_READY: OrganoidEnv Verification Test Suite

**Status:** Complete & Authoritative (Track T_E2E)  
**Date:** 2026-11-09  
**Execution Environment:** Python 3.11.5 (`py -3.11`) on Windows / Linux  
**Target Specifications:**
- `OrganoidEnv_Master_Verifier_Plan.md` (Stages 0–5)
- `OrganoidEnv_Master_Revision_Plan.md` (Unified Manuscript Strategy)
- `PROJECT.md` (Milestones M1–M5 & Track T_E2E)

---

## 1. Quick Start / Test Runner Commands

Run the full verification test suite:
```bash
py -3.11 -m pytest tests/ -v
```

Run stage-specific verification test suites:
```bash
# Stage 0: Platform Integrity, Seeding & Gymnasium Compliance
py -3.11 -m pytest tests/test_stage0_platform.py -v

# Stage 1: Biophysical & Mathematical Invariants
py -3.11 -m pytest tests/test_stage1_biophysical.py -v

# Stage 2: Checkpoint Completeness & Resume-Equivalence
py -3.11 -m pytest tests/test_stage2_checkpoints.py -v

# Stage 3: Profiling, Optimization & Throughput
py -3.11 -m pytest tests/test_stage3_performance.py -v

# Stage 4: Scientific Discoveries, Ablations & Baselines
py -3.11 -m pytest tests/test_stage4_benchmarks.py -v

# Stage 5: Pre-Submission Paper, Figures & Data Integrity Gate
py -3.11 -m pytest tests/test_stage5_publication.py -v
```

Run fast unit and invariant checks (excluding long-running simulation benchmarks):
```bash
py -3.11 -m pytest tests/ -k "not (latency or memory_leak or resume_equivalence)" -v
```

---

## 2. Test Suite Architecture & Coverage Matrix

| Test Module | Stage | Spec Ref | Tiers Covered | Tests Count | Current Status | Key Features Verified |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `tests/test_stage0_platform.py` | Stage 0 | Verifier §0.1–§0.4 | T1, T2, T3, T4 | 16 | **16 / 16 PASSED (100%)** | Multi-engine seeding (Python/NumPy/PyTorch/Brian2), Gymnasium API compliance, `check_env` validation, atomic checkpoint write |
| `tests/test_stage1_biophysical.py` | Stage 1 | Verifier §1.1–§1.5 | T1, T2, T3, T4 | 12 | **12 / 12 PASSED (100%)** | Brian2 ODE stability, metabolic governor binding, 2.0 ms refractory period, Dale's principle (80% RS / 20% FS), dual trace decay |
| `tests/test_stage2_checkpoints.py` | Stage 2 | Verifier §2.1–§2.2 | T1, T2, T3, T4 | 5 | **4 / 5 PASSED** | 12-component schema validation, array shapes/bounds, missing key rejection, corrupt file rejection, resume-equivalence |
| `tests/test_stage3_performance.py` | Stage 3 | Verifier §3.1–§3.4 | T1, T2, T3, T4 | 5 | **3 / 5 PASSED** | Zero Python neuron/synapse loops (AST verified), heap memory leak bounded (<50 MB), CPU replay buffer; step latency target |
| `tests/test_stage4_benchmarks.py` | Stage 4 | Verifier §4.1–§4.5 | T1, T2, T3, T4 | 10 | **10 / 10 PASSED (100%)** | Firing rate range, branching ratio ($\sigma \sim 1.0$), DSI polar tuning limits, 6-way ablation toggles, 2D stability sweep, baselines |
| `tests/test_stage5_publication.py` | Stage 5 | Verifier §5.1–§5.4 | T1, T2, T3, T4 | 5 | **4 / 5 PASSED** | `rliable` IQM and 95% bootstrap CIs, 12 vector figure formats, reproduction CLI parser; manuscript parameter consistency |
| **Total** | **Stages 0–5** | **Master Roadmaps** | **Tiers 1–4** | **53** | **49 PASSED (92.5%)** | Complete end-to-end coverage across platform, biophysics, and publication |

---

## 3. Discovered Implementation Defects & Escalations

In accordance with the test writer role ("test code only — escalate implementation bugs to the implementing agent"), the following items have been identified and formally escalated:

1. **Stage 2 Resume-Equivalence Brian2 RNG State Persistence (Milestone M2)**:
   - *Observation*: While all 12 checkpoint keys persist cleanly and PyTorch/NumPy RNGs restore, resuming mid-trajectory requires Brian2's `PoissonInput` noise generator and clock state to be synchronized across checkpoint restore.
   - *Escalation to*: Milestone M2 implementer (`save_full_checkpoint` / `load_full_checkpoint`).
2. **Stage 3 Throughput Optimization (Milestone M3)**:
   - *Observation*: Python/NumPy execution on Windows currently averages ~1.9s per step (~28s per episode). AST confirms zero Python loops over individual neurons, but ODE integration requires Cython/batch vectorization to reach the sub-10s/episode target.
   - *Escalation to*: Milestone M3 implementer (Stage 3 vectorization & Cython build).
3. **Stage 5 LaTeX Parameter Alignment (Milestone M5)**:
   - *Observation*: `test_manuscript_code_consistency` flags that `submissions/frontiers_in_neuroinformatics/organoidenv_manuscript.tex` still lists legacy `\tau_{recovery} = 100 ms` instead of the unified `1000 ms` (exc) / `500 ms` (inh).
   - *Escalation to*: Milestone M5 implementer (manuscript LaTeX revision).

---

## 4. Verification Gate Readiness

The test infrastructure in `tests/` is fully operational, self-contained, isolated with Brian2 scope management and sandbox temporary directories, and ready to serve as the continuous regression and verification gate for Milestones M1 through M5.
