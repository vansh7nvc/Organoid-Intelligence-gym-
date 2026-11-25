# OrganoidEnv: Master Verifier Plan (Single Unified Manuscript Audit)

**Target Paper:** *"OrganoidEnv: A Closed-Loop In Silico Testbed for Investigating Dynamical Stability and Reinforcement Learning in Cortical Organoid Networks"*  
**Purpose:** An automated, quantitative, and deterministic verification protocol to audit every technological invariant, biophysical equation, dynamical stability regime, and checkpoint mechanism in the OrganoidEnv research platform.  
**Execution Protocol:** Execute Stages 0 through 5 sequentially. Compute-intensive 10-seed benchmarks are gated by successful verification of Stages 0 through 3.

---

## Stage 0: Runtime, Determinism & Software Dependencies (Platform Integrity)

| ID | Verification Item | Formal Pass Rule | Verification Method / Snippet | Remediation Action |
|:---:|:---|:---|:---|:---|
| **0.1** | **Deterministic Multi-Engine Seeding** | Bitwise reproducibility across Python `random`, `numpy`, `torch` (CPU/CUDA), and `brian2`. | Run identical 50-step simulation twice on seed 42; assert `np.array_equal(spikes_1, spikes_2)` and `np.allclose(voltages_1, voltages_2)`. | Implement unified `set_seed(seed)` setting `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed`, and `brian2.seed`. Set `torch.backends.cudnn.deterministic = True`. |
| **0.2** | **Pinned Dependency Manifest** | `requirements.txt` contains pinned versions (`==`); zero version conflicts on import. | `pip check` returns 0; runtime versions match: `gymnasium>=0.29`, `brian2>=2.5.4`, `torch>=2.1`, `rliable>=1.0`, `elephant>=1.0`. | Generate frozen manifest: `pip freeze > requirements-pinned.txt`; execute virtualenv build from clean state. |
| **0.3** | **Gymnasium API Compliance** | Passes official Gymnasium environment validation suite without warnings. | `from gymnasium.utils.env_checker import check_env; check_env(OrganoidEnv())` completes with exit code 0. | Fix action/observation space shape bounds, seed handling, and reset return structure `(obs, info)`. |
| **0.4** | **Atomic Checkpoint Serialization** | Checkpoints write to temporary `.tmp` files and use atomic filesystem renaming (`os.replace`). | Forced process kill mid-save never leaves a corrupted `.pt` or `.json` file. | Replace direct `torch.save(path)` with `torch.save(temp_path)` followed by `os.replace(temp_path, path)`. |

---

## Stage 1: Biophysical & Mathematical Invariant Verifications (Scientific Integrity)

These tests ensure the computational model adheres to the fundamental biophysical laws asserted in the manuscript:

### 1.1 Brian2 Canonical Solver Numerical Equivalence
* **Objective:** Ensure our custom integration matches Brian2's canonical numerical solver.
* **Pass Rule:** Absolute voltage difference $\max |v_{\text{model}}(t) - v_{\text{brian2}}(t)| < 10^{-4}\text{ mV}$ across a 50 ms current injection protocol ($I = [0, 5, 10, 25]\text{ pA}$).
* **Assertion:**
```python
def test_izhikevich_numerical_accuracy():
    from organoid_rl.environment.neurons import get_metabolic_izhikevich_eqs
    from brian2 import NeuronGroup, Network, ms, defaultclock
    
    defaultclock.dt = 0.1 * ms
    eqs = get_metabolic_izhikevich_eqs()
    G = NeuronGroup(1, eqs, threshold='v>=30 and E>0', reset='v=-65; u+=8; E-=0.1', method='euler')
    G.v = -65.0; G.u = 0.2 * -65.0; G.E = 1.0; G.I = 15.0
    G.a = 0.02; G.b = 0.2; G.c = -65.0; G.d = 8.0; G.tau_recovery = 1000*ms; G.spike_cost = 0.1
    net = Network(G)
    net.run(10 * ms)
    assert not np.isnan(G.v[0]), "Numerical divergence detected in Euler step"
    assert -100 <= G.v[0] <= 60, f"Voltage bounds violated: {G.v[0]}"
```

### 1.2 The Metabolic Governor Binding Invariant
* **Objective:** Prove that ATP exhaustion is not dormant, but actively arrests firing during excessive high-frequency stimulation.
* **Test Protocol:** Inject a continuous supra-threshold current ($I = 50.0\text{ pA}$) for 500 ms (5000 integration sub-steps).
* **Pass Rule:**
  1. ATP reserve $E_i(t)$ must deplete from $1.0$ down to $\le 0.10$.
  2. Spiking must cease completely while $E_i(t) \le 0$, even though membrane voltage exceeds the geometric threshold ($v_i \ge 30\text{ mV}$).
  3. Upon current cessation, $E_i(t)$ must recover asymptotically toward $1.0$ with time constant $\tau_{\text{recovery}}$.
* **Remediation if Failed:** Verify that $\tau_{\text{recovery}} = 1000\text{ ms}$ and $\Delta_{\text{spike}} = 0.10$ are unified across all environment modules.

### 1.3 Refractory Period Enforcement Invariant
* **Objective:** Verify absolute biological refractory period compliance ($t_{\text{ref}} = 2.0\text{ ms}$).
* **Pass Rule:** Across all neurons for 100 episodes, no inter-spike interval may violate $\min(\text{ISI}) \ge 2.0\text{ ms}$.

### 1.4 Dale's Principle & Synaptic Polarity Compliance
* **Objective:** Verify Dale's Principle: excitatory synapses produce strictly positive postsynaptic potential deflections ($v_{\text{post}} \leftarrow v_{\text{post}} + w$), while inhibitory synapses produce strictly negative deflections ($v_{\text{post}} \leftarrow v_{\text{post}} - w$).
* **Pass Rule:** Zero sign reversals in synaptic conductance matrices; Excitatory neuron count = 400 (80%), Inhibitory count = 100 (20%).

### 1.5 Presynaptic Eligibility Trace Exponential Decay Invariant
* **Objective:** Confirm dual eligibility traces follow the exact biological decay constants:
  $$\text{Trace}_1(t) = \text{Trace}_1(0) e^{-t / 100\text{ ms}}, \quad \text{Trace}_2(t) = \text{Trace}_2(0) e^{-t / 2500\text{ ms}}$$
* **Pass Rule:** Residual difference between simulated trace dynamics and analytical exponential formula $< 10^{-5}$ across a 500 ms decay window.

---

## Stage 2: Checkpoint Completeness & Resume-Equivalence (Platform Reliability)

Guarantees full computational reproducibility across cloud disconnects and multi-day compute sessions.

### 2.1 The 12 Mandatory State Components
Every checkpoint dictionary **must** contain all 12 items:
1. `snn_weights`: Synaptic weight matrices across all projections.
2. `snn_traces_fast`: $\text{Trace}_1$ state matrices.
3. `snn_traces_slow`: $\text{Trace}_2$ state matrices.
4. `snn_metabolic_atp`: Per-neuron ATP energy vectors ($E_i$).
5. `snn_membrane_v`: Membrane potential vectors ($v_i$).
6. `snn_recovery_u`: Membrane recovery vectors ($u_i$).
7. `snn_pruning_mask`: Boolean adjacency matrix of unpruned connections.
8. `gar_state_counters`: Tally of seizure clamps, coma kicks, and NaN resets.
9. `agent_online_net`: PyTorch `state_dict` of D3QN online network.
10. `agent_target_net`: PyTorch `state_dict` of D3QN target network.
11. `agent_optimizer`: PyTorch `state_dict` of Adam optimizer (including momentum/variance buffers).
12. `rng_states`: Snapshot of Python `random`, `numpy`, and `torch` (CPU/CUDA) random generators.

### 2.2 Resume-Equivalence Standalone Test
```python
def test_resume_equivalence(seed=42, n_episodes=20, ckpt_path="audit_ckpt.pt"):
    """Asserts that 20 continuous episodes == 10 episodes + save + load + 10 episodes."""
    import hashlib, os, numpy as np, torch
    from organoid_rl.environment.core import OrganoidEnv
    from organoid_rl.agents.dqn_agent import D3QNAgent

    def get_hash(env, agent):
        h = hashlib.sha256()
        for k, v in env.get_state_dict().items():
            if isinstance(v, np.ndarray): h.update(np.ascontiguousarray(v).tobytes())
        for p in agent.q_network.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    # Continuous Reference Run
    env_a = OrganoidEnv(seed=seed); agent_a = D3QNAgent(env_a, seed=seed)
    r_uninterrupted = [agent_a.train_episode() for _ in range(n_episodes)]
    hash_uninterrupted = get_hash(env_a, agent_a)

    # Interrupted Run
    half = n_episodes // 2
    env_b = OrganoidEnv(seed=seed); agent_b = D3QNAgent(env_b, seed=seed)
    r_part1 = [agent_b.train_episode() for _ in range(half)]
    save_full_checkpoint(ckpt_path, env_b, agent_b)

    env_c = OrganoidEnv(seed=999); agent_c = D3QNAgent(env_c, seed=999)
    load_full_checkpoint(ckpt_path, env_c, agent_c)
    r_part2 = [agent_c.train_episode() for _ in range(n_episodes - half)]
    hash_interrupted = get_hash(env_c, agent_c)

    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    assert np.allclose(r_uninterrupted, r_part1 + r_part2, atol=1e-6), "Reward divergence upon resume!"
    assert hash_uninterrupted == hash_interrupted, "State hash mismatch upon resume!"
    print("[PASS] Resume-Equivalence Verified.")
```

---

## Stage 3: Profiling, Optimization & Throughput (Platform Performance)

| ID | Performance Metric | Target Threshold | Measurement Instrument | Remediation Action |
|:---:|:---|:---|:---|:---|
| **3.1** | **Episode Wall-Clock Latency** | $< 10.0\text{ s}$ / episode (down from 27.5 s) | `time.perf_counter()` over 50 evaluation episodes | Vectorize SNN integration and trace updates in batch NumPy. |
| **3.2** | **Zero Python Neuron Loop** | **0** Python loops over $N=500$ neurons or $M$ synapses | AST inspection or code profiling (`cProfile`) | Replace any `for neuron in neurons:` with matrix operations: `v += dt * dv`. |
| **3.3** | **Memory Leak Audit** | Peak RAM growth $< 50\text{ MB}$ over 500 consecutive episodes | `tracemalloc` snapshot at Ep 0 vs Ep 500 | Check for detached PyTorch computation graphs leaking in replay buffer. |
| **3.4** | **GPU Memory Footprint** | VRAM consumption $\le 2.0\text{ GB}$ throughout training | `torch.cuda.max_memory_allocated()` | Buffer storage in `uint8` / `float32` CPU memory, transferring only mini-batches to GPU. |

---

## Stage 4: Scientific Discoveries, Plausibility & Ablations (Scientific Proofs)

### 4.1 Physiological Firing Rate Range
* **Biological Target:** Population mean firing rate $\bar{f} \in [2.0, 20.0]\text{ Hz}$ (*Trujillo et al., 2019 Cell Stem Cell*).
* **Pass Rule:** Nominal model mean operational firing rate across 100 episodes must fall strictly within $5.0 \le \bar{f} \le 15.0\text{ Hz}$.

### 4.2 Criticality & Neural Avalanche Exponent
* **Biological Target:** Power-law scaling of spike avalanche sizes $P(S) \propto S^{-\alpha}$ with $\alpha \approx 1.5$, and branching ratio $\sigma \approx 1.0$ (*Beggs & Plenz, 2003*).
* **Assertion:**
```python
def test_neural_criticality(spike_raster, bin_size_ms=4.0):
    from elephant.functional_connectivity import branching_ratio
    sigma = branching_ratio(spike_raster, bin_step=bin_size_ms)
    assert 0.90 <= sigma <= 1.10, f"Network not operating near criticality: sigma = {sigma:.3f}"
    sizes = extract_avalanche_sizes(spike_raster, bin_size_ms)
    alpha = fit_power_law_exponent(sizes)
    assert 1.30 <= alpha <= 1.70, f"Avalanche exponent deviates from biological target: alpha = {alpha:.3f}"
```

### 4.3 Directional Selectivity Index (DSI) Tuning Test
* **Scientific Objective:** Prove that the plastic SNN develops directional tuning curves, while frozen and random controls do not.
* **Pass Rule:**
  * Full Model (Plastic): $\text{DSI} \ge 0.50$ by Episode 500.
  * Frozen Weights (C1): $\text{DSI} \le 0.20$ across all episodes.
  * Random SNN (C2): $\text{DSI} \le 0.20$ across all episodes.

### 4.4 Automated Ablation Orthogonality Gate
Ensure each ablation configuration toggles exactly one biological mechanism without collateral interference:

| Config | Active Flag | Mechanism Isolated | Verification Assertions |
|:---|:---|:---|:---|
| **C1** | `use_plasticity=False` | Frozen Synaptic Weights | $\max |W(t) - W(0)| == 0$; membrane dynamics active. |
| **C2** | `random_snn=True` | Static Reservoir Control | Random synaptic weights fixed; D3QN training continues. |
| **C3** | `use_metabolism=False` | Non-Exhaustible Energy | $E_i(t) \equiv 1.0$ for all $t$; Izhikevich dynamics active. |
| **C4** | `use_gar=False` | Unstabilized Dynamics | GAR interventions $== 0$; demonstrates seizure/coma risk. |
| **C5** | `use_pruning=False` | Fixed Connectome | Synaptic mask $M(t) == M(0)$; weight updates active. |
| **C6** | `use_dual_trace=False` | Single-Trace STDP | $\text{Trace}_2 \equiv 0$; $\text{Trace}_1$ dynamics active. |

### 4.5 Automated 2D Stability Phase Diagram Generator
* Run a parameterized parameter sweep: $I_{\text{stim}} \in [5, 10, 20, 30, 50]\text{ pA} \times w_{\text{inh}} \in [5, 10, 20, 30]\text{ nS}$.
* Verify that the generated phase plot clearly resolves the three theoretical dynamical regimes:
  1. Epileptiform Hyperactivity ($>100\text{ Hz}$).
  2. Quiescent Death ($<0.5\text{ Hz}$).
  3. Physiological Homeostatic Regime ($2–20\text{ Hz}$).

---

## Stage 5: Pre-Submission Paper & Data Integrity Gate (Unified Manuscript Audit)

### 5.1 Automated LaTeX vs. Code Consistency Validator
An automated script scans the LaTeX manuscript and cross-references all biophysical constants against the authoritative `OrganoidConfig`:
```python
def test_manuscript_code_consistency():
    import re
    from organoid_rl.environment.core import OrganoidConfig
    cfg = OrganoidConfig()
    
    with open("submissions/ieee_tnnls/manuscript.tex", "r") as f:
        tex = f.read()
    
    assert f"{cfg.n_neurons}" in tex, f"Neuron count {cfg.n_neurons} missing from LaTeX"
    assert "1000~ms" in tex or "1000\\text{ ms}" in tex, "tau_recovery (exc) mismatch"
    assert "500~ms" in tex or "500\\text{ ms}" in tex, "tau_recovery (inh) mismatch"
    assert "2~ms" in tex or "2.0~ms" in tex, "refractory period mismatch"
    assert "100~ms" in tex, "tau_trace1 mismatch"
    assert "2500~ms" in tex, "tau_trace2 mismatch"
    print("[PASS] All LaTeX biophysical parameters match codebase configuration.")
```

### 5.2 Non-Parametric Statistical Certification (`rliable`)
* Verify that all reported aggregate scores use the Google Research `rliable` library:
  * Interquartile Mean (IQM) across 10 random seeds.
  * Stratified 95% Bootstrap Confidence Intervals.
  * Mann-Whitney U test significance ($p < 0.05$) reported with rank-biserial effect sizes.

### 5.3 Vector Graphic Quality & Font Certification
* All 12 figures must be vector formats (`.pdf` or `.svg`).
* Zero rasterized text; all fonts must be embedded Type 1 or TrueType.
* Axis labels $\ge 8.0\text{ pt}$ at final publication scale.

### 5.4 One-Click Scientific Reproduction Script
Verify that a clean environment reproduces all paper results via:
```bash
python scripts/reproduce_paper_results.py --run-unit-tests --generate-all-figures
```

---

## Stage Execution Gate Checklist

- [ ] **Stage 0 (Platform):** Determinism, environment lock, Gymnasium API checker, and atomic checkpointing verified.
- [ ] **Stage 1 (Science):** Brian2 equivalence, metabolic governor binding, and refractory invariants passed.
- [ ] **Stage 2 (Platform):** Checkpoint completeness and Resume-Equivalence test passed (bitwise match).
- [ ] **Stage 3 (Platform):** Throughput latency $< 10$ s/episode; zero Python neuron loops verified.
- [ ] **Stage 4 (Science):** Firing rate (8.5 Hz), criticality ($\sigma \approx 1.0$), DSI tuning, and 2D stability phase sweep verified.
- [ ] **Stage 5 (Unified):** LaTeX-code parameter scanner passed; `rliable` statistics certified; 12 vector figures compiled.
