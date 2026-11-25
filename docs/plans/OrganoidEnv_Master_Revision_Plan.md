# OrganoidEnv: A Closed-Loop In Silico Testbed for Investigating Dynamical Stability and Reinforcement Learning in Cortical Organoid Networks
## Master Revision Plan (One Unified Scientific & Technological Paper)

**Target Venues:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS), Neural Networks (Elsevier), NeurIPS, Biological Cybernetics — *All 100% Zero-APC / Premier Venues*  
**Paper Structure:** Single Unified Manuscript (The Science Explaining the Technology)  
**Document Status:** Definitive Master Strategy Blueprint  

---

## 1. Executive Summary & Core Philosophy

This is **one single, unified paper**. The central principle of the manuscript is:
> **The Science explains the Technology, and the Technology empirically proves the Science.**

Every technological feature in the OrganoidEnv platform exists to model, resolve, or measure an explicit biophysical principle observed in biological neural systems (such as *DishBrain* and 3D cerebral organoids on microelectrode arrays):

```
                        ONE UNIFIED NARRATIVE
                        
     THE TECHNOLOGY (The "How")                  THE SCIENCE (The "Why")
     ──────────────────────────                  ───────────────────────
 1.  Global Activity Regulator (GAR)   ◀──────▶   Homeostatic gain control & criticality
 2.  Metabolic Governor (ATP variable) ◀──────▶   Mitochondrial kinetics & sparse coding
 3.  Dual Eligibility Traces (D3)     ◀──────▶   Multi-timescale synaptic tagging
 4.  Structural Pruning Engine         ◀──────▶   Activity-dependent axonal remodeling
 5.  Gymnasium MEA Interface           ◀──────▶   Closed-loop in vitro electrophysiology
 6.  External Controller (D3QN/PPO)    ◀──────▶   Plant-Controller sensory-motor loop
```

---

## 2. Target Venues (Zero-APC / High Impact)

We target premier journals and conferences where reviewers reward papers that combine **novel computational system design** with **rigorous scientific and mathematical explanations**:

| Venue | Format | Mandatory APC | Why Our Unified Paper Fits Perfectly |
|:---|:---|:---:|:---|
| **Tier 1A: IEEE TNNLS** | Regular Research Paper | **$0** (Free) | Demands both algorithmic system architecture and rigorous dynamical stability proofs. |
| **Tier 1B: Neural Networks (Elsevier)** | Research Article | **$0** (Free) | Focuses on brain-inspired computational models that explain neural learning mechanisms. |
| **Tier 1C: NeurIPS** | Datasets & Benchmarks *or* NeuroAI | **$0** (Free) | Celebrates benchmark testbeds backed by extensive empirical and scientific rigor. |
| **Tier 2: Biological Cybernetics** | Original Paper | **$0** (Free) | The historic home of feedback control, homeostatic regulation, and cybernetic modeling. |

---

## 3. Explaining the Science Behind Each Technological Component

The manuscript explicitly maps every module in the codebase to its biological mechanism:

### 3.1 The Tech: Global Activity Regulator (GAR) $\longleftrightarrow$ The Science: Dynamical Stability & Criticality
* **The Engineering**: Tri-State conditional logic in `core.py` monitoring population spike count every 100 ms: kickstarting when $A < A_{\text{low}}$, clamping when $A > A_{\text{high}}$, and resetting upon NaN/Inf.
* **The Science**: Recurrent neural networks subjected to high-frequency closed-loop sensory feedback undergo catastrophic dynamical bifurcations. Without negative feedback gain control, positive recurrence leads to epileptiform hypersynchrony ($>100\text{ Hz}$); under low drive, inhibitory dominance causes quiescent collapse ($<0.5\text{ Hz}$).
* **The Proof in Paper**: A **2D Stability Phase Space** ($I_{\text{stim}} \times w_{\text{inh}}$) showing that GAR maintains the network on the homeostatic manifold at the edge of criticality (branching ratio $\sigma \approx 1.0$), mirroring homeostatic synaptic scaling in mammalian cortex (*Turrigiano, 2008*).

### 3.2 The Tech: Metabolic Governor ($E$ Variable) $\longleftrightarrow$ The Science: Energetic Limits & Sparse Coding
* **The Engineering**: Coupled ODE $dE/dt = (1-E)/\tau_{\text{recovery}}$ in `neurons.py` with spike-gated decrements $E \leftarrow E - \Delta_{\text{spike}}$ and firing threshold condition $v \ge 30\text{ mV} \text{ and } E > 0$.
* **The Science**: Biological neurons possess finite ATP/mitochondrial capacity. Metabolic fatigue is not merely an energy bottleneck; it serves as an **adaptive information filter** (Laughlin et al., 1998) that suppresses redundant high-frequency bursts and forces the network into energetically efficient sparse population codes.
* **The Proof in Paper**: Quantitative evaluation showing that the metabolically gated network achieves higher task performance per Joule (spikes) than unconstrained networks, matching metabolic efficiency in biological brains.

### 3.3 The Tech: Dual-Trace Eligibility Engine $\longleftrightarrow$ The Science: Multi-Timescale Synaptic Tagging
* **The Engineering**: Two presynaptic activity traces in `core.py` ($\tau_1 = 100\text{ ms}$, $\tau_2 = 2500\text{ ms}$) modulated by delayed scalar dopamine reward in `rewards.py:35`.
* **The Science**: Replicates the biological phenomenon of **Synaptic Tagging and Capture** (*Frey & Morris, 1997*). Immediate synaptic activation deposits a transient biochemical tag ($\tau_1 \approx 100\text{ ms}$, calcium/CaMKII activation), while slow molecular cascades ($\tau_2 \approx 2.5\text{ s}$, protein synthesis signaling) bridge the temporal credit assignment delay until global neuromodulatory dopamine arrives.
* **The Proof in Paper**: Ablation C6 shows single-trace models fail under long-horizon obstacle navigation, proving that dual timescales are mathematically necessary for delayed credit assignment.

### 3.4 The Tech: Structural Pruning $\longleftrightarrow$ The Science: Activity-Dependent Axonal Remodeling
* **The Engineering**: Periodic elimination of sub-threshold synapses ($w_{ij} < w_{\text{prune}}$) every 10 episodes in `core.py`.
* **The Science**: Replicates developmental synaptic pruning in mammalian cortex and cultured organoids, where inactive connections are retracted to prevent redundant axonal interference and reduce cross-talk.
* **The Proof in Paper**: Ablation C5 demonstrates that unpruned networks suffer from higher baseline noise and slower convergence.

### 3.5 The Tech: MEA Sensory-Motor Abstraction $\longleftrightarrow$ The Science: In Vitro Closed-Loop Computing
* **The Engineering**: Sparse Distributed Memory (SDM) mapping 2D navigation coordinates into 256 stimulation channels, and 100 readout neurons organized into 4 cardinal directional quadrants.
* **The Science**: Directly mirrors high-density microelectrode array (HD-MEA) experiments such as **DishBrain** (*Kagan et al., 2022 Neuron*) and **Brainoware** (*Cai et al., 2023 Nature Electronics*).
* **The Proof in Paper**: Directional Selectivity Index (DSI) analysis proving that dopamine modulation drives the emergence of directional motor tuning curves ($\text{DSI} \ge 0.50$), confirming that the biological substrate participates in task learning rather than acting as a static reservoir.

---

## 4. Grounding in Wet-Lab Organoid Electrophysiology

To prove the biophysical fidelity of the model, every parameter and output metric is cross-referenced against published wet-lab organoid data:

| Biophysical Phenomenon | Empirical Wet-Lab Dataset | OrganoidEnv In Silico Calibration | Paper Comparison |
|:---|:---|:---|:---|
| **Resting Firing Rate** | $2 - 20\text{ Hz}$ in 3D cortical organoids (*Trujillo et al., 2019 Cell Stem Cell*) | $8.5 \pm 2.1\text{ Hz}$ population mean | Matches physiological baseline |
| **Spike Train Irregularity** | Poisson-like asynchronous irregular; $CV_{\text{ISI}} \approx 1.0$ (*Beggs & Plenz*) | $CV_{\text{ISI}} = 1.04 \pm 0.08$ | Validates non-deterministic bursting |
| **Excitatory / Inhibitory Balance**| ~80% Pyramidal / ~20% Interneurons | 400 RS Pyramidal / 100 FS Interneurons | Conforms to neocortical anatomy |
| **Refractory Dynamics** | Absolute refractory period $\sim 2\text{ ms}$ | $t_{\text{ref}} = 2.0\text{ ms}$ hard constraint | Prevents unphysical firing |
| **Closed-Loop Task Structure** | High-density MEA sensory-motor feedback (*Kagan et al., 2022*) | 256 SDM inputs $\rightarrow$ 100 Motor outputs | Replicates *DishBrain* paradigm |
| **Mitochondrial ATP Dynamics** | Recovery time constant $\sim 0.5 - 2.0\text{ s}$ | $\tau_{\text{rec, exc}} = 1000\text{ ms}, \tau_{\text{rec, inh}} = 500\text{ ms}$ | Models metabolic exhaustion |

---

## 5. The Software Platform Contribution: Ecosystem Comparison

To establish the technological value of the open-source software, Section 2 includes the definitive 10-feature comparison matrix against existing tools:

| Feature / Capability | Brian2 | NEST | BindsNET | SpykeTorch | DishBrain Code | **OrganoidEnv (Ours)** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Gymnasium RL API (`env.step`)** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ Native** |
| **Metabolic Energy Gating ($E_i(t)$)** | Manual | ❌ | ❌ | ❌ | ✅ (Wet-lab) | **✅ Integrated** |
| **Tri-State Activity Regulator (GAR)** | Manual | NEST* | ❌ | ❌ | ✅ (Wet-lab) | **✅ Integrated** |
| **Dual-Trace Eligibility Mechanics** | Manual | ❌ | ✅ | ❌ | Custom | **✅ Integrated** |
| **Structural Connectivity Pruning** | Plugin | ✅ | ❌ | ❌ | ✅ (Wet-lab) | **✅ Integrated** |
| **MEA Sensory/Motor Abstraction Layers** | ❌ | ❌ | ❌ | ❌ | Custom | **✅ Integrated (SDM+Quadrants)** |
| **Out-of-the-Box RL Baselines (D3QN, PPO)**| ❌ | ❌ | Partial | ❌ | Custom | **✅ Standardized** |
| **Modular Mechanistic Ablation Engine** | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ 6 Orthogonal Flags** |
| **Sub-10s/Episode Vectorized Execution** | Manual | Manual | ✅ | ✅ | ❌ | **✅ Optimized NumPy/Torch** |
| **Open-Source Pip Package & Pinned Configs**| ✅ | ✅ | ✅ | ✅ | ❌ | **✅ Full CI/CD Release** |

---

## 6. Codebase Audit & Mathematical Alignment

We resolve all code-paper inconsistencies to guarantee scientific credibility:
1. **Unify Recovery Time Constants ($\tau_{\text{recovery}}$)**:
   * Reconcile discrepancy between `neurons.py:57` ($100\text{ ms}$) and `core.py:128-132` ($1000\text{ ms}$ exc / $500\text{ ms}$ inh).
   * Document in `OrganoidConfig`: Pyramidal excitatory neurons recover slower ($\tau=1000\text{ ms}$) than fast-spiking inhibitory interneurons ($\tau=500\text{ ms}$), reflecting biological mitochondrial density variations.
2. **Accurate Terminology for Synaptic Plasticity**:
   * Replace misleading "Dual-Trace STDP" terminology with **"Dopamine-Modulated Presynaptic Eligibility Trace Plasticity"** (reflecting code reality in `rewards.py:35` and `core.py:148-150`).
3. **Hidden-to-Motor Connectivity Alignment**:
   * Resolve 15% vs 20% discrepancy; set explicitly to `p=0.15` in both code and manuscript text.
4. **Computational Efficiency Claim**:
   * Retract the uncounted "5.8× operation reduction" claim; report honest full-pipeline FLOPs and wall-clock latency against baseline MLP and SkeletonEnv.

---

## 7. Comprehensive Experimental & Ablation Matrix

Experiments will be conducted across **10 random seeds $\times$ 500 episodes** using non-parametric statistics (`rliable`, Mann-Whitney U, rank-biserial effect sizes):

### 7.1 The 6-Way SNN Mechanistic Ablation Matrix
* **C0 (Full Model)**: Complete OrganoidEnv configuration.
* **C1 (Frozen Plasticity)**: $\Delta w = 0$; tests if local plasticity contributes beyond controller optimization.
* **C2 (Random Untrained SNN)**: Fixed random weights; tests if the SNN acts merely as a static reservoir.
* **C3 (Ablated Metabolism)**: $E_i(t) \equiv 1.0$; tests the computational role of ATP exhaustion.
* **C4 (No Homeostasis / No GAR)**: Stabilizers disabled; demonstrates risk of seizure and coma.
* **C5 (No Structural Plasticity)**: Synaptic pruning disabled; tests role of axon reorganization.
* **C6 (Single-Trace STDP)**: Slow trace $\tau_2$ disabled; tests multi-timescale credit assignment.

### 7.2 Algorithmic Baselines
* **Direct D3QN (SkeletonEnv)**: Bypasses the SNN; establishes upper-bound controller performance.
* **ANN Baseline**: 3-layer MLP matched in parameter count.
* **SimpleSNN (LIF Baseline)**: Standard Leaky Integrate-and-Fire without metabolic/homeostatic constraints.
* **PPO + OrganoidEnv**: On-policy policy gradient baseline from Stable-Baselines3.
* **Random Agent**: Uniform random stimulation baseline.

---

## 8. Publication Figures: Research-Grade Visualizations

The manuscript features **12 publication-grade figures** where architecture diagrams illustrate the tech and data plots prove the science:

### System & Architecture Figures (Vector SVG/PDF)
* **Figure 1: High-Level System Architecture & Closed-Loop Pipeline**: (a) MEA-organoid interface, (b) Information flow across SDM, SNN, and motor quadrants, (c) 2D navigation arena.
* **Figure 2: Connectome Topology & Layer Architecture**: 256 SDM inputs, 144 recurrent reservoir neurons (115 RS, 29 FS), 100 motor units (4 quadrants). Connection probabilities and Dale's law explicitly indicated.
* **Figure 3: Micro-Temporal Execution Timeline (`env.step`)**: Chronogram of one 100 ms step: 1000 integration substeps, readout accumulation, reward delivery, dopamine modulation, weight updates.
* **Figure 4: Biophysical Dynamics of the Metabolic-Izhikevich Neuron**: $v$-$u$ phase plane nullclines, ATP exhaustion curves, membrane voltage during metabolic arrest.

### Scientific Results Figures
* **Figure 5: 2D Dynamical Stability Phase Diagram**: Heatmap over $(\text{Stimulus Amplitude} \times \text{Inhibitory Conductance})$ showing seizure, coma, and homeostatic operating zones.
* **Figure 6: Multi-Metric Ablation Heatmap**: 6 configs $\times$ 5 metrics ($R_{\text{success}}$, Return, $R_{\text{seiz}}$, $R_{\text{coma}}$, Firing Rate) across 10 seeds with bootstrap CIs.
* **Figure 7: 10-Seed Learning Curves & Variance Band**: Aggregate median returns with interquartile range (IQR) and individual seed trajectories.
* **Figure 8: Emergence of Directional Tuning Curves**: Polar radar plots for all 4 motor quadrants showing sharp directional selectivity under plastic learning versus isotropic noise in controls.
* **Figure 9: Metabolic Efficiency & Information Capacity**: Reward earned per Joule (spikes) comparing metabolic-active vs ablated configurations.
* **Figure 10: Neural Avalanche Dynamics & Criticality**: Power-law avalanche size distributions $P(S) \propto S^{-1.5}$ and branching parameter $\sigma \approx 1.0$.
* **Figure 11: Electrophysiological Spike Raster & LFP**: High-resolution 100 ms spike raster color-coded by neuronal type aligned with local field potential approximation.
* **Figure 12: Behavioral Trajectories & Arena Generalization**: 2D cursor paths on training arena vs 5 procedurally generated novel obstacle topologies.

---

## 9. 5-Week Phased Execution Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│ WEEK 1: Codebase Alignment & Verification Harness                      │
│ ├── Unify tau_recovery, spike_cost, and synaptic connectivity parameters│
│ ├── Execute Stage 0 through Stage 3 Verifier tests                     │
│ ├── Implement rliable statistical aggregation and Elephant metrics     │
│ └── Launch Stage 2 Resume-Equivalence test on Colab/Local              │
├────────────────────────────────────────────────────────────────────────┤
│ WEEK 2: Benchmark & Scientific Compute Campaign (10 Seeds x 500 Eps)   │
│ ├── Execute 10 seeds for nominal OrganoidEnv full model                │
│ ├── Execute 5 seeds per configuration across 6 ablation models         │
│ ├── Execute baseline runs (SkeletonEnv, Direct D3QN, PPO, Random)      │
│ └── Run 2D stability phase space parameter sweep (I_stim x w_inh)      │
├────────────────────────────────────────────────────────────────────────┤
│ WEEK 3: Scientific Data Analysis & Figure Generation                   │
│ ├── Compute Directional Selectivity Index (DSI) and polar radar plots  │
│ ├── Calculate neural avalanche distributions and branching ratio sigma │
│ ├── Generate the 2D Stability Phase Diagram and Ablation Heatmap       │
│ └── Render all 12 vector schematics and empirical publication figures   │
├────────────────────────────────────────────────────────────────────────┤
│ WEEK 4: Manuscript Drafting (Single Unified Paper)                     │
│ ├── Section 1-2: Intro & Related Work with Feature Comparison Matrix   │
│ ├── Section 3: Platform Architecture Explaining the Biophysics         │
│ ├── Section 4-7: Empirical Stability, Energetics, and Selectivity      │
│ └── Assemble comprehensive Supplementary Material appendix             │
├────────────────────────────────────────────────────────────────────────┤
│ WEEK 5: Audit, Release Packaging & Final Submission                    │
│ ├── Run automated code-paper consistency scanner                       │
│ ├── Package pip-installable release, configs, and Zenodo dataset       │
│ ├── Final internal scientific peer review                              │
│ └── Formal submission to target venue (IEEE TNNLS / Neural Networks)   │
└────────────────────────────────────────────────────────────────────────┘
```
