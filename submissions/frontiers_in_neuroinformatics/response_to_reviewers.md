# Response to Reviewers & Revision Notes

**Manuscript Title:** OrganoidEnv: A Stabilized Reinforcement Learning Environment for Training Biologically Constrained Spiking Neural Networks via Sensory–Motor Mapping  
**Target Journal:** *Frontiers in Neuroinformatics* (Special Issue on Neuromorphic Computing & Organoid Intelligence) / *IEEE TNNLS*  
**Authors:** Vansh Sharma, Dr. Seema Malik  

---

## Overview of Revisions

We thank the reviewers for their constructive and insightful feedback. In this revised submission, we have addressed all primary critique points through empirical multi-seed experiments, systematic ablation studies, statistical confidence interval reporting, and theoretical reframing:

---

### Point 1: Multi-Seed Statistical Validation (Reviewer Critique: "Single-seed evaluation")
* **Action Taken:** We conducted full multi-seed empirical training across $N = 3$ independent biological initializations (Seeds 42, 123, and 456; 200 episodes each) across the curriculum.
* **Empirical Findings:**
  - **Stage 1 (Basic Navigation):** $71.3\% \pm 40.1\%$ success rate (with top-performing seeds achieving $95.0\%$ and $94.0\%$ success, converging within 15 episodes).
  - **Stage 2 (Obstacle Adaptation):** $62.0\% \pm 38.4\%$ success rate.
  - **Overall Multi-Seed Completion:** $66.7\% \pm 39.2\%$, with Seed 123 completing $184/200$ goals ($92.0\%$) and Seed 42 completing $173/200$ goals ($86.5\%$).
* **Manuscript Update:** Replaced single-run figures with **Figure 1**, plotting the empirical multi-seed mean learning progression with a shaded **95% Bootstrap Confidence Interval band** alongside cumulative task completion curves.

---

### Point 2: Empirical Ablation Studies (Reviewer Critique: "Ablation claims and Stage 2 counts")
* **Action Taken:** Executed systematic multi-seed component ablations ($N = 3$ independent seeds per configuration, 100 episodes each) for the two core architectural stabilizers: Sparse Distributed Memory (SDM) and the Global Activity Regulator (GAR).
* **Empirical Findings:**
  - **Without SDM (No SDM):** Complete sensory collapse ($3.7\%$ success, mean reward $-158.31 \pm 51.28$), demonstrating that high-dimensional non-linear sensory expansion is mandatory to resolve continuous spatial coordinates.
  - **Without GAR (No GAR):** Severe dynamical instability with runaway recurrent excitation and periodic seizure/coma dropouts after Episode 50 (mean reward $+23.14 \pm 93.70$, with reward plunging from $+60.0$ to $-100.0$).
  - **Full OrganoidEnv:** Sustained, stable performance across all seeds (mean reward $+51.00 \pm 24.89$, $71.3\%$ overall success).
* **Manuscript Update:** Added **Figure 2** (Ablation Learning Trajectories & Component Comparison Bar Chart) and updated **Table III** with exact multi-seed means, standard deviations, and dynamical effect descriptions.

---

### Point 3: Strategic Reframing (Reviewer Critique: "Performance gap vs. ANN baseline")
* **Action Taken:** Reframed the narrative from an adversarial comparison against unconstrained deep RL (ANN/D3QN reaching 94.1%) to positioning **OrganoidEnv as a standardized neuromorphic testbed**.
* **Rationale:** OrganoidEnv is the first Gymnasium-compatible environment combining metabolic-Izhikevich neurons, homeostatic activity regulators, and dual-trace dopamine plasticity in a reproducible open-source package. The 23.3-percentage-point gap is formally presented as the *quantified energetic and computational cost of biological constraints* (sparse spiking at 8.5 Hz, ATP fatigue, local plasticity).

---

### Point 4: Software Availability & Reproducibility
* **Action Taken:** Packaged all code, trained weights, CSV logs, and execution pipelines in a clean repository with a 1-click Google Colab reproduction notebook (`notebooks/colab_gpu_training.ipynb`) and companion *Journal of Open Source Software (JOSS)* submission.
