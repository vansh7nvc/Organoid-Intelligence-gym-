---
title: 'OrganoidEnv: An Open, Ablatable Spiking Testbed for Studying Metabolic and Homeostatic Constraints in Organoid-Inspired Closed-Loop Reinforcement Learning'
tags:
  - Python
  - C99
  - neuroscience
  - neuromorphic computing
  - reinforcement learning
  - spiking neural networks
  - brian2
  - gymnasium
  - organoid intelligence
authors:
  - name: Vansh Sharma
    orcid: 0009-0002-1825-0097
    affiliation: 1
  - name: Seema Malik
    affiliation: 1
affiliations:
 - name: Department of Computer Science & Engineering
   index: 1
date: 25 November 2026
bibliography: paper.bib
---

# Summary

Bridging the divide between artificial neural networks trained with global backpropagation and biologically grounded spiking neural networks (SNNs) remains a central challenge in neuromorphic engineering and computational neuroscience. Living neural tissue operating *in vitro* or *in vivo* faces strict physical constraints: non-differentiable event-driven action potentials, local eligibility-modulated plasticity, continuous energy-fatigue expenditure, and susceptibility to hyperactive or quiescent regimes [@kagan2022in_vitro; @smirnova2023organoid].

`OrganoidEnv` is an open-source, Gymnasium-compliant Python and C99 software package designed to simulate, ablate, and benchmark organoid-inspired recurrent spiking networks in closed-loop reinforcement learning (RL). It embeds a 500-neuron recurrent Izhikevich network (80/20 excitatory/inhibitory Dale's split) [@izhikevich2003simple] inside a 2D obstacle-navigation environment, paired with a high-speed native C99 integration kernel, a multi-session FastAPI + Three.js interactive 3D workbench, and Dockerized reproducibility workflows.

# Statement of Need

Researchers studying closed-loop biological and neuromorphic control lack standardized, single-flag-ablatable testbeds that isolate metabolic fatigue, homeostatic gain control, and multi-timescale three-factor plasticity. Existing SNN frameworks such as BindsNET, snnTorch, and SpikingJelly focus on surrogate-gradient backpropagation or general-purpose simulation without built-in energy-fatigue bounds, tri-state activity regulators, or matched RL baselines.

`OrganoidEnv` addresses this need by providing:
1. **Gymnasium Compliance & Deterministic 12-Key Seeding**: Standard `reset()` and `step()` APIs [@towers2023gymnasium] with bit-exact reproducibility across `random`, `numpy`, `torch`, Brian2 [@brian2], and the native C99 kernel.
2. **Native C99 Kernel Cross-Checked Against Brian2**: Achieves a subthreshold voltage residual of $|\Delta v| = 1.42 \times 10^{-14}\text{ mV}$ and a $191.3\times$ step speedup over pure Brian2 execution.
3. **Single-Flag Ablation Architecture**: Independent flags (`use_metabolism`, `use_stabilizer`, `use_plasticity`, `use_dual_trace`, `use_pruning`) allow controlled isolation of every biophysical mechanism.
4. **Scalable Multi-Session 3D Workbench & Containerization**: A FastAPI + WebSocket + Three.js web platform with isolated per-user sessions (`SessionManager`), alongside `Dockerfile` and `docker-compose.yml` for one-command reproduction.

# Architectural Overview

```
[15D Spatial Observation]
         │
         ▼
[Top-k Sparse Expansion Encoder] (256 Exc Channels, k=26 / 10.2% active)
         │
         ▼
[Recurrent Organoid-Inspired SNN] (400 RS Exc / 100 FS Inh, Energy Variable E)
         │
         ▼
[Clustered Motor Quadrants] (Up, Down, Left, Right) ──► [2D Kinematic Displacement]
         ▲
         │
[Global Activity Regulator (GAR)] (Tri-state Coma Kick <50 / Seizure Clamp >5000 spikes)
```

## 1. Energy-Constrained Izhikevich Neurons
Neurons evolve in dimensionless Izhikevich units extended with an abstract energy-fatigue variable $E \in [-0.1, 1.0]$:
$$\frac{dv}{dt} = \mathbb{I}[E>0]\!\left(0.04v^2 + 5v + 140 - u + I\right) + \mathbb{I}[E\le 0]\!\left(\frac{-80 - v}{0.5\text{ ms}}\right), \quad \frac{dE}{dt} = \frac{1 - E}{\tau_{\text{rec}}}$$
Spikes ($v \ge 30$, $E > 0$) deduct $\Delta_E = 0.1$ ($\tau_{\text{rec,exc}} = 1000\text{ ms}$, $\tau_{\text{rec,inh}} = 500\text{ ms}$), yielding an analytic steady-state firing ceiling $f_{\max} = 1/(\Delta_E \tau_{\mathrm{rec}}) = 12.0\text{ Hz}$ population-weighted.

## 2. Presynaptic Dual-Trace Three-Factor Plasticity
Synapses maintain fast ($\tau_1 = 100\text{ ms}$) and slow ($\tau_2 = 2500\text{ ms}$) presynaptic eligibility traces modulated by scalar step reward $R_t$:
$$\Delta w_{ij} = \eta_{\mathrm{DA}} R_t \left(\mathrm{Trace1}_{ij} + 0.5\,\mathrm{Trace2}_{ij}\right)$$

## 3. Tri-State Global Activity Regulator (GAR)
Over each $100\text{ ms}$ window ($N_{\mathrm{recent}}$ population spikes across 500 neurons), GAR executes safety NaN resets, coma kicks ($N_{\mathrm{recent}} < 50$, $<1.0\text{ Hz}$), and seizure clamps ($N_{\mathrm{recent}} > 5000$, $>100.0\text{ Hz}$).

# Reproducibility, Empirical Verification & Benchmarks

`OrganoidEnv` includes an automated verification suite (`pytest tests/` — 84+ unit, biophysical, adversarial, and platform API tests) and `scripts/verify_reproducibility.py`. Across an 18-condition, 180-run empirical campaign ($10$ equalized seeds per condition, $500$ episodes per seed) evaluated with `rliable` IQMs and 60-second stability sweeps:
- **Metabolic Rate Bounding:** Energy fatigue caps sustained sweep firing to $12.2\text{--}12.6\text{ Hz}$ over $60\text{ s}$, whereas unregulated networks reach $139.1\text{ Hz}$.
- **Transparent Null & Baseline Findings:** GAR stays dormant under nominal drive ($0$ seizure clamps across $55{,}000$ spiking episodes), structural pruning is null ($p_{\mathrm{FDR}} = 0.362$), static/fast-trace SNNs match or outperform dual-trace `C0` on in-distribution return, and step-matched (`0.05`) silicon baselines recover Stage-4 and out-of-distribution goal completion.

# Software Availability

`OrganoidEnv` is hosted on GitHub under the MIT License (`https://github.com/vansh7nvc/Organoid-Intelligence-gym-`) with GitHub Actions CI across Ubuntu and Windows (Python 3.9–3.11) and Docker support (`docker compose up platform` / `docker compose --profile verify run reproduce`).

# Acknowledgements

The authors acknowledge the open-source communities behind Brian 2 [@brian2], Gymnasium [@towers2023gymnasium], PyTorch, FastAPI, and Three.js.

# References
