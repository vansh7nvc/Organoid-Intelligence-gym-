---
title: 'OrganoidEnv: A Neuromorphic Testbed for Reinforcement Learning with Biologically Constrained Spiking Neural Networks'
tags:
  - Python
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
date: 28 August 2026
bibliography: paper.bib
---

# Summary

Bridging the divide between high-performing artificial neural networks trained with mathematical backpropagation and biologically grounded spiking neural networks (SNNs) remains a fundamental grand challenge in neuromorphic engineering and computational neuroscience. While artificial agents master complex tasks through non-local gradient optimization, biological neural tissue operating *in vitro* or *in vivo* faces severe physical constraints: non-differentiable event-driven action potentials, local synaptic plasticity, continuous metabolic expenditure, and susceptibility to pathological dynamics such as paroxysmal seizures or hypoactive comas [@kagan2022in_vitro; @smirnova2023organoid].

`OrganoidEnv` is an open-source, Gymnasium-compatible Python package designed to simulate, control, and benchmark *in silico* biological neural organoids. It embeds a 500-neuron recurrent spiking neural network modeled via metabolic-extended Izhikevich dynamics [@izhikevich2003simple] within an embodied 2D closed-loop navigation ecosystem. An external reinforcement learning optimization agent (acting as an "automated experimentalist") guides the biological network through sensory-motor electrical stimulation while internal synaptic weights adapt autonomously via dopamine-modulated plasticity.

# Statement of Need

Neuromorphic researchers currently lack standardized, reproducible software environments to benchmark biological learning rules on embodied closed-loop tasks. Existing SNN libraries such as BindsNET, SNN-Torch, or BrainPy primarily focus on either converting deep ANNs to SNNs via backpropagation-through-time (BPTT) or simulating biophysical networks without standard RL interfaces.

`OrganoidEnv` fills this critical gap by providing:
1. **Gymnasium Compliance**: A standard interface (`reset()`, `step()`, `observation_space`, `action_space`) seamlessly compatible with modern RL ecosystems [@towers2023gymnasium].
2. **Biological Realism**: True local synaptic learning rules without non-local gradient backpropagation.
3. **Metabolic Fatigue Modeling**: Action potentials consume discrete quanta of metabolic energy ($E$), penalizing hyper-synchronous firing and naturally producing sparse dynamics ($\approx 8.5\,\text{Hz}$).
4. **Autonomous Homeostatic Regulation**: A built-in *Global Activity Regulator* (GAR) preventing seizure and coma states.
5. **Reproducible Benchmarking**: Multi-seed pipelines, statistical confidence estimation, and 1-click Google Colab reproduction notebooks.

# Architectural Overview

The core architecture consists of four interconnected biological modules:

```
[21D Spatial Observation]
         │
         ▼
[Sparse Distributed Memory (SDM)] (256 Neurons, 10% Expansion)
         │
         ▼
[Recurrent Cortical Mesh] (500 RS/FS Metabolic-Izhikevich Neurons)
         │
         ▼
[Clustered Motor Quadrants] (Up, Down, Left, Right) ──► [2D Kinematic Movement]
         ▲
         │
[Global Activity Regulator (GAR)] (Homeostatic Gain Modulation & Reset Clamping)
```

## 1. Metabolic-Izhikevich Dynamics
Neurons follow two-variable differential equations extended with an ATP-like dynamic variable $E \in [0, 1]$:
$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I, \quad \frac{du}{dt} = a(bv - u)$$
$$\frac{dE}{dt} = \frac{1 - E}{\tau_{\text{recovery}}}$$
Spiking occurs when $v \ge 30\,\text{mV}$ and $E > 0$. Upon spiking, $v \leftarrow c$, $u \leftarrow u + d$, and $E \leftarrow E - 0.1$.

## 2. Sparse Distributed Memory (SDM)
To resolve spatial coordinates without catastrophic aliasing, continuous observations are expanded into a 256-neuron high-dimensional non-linear projection layer with 10% sparse activation.

## 3. Dopamine-Modulated Dual-Trace Plasticity ($D^3$)
Synaptic weights are updated according to an eligibility trace bridging millisecond-level STDP with delayed navigation rewards:
$$\Delta w = \eta \cdot R \cdot \left(\text{Trace}_{\text{fast}} + 0.5 \cdot \text{Trace}_{\text{slow}}\right)$$
where $\tau_{\text{fast}} = 100\,\text{ms}$ and $\tau_{\text{slow}} = 2{,}500\,\text{ms}$.

## 4. Global Activity Regulator (GAR)
GAR continuously tracks mean population firing frequency $\bar{f}$. When $\bar{f} > 35\,\text{Hz}$, inhibitory conductance is scaled up to prevent runaway seizures. When $\bar{f} < 1.0\,\text{Hz}$, stochastic Poisson current is injected to prevent hypoactive coma collapse.

# Empirical Benchmarks & Performance

`OrganoidEnv` was validated across $N=3$ independent biological seeds (200 episodes each) and component ablations. Key empirical findings:
- **Task Success:** The full model achieves $71.3\% \pm 40.1\%$ Stage 1 success and $62.0\% \pm 38.4\%$ Stage 2 (obstacle navigation) success.
- **Ablation Criticality:** Disabling SDM reduces success to $3.7\%$, while disabling GAR results in high variance ($\sigma = \pm 93.7$) and severe mid-training seizure collapse.
- **Neuromorphic Energy Efficiency:** With an average firing rate of $8.5\,\text{Hz}$, the SNN requires only $\approx 4.2 \times 10^3$ synaptic operations (SOPs) per step, yielding an estimated $>190\times$ theoretical energy reduction over traditional deep RL on neuromorphic hardware (e.g., Intel Loihi 2).

# Software Availability & Verification

`OrganoidEnv` is hosted on GitHub under the MIT License:  
`https://github.com/vansh7nvc/Organoid-Intelligence-gym-`

The repository includes:
- Automated GitHub Actions test matrix across Python 3.9, 3.10, and 3.11.
- Complete documentation, architectural diagrams, and interactive Jupyter / Google Colab notebooks.
- Pre-trained checkpoints and reproducible figure generation pipelines.

# Acknowledgements

The authors acknowledge the open-source communities behind Brian 2 [@brian2], Gymnasium [@towers2023gymnasium], and PyTorch.

# References
