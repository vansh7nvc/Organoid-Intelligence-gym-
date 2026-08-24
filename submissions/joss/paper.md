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
authors:
  - name: Vansh Sharma
    orcid: 0000-0000-0000-0000 # Replace with actual ORCID
    affiliation: 1
  - name: Seema Malik
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 21 August 2026
bibliography: paper.bib
---

# Summary

Bridging the gap between high-performing deep reinforcement learning (RL) and biologically grounded spiking neural networks (SNNs) remains a fundamental challenge in neuromorphic computing. Most existing approaches either compromise on biological plausibility by using surrogate gradients and non-local learning rules, or they fail to scale to complex delayed-reward tasks. Furthermore, the burgeoning field of *organoid intelligence* (OI) requires *in silico* models that can simulate the constraints of living neural tissue—such as metabolic fatigue, continuous structural plasticity, and homeostatic regulation—to guide future *in vitro* experiments.

`OrganoidEnv` is a Gymnasium-compatible reinforcement learning environment that encapsulates a highly biologically constrained SNN as the "actor" within a simulated ecosystem. It is the first open-source testbed to combine metabolic-Izhikevich neurons, homeostatic stabilizers (Global Activity Regulator), and dopamine-modulated dual-trace plasticity in a single, reproducible package. 

# Statement of need

Neuromorphic engineers and computational neuroscientists lack standardized platforms to benchmark biological learning rules in embodied tasks. `OrganoidEnv` addresses this need by providing an external deep RL agent (e.g., a D3QN) that acts as an "automated experimentalist," interacting with the SNN via sensory-motor current stimulation. This architecture allows the internal dynamics of the SNN to remain strictly local, temporally precise, and metabolically constrained, while the external RL agent optimizes the stimulation protocol. 

Built on `Brian2` [@brian2] and the `Gymnasium` API, `OrganoidEnv` allows researchers to:
1. Evaluate new synaptic plasticity and structural pruning rules.
2. Quantify the energetic and computational costs of biological constraints.
3. Rapidly prototype training protocols for future brain-computer interfaces and biological organoids.

# Implementation Details

The core network consists of 500 Izhikevich neurons (80/20 excitatory/inhibitory ratio) structured into functional layers: Exteroceptive, Spatial Distributed Memory (SDM), Hidden, and Motor layers. Synaptic weights are modified using a novel Dual-Trace STDP rule that bridges millisecond-scale spike timing with second-scale reward signals. The environment incorporates an ATP-like metabolic variable that enforces sparse firing, alongside a Global Activity Regulator (GAR) that injects noise or triggers resets to prevent pathological seizure or coma states.

# Acknowledgements

The authors acknowledge the open-source communities behind Brian2, Gymnasium, and Ray Tune, which made this framework possible.

# References
