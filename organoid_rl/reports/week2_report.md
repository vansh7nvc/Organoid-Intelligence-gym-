# Week 2 Report: The "Seizure" Test (Scaling Up)

## 1. Objective

The goal of Week 2 was to scale the neural simulation to a "mesoscale" population (1000 neurons) and investigate the role of random connectivity, synaptic weights, and metabolic constraints in maintaining network stability.

## 2. Implementation Overview

### Population Scaling

- **Total Neurons**: 1000
- **Excitatory (RS)**: 800 (80%)
- **Inhibitory (FS)**: 200 (20%)
- **Model**: Metabolic Izhikevich Model (incorporating energy variable `E`).

### Connectivity and Wiring

- **Topology**: Random sparse connectivity ($P = 0.1$).
- **Excitatory Weights ($w_{exc}$)**: $+15mV$
- **Inhibitory Weights ($w_{inh}$)**: $-25mV$
- **Inputs**: Poisson noise (100Hz) applied to all neurons to maintain stochastic drive.

## 3. Findings

### The "Crash" Phase (Day 3)

During initial testing without a refractory period and with fast energy recovery, the network exhibited **numerical instability**. High-frequency firing led to membrane potentials exploding to infinity (NaN errors), simulating a massive epileptic seizure that crashed the simulation.

### Tuning & Stability (Day 4)

To stabilize the network, the following adjustments were made:

- **Refractory Period**: A $2ms$ refractory period was added to prevent mathematical singularities.
- **Metabolic Governor**: `tau_recovery` was increased (1.0s for Exc, 0.5s for Inh) to simulate slower ATP replenishment.
- **Dynamic Regulation**: The system now self-regulates; high firing rates deplete energy `E`, which automatically lowers synaptic efficacy, forcing the network back into a quiescent state.

## 4. Final Deliverable Results

The final 60-second simulation demonstrates **sustained, stable activity**. The metabolic system prevents runaway excitation, keeping the average firing rate at a sustainable level (~1Hz sustained across the whole population).

### Visualization

![Week 2 Raster Plot](file:///c:/Users/Acer/OneDrive/Desktop/Organoid%20Intelligence/organoid_rl/experiments/02_seizure_test_result.png)

## 5. Conclusion

Week 2 successfully demonstrated that metabolic constraints are sufficient to prevent catastrophic seizures in large-scale random networks. This foundation is essential for the Pavlovian tasks in Week 3, where specific pathways will be trained using STDP.
