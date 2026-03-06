# Week 1 Report: Organoid Intelligence - High-Performance Neural Environment

## Executive Summary
During Week 1, we successfully initialized the **OrganoidRL** research environment. We moved from standard, infinite-energy neuron models to a biologically constrained **Metabolic Izhikevich** model, optimized for high-performance execution on Windows using a custom Cython backend.

---

## 1. Technical Infrastructure: Cython Backend
To support large-scale organoid simulations, we bypassed the standard Python interpreter overhead by configuring a **C++ Cython backend**.

- **Compiler:** `MinGW-w64 (GCC 14.2.0)`
- **Optimization:** Forced `MS_WIN64` macros and `mingw32` linkage to ensure 64-bit stability on Windows.
- **Performance Gain:** Initial benchmarks show significantly faster execution compared to the default `numpy` backend, enabling real-time simulation of complex spiking patterns.

---

## 2. Core Model: The Metabolic Governor (Day 2)
Standard spiking models (like the original Izhikevich 2003) ignore the metabolic cost of firing. We implemented the **Metabolic Governor** to simulate ATP depletion.

### Key Equations:
- **Energy Depletion:** $E \leftarrow E - \text{spike\_cost}$ on every spike.
- **Energy Recovery:** $\frac{dE}{dt} = \frac{1 - E}{\tau_{recovery}}$
- **The Constraint:** A neuron cannot fire if $E \le 0$, regardless of membrane potential.

**Benefit:** This prevents "Numerical Seizures" where neurons fire at unrealistic frequencies forever, providing a natural homeostatic stabilizer for the RL agent.

---

## 3. Experimental Findings

### Experiment 01: The Stress Test (Day 3)
We subjected a single neuron to a "dangerously high" input current ($I=25.0$) for 5 seconds.
- **Finding:** The neuron exhibited three distinct phases:
    1. **Fresh Phase:** Rapid Gamma-band firing (~55Hz).
    2. **Depletion Phase:** Noticeable slowing as energy $E$ dropped.
    3. **Exhaustion Phase:** Minimal firing sustained only by the recovery rate.
- **Conclusion:** Our model successfully self-regulates activity under extreme stress, mimicking biological fatigue.

### Experiment 02: Synaptic Transmission & Delays (Day 4-5)
We established the first synaptic link between two metabolic neurons.
- **Axonal Delays:** Implemented randomized delays ($1\text{ms} + \text{jitter}$).
- **Causality:** Proved that transmission delays are correctly handled by the Cython backend.
- **Significance:** Without these delays, Spike-Timing Dependent Plasticity (STDP) is mathematically impossible as "cause" and "effect" would coincide. We have now laid the groundwork for learning.

---

## 4. Current Repository Structure
- `organoid_rl/environment/neurons.py`: Core Metabolic Izhikevich implementation.
- `organoid_rl/experiments/01_stress_test.py`: Validation of metabolic stability.
- `organoid_rl/experiments/02_synapse_test.py`: Validation of causal synaptic links.
- `organoid_rl/sanity_check.py`: Automated toolchain verification script.

---

## Next Steps: Week 2
- **Synaptic Plasticity:** Implementing STDP (Spike-Timing Dependent Plasticity).
- **Environment Integration:** Wrapping the spiking network into a Gymnasium-compatible `OrganoidEnv`.
- **Sensory Input:** Mapping external observations to spike trains.

**Status: Week 1 Complete - Environment Stabilized.**
