# Organoid Intelligence: Month 1 Progress Report

## 1. Executive Summary

Over the past month, we have successfully established the foundational architecture for the **Organoid Reinforcement Learning (RL)** environment. We progressed from single-neuron models to a **mesoscale network of 1000 metabolic neurons** integrated with the **Gymnasium** standard. While the structural implementation of the learning mechanism (Eligibility Traces & Dopamine) is complete, **numerical stability** remains a critical challenge for the metabolic model under stimulation.

## 2. Weekly Achievements

### Week 1: Foundation & Modeling

- **Objective**: Set up the project and implement the core neuron model.
- **Achievements**:
  - Established project structure (`organoid_rl` repository).
  - Implemented the **Basic Izhikevich Model**.
  - Configured **Brian2** with **Cython** backend for high-performance simulation.

### Week 2: Scaling & Metabolism

- **Objective**: Scale to 1000 neurons and prevent runaway "seizures".
- **Achievements**:
  - **Metabolic Governor**: Integrated energy variable `E` to simulate ATP constraints.
  - **Network Scaling**: 1000 Neurons (800 Excitatory / 200 Inhibitory).
  - **Stability Tuning**: Tuned `tau_recovery` and refractory periods to stabilize the resting network at ~1Hz.
  - **Visualization**: Generated raster plots proving sustained, non-epileptic activity.

### Week 3: The Interface

- **Objective**: Create a standard RL interface for agent interaction.
- **Achievements**:
  - **Gymnasium Integration**: Implemented `OrganoidEnv` in `environment/core.py`.
    - **Action Space**: `Discrete(8)` (Stimulation of neuron subgroups).
    - **Observation Space**: `Box(10,)` (Normalized firing rates).
  - **"God Mode" Stabilizer**: Implemented a fail-safe system (`stimulator.py`) to inject noise (kickstart) or inhibition (calming) based on global activity.

### Week 4: Memory & Learning

- **Objective**: Implement biological learning rules (STDP + Dopamine).
- **Achievements**:
  - **Plasticity Architecture**: Updated synapses to support learnable weights (`w`) and eligibility traces (`Trace`).
  - **Learning Rule**: Implemented `apply_dopamine` (`w += lr * Reward * Trace`).
  - **Validation**: Attempted "Pavlovian Conditioning" (Delayed Reward) task.

## 3. Key Technical Challenges

### Numerical Instability in Metabolic Model

The primary hurdle encountered in Week 4 is the **numerical sensitivity** of the Metabolic Izhikevich equations during strong stimulation (the "Action" phase).

- **Issue**: Injection of stimulation current causes the membrane potential `v` to diverge to `NaN` or `Infinity`.
- **Root Cause**: The coupled differential equations for voltage and energy become stiff or unstable when driven hard, especially with the simplified `euler` or `rk2` integration methods required for speed.
- **Current Mitigation**: Reduced stimulation intensity and implemented automatic state resetting ("God Mode") when NaNs are detected.

## 4. Month 2 Roadmap

1.  **Stabilization (Critical)**:
    - Tune metabolic parameters (`a`, `b`, `epsilon`) to handle stimulation robustly.
    - Investigate "soft thresholds" or alternative integration methods (e.g., exponential Euler) if supported by Brian2/Cython.
2.  **Learning Validation**:
    - successfully demonstrate weight strengthening in the Delayed Reward task.
3.  **Closed-Loop Tasks**:
    - Train the organoid to solve a simple control task (e.g., "Keep activity within range" or "Move a cursor").
