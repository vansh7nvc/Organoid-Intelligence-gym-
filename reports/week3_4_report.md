# Week 3 & 4 Progress Report: Interface and Memory

## 1. Objectives

- **Week 3**: Implement Gymnasium interface (`OrganoidEnv`) and "God Mode" stabilizer.
- **Week 4**: Implement Eligibility Traces and Dopamine-modulated STDP for reinforcement learning.

## 2. Achievements

### Week 3: Gymnasium Integration

- **`OrganoidEnv` Class**: Fully implemented in `environment/core.py`.
  - **Action Space**: `Discrete(8)` mapping to current injection in neuron subgroups.
  - **Observation Space**: `Box(10,)` representing normalized firing rates.
  - **Step Function**: Integrated with Brian2 simulation loop.
- **Verification**: `experiments/03_pavlov_task.py` successfully runs the environment loop and produces valid observations.

### Week 4: Memory & Learning

- **Plasticity Mechanism**: Implemented in `environment/rewards.py` (`apply_dopamine`).
- **Eligibility Traces**: Added `Trace` variable to synapses in `OrganoidEnv`.
- **Synaptic Model**: Updated synapses to include `w` (weight) and `Trace` dynamics.
  - _Note_: Simplified to non-differential `Trace : 1` due to Brian2 `stateupdater` issues with the complex metabolic model equations.

### Stability & "God Mode"

- Implemented `check_and_stabilize` in `environment/stimulator.py`.
- Features:
  - **Activity Regulation**: Injects noise if silent, applies inhibition if seizing.
  - **Crash Recovery**: Detects `NaN` values (numerical precision errors) and resets neuron states to resting potential to prevent simulation failure.

## 3. Challenges & Findings

### Numerical Instability

- The Metabolic Izhikevich model proves to be highly sensitive to stimulation.
- **Symptoms**: `v` (membrane potential) diverges to `NaN` or `Infinity` during strong stimulation, causing "overflow encountered" warnings.
- **Mitigation Attempts**:
  - Reduced `dt` to 0.05ms.
  - Switched integration method to `rk2`.
  - Reduced stimulation current from 100.0 to 20.0.
  - Implemented automatic NaN reset in `step()`.
- **Current Status**: The simulation runs, but instability often interrupts learning events, preventing the "Delayed Reward" validation from showing clear weight strengthening.

## 4. Next Steps

1. **Stabilize Neuron Model**:
   - Further tune `a`, `b`, `c`, `d` parameters for the Metabolic model.
   - Consider clamping `v` within the equations (soft thresholds).
2. **Refine Traces**:
   - Restore differential equation `dTrace/dt = -Trace/tau` once stability is achieved.
3. **Learning Validation**:
   - Re-run `04_delayed_reward.py` once the network can sustain stimulation without resetting.
