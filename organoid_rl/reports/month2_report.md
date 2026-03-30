# Month 2 Progress Report: The Leap to Learning

## Executive Summary

Month 2 marks a fundamental transition from a static neural simulation to a **learning-capable organoid agent**. We have successfully resolved the catastrophic numerical instabilities of Month 1 and established a closed-loop system where the organoid can sense a target and move a virtual cursor.

---

## 🛠️ Technical Breakthroughs

### 1. Mesoscale Numerical Stability

The metabolic Izhikevich model is now robust under high-intensity stimulation.

- **Voltage Clipping**: Prevented quadratic term explosion ($v^2$) by introducing a metabolic-safe `v_clipped` variable.
- **Euler Integration**: Switched to first-order integration for synapses to ensure stable eligibility trace decay.
- **Stabilizer Refinement**: Tuned "God Mode" thresholds to 5,000 spikes (50Hz), allowing natural network activity while suppressing pathological seizures.

### 2. Biological Learning Infrastructure

The network now implements **Dopamine-Modulated STDP** (Eligibility Traces).

- **Eligibility Traces**: Every pre-synaptic spike leaves a "tag" (Trace) on the synapse that decays over $250ms$.
- **Global Reward Signal**: Rewards (Dopamine) are applied globally to all synapses, but only those with active traces (recent activity) undergo weight changes.
- **Weight Homeostasis**: Implemented slow weight decay back to baseline ($w_0$) and hard capping (0–100) to prevent runaway excitation.

### 3. Closed-Loop Motor Mapping

We have mapped the organoid's output to a 2D virtual environment.

- **Motor Cortex**: The last 100 neurons (900-999) act as "muscles." Activity in these quadrants moves the cursor UP, DOWN, LEFT, or RIGHT.
- **Proprioception**: The organoid now "sees" its own cursor position and the target position as part of its observation vector.

---

## 📈 Validation Results

### Pavlovian Conditioning (`04_delayed_reward.py`)

- **Control**: +0.18 weight change (background noise).
- **Target**: **+0.32 weight change** (reinforced).
- **Verdict**: The network can successfully form and retain memories of rewarded stimulation.

### Goal-Directed Navigation (Initial Training)

- Successfully established a 14-dimensional observation space.
- Implemented **Distance-Based Reward Shaping** (Dopamine spikes when moving closer to target).
- **Current Status**: Infrastructure for 100-episode training runs is fully functional.

---

## 🔗 Repository Status

The project is now officially connected and synced with:
[https://github.com/vansh7nvc/Organoid-Intelligence-gym-](https://github.com/vansh7nvc/Organoid-Intelligence-gym-)

## 📅 Next Milestones

- [ ] Complete the 100-episode training curriculum.
- [ ] Measure "Navigation Efficiency" (steps taken per goal reached).
- [ ] Implement inhibitory homeostasis to balance excitatory strengthening.

---

_Report generated on 2026-03-07 by Antigravity._
