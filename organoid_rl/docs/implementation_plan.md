# Month 6: Fresh Start — Full Bug Audit & Fix Plan

## Root Cause: Why Training Never Scored Goals

The diagnostic proved the cursor moves **0.06 units in 80 steps** — a random walk that can never reach a goal 0.7 units away. Here's every bug causing this:

---

## Bug #1: Actions Don't Control Motor Neurons (CRITICAL)

**`_apply_action` stimulates the WRONG neurons.**

| Action | Neurons Stimulated | What They Actually Are |
|:-------|:-------------------|:-----------------------|
| 0 | 0-49 | **SDM neurons** (sensory!) |
| 1 | 50-99 | **SDM neurons** |
| 2-5 | 100-299 | **SDM + Hidden neurons** |
| 6-7 | 300-399 | **Hidden neurons** |
| ❌ NONE | **400-500** | **Motor neurons** (the ones that actually move the cursor!) |

The agent's 8 actions never touch motor neurons (400-500). Movement is entirely driven by indirect spike propagation, which is too weak and noisy.

**Fix**: Remap actions 0-3 to directly stimulate the 4 motor quadrants (Up/Down/Left/Right at 400-500), and actions 4-7 to stimulate diagonal combinations.

---

## Bug #2: Motor Sensitivity Too Low (CRITICAL)

`sensitivity = 0.02` means each net spike difference produces 0.02 units of movement. With ~5-10 net spikes per step, the cursor moves ~0.10-0.20 per step. But because the 4 motor clusters fire roughly equally (no directed input!), the net movement is ~0.02 per step in a random direction.

**Fix**: Increase `sensitivity` from `0.02` → `0.05`.

---

## Bug #3: SDM and Action Stimulation Conflict (MODERATE)

Both `_apply_action()` and `_stimulate_sdm()` inject current into neurons 0-255. The SDM sensory drive and action stimulation fight each other, corrupting both signals.

**Fix**: Once actions target motor neurons (Bug #1 fix), this conflict disappears automatically.

---

## Bug #4: No Directional Action Mapping (MODERATE)

Even if we fix the target neurons, the current 8-action scheme has no concept of "move right." The agent picks action 0-7, which stimulates an arbitrary neuron block. None of these actions map to a specific direction.

**Fix**: Create a **directional action space**:
- Action 0 = Stimulate **motor_up** (400-425) → cursor moves up
- Action 1 = Stimulate **motor_down** (425-450) → cursor moves down  
- Action 2 = Stimulate **motor_left** (450-475) → cursor moves left
- Action 3 = Stimulate **motor_right** (475-500) → cursor moves right
- Actions 4-7 = Diagonal combos (up+right, up+left, down+right, down+left)

---

## Bug #5: prev_dist Updated After Reward Calc (MINOR)

In `step()`, `prev_dist` is updated at line 276 **after** `_calculate_reward()` at line 273. This means the potential-based shaping `F = γ·Φ(s') - Φ(s)` uses the correct `prev_dist` for Φ(s), which is actually fine. ✅ Not a bug.

---

## Proposed Changes

### [MODIFY] [core.py](file:///c:/Users/Acer/OneDrive/Desktop/Organoid/Intelligence/organoid_rl/environment/core.py)

#### 1. Rewrite `_apply_action` — Target Motor Neurons Directly
```python
def _apply_action(self, action):
    """Maps action ID to directional motor neuron stimulation."""
    stim_strength = 25.0
    if action == 0:    # Up
        self.neurons.I[400:425] += stim_strength
    elif action == 1:  # Down
        self.neurons.I[425:450] += stim_strength
    elif action == 2:  # Left
        self.neurons.I[450:475] += stim_strength
    elif action == 3:  # Right
        self.neurons.I[475:500] += stim_strength
    elif action == 4:  # Up-Right
        self.neurons.I[400:425] += stim_strength
        self.neurons.I[475:500] += stim_strength
    elif action == 5:  # Up-Left
        self.neurons.I[400:425] += stim_strength
        self.neurons.I[450:475] += stim_strength
    elif action == 6:  # Down-Right
        self.neurons.I[425:450] += stim_strength
        self.neurons.I[475:500] += stim_strength
    elif action == 7:  # Down-Left
        self.neurons.I[425:450] += stim_strength
        self.neurons.I[450:475] += stim_strength
```

#### 2. Increase Motor Sensitivity
```python
sensitivity = 0.05  # Was 0.02
```

### [MODIFY] [12_month6_training.py](file:///c:/Users/Acer/OneDrive/Desktop/Organoid/Intelligence/organoid_rl/experiments/12_month6_training.py)

- New tag: `"month6_final"`
- Keep: `total_episodes=500`, `steps_per_episode=80`
- Keep: Brain checkpointing ✅

## Verification Plan

1. **Run diagnostic** first — confirm cursor covers >0.3 units in 80 random steps  
2. **Launch full training** — expect first goals within 10-20 episodes  
3. **Monitor** Episode 50 checkpoint for goal count >5
