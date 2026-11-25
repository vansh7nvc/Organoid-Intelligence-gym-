# OrganoidEnv: Environment Verifier Plan (Colab)

**Purpose:** Check that your Colab setup meets the requirements from the revision plan. Each check has a **Pass rule** and a **Fix if failed**. Run Stages A to E in order; do not start long runs until all pass.

**Confidence:** [Certain] = deterministic test you can run; [Likely] = depends on your code, which I have not seen.

---

## Stage A: Runtime and environment

| ID | Check | Pass rule | Fix if failed |
|----|-------|-----------|---------------|
| A1 | Hardware | Record CPU count (`os.cpu_count()`), RAM, GPU or none | Log it in every run; adjust parallelism to the real core count |
| A2 | Drive mounted | Writable folder on Drive for checkpoints and logs | Mount Drive; never write checkpoints to `/content` only |
| A3 | Pinned versions | `requirements.txt` exists with exact versions; installed versions match | `pip freeze > requirements.txt`; install from it at session start |
| A4 | Version logging | Python, numpy, torch, gymnasium versions saved in every run's metadata | Add a metadata dict to each log file |
| A5 | Seeding | One function sets Python, NumPy, torch (CPU and CUDA) seeds and is called at run start | Write `set_all_seeds(seed)`; call before building env, SNN, and agent |

## Stage B: Checkpoint completeness [Certain]

The checkpoint must contain **all** of the following. Mark each as present or missing.

- [ ] SNN weights (all layers)
- [ ] Fast and slow eligibility traces for every synapse
- [ ] ATP/energy vector
- [ ] Membrane `v` and recovery `u` state (or confirmed reset at episode start)
- [ ] Inhibitory scaling state and pruning mask
- [ ] Structural-plasticity episode counter (pruning every 10 episodes)
- [ ] D3QN online and target networks
- [ ] Adam optimizer state
- [ ] Replay buffer, including PER priorities and max priority, and HER-generated samples
- [ ] RNG states: Python `random`, NumPy, torch, and any env-level generators
- [ ] Episode number, curriculum stage, step counters, epsilon/exploration schedule
- [ ] Per-episode logs so far (reward, success, firing rate, GAR events)

**Fix if any missing:** add it to `save_ckpt` and `load_ckpt`. Missing RNG state or replay buffer are the two most damaging omissions.

## Stage C: Resume-equivalence test [Certain]

This is the decisive test. Same seed, interrupted vs uninterrupted, must give identical results.

1. Run seed 42 for **20 episodes** straight. Save the per-episode reward list and a hash of final weights.
2. Run seed 42 for **10 episodes**, save a checkpoint, **restart the Python process**, load, run **10 more**.
3. Compare.

**Pass rule:** Rewards identical and weight hashes equal (or equal within 1e-6 if GPU nondeterminism is present, with that noted).

**Fix if failed:** bisect. Save and reload only one component at a time until the divergence disappears. Typical culprits: RNG state, empty replay buffer after resume, traces reset to zero, optimizer moments lost.

Template (adapt the function names to your code):

```python
import hashlib, json, numpy as np, torch

def state_hash(run):
    h = hashlib.sha256()
    for name, arr in run.snn_state_arrays().items():      # weights, traces, ATP
        h.update(np.ascontiguousarray(arr).tobytes())
    for p in run.agent.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def resume_test(make_run, train, save_ckpt, load_ckpt, seed=42, n=20, path="/content/drive/MyDrive/ckpt_test.pt"):
    # Straight run
    a = make_run(seed); ra = train(a, n)
    # Interrupted run (restart the process between the two halves for a true test)
    b = make_run(seed); rb1 = train(b, n // 2); save_ckpt(b, path)
    c = make_run(seed); load_ckpt(c, path);     rb2 = train(c, n - n // 2)
    same_rewards = np.allclose(ra, rb1 + rb2, atol=1e-6)
    same_state = state_hash(a) == state_hash(c)
    print("rewards match:", same_rewards, "| state match:", same_state)
    return same_rewards and same_state
```

## Stage D: Performance profile [Likely]

| ID | Check | Pass rule | Fix if failed |
|----|-------|-----------|---------------|
| D1 | Profile one episode with `cProfile` | Time split reported for SNN substep loop, plasticity update, D3QN update, env | Needed before any optimisation |
| D2 | SNN substep loop | No Python-level loop over neurons; operations are vectorised arrays | Vectorise with NumPy or move to torch tensors |
| D3 | Plasticity update | Trace updates are array ops, not per-synapse loops | Use matrices or sparse tensors |
| D4 | Seconds per episode | Target is under 10 s per episode (from 27.5 s) if feasible | Optimise the largest profile bucket first; stop when returns diminish |
| D5 | Numerical equivalence after optimisation | Optimised and original code match on a fixed seed for 5 episodes within tolerance | Re-run Stage C after any optimisation |

## Stage E: Experiment orchestration [Likely]

| ID | Check | Pass rule | Fix if failed |
|----|-------|-----------|---------------|
| E1 | One seed per session | Each seed writes its own log and checkpoint path | Parameterise paths by seed and config |
| E2 | Auto-resume | Starting the notebook again continues from the latest checkpoint | Add a "find latest checkpoint" step at start |
| E3 | Checkpoint frequency | Every 1 to 5 episodes; at most 15 minutes of work at risk | Lower the interval |
| E4 | Atomic writes | Write to temp file, then rename, so a disconnect cannot leave a corrupt checkpoint | Save to `.tmp`, then `os.replace` |
| E5 | Run manifest | A CSV listing every seed and config with status (pending, running, done) | Create it; update after each run |
| E6 | Idle/disconnect handling | Run state survives a forced disconnect | Test by manually disconnecting the runtime mid-run |

## Stage F: Pre-launch gate

Launch the full 10-seed and ablation runs only when:

- [ ] Stage A all pass
- [ ] Stage B all boxes ticked
- [ ] Stage C passes
- [ ] Stage D profile recorded and D5 passes if optimised
- [ ] Stage E1 to E6 pass

Save this checklist's results (a JSON of pass/fail per ID) alongside the code and cite it in the paper's reproducibility section.

## If you use a different technique

If your current approach differs from the above, keep it only if it passes Stage C. The resume-equivalence test, not the method, is the criterion.
