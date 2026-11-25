# OrganoidEnv: Revision Plan

**Goal:** Turn the manuscript into a rigorous *resource/benchmark* paper by cutting unsupported claims and proving the remaining ones.
**Confidence tags:** [Certain] = verifiable from the manuscript or arithmetic; [Likely] = strong inference; [Guessing] = gap-filling.
**Effort:** S = under 1 day, M = 1-3 days, L = a week or more (CPU-only assumed).

---

## Phase 1: Reframe the claims (S, do first)

| # | Section | Current | Change | Done when |
|---|---------|---------|--------|-----------|
| 1.1 | Title/Abstract | "RL can effectively train a biologically constrained SNN" | Reword to "a reproducible testbed for RL-SNN co-training." [Likely] | No sentence claims the SNN *learns* unless Phase 2 proves it |
| 1.2 | Abstract, Contribution, Conclusion | "First to combine metabolic-Izhikevich, stabilizers, dual-trace plasticity" | Delete, or support with a documented survey table of existing tools. DishBrain (Kagan et al., 2022) must be addressed. [Likely] | Zero unsupported "first" claims |
| 1.3 | Abstract, Discussion | Seed variance "resembles organoid variability" | Remove or mark as speculation. It is unfalsifiable. [Certain] | Variance is reported as a result, not a feature |
| 1.4 | Throughout | "Organoid" framing | State that nothing in the model is organoid-specific; it is an analogy. [Likely] | Explicit disclaimer in Introduction |
| 1.5 | Sec. III-A | "SDM" layer | Say it is a sparse random projection, not Kanerva's Sparse Distributed Memory, or rename it. [Likely] | Name and description match |
| 1.6 | Abstract | "±25-50% perturbations did not destabilize" | Replace with actual Table V results (drops of ~18-22 points at alpha=1.0 and f_seizure=30 Hz). [Certain] | Text matches table |

---

## Phase 2: New experiments (the part reviewers weigh most)

### 2.1 SNN contribution ablations (M)
[Certain] None of your stated contributions except GAR and SDM are currently ablated.

| Config | What to change | Question answered |
|--------|----------------|-------------------|
| Frozen weights | Disable Eqs. 5-7 | Does dopamine plasticity help? |
| Random untrained SNN | Fixed random weights, same D3QN | Is the SNN a noisy relay? |
| No metabolism | E_i fixed at E_max | Does ATP fatigue matter? |
| No homeostasis | Remove alpha scaling | Is inhibitory scaling needed? |
| No structural plasticity | Disable pruning | Does pruning matter? |
| No dual-trace (single trace) | Remove slow trace | Does the slow trace matter? |

**Done when:** a table shows each config over 10 seeds with the metrics in 2.3.

### 2.2 Seeds and statistics (L, mostly compute time)
- [Certain] Run **10 seeds x 500 episodes** for the full model. At 27.5 s/episode that is about 38 CPU-hours total, less if parallelised.
- Report **median, IQR, and a per-seed table**. Drop the bootstrap CI on N=3.
- Run **5 seeds per ablation** minimum, with Mann-Whitney U or similar and effect sizes.
- **Diagnose seed 456:** initial weight statistics, firing rate trajectory, GAR event counts, quadrant selectivity. [Certain]
- Apply the same seed count to baselines (they already use 3; raise to 10).

### 2.3 Stability metrics defined upfront (S)
Define in Methods, before results:
- Fraction of episodes with at least one seizure reset
- Fraction of episodes with at least one coma rescue
- Firing-rate mean and variance per episode
- Reward variance

Re-report No-GAR on these and **delete the "success rate is misleading" footnote**. [Certain] Say plainly that No-GAR had higher raw success (86.0% vs 71.3%) and was less stable.

### 2.4 Metabolism evidence (S)
- [Likely] By Eq. 4, recovery is about 0.001 per 0.1 ms step and a spike costs 0.01, so a neuron could sustain roughly 1000 Hz. At 8.5 Hz the constraint probably never binds.
- Plot ATP distributions over training. If energy never drops, either retune delta_spike/tau_recovery so it binds, or state clearly that it is inert at these rates.

### 2.5 Sensitivity analysis upgrade (M)
- Multiple seeds (3-5) per setting; error bars on Table V.
- Note that nominal values are best everywhere, which suggests tuning on the test setting. Describe how nominals were chosen.

### 2.6 Fair baselines (M)
- Report tuning budget for every method.
- Include the random-untrained-SNN control (2.1) as a baseline row in Table IV.
- Make the SimpleSNN comparison honest: it differs in neuron model, learning rule, and stabilizers, so state that no mechanism can be attributed.

### 2.7 Generalization (M)
- Procedurally generated obstacle layouts and novel goal positions at evaluation time.
- Report train vs held-out performance.

### 2.8 Selectivity control (S)
- [Likely] The 0.64 selectivity index and 14.2 vs 3.1 Hz may mostly reflect the 25 pA injection. Compute selectivity (a) with plasticity frozen and (b) with an untrained SNN. Only the *difference* is evidence of learning.

---

## Phase 3: Fix errors and inconsistencies (S-M)

| # | Issue | Where | Fix |
|---|-------|-------|-----|
| 3.1 | 8.5 Hz claim vs Fig. 6 annotations (~20-22 Hz baseline, 113 Hz transient) | Abstract, IV-D, Fig. 6 | Reconcile; state what window and population each number refers to |
| 3.2 | Hidden-to-Motor connectivity: 20% vs 15% | III-A vs III-C | Use one value, verify against code |
| 3.3 | Stage 4 "60.7% plateau" vs 70.8% headline | V vs IV-C | Report per-stage success in a table |
| 3.4 | Multi-seed 66.7% vs full-curriculum 70.8% sit in different tables | IV-A, Table IV | Never place side by side without the stage coverage stated |
| 3.5 | Garbled/overlapping text | Figs. 1, 2, 4 | Regenerate as vector graphics with legible fonts |
| 3.6 | Table III header says "100 episodes each" but the text implies a different protocol | Table III, IV-B | State exact episodes, seeds, stage coverage |
| 3.7 | No-GAR success 86.0% vs the text implying failure | Table III | Handled in 2.3 |
| 3.8 | STDP time constants and clipping | Eqs. 5-7 | Confirm the equations match the code, including sign handling for inhibitory synapses |

---

## Phase 4: Efficiency claim (S-M)

[Likely] The 5.8x operation reduction is not valid as stated: the D3QN still runs in the system, so SNN operations add to its MACs rather than replace them. Neuron-state updates (500 neurons x 500 substeps) also appear uncounted.

Choose one:
- **Option A (recommended):** Delete the claim from Abstract, Discussion, and Conclusion.
- **Option B:** Count the full pipeline (D3QN + neuron updates + synaptic events) and compare against the D3QN-only baseline on the same footing.

---

## Phase 5: Reproducibility and release (M)

- [Certain] Put the repository URL in the paper. It is currently absent.
- Versioned release archived on Zenodo with a DOI.
- Pinned dependencies, config files for every reported experiment, seed lists.
- One-command script to reproduce each table and figure.
- Unit tests and CI; a minimal "hello world" example in the README.
- State license, Python version, and Gymnasium version.

---

## Phase 6: Related work and positioning (S-M)

- Add: DishBrain / closed-loop in vitro learning, RL-SNN hybrid literature, existing neuromorphic RL environments.
- Add a comparison table of OrganoidEnv vs Brian2, NEST, BindsNET, SpykeTorch showing which features each has (be exact and cite versions).
- Justify the toy task: explain what the benchmark is *for*, since an MLP reaches 98% on it. [Likely]
- Target a **resource/software venue** (e.g., Frontiers in Neuroinformatics, a benchmarks/datasets track) rather than a methods-claim venue.

---

## Phase 7: Move to Limitations (do not attempt)

Loihi 2 measurements, 10x-100x larger networks, glia/astrocyte modeling, synaptic delay and receptor dynamics, synaptogenesis. [Certain] Claiming these without hardware or compute would hurt credibility.

---

## Suggested order of work

1. Phase 1 (reframing) and Phase 3 (error fixes): quick and removes the most visible weaknesses.
2. Phase 2.3 and 2.4 (stability metrics, metabolism check): cheap, high impact.
3. Phase 2.1, 2.2, 2.8 (ablations, seeds, selectivity control): the long compute jobs. Start these running early.
4. Phase 4 and 5 (efficiency claim, code release).
5. Phase 2.5-2.7 and Phase 6 (sensitivity, baselines, generalization, related work).
6. Final pass: regenerate figures, re-read every number against the results.

## Final pre-submission checklist

- [ ] Every claim in the Abstract is backed by a table or figure
- [ ] No "first," "effectively trains," or "resembles organoid" claims remain unsupported
- [ ] 10-seed full-curriculum results with median/IQR
- [ ] All ablations including plasticity-off and random-SNN control
- [ ] Stability metrics defined before results
- [ ] Efficiency claim deleted or recomputed
- [ ] All inconsistencies in Phase 3 resolved
- [ ] Repository link, DOI, and reproduction script in the paper
- [ ] Figures regenerated and legible
- [ ] Venue chosen to match the resource-paper framing
