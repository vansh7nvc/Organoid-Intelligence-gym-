# OrganoidEnv: Revision Plan

Based on a review of manuscript.tex, references.bib and manuscript.pdf from the ieee\_tnnls\_package. Actions are tagged KEEP, SOFTEN, DROP, ADD or FIX. Items are numbered so you can tick them off.

## 0. Ground rules

- The framing must follow the data. Do Phase A (existing data) and Phase B (new runs) before finalizing Sections IV to VII. Edits marked \[after data\] depend on outcomes.
- Target framing: a platform/testbed paper with modest, defensible findings, not a paper that proves biology.
- A metric supports a mechanism only if removing that mechanism changes the metric. Otherwise it is descriptive.
- Negative and null results (C0 trailing on return, GAR null, pruning null) are reported plainly. Reviewers respect that more than spin.

## 1. Target framing

**New title (pick one):**

- OrganoidEnv: An Open, Ablatable Spiking Testbed for Studying Metabolic and Homeostatic Constraints in Organoid-Inspired Closed-Loop Reinforcement Learning
- OrganoidEnv: A Gymnasium Testbed for Metabolically Constrained Spiking Control

**One-sentence contribution:** a fast, reproducible, single-flag-ablatable spiking RL environment, plus a characterization of what energy-based fatigue and homeostatic regulation do to its dynamics and control behavior.

**Contributions list (replaces the slogan paragraph in the Introduction):**

1. An open Gymnasium environment with an organoid-inspired E/I network, an abstract energy-fatigue variable, a three-factor plasticity rule, and a sparse sensory encoder.
2. A numerically cross-checked C kernel against Brian2 (with the protocol and speedup baseline stated).
3. A single-flag ablation suite (C1 to C6, plus any added) and baselines with matched protocols.
4. Findings: \[after data\] the analytic rate bound from metabolism, GAR's measured role, plasticity's measured role, and OOD behavior.

## 2. Claim ledger

| # | Claim (where) | Action | How |
| --- | --- | --- | --- |
| 1 | “Science explains the technology / technology proves the science” (Abstract, Intro) | DROP | Delete both occurrences. Replace with “inspired by” language. |
| 2 | Brian2 bit-exact, 2e-12 mV, \~200x (Abstract, Intro, Conclusion, Table I) | SOFTEN | “Numerically cross-checked against Brian2 (max residual 2e-12 mV).” State the protocol and what 200x is measured against. Drop “bit-exact” unless literally true. |
| 3 | Conductance-partitioned neurons (Abstract) | FIX | The model uses delta-jump synapses and Izhikevich arbitrary-unit current. Say that, and remove pA/nS or add a unit-mapping note. |
| 4 | Metabolism/GAR prevent epileptiform runaway | SOFTEN | Metabolism bounds rates (analytic result, see III-A). GAR's contribution is unshown. See A1, B3. |
| 5 | Rates inside 2 to 20 Hz organoid range | SOFTEN | Verify the band against Trujillo 2019 (A6). Say “within the band we adopt (cite)”. |
| 6 | Near-critical avalanches | DROP | sigma = 0.49 is subcritical. Report sigma and alpha descriptively. Note alpha is about 1.3 in every condition. |
| 7 | SDM sparsity verified (Fig. 3) | SOFTEN | Top-26 enforces about 10% by construction. Move to appendix as an implementation check. |
| 8 | Emergent directional selectivity, 3x DSI | SOFTEN or DROP | Contingent on A2. Random policy has DSI 0.678 and C3 0.492, both above C0. |
| 9 | Dual-trace plasticity gives credit assignment (STC mapping) | SOFTEN | Call it a presynaptic dual-trace three-factor rule. Not STDP. STC is only loose inspiration. |
| 10 | Pruning stabilizes / refines | DROP the claim, KEEP the module | C5 shows no effect (p = 0.68). Report as null. |
| 11 | Plasticity–stationarity trade-off | SOFTEN | Replace with the plain result: static SNNs match or beat C0 on return and goal rate. Discuss why. |
| 12 | Metabolism improves efficiency (9.98 vs 2.86 utility per 1000 spikes) | KEEP if defined | Define utility. Check it is not an artifact of negative return. |
| 13 | Zero-shot OOD robustness (T4) | KEEP conditionally | Needs more episodes and matched baseline action granularity (B4, B5). |
| 14 | Platform outperforms prior testbeds (Table I) | SOFTEN | Compare to named frameworks or relabel as in-house baselines. |
| 15 | Seeds: 10-seed training (Abstract) | FIX | State n per condition explicitly, then equalize (B1). |

## 3. Phase A: diagnostics on existing data (no new training)

- **A1. GAR threshold audit.** Confirm what N\_recent counts. If it is the population spike count over 100 ms across 500 neurons, the seizure clamp (>500) equals a 10 Hz mean rate, which is below the 12 to 14.5 Hz phase-space results and 10x below the >100 Hz epileptiform definition. Either fix the threshold, fix the description, or both. Log how often each GAR branch (NaN reset, coma kick, seizure clamp) fires per condition and report it.
- **A2. DSI confound.** Recompute DSI with stimulation matched across conditions, or using motor-cluster responses to a fixed probe stimulus set. Check whether stimulation itself imposes preferred-quadrant firing. Explain why the random policy has the highest DSI.
- **A3. Return decomposition.** Break return into goal bonus, step penalty, collision penalty, shaping terms (and any spike-cost term). Explain why C3 has 0% goal rate but better return than C0, and why C0 has a higher goal rate than B1 but far lower return.
- **A4. Metric definitions.** Write exact formulas for return, goal rate, utility per 1000 spikes, DSI, sigma\_MR, alpha, Fano factor, and the binning and subsampling settings.
- **A5. Seed audit.** Confirm actual n per condition and whether the 5 ablation seeds are a subset of the 10 C0 seeds.
- **A6. Organoid band check.** Verify what Trujillo 2019 reports. Replace the “2 to 20 Hz wet-lab range” sentence if the paper does not support it as stated.
- **A7. Baseline action mapping.** Document how SkeletonEnv step sizes differ from SNN displacement 0.05\*(N\_right − N\_left). This is the precondition for B4.
- **A8. Brian2 cross-check.** Re-run and save the residual measurement and the timing comparison (Brian2 runtime target, hardware, run length). You need this as a table or appendix item.

## 4. Phase B: new experiments

- **B1. Equal seeds.** 10 seeds for every condition (C0 to C6, B1 to B5): about 55 extra runs. Go to 15 to 20 for conditions you make claims about. Report per-seed points, not just IQM.
- **B2. Matched single-trace ablation.** Add C6b: single trace with tau = 100 ms, and C6c: tau = 2500 ms, with the same weighting as C0. The current C6 changes tau, trace count and weighting at once.
- **B3. GAR-only condition (C7).** GAR on, metabolism off. The phase space currently shows neither, metabolism-only and both, so GAR's own effect is never isolated.
- **B4. Action-matched baselines.** Rerun B1, B2, B4 with action granularity equal to the SNN's effective displacement. Decide whether the T4 advantage survives.
- **B5. OOD scale-up.** At least 50 to 100 evaluation episodes per topology (spread across seeds), exploration off. Current values are multiples of 6.67%, so n is about 15.
- **B6. Plasticity attribution.** Compare C0 against C1 (frozen) and C2 (random) with the larger seed count. This tests whether SNN plasticity contributes beyond the controller.
- **B7. Sensitivity sweep.** Vary Delta\_E, tau\_rec, and GAR thresholds. Report as one table or figure. This shows the results are not tuned to a single point.
- **B8. Optional calibration.** Compare firing-rate distributions, burst statistics or avalanche statistics against a published organoid MEA dataset. If you can’t, say “organoid-inspired” everywhere.

## 5. Decision branches \[after data\]

| Outcome | Then |
| --- | --- |
| DSI fix still shows C0 above C1/C2 | Keep “learned selectivity” as a finding with CIs and a test. |
| DSI fix removes the gap | Drop emergent-selectivity claim. Report DSI as descriptive. |
| Matched-action baselines still collapse on T4 | OOD becomes a headline result. |
| Matched-action baselines recover on T4 | Report that the advantage was action granularity. |
| GAR-only adds nothing beyond metabolism | Report GAR as a safety mechanism with a null effect in tested regimes. |
| C0 still trails C2 on return with more seeds | State plainly. Frame constraints as buying rate bounds and per-spike efficiency, only if those metrics support it. |

## 6. Section-by-section edit list

### Title, authors, keywords

- FIX title per Section 1. Remove “Cortical Organoid Networks” as a literal claim unless B8 is done.
- FIX author block: add affiliations, emails and IEEE membership if applicable. Replace the \\thanks line (“authors of the platform”) with funding, received date, or delete it.
- Keywords: replace “dual-trace STDP” with “three-factor plasticity”. Add “benchmark”, “Gymnasium”. Keep “organoid intelligence”, “spiking neural networks”, “homeostatic plasticity”, “reinforcement learning”.

### Abstract

- Target 150 to 250 words, no parameter dump, no citation-style tokens.
- Structure: context (closed-loop organoid computing), gap (no open, ablatable in silico testbed with metabolic and homeostatic constraints), contribution (OrganoidEnv), findings, availability.
- Draft skeleton: “Closed-loop organoid-computing experiments are slow and hard to observe. We present OrganoidEnv, an open Gymnasium environment of an organoid-inspired spiking network (500 Izhikevich neurons, 80/20 E/I) with an energy-fatigue variable, a global activity regulator, a presynaptic dual-trace three-factor rule, and a sparse sensory encoder, with a numerically cross-checked C kernel. Across \[N\] seeds, a \[4x5\] stimulation–inhibition sweep, \[N\] ablations and \[N\] baselines, we find that \[after data: metabolic fatigue bounds firing to \~10 to 20 Hz by construction; GAR adds / does not add X; plasticity adds / does not add Y; OOD result on T4 with n episodes\]. We report limitations, including \[C0 trailing on return\]. Code and seeds are available at \[DOI\].”

### I. Introduction

- KEEP paragraph 1 (wet-lab limits and the case for simulation).
- FIX paragraph 2: Beggs & Plenz 2003 does not support epileptiform hypersynchrony. Find a proper citation for each of the two pathologies.
- SOFTEN the five-item mapping list: change “↔” to “is inspired by”. Item 4 (pruning) currently has no citation. Add one or drop the mapping.
- DROP the paragraph “In this unified paper, the science explains the technology...”. Replace with the contributions list from Section 1 and a one-paragraph paper roadmap.
- FIX the Brian2 sentence: “numerically cross-checked”, not “validated”.

### II. Related Work

- ADD subsections: (a) biological closed-loop systems (DishBrain, Brainoware, OI); (b) SNN and neuromorphic RL libraries and benchmarks (verify and cite BindsNET, Norse, snnTorch, SpikingJelly, Brian2/NEST-based work); (c) reward-modulated plasticity and eligibility traces (Izhikevich 2007, Florian 2007, Legenstein 2008, Fremaux & Gerstner 2016, Bellec 2020); (d) energy-constrained and homeostatic spiking models.
- DROP “typically rely on non-spiking surrogate gradients... lack explicit ATP dynamics” unless cited.
- Table I: either replace rows with named frameworks (verify each cell from their docs and cite) or relabel it “Design-space comparison with in-house baselines”. Define “12-Key”. Remove “Rarely/Sometimes/Unverified” cells that cannot be sourced.

### Table II (constants)

- FIX caption: “Network and simulation parameters”. Drop “verified against codebase”.
- FIX the “Synaptic Weight Bounds” row (the symbol column shows a function name).
- ADD: w\_bg, Poisson rate, eta\_DA, weight-decay tau, GAR thresholds, GAR actions, homeostatic E/I target 1.5 and rate 0.1, k = 26, stimulation amplitude, pruning threshold and period.

### III intro and Fig. 1

- KEEP Fig. 1 and the controller description (D3QN, NoisyNet, PER, HER, 3-step).
- ADD one sentence: the controller selects actions and the SNN converts stimulation into displacement. Say explicitly where learning happens (controller, SNN weights, or both).
- ADD citations for D3QN components (dueling, double DQN, NoisyNet, PER, PPO, SB3, Gymnasium, Neo/Elephant).

### III-A Neuron dynamics

- FIX E range: spikes occur only when E > 0 and cost 0.1, so E is bounded below by about −0.1, not −infinity. Correct the stated range.
- ADD the synapse equation (delta jumps to v or I?) and state units are Izhikevich dimensionless.
- SOFTEN “mitochondrial ATP” to “an abstract energy-fatigue variable inspired by ATP-sensitive potassium channel dynamics”.
- ADD a proposition: with Delta\_E = 0.1, steady-state E = 1 − Delta\_E · f · tau\_rec, so sustained rates above 1/(Delta\_E · tau\_rec) are not possible: about 10 Hz for excitatory (tau = 1000 ms) and 20 Hz for inhibitory (tau = 500 ms). This turns the circularity into an analytic result and explains the 12 to 14 Hz plateau. Verify against simulation.
- FIX the Ashwood2014KATP bib key (authors are Tanner et al. 2011).

### III-B Compartments and Dale

- DROP “strictly”. State the counts plainly: 256 input RS, 44 hidden RS, 100 FS, 100 motor RS.
- ADD one sentence on the bottleneck: 256 inputs project onto only 44 hidden excitatory units. Discuss in limitations.
- FIX wording “conductance-partitioned” (see claim 3).

### III-C GAR

- ADD the definition of N\_recent and the per-neuron equivalent of each threshold.
- ADD GAR branch trigger frequencies per condition (A1).
- SOFTEN language: “regulator that intervenes when population activity leaves a band”. Do not call it a model of criticality control.
- Keep the inter-episode E/I homeostatic scaling, but state whether it is active in C0 and in C4.

### III-D Plasticity and pruning

- RENAME: “presynaptic dual-trace three-factor rule”. State clearly it is not STDP (no postsynaptic term).
- ADD the exact reward function R\_t, its range, and when it is applied.
- SOFTEN the STC claim: “loosely inspired by”, with Frey & Morris 1997.
- FIX “millisecond spike timing” (tau\_1 = 100 ms is not millisecond timing).
- KEEP pruning as an implementation detail. State pruning period and threshold. No mechanistic claim.

### III-E Sensory encoder and motor readout

- RENAME “SDM” to “sparse expansion encoder (top-k)”. Mention Kanerva as inspiration only. Add sparse-expansion literature as the proper reference.
- ADD environment spec subsection (can live in Section IV): arena size, obstacle geometry for every stage and topology, goal radii, start distribution, max steps, termination, exact observation vector (15-D).
- ADD the exact motor protocol: what stimulates which quadrant, and how displacement is read. Needed for the DSI fix.

### IV Experimental design

- ADD n per condition table; equalize per B1.
- ADD evaluation protocol: training-time metrics versus separate evaluation episodes with exploration noise off. State which is reported.
- ADD metric definitions (A4).
- KEEP rliable IQM, BH-FDR, rank-biserial, 5000-sample bootstrap. ADD per-seed plots and pre-specified primary comparisons (for example C0 vs C1, C2, C3, C4, C6b, and C0 vs B1 to B4).
- ADD the matched-baseline protocol (B4) and sensitivity sweep (B7).

### V-A Biophysics

- KEEP Fig. 2 (raster, rate, E(t)). Fix the band sentence per A6.
- MOVE Fig. 3 (SDM occupancy) to the appendix with the by-construction note.
- REWRITE the avalanche paragraph: report sigma\_MR and alpha for all conditions, note sigma is subcritical, alpha is about 1.3 in all conditions including untrained ones, and fit tests are \[cite Clauset 2009 and report them or drop alpha\].
- KEEP the C3 Fano result (135 vs 19, about 7x) as a descriptive finding about metabolism.

### V-B Phase space

- REPORT four conditions: neither, metabolism-only, GAR-only (new), both.
- FIX the inconsistency between 12.3 to 14.4 Hz (text) and 12.5 to 14.5 Hz (caption).
- Say explicitly that the metabolism-only result already lies within 2 to 20 Hz, and state what GAR adds \[after data\].

### V-C Directional selectivity and plasticity \[after data\]

- Contingent on A2 and B6 (see Section 5). Include the full DSI list: C0, C1 to C6, B3, B5, not a selected subset.
- Fig. 8: KEEP the kernel panel. For the weight spectrum, add the C0 vs C5 test or label it descriptive.

### V-D Learning, ablation, efficiency, OOD

- REWRITE finding 1: report that C3 improves return (p = 0.035) but collapses goal rate, and explain with A3.
- REWRITE finding 2: report that C2 and C1 match or beat C0, and that B2, B4 beat C0 in-distribution.
- REWRITE finding 3 per B4 and B5. Report n, CIs and unadjusted and adjusted tests.
- Table III: KEEP (it is honest). ADD n, mark non-significant results, fix caption wording.
- Figures 9 to 12: check axes, units, legends, color-blind-safe palette, per-seed points, consistent condition colors, captions that stand alone. Confirm Fig. 9 shows C0 learning curves actually improving.

### VI Discussion and limitations

- ADD a short “What the results show” paragraph with the findings that survive.
- ADD threats to validity: no organoid calibration (or state B8), 44-unit hidden layer, controller-side credit assignment, 100 ms control step, single 500-neuron size, small seeds.
- ADD negative results explicitly (GAR, pruning, C0 return).
- KEEP the limitation on conduction delays and the hardware-in-the-loop future work.

### VII Conclusion

- REWRITE to match the findings. Remove “verify how metabolic limits, multi-timescale traces and homeostatic gain control cooperate”. Say what is bounded, what is not demonstrated, and what is released.

### Appendices

- A (hyperparameters): KEEP and extend with SNN and GAR parameters, environment spec, PPO and D3QN details.
- B (criticality table): KEEP with a note that sigma is subcritical and alpha is non-discriminating.
- C (OOD table): ADD n episodes per cell and CIs.
- ADD: Brian2 cross-check protocol (A8), sensitivity sweep (B7), Fig. 3.
- DELETE the %%BEGIN/END%% marker comments.

## 7. Bibliography

- FIX: add volume, pages, DOI, publisher (Kanerva missing publisher). Rename the Ashwood2014KATP key.
- CITE Clauset2009PowerLaw where alpha is fitted, or delete it if unused.
- ADD (verify each reference before adding): Izhikevich 2007, Florian 2007, Legenstein 2008, Fremaux & Gerstner 2016, Bellec 2020, dueling and double DQN, NoisyNet, PER, PPO, Stable-Baselines3, Gymnasium, Neo/Elephant, BindsNET, Norse, snnTorch, SpikingJelly, a sparse-expansion coding reference.
- CHECK each claim-to-citation pairing (Laughlin 1998 for “ATP exhaustion kinetics” is loose; Beggs & Plenz for epileptiform is wrong).

## 8. Formatting and submission hygiene

- Abstract 150 to 250 words. Consistent notation (Hz, ms, pA only if units are real).
- Check the current TNNLS author guidelines for page limit, required sections and biography requirements.
- Figures: embedded fonts at readable size, vector PDF, consistent style. Keep SVG sources out of the submitted package unless asked.
- Do not ship IEEEtran.cls unless required. Compile cleanly with no overfull boxes.
- ADD a code and data availability statement with a versioned DOI (for example a Zenodo archive of the repo, seeds and configs).
- Reproducibility checklist: seeds, versions (Brian2, torch, SB3, Python), hardware, runtime per condition.

## 9. Suggested order of work

1. Phase A diagnostics (A1 to A8). A1 and A2 first: they can change the story.
2. Fix any bugs found, then freeze the code and configs.
3. Phase B runs, once, on the frozen setup (B1 to B7).
4. Decide branches (Section 5).
5. Rewrite Sections III, IV, V, VI, VII, abstract, intro, related work.
6. Bibliography and formatting pass.
7. Internal review: someone outside the project reads it cold and lists what claim each figure supports.

## 10. Final pre-submission checklist

- Every claim in the abstract appears in a table or figure that supports it.
- No “prove”, “strictly”, “bit-exact”, “near-critical”, “emergent” without direct evidence.
- n per condition is stated and equal, and per-seed data is shown.
- Every metric has a formula. Every baseline has a matched protocol.
- Every figure and table is cited in the text and readable alone.
- All references are complete and verified. Code and data are archived with a DOI.
