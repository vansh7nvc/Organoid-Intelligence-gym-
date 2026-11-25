# OrganoidEnv: Title and Framing Guide

**Principle:** The title should contain only words you can prove. The current title promises "biologically constrained SNNs" trained through "sensory-motor mapping," but the evidence shows a D3QN controlling a network whose own learning was never isolated. [Likely]

**Confidence tags:** [Certain] = verifiable from the manuscript; [Likely] = strong inference; [Guessing] = gap-filling.

---

## Title options (choose after Phase 2 results)

| Outcome of ablations | Suggested title | What it claims |
|----------------------|-----------------|----------------|
| SNN contributes nothing measurable | **OrganoidEnv: A Gymnasium Testbed for Closed-Loop RL Control of Metabolic Spiking Networks** | The software only (safest) |
| Plasticity-on beats frozen and random-SNN controls | **OrganoidEnv: A Benchmark Environment for RL-Driven Stimulation of Homeostatically Regulated Spiking Networks** | The testbed plus a measured effect |

**Avoid:** leading with "organoid" unless explained. [Likely] It attracts neuroscience reviewers, who will note nothing in the model is organoid-specific.

**Decision rule:** [Certain] Choose the title last. If the random-SNN control matches the full model, only the first title is honest, and the paper becomes a software contribution.

---

## Framing changes

### Problem statement
Replace "SNN + RL is hard" with a concrete gap: no standard environment exists where an external agent controls a spiking substrate with built-in stability mechanisms, so results across papers cannot be compared. [Likely] This must be backed by the survey/feature table (Brian2, NEST, BindsNET, SpykeTorch, DishBrain-related work).

### Contribution list
Cut to what is verifiable:
1. The open-source environment (with repository link and DOI)
2. Stabilizer ablations with stability metrics defined upfront
3. A 10-seed benchmark with baselines

Add plasticity and metabolism **only if** their ablations show an effect.

### Role of the SNN
[Certain] State plainly that the D3QN performs long-horizon control and the SNN is the plant being controlled. The current text implies the SNN learns the task.

### Seed variance
Present it as a benchmark property: "initialization sensitivity is high, so we report 10 seeds." Drop the organoid-variability analogy.

### Accuracy versus baselines
Frame the gap as the cost of biological constraints. Do not use the 5.8x operation-count claim unless recomputed for the full pipeline.

### Limitations
Lead with the toy task, the small network, and the unproven SNN contribution, rather than burying them.

---

## Recommended structure

1. Introduction (the gap)
2. Related work (with feature comparison table)
3. Environment design
4. Experimental protocol (metrics defined before results)
5. Results: ablations, then baselines, then sensitivity
6. Limitations
7. Release details (repository, DOI, reproduction instructions)

---

## Abstract skeleton (fill in after results)

**If the SNN contributes:** "We present OrganoidEnv, a Gymnasium-compatible testbed in which an external RL agent controls a metabolic-Izhikevich spiking network with homeostatic stabilizers. Across 10 seeds, removing plasticity reduced success from X% to Y% (p = Z), and removing the activity regulator increased seizure-like episodes from A% to B%. We release code, configurations, and a reproduction script."

**If the SNN does not contribute:** "We present OrganoidEnv, a Gymnasium-compatible testbed for studying closed-loop RL control of spiking networks with metabolic and homeostatic constraints. Across 10 seeds, an untrained network performed comparably to the plastic network, indicating that task performance is dominated by the external controller; we report stability metrics for each stabilizer and release the code as a baseline for future learning rules."
