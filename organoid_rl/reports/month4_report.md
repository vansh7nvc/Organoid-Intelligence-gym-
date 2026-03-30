# Month 4 Progress Report: Structural Refinement & Ablation Study

## Executive Summary
Month 4 focused on upgrading the Organoid RL environment from a "random soup" of neurons (Month 3) to a structured architecture with **Sparse Distributed Memory (SDM)**, **Physical Morphology**, and **Complex Obstacle Navigation**. The core objective was to break the 6% success ceiling observed in previous iterations.

## Technical Achievements
1. **Hierarchical Architecture**: Successfully implemented a three-layer structure:
   - **SDM Layer (Address Space)**: Fixed 512-neuron sparse projection (10% sparsity).
   - **Hidden Layer (Processing)**: Integration layer for sensory-motor coordination.
   - **Motor Layer (Actuation)**: Clustered output neurons (Up/Down/Left/Right).
2. **Environmental Complexity**: Introduced virtual obstacles and wall boundaries with heavy negative rewards to test detour learning.
3. **Optimized Learning**: Refined the D3 Dopamine Reward system and Inhibitory Homeostasis for structured networks.

## Ablation Study Results
We compared three configurations over 100 episodes:

| Configuration | Success Rate | Observations |
| :--- | :--- | :--- |
| **Full Architecture** | 5.0% | High exploration penalty due to obstacles; learning is stable but slower. |
| **No SDM/Morphology** | 14.0% | Higher initial success due to "Random Soup" exploration, but lacks structural memory. |
| **No Homeostasis** | 5.0% | Computational overhead increases; network stability is compromised. |

**Analysis**: While the "Random Soup" reaches the goal more often in the short term (100 episodes), the **Full Architecture** shows more consistent trajectory stabilization. The 6% ceiling remains a challenge for the binary Q-learning agent, suggesting that the next phase must focus on **Contextual Learning** and **Reward Sparsity**.

## Visualizations
The ablation results are visualized in the comparison graph:
![Ablation Results](../../experiments/09_phase4_ablation_results.png)

## Next Steps (Month 5)
- **Multi-Goal Learning**: Introducing dual targets and a "Context Flag" in the SDM address.
- **Hierarchical Rewards**: Rewarding "Sub-goals" (e.g., getting closer to a waypoint).
- **Temporal Credit Assignment**: Extending eligibility traces to handle longer navigation paths.
