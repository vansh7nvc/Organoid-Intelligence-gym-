# Month 5 Progress Report: Multi-Goal Navigation

## Executive Summary
Month 5 introduced **multi-goal navigation with context switching** — the organoid must reach one of two targets `(0.8, 0.8)` or `(0.2, 0.2)` depending on a context signal embedded in the observation. This tests the network's capacity for conditional behavior.

## Technical Achievements
1. **Context-Aware Q-Learning**: Extended the Q-table agent with a 3D state representation `(distance_bin, angle_bin, context)`, enabling separate policies for each target.
2. **Dual-Target Environment**: The `OrganoidEnv` now randomly selects between two targets on each episode, with the context signal passed through the observation vector (dim 14).
3. **Extended Episodes**: Increased steps per episode to 100 (from 80) for longer exploration paths.

## Training Results
| Metric | Value |
|:--|:--|
| **Episodes** | 200 |
| **Goals Reached** | 0 / 200 (0.0%) |
| **Context 0 (0.8, 0.8)** | 0 |
| **Context 1 (0.2, 0.2)** | 0 |
| **Training Duration** | ~66 minutes |

## Analysis
The Q-Table agent was unable to reliably reach either target within the expanded environment. Two factors contributed:
- **State Space Explosion**: The 3D discretized state space is too coarse for the 21D observation, causing aliasing.
- **Sparse Rewards**: With a goal radius of 0.15, the shaping signal alone is insufficient for the tabular agent to discover reliable paths.

This directly motivated the switch to a **neural network-based agent** (Dueling DQN) in Month 6, which can handle continuous observations natively.

## Visualizations
Training reward curve:
![Month 5 Results](../../experiments/results/month5_results.png)

## Next Steps (Month 6)
- Replace Q-Table with Dueling Double DQN + Prioritized Experience Replay.
- Add Curriculum Learning with 4 difficulty stages.
- Implement Hindsight Experience Replay (HER) for sparse reward recovery.
- Introduce Structural Plasticity (synapse pruning).
