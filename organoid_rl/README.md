<p align="center">
  <h1 align="center">🧠 OrganoidEnv</h1>
  <p align="center">
    <em>A Stabilized Reinforcement Learning Environment for Biological Spiking Networks</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Brian2-spiking_sim-orange?logo=data:image/png;base64," alt="Brian2">
    <img src="https://img.shields.io/badge/Gymnasium-RL_env-green?logo=openaigym&logoColor=white" alt="Gymnasium">
    <img src="https://img.shields.io/badge/PyTorch-DQN_agent-red?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  </p>
</p>

---

**OrganoidEnv** is a Gymnasium-compatible reinforcement learning environment that wraps a biologically-realistic spiking neural network built with [Brian2](https://brian2.readthedocs.io/). The "organoid" learns to navigate a 2D space with obstacles by modulating its own neural activity — no gradient descent involved.

## ✨ Key Features

| Feature | Description |
|:--|:--|
| **Metabolic Izhikevich Neurons** | Energy-gated spiking model with ATP recovery dynamics |
| **Sparse Distributed Memory** | Cerebellum-like 256-neuron expansion layer for state encoding |
| **Clustered Motor Output** | Quadrant-based motor neurons (Up/Down/Left/Right) |
| **Dopamine D3 Learning Rule** | Dual-trace eligibility (fast 100ms + slow 2500ms) with reward modulation |
| **God Mode Stabilizer** | Automatic seizure suppression and activity kickstart |
| **Curriculum Learning** | 4-stage difficulty ramp (easy → obstacles → multi-goal → full) |
| **Structural Plasticity** | Synapse pruning for weak connections |
| **Dueling Double DQN** | PER, Noisy Nets, N-step returns, and HER for the outer agent |

## 🏗️ Architecture

```mermaid
graph LR
    subgraph OrganoidEnv
        A["Input (21D Obs)"] --> B["SDM Layer\n256 neurons\n10% sparse"]
        B --> C["Hidden Layer\n144 neurons"]
        C --> D["Motor Layer\n100 neurons"]
        D --> E["🎯 Cursor\nMovement"]
    end
    
    subgraph Learning
        F["RL Agent\n(DQN / Q-Table)"] -->|Action 0-7| A
        E -->|Reward| F
    end
    
    subgraph Stabilizers
        G["God Mode"] -.->|Monitor & Fix| B
        H["Homeostasis"] -.->|E/I Balance| C
        I["Structural\nPlasticity"] -.->|Prune/Grow| C
    end
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/vansh-sharma/organoid-rl.git
cd organoid-rl
pip install -e .
```

Or with Docker:

```bash
docker build -t organoid-rl .
docker run organoid-rl
```

### Run a Quick Test

```bash
# Verify Brian2 works
python tests/sanity_check.py

# Watch the organoid navigate (80 steps)
python tests/diagnostic_cursor.py
```

### Training

```bash
# Month 3: Basic cursor navigation (Q-Learning)
python experiments/month3_navigation/07_month3_training.py

# Month 4: SDM + Obstacles + Ablation study
python experiments/month4_6_advanced/09_month4_training.py
python experiments/month4_6_advanced/10_month4_ablation_study.py

# Month 5: Multi-goal navigation with context switching
python experiments/month4_6_advanced/11_month5_multigoal_training.py

# Month 6: Grand Unification (Dueling DDQN + Curriculum + HER)
python experiments/month4_6_advanced/12_month6_training.py
```

## 📊 Results Summary

| Phase | Agent | Key Milestone | Episodes | Status |
|:--|:--|:--|--:|:---|
| Month 1 | — | Stable 500-neuron spiking network (5-10 Hz) | — | ✅ Done |
| Month 2 | Random | Pavlov conditioning: Bell → Saliva pathway | 50 | ✅ Done |
| Month 3 | Q-Table | First cursor-to-goal navigation | 150 | ✅ Done |
| Month 4 | Q-Table | SDM + Obstacles + Ablation proof | 100 | ✅ Done |
| Month 5 | Q-Table | Multi-goal with context switching | 250 | ✅ Done |
| Month 6 | **Dueling DDQN** | **Grand Unification: 70.8% Success Rate** | **500** | 🚀 **Winner** |

> **Note**: The final Month 6 success was driven by a "Motor Mapping Breakthrough" (direct quadrant stimulation), achieving **97% accuracy** on Stage 2 (Obstacle Navigation).

## 📁 Project Structure

```
organoid-rl/
├── environment/
│   ├── core.py          # OrganoidEnv (Gymnasium wrapper)
│   ├── neurons.py       # Metabolic Izhikevich model
│   ├── stimulator.py    # God Mode stabilizer
│   └── rewards.py       # D3 dopamine learning rule
├── agents/
│   └── dqn_agent.py     # Dueling Double DQN + PER + HER
├── experiments/
│   ├── month1_2_basics/ # Foundation tests
│   ├── month3_navigation/ # Cursor navigation training
│   ├── month4_6_advanced/ # SDM, obstacles, multi-goal, grand unification
│   └── publication/     # Publication figure generator
├── tests/               # Diagnostics and sanity checks
├── docs/                # Project roadmap and architecture planning
├── reports/             # Monthly progress reports
├── paper/               # Publication draft and outline
├── setup.py
├── requirements.txt
├── Dockerfile
└── LICENSE              # MIT
```

## 🔬 How It Works

1. **The Organoid**: 500 Metabolic-Izhikevich neurons connected via STDP synapses with energy-gated firing.
2. **The Task**: Navigate a virtual cursor from `(0.1, 0.9)` to target `(0.8, 0.8)` while avoiding obstacles.
3. **The Agent**: Stimulates specific neuron groups (actions 0-7). The organoid's motor neuron clusters translate spiking activity into cursor movement.
4. **The Learning**: Reward signals modulate synapse weights through eligibility traces — recent pre-synaptic activity gets credit for good outcomes.

## 📄 Citation

```bibtex
@software{sharma2026organoidenv,
  title  = {OrganoidEnv: A Stabilized Reinforcement Learning Environment 
            for Biological Spiking Networks},
  author = {Sharma, Vansh},
  year   = {2026},
  url    = {https://github.com/vansh-sharma/organoid-rl},
}
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
