# 🧠 OrganoidEnv: A Stabilized Reinforcement Learning Environment for Biological Spiking Networks

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Brian2](https://img.shields.io/badge/Brian2-spiking_sim-orange)](https://brian2.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_env-green)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DQN_agent-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OrganoidEnv** is an open-source, Gymnasium-compatible neuromorphic testbed that simulates an *in silico* brain organoid. It combines **metabolic-Izhikevich spiking neurons**, **homeostatic stabilizers (Global Activity Regulator)**, and **dopamine-modulated dual-trace plasticity** in a reproducible Python package.

---

## 📁 Repository & Submission Organization

```
Organoid Intelligence/
├── submissions/
│   ├── frontiers_in_neuroinformatics/       # Primary Journal Manuscript
│   │   ├── organoidenv_manuscript.tex       # Complete LaTeX source (empirical multi-seed & ablations)
│   │   ├── response_to_reviewers.md         # Detailed response to peer reviews
│   │   └── figures/                         # Camera-ready vector PDF & 300 DPI PNG figures
│   │       ├── fig1_multiseed_progression.pdf
│   │       ├── fig2_ablation_comparison.pdf
│   │       └── ...
│   │
│   └── joss/                                # Journal of Open Source Software
│       ├── paper.md                         # JOSS submission draft
│       ├── paper.bib                        # BibTeX references
│       └── CITATION.cff                     # Citation metadata
│
├── organoid_rl/                             # Python Package Source Code
│   ├── environment/                         # Core SNN & Gymnasium Environment
│   │   ├── core.py                          # OrganoidEnv implementation
│   │   ├── neurons.py                       # Metabolic Izhikevich neuron model
│   │   ├── stimulator.py                    # Global Activity Regulator (GAR)
│   │   └── rewards.py                       # Dopamine D3 Dual-Trace STDP rule
│   ├── agents/                              # RL Optimization
│   │   └── dqn_agent.py                     # Dueling Double DQN (PER + NoisyNet + HER)
│   └── experiments/                         # Training & Validation Pipelines
│       ├── publication/                     # Multi-seed & ablation scripts
│       └── results/                         # Empirical CSV logs & trained weights
│
├── notebooks/
│   └── colab_gpu_training.ipynb             # 1-Click Google Colab Training & Reproduction
│
├── requirements.txt                         # Dependency requirements
└── setup.py                                 # Package setup
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/vansh7nvc/Organoid-Intelligence-gym-.git
cd Organoid-Intelligence-gym-
pip install -e .
```

### 2. Basic Usage
```python
import gymnasium as gym
from organoid_rl.environment.core import OrganoidEnv

# Initialize the biologically-constrained SNN environment
env = OrganoidEnv(
    use_sdm=True, 
    use_morphology=True, 
    use_dual_trace=True, 
    use_stabilizer=True
)

obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Sensory stimulation pattern (0-7)
    next_obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 3. Run Multi-Seed Training in 1-Click (Colab)
Open [`notebooks/colab_gpu_training.ipynb`](notebooks/colab_gpu_training.ipynb) on Google Colab and click **Runtime → Run All**.

---

## 🔬 Scientific Highlights & Benchmarks

| Configuration | Mean Reward | Success Rate (%) | Dynamical Behavior |
| :--- | :---: | :---: | :--- |
| **OrganoidEnv (Full)** | $\mathbf{+51.00 \pm 24.89}$ | $\mathbf{71.3\%}$ | **Stable, sustained learning** ($N=3$ seeds; top seeds: $95\%$ & $94\%$). |
| **No GAR (Homeostasis)** | $+23.14 \pm 93.70$ | $86.0\%$ | **Severe dynamical instability ($\sigma = \pm 93.7$)**; periodic seizure dropouts. |
| **No SDM (Sparse Mem.)** | $-158.31 \pm 51.28$ | $3.7\%$ | **Complete sensory collapse** due to spatial aliasing in continuous coordinates. |

---

## 📜 Citation

If you use OrganoidEnv in your research, please cite:

```bibtex
@article{Sharma2026OrganoidEnv,
  author    = {Sharma, Vansh and Malik, Seema},
  title     = {OrganoidEnv: A Stabilized Reinforcement Learning Environment for Training Biologically Constrained Spiking Neural Networks via Sensory--Motor Mapping},
  journal   = {Frontiers in Neuroinformatics},
  year      = {2026}
}
```

## 📄 License
MIT License. Free for academic and commercial research use.
