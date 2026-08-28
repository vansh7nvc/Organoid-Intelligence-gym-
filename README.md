<p align="center">
  <h1 align="center">🧠 OrganoidEnv</h1>
  <p align="center">
    <strong>A Stabilized Gymnasium Testbed for Training Biologically Constrained Spiking Neural Networks via Reinforcement Learning</strong>
  </p>
  <p align="center">
    <a href="https://github.com/vansh7nvc/Organoid-Intelligence-gym-/actions/workflows/ci.yml"><img src="https://github.com/vansh7nvc/Organoid-Intelligence-gym-/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
    <a href="https://colab.research.google.com/github/vansh7nvc/Organoid-Intelligence-gym-/blob/main/notebooks/colab_gpu_training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://brian2.readthedocs.io/"><img src="https://img.shields.io/badge/Brian2-v2.8%2B-FF6F00" alt="Brian2"></a>
    <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-v0.29%2B-008080" alt="Gymnasium"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
    <a href="https://joss.theoj.org"><img src="https://img.shields.io/badge/JOSS-Submitted-blue" alt="JOSS Submission"></a>
  </p>
</p>

---

## 📌 Executive Summary

**OrganoidEnv** is an open-source, Gymnasium-compatible neuromorphic computing testbed designed to simulate, control, and benchmark *in silico* neural organoids. 

While modern deep reinforcement learning (RL) achieves remarkable artificial game scores through mathematical backpropagation, biological neural tissue operates under strict physical constraints: **non-differentiable spiking**, **metabolic fatigue (ATP exhaustion)**, **local synaptic plasticity**, and **vulnerability to seizure or coma states**. 

`OrganoidEnv` bridges this divide. It provides an embodied 2D closed-loop navigation ecosystem where an external RL optimization agent (acting as an "automated experimentalist") guides a 500-neuron recurrent Izhikevich spiking network via sensory–motor electrical stimulation.

<p align="center">
  <img src="submissions/frontiers_in_neuroinformatics/figures/fig1_multiseed_progression.png" width="85%" alt="Multi-Seed Learning Progression">
</p>

---

## ✨ Key Architectural Pillars

```
+----------------------------------------------------------------------------------------------------+
|                                      ORGANOIDENV ARCHITECTURE                                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [21D Spatial Obs] ---> [Sparse Distributed Memory] ---> [Recurrent Mesh] ---> [Motor Quadrants]   |
|   (Cursor, Target,        (256 Neurons, Non-Linear         (500 RS/FS Neurons,   (Left, Right,     |
|    Proximity, Context)     Cerebellar Expansion)           Metabolic ATP E)       Up, Down Firing) |
|                                                                                         |          |
|                                                                                         v          |
|  [Dopamine D3 Plasticity] <--- [Reward Engine] <-------------------------------- [Cursor Action]   |
|   (Fast 100ms + Slow 2.5s       (Curriculum Distance +                             (2D Kinematics, |
|    Dual-Trace STDP)              Obstacle Avoidance)                                Obstacles)     |
|                                                                                                    |
|  ============================== [GLOBAL ACTIVITY REGULATOR (GAR)] ==============================  |
|  * Autonomous Seizure Suppression: Injects homeostatic IPSCs if firing > 35 Hz                     |
|  * Autonomous Silence Kickstart: Injects stochastic current if firing < 1.0 Hz                     |
+----------------------------------------------------------------------------------------------------+
```

### 1. Metabolic-Izhikevich Neurons
Neurons incorporate a dynamic **ATP-like metabolic energy variable ($E$)**:
$$\frac{dv}{dt} = \frac{0.04v^2 + 5v + 140 - u + I}{ms}, \quad \frac{dE}{dt} = \frac{1 - E}{\tau_{\text{recovery}}}$$
Spiking requires $v \ge 30\,\text{mV}$ **and** $E > 0$. Every action potential expends a discrete quantum of metabolic energy ($\Delta E = -0.1$), naturally penalizing continuous seizure-like firing and enforcing sparse, energy-efficient biological dynamics ($\approx 8.5\,\text{Hz}$).

### 2. Cerebellar Sparse Distributed Memory (SDM)
Direct low-dimensional stimulation of recurrent spiking meshes causes catastrophic spatial aliasing. OrganoidEnv projects continuous coordinates into a **256-neuron high-dimensional non-linear expansion layer** (10% sparse activation), enabling the SNN to resolve complex spatial topologies.

### 3. Global Activity Regulator (GAR)
Biological neural cultures spontaneously collapse into non-functional states without homeostatic regulation. GAR continuously monitors network firing:
- **Paroxysmal Seizure Prevention:** When population firing exceeds $35\,\text{Hz}$, GAR dynamically scales inhibitory synaptic gains.
- **Hypoactive Coma Kickstart:** When population firing drops below $1.0\,\text{Hz}$, GAR delivers Poisson background stimulation to restore self-sustained dynamics.

### 4. Dopamine-Modulated Dual-Trace STDP ($D^3$)
Bridges millisecond-scale spike-timing-dependent plasticity (STDP) with second-scale reinforcement signals:
$$\Delta w = \eta \cdot R \cdot \left(\text{Trace}_{\text{fast}} + 0.5 \cdot \text{Trace}_{\text{slow}}\right)$$
where $\tau_{\text{fast}} = 100\,\text{ms}$ captures immediate causal dynamics and $\tau_{\text{slow}} = 2{,}500\,\text{ms}$ enables long-horizon credit assignment for delayed navigation goals.

---

## 📊 Empirical Benchmarks & Statistical Results

OrganoidEnv was benchmarked across **$N=3$ independent biological initializations** (Seeds 42, 123, and 456; 200 episodes each) and systematic multi-seed component ablations (100 episodes each).

### 1. Multi-Seed Training Progression
| Metric | Seed 123 | Seed 42 | Seed 456 | **Mean $\pm$ Std** |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 Success (Basic Nav, Ep 0–100)** | **$95.0\%$** | **$94.0\%$** | $25.0\%$ | **$71.3\% \pm 40.1\%$** |
| **Stage 2 Success (Obstacles, Ep 100–200)** | **$89.0\%$** | **$79.0\%$** | $18.0\%$ | **$62.0\% \pm 38.4\%$** |
| **Cumulative Goals Reached (/200)** | **$184 / 200$ ($92.0\%$)** | **$173 / 200$ ($86.5\%$)** | $43 / 200$ ($21.5\%$) | **$66.7\% \pm 39.2\%$** |

<p align="center">
  <img src="submissions/frontiers_in_neuroinformatics/figures/fig2_ablation_comparison.png" width="85%" alt="Ablation Benchmark Comparison">
</p>

### 2. Multi-Seed Component Ablations
| Configuration | Mean Episode Reward | Task Success (%) | Empirical Mechanism & Dynamical Effect |
| :--- | :---: | :---: | :--- |
| **OrganoidEnv (Full Model)** | $\mathbf{+51.00 \pm 24.89}$ | $\mathbf{71.3\%}$ | **Stable, sustained learning** across random biological seeds. |
| **No GAR (No Homeostasis)** | $+23.14 \pm 93.70$ | $86.0\%$ | **Severe dynamical instability ($\sigma = \pm 93.7$)**; periodic seizure/coma dropouts after Episode 50. |
| **No SDM (No Sparse Memory)** | $-158.31 \pm 51.28$ | $3.7\%$ | **Complete sensory collapse**; unable to decode continuous spatial coordinates. |

---

## ⚡ Energy & Neuromorphic Efficiency

| Metric | Traditional Deep RL (ANN) | OrganoidEnv on CPU | OrganoidEnv on Neuromorphic (Loihi 2 estimate) |
| :--- | :---: | :---: | :---: |
| **Average Firing Rate** | Continuous | $8.5\,\text{Hz}$ (Sparse) | $8.5\,\text{Hz}$ (Sparse Event-Driven) |
| **SOPs per Step** | $\approx 10^6$ MACs | $\approx 4.2 \times 10^3$ SOPs | $\approx 4.2 \times 10^3$ SOPs |
| **Theoretical Energy / Step** | $\approx 1.2\,\mu\text{J}$ | CPU simulation overhead | **$\approx 6.3\,\text{nJ}$ ($>190\times$ efficiency gain)** |

---

## 📂 Repository Structure

```
Organoid-Intelligence-gym-/
├── .github/
│   ├── ISSUE_TEMPLATE/                      # GitHub issue templates (Bug / Feature)
│   ├── workflows/
│   │   ├── ci.yml                           # Automated multi-OS/Python CI workflow
│   │   └── paper.yml                        # JOSS paper build verification workflow
│   └── pull_request_template.md             # Pull request checklist template
│
├── docs/                                    # Architectural diagrams, roadmaps, and review records
│   ├── architecture_flow.mmd                # Architecture Mermaid source
│   ├── architecture_flow.tex                # TikZ architecture source
│   ├── OrganoidRL_Roadmap.pdf               # Project development roadmap
│   ├── OrganoidEnv_Revision_Checklist.docx  # Revision checklist
│   ├── drafts/                              # Historical draft archive
│   └── reviews/                             # Reviewer comments & logs
│
├── examples/                                # Standalone research examples
│   ├── 01_quickstart.py                     # Minimal environment interaction
│   ├── 02_evaluate_pretrained_agent.py      # Load checkpoint & evaluate policy
│   └── 03_run_ablation_comparison.py        # Compare full model vs No-GAR vs No-SDM
│
├── notebooks/
│   └── colab_gpu_training.ipynb             # 1-Click Google Colab Training & Reproduction
│
├── organoid_rl/                             # Core Python Package
│   ├── __init__.py                          # Exports: OrganoidEnv, DQNAgent, __version__
│   ├── environment/
│   │   ├── core.py                          # Main OrganoidEnv class (Gymnasium API)
│   │   ├── neurons.py                       # Metabolic-Izhikevich equations
│   │   ├── rewards.py                       # Dual-Trace STDP engine
│   │   ├── stimulator.py                    # Global Activity Regulator (GAR)
│   │   └── baseline_envs.py                 # Gymnasium baseline wrappers
│   ├── agents/
│   │   ├── dqn_agent.py                     # Dueling Double DQN (PER + NoisyNet + HER)
│   │   ├── baseline_agent.py                # MLPBaselineAgent
│   │   └── sb3_baselines.py                 # Stable-Baselines3 baselines
│   ├── experiments/                         # Experiment Suite & Pipelines
│   │   ├── month1_2_basics/
│   │   ├── month3_navigation/
│   │   ├── month4_6_advanced/
│   │   ├── publication/                     # Multi-seed & ablation evaluation scripts
│   │   └── results/                         # Data logs, checkpoints (.pth), figures
│   └── tests/                               # Test Suite
│       ├── test_environment.py              # Standard unittest / pytest suite
│       ├── sanity_check.py                  # Brian2 & C++ backend sanity check
│       ├── quick_test.py                    # Simulation step test
│       ├── test_metabolic.py                # Metabolic dynamics verification
│       ├── diagnostic.py                    # SNN diagnostic
│       ├── diagnostic_cursor.py             # Visual cursor simulation
│       └── benchmark_speed.py               # Simulation FPS benchmark
│
├── reports/                                 # Milestone Development Reports (Weeks 1-4, Months 1-6)
│
├── submissions/                             # Academic Manuscripts
│   ├── frontiers_in_neuroinformatics/       # Primary Journal Submission
│   │   ├── organoidenv_manuscript.tex       # Full LaTeX source
│   │   ├── organoidenv_supplementary.tex    # Supplementary material
│   │   ├── response_to_reviewers.md         # Rebuttal & revision notes
│   │   └── figures/                         # Vector PDF & 300 DPI PNG figures
│   └── joss/                                # Journal of Open Source Software (JOSS)
│       ├── paper.md                         # JOSS submission manuscript
│       ├── paper.bib                        # Complete BibTeX bibliography with DOIs
│       └── CITATION.cff                     # Citation metadata
│
├── .gitignore                               # Production gitignore
├── CHANGELOG.md                             # Version history & release notes
├── CITATION.cff                             # GitHub repository citation metadata
├── CODE_OF_CONDUCT.md                       # Contributor Covenant v2.1
├── CONTRIBUTING.md                          # Contribution & development guide
├── Dockerfile                               # Reproducible Docker container setup
├── LICENSE                                  # MIT License
├── pyproject.toml                           # PEP 517/518 build & tool configuration
├── requirements.txt                         # Runtime dependencies
├── requirements-dev.txt                     # Development & test dependencies
├── SECURITY.md                              # Vulnerability disclosure policy
└── setup.py                                 # Setuptools packaging metadata
```

---

## 🚀 Quick Start & Usage

### 1. Installation

```bash
# Clone repository
git clone https://github.com/vansh7nvc/Organoid-Intelligence-gym-.git
cd Organoid-Intelligence-gym-

# Install in editable mode with development tools
pip install -e ".[dev]"
```

### 2. Basic Environment Interaction

```python
import gymnasium as gym
from organoid_rl import OrganoidEnv

# Initialize the biologically-constrained SNN environment
env = OrganoidEnv(
    use_sdm=True,           # Enable 256-neuron Sparse Distributed Memory
    use_morphology=True,    # Enable functional layered morphology
    use_dual_trace=True,    # Enable Dual-Trace Dopamine Plasticity
    use_stabilizer=True     # Enable Global Activity Regulator (GAR)
)

obs, info = env.reset(seed=42)
print(f"Observation space dimension: {obs.shape}")  # 21-dimensional state

for step in range(100):
    action = env.action_space.sample()  # Action: 0-7 stimulation pattern
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished at step {step} with reward {reward:.2f}")
        obs, info = env.reset()
```

### 3. Run Standalone Research Examples

```bash
# 1. Quickstart demonstration
python examples/01_quickstart.py

# 2. Evaluate pre-trained DQN agent
python examples/02_evaluate_pretrained_agent.py

# 3. Run component ablation comparison
python examples/03_run_ablation_comparison.py
```

### 4. Running Tests

```bash
# Run unit test suite
python -m unittest discover -s organoid_rl/tests -p "test_*.py" -v

# Run Brian2 sanity check
python organoid_rl/tests/sanity_check.py
```

### 5. Run Training with 1 Click

Open [`notebooks/colab_gpu_training.ipynb`](notebooks/colab_gpu_training.ipynb) in Google Colab, connect to a CPU/T4 runtime, and select **Runtime ➔ Run All**. Checkpoints and logs will automatically sync with your Google Drive.

---

## 📜 Citation

If you use **OrganoidEnv** in your research, please cite:

```bibtex
@article{Sharma2026OrganoidEnv,
  author    = {Sharma, Vansh and Malik, Seema},
  title     = {OrganoidEnv: A Stabilized Reinforcement Learning Environment for Training Biologically Constrained Spiking Neural Networks via Sensory--Motor Mapping},
  journal   = {Frontiers in Neuroinformatics},
  year      = {2026}
}
```

For software attribution (JOSS submission):
```bibtex
@article{Sharma_OrganoidEnv_JOSS_2026,
  author    = {Sharma, Vansh and Malik, Seema},
  title     = {{OrganoidEnv: A Neuromorphic Testbed for Reinforcement Learning with Biologically Constrained Spiking Neural Networks}},
  journal   = {Journal of Open Source Software},
  year      = {2026},
  url       = {https://github.com/vansh7nvc/Organoid-Intelligence-gym-},
  version   = {1.0.0}
}
```

---

## 📄 License & Attribution
Distributed under the **MIT License**. Free for academic, scientific, and commercial use.
