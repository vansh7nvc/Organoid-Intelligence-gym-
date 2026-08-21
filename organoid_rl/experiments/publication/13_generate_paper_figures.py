"""
Publication Figure Generator for OrganoidEnv.

Loads training results from experiments/results/ and generates
publication-quality matplotlib figures for the paper.

Usage:
    python experiments/13_generate_paper_figures.py

Author: Vansh Sharma
License: MIT
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Configuration ---
# Results may live in organoid_rl/experiments/results/ or in the
# project-root experiments/results/ directory.
_local_results = os.path.join(os.path.dirname(__file__), "results")
_root_results = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")
RESULTS_DIR = _local_results if os.path.isdir(_local_results) and os.listdir(_local_results) else os.path.abspath(_root_results)
# Prefer whichever directory actually contains JSON files
if not any(f.endswith('.json') for f in os.listdir(RESULTS_DIR) if os.path.isfile(os.path.join(RESULTS_DIR, f))):
    alt = _root_results if RESULTS_DIR == _local_results else _local_results
    if os.path.isdir(alt):
        RESULTS_DIR = os.path.abspath(alt)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def load_json(filename):
    """Safely load a JSON results file."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found")
        return None
    with open(path) as f:
        return json.load(f)


def figure1_learning_progression():
    """
    Figure 1: Learning Progression Across Months.
    Plots reward curves from Month 4 (full), Month 5, and Month 6.
    """
    print("[Fig 1] Learning Progression...")
    
    datasets = {
        'Month 4 (Full Arch.)': load_json('results_full.json'),
        'Month 5 (Multi-Goal)': load_json('results_month5_multigoal.json'),
        'Month 6 (Grand Unif.)': load_json('results_month6_grand.json'),
    }
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for (label, data), color in zip(datasets.items(), colors):
        if data is None or 'rewards' not in data:
            continue
        rewards = data['rewards']
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)
        # Moving average
        if len(rewards) > 15:
            window = min(20, len(rewards) // 3)
            ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), ma, color=color,
                    linewidth=2, label=f'{label} ({len(rewards)} ep)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Progression: Month 4 → Month 6')
    ax.legend(loc='upper left', framealpha=0.9)
    
    path = os.path.join(OUTPUT_DIR, 'fig1_learning_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def figure2_ablation_comparison():
    """
    Figure 2: Month 4 Ablation Study.
    Bar chart comparing Full Architecture vs. No SDM/Morphology vs. No Homeostasis.
    """
    print("[Fig 2] Ablation Comparison...")
    
    configs = {
        'Full\nArchitecture': load_json('results_full.json'),
        'No SDM /\nMorphology': load_json('results_no_sdm_morph.json'),
        'No\nHomeostasis': load_json('results_no_homeostasis.json'),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Panel A: Success Rate
    ax = axes[0]
    labels, rates, totals = [], [], []
    for label, data in configs.items():
        labels.append(label)
        if data and 'goals_reached' in data and 'episodes' in data:
            rates.append(data['goals_reached'] / data['episodes'] * 100)
            totals.append(f"{data['goals_reached']}/{data['episodes']}")
        else:
            rates.append(0)
            totals.append('N/A')
    
    bars = ax.bar(labels, rates, color=colors, edgecolor='white', linewidth=1.2)
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                total, ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(A) Goal Completion Rate')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 20)
    
    # Panel B: Average Reward
    ax = axes[1]
    avg_rewards = []
    for label, data in configs.items():
        if data and 'rewards' in data:
            avg_rewards.append(np.mean(data['rewards']))
        else:
            avg_rewards.append(0)
    
    bars = ax.bar(labels, avg_rewards, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, avg_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('(B) Average Reward')
    
    fig.suptitle('Ablation Study: Component Contribution (Month 4)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, 'fig2_ablation_study.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def figure3_month6_dashboard():
    """
    Figure 3: Month 6 Curriculum Stage Success Rates.
    """
    print("[Fig 3] Month 6 Dashboard...")
    
    data = load_json('results_month6_grand.json')
    if data is None:
        # Try v2
        data = load_json('results_month6_grand_v2.json')
    if data is None:
        print("  [SKIP] No Month 6 results found")
        return
    
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[2, 1, 1])
    
    # Panel A: Reward Curve with Curriculum Markers
    ax = fig.add_subplot(gs[0])
    rewards = data.get('rewards', [])
    if rewards:
        ax.plot(rewards, alpha=0.2, color='steelblue', linewidth=0.5)
        if len(rewards) > 20:
            ma = np.convolve(rewards, np.ones(20) / 20, mode='valid')
            ax.plot(range(19, len(rewards)), ma, color='navy', linewidth=2, label='20-ep MA')
        for ep, label in [(100, 'Stage 2'), (200, 'Stage 3'), (350, 'Stage 4')]:
            if ep < len(rewards):
                ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
                ax.text(ep + 3, max(rewards) * 0.85, label, fontsize=8, color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('(A) Training Reward Curve')
        ax.legend(loc='lower right')
    
    # Panel B: Stage Success Rates
    ax = fig.add_subplot(gs[1])
    stage_goals = data.get('stage_goals', {})
    stage_episodes = data.get('stage_episodes', {})
    stages = ['1', '2', '3', '4']
    stage_rates = []
    for s in stages:
        goals = stage_goals.get(s, 0)
        eps = stage_episodes.get(s, 1)
        stage_rates.append(goals / max(1, eps) * 100)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar([f'S{s}' for s in stages], stage_rates, color=colors)
    for bar, rate in zip(bars, stage_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(B) By Stage')
    ax.set_ylim(0, 110)
    
    # Panel C: Context Split
    ax = fig.add_subplot(gs[2])
    ctx = data.get('context_success', {})
    ctx_vals = [ctx.get('0', ctx.get(0, 0)), ctx.get('1', ctx.get(1, 0))]
    ax.bar(['Target A\n(0.8, 0.8)', 'Target B\n(0.2, 0.2)'], ctx_vals,
           color=['#3498db', '#e67e22'])
    ax.set_ylabel('Goals Reached')
    ax.set_title('(C) By Context')
    
    fig.suptitle('Month 6: Grand Unification Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, 'fig3_month6_dashboard.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def figure4_architecture_schematic():
    """
    Figure 4: Architecture schematic as a text-based diagram.
    """
    print("[Fig 4] Architecture Schematic...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('OrganoidEnv Architecture', fontsize=15, fontweight='bold', pad=20)
    
    # Boxes
    boxes = [
        (1.0, 4.5, 2.0, 1.5, '#3498db', 'SDM Layer\n256 neurons\n(10% sparse)', 'white'),
        (4.0, 4.5, 2.0, 1.5, '#2ecc71', 'Hidden Layer\n144 neurons\n(integration)', 'white'),
        (7.0, 4.5, 2.0, 1.5, '#e74c3c', 'Motor Layer\n100 neurons\n(clustered)', 'white'),
        (1.0, 1.5, 2.0, 1.2, '#f39c12', 'Observation\n21D Input', 'white'),
        (4.0, 1.5, 2.0, 1.2, '#9b59b6', 'RL Agent\nDDQN + PER', 'white'),
        (7.0, 1.5, 2.0, 1.2, '#1abc9c', 'Cursor\nMovement', 'white'),
    ]
    
    for x, y, w, h, color, text, tc in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.85, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color=tc, zorder=3)
    
    # Arrows (neural pathway)
    for x1, x2, y in [(3.0, 4.0, 5.25), (6.0, 7.0, 5.25)]:
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    
    # Arrows (agent loop)
    ax.annotate('', xy=(1.0, 4.5), xytext=(2.0, 2.7),  # Obs -> SDM
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, ls='--'))
    ax.annotate('', xy=(4.0, 1.5), xytext=(4.0, 4.5),  # Agent -> actions
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, ls='--'))
    ax.annotate('', xy=(7.0, 2.7), xytext=(9.0, 4.5),  # Motor -> Cursor
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, ls='--'))
    
    # Labels
    ax.text(5.0, 6.5, 'Spiking Neural Network (Brian2)', ha='center',
            fontsize=12, fontstyle='italic', color='#2c3e50')
    ax.text(5.0, 0.5, 'Agent-Environment Loop', ha='center',
            fontsize=12, fontstyle='italic', color='#7f8c8d')
    
    path = os.path.join(OUTPUT_DIR, 'fig4_architecture.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=" * 50)
    print("  Generating Publication Figures")
    print("=" * 50)
    
    figure1_learning_progression()
    figure2_ablation_comparison()
    figure3_month6_dashboard()
    figure4_architecture_schematic()
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("=" * 50)
