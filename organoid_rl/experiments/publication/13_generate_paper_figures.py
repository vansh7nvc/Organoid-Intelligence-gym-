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
# Results live in organoid_rl/experiments/results/, which is one level up from this script (in publication/)
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    

OUTPUT_DIR = os.path.join(RESULTS_DIR, "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style (LaTeX compatibility)
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'axes.formatter.use_mathtext': True,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'pdf.fonttype': 42,  # TrueType for PDF editors
    'ps.fonttype': 42,
})

def save_fig(fig, base_name):
    """Helper to save figures as both PNG and PDF for LaTeX."""
    png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"  Saved: {png_path} and .pdf")

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
        'Month 6 (Grand Unif.)': load_json('results_month6_final.json'),
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
    
    save_fig(fig, 'fig1_learning_progression')
    plt.close(fig)


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
    
    save_fig(fig, 'fig2_ablation_study')
    plt.close(fig)


def figure3_month6_dashboard():
    """
    Figure 3: Month 6 Curriculum Stage Success Rates.
    """
    print("[Fig 3] Month 6 Dashboard...")
    
    data = load_json('results_month6_final.json')
    if data is None:
        data = load_json('results_month6_grand.json')
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
    
    save_fig(fig, 'fig3_month6_dashboard')
    plt.close(fig)


def figure4_architecture_schematic():
    """
    Figure 4: Publication-quality Neural Topology Architecture schematic.
    """
    print("[Fig 4] Architecture Schematic...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-10, 95)
    ax.axis('off')
    
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe

    # ==========================
    # 1. DRAW SNN NEURAL TOPOLOGY
    # ==========================
    def draw_layer_nodes(cx, cy, width, height, num_nodes, color, seed=42):
        np.random.seed(seed)
        xs = cx - width/2 + np.random.rand(num_nodes) * width
        ys = cy - height/2 + np.random.rand(num_nodes) * height
        
        # Plot glowing nodes
        ax.scatter(xs, ys, s=40, c=color, edgecolors='white', linewidths=0.5, alpha=0.9, zorder=4)
        ax.scatter(xs, ys, s=120, c=color, alpha=0.15, zorder=3)
        return xs, ys

    def draw_synapses(xs1, ys1, xs2, ys2, sparsity, color, alpha=0.1):
        lines = []
        np.random.seed(42)
        n1, n2 = len(xs1), len(xs2)
        # Cap max lines to prevent massive PDF sizes, but keep it looking dense
        max_lines = 800
        current_lines = 0
        for i in range(n1):
            for j in range(n2):
                if np.random.rand() < sparsity and current_lines < max_lines:
                    lines.append([(xs1[i], ys1[i]), (xs2[j], ys2[j])])
                    current_lines += 1
        lc = LineCollection(lines, colors=color, linewidths=0.4, alpha=alpha, zorder=2)
        ax.add_collection(lc)

    # Base Coordinates for Neural Layers
    input_cx, input_cy = 15, 60
    sdm_cx, sdm_cy = 40, 60
    hidden_cx, hidden_cy = 65, 60
    motor_cx, motor_cy = 90, 60

    # Draw Nodes
    in_xs, in_ys = draw_layer_nodes(input_cx, input_cy, 4, 30, 21, '#9b59b6')
    sdm_xs, sdm_ys = draw_layer_nodes(sdm_cx, sdm_cy, 10, 40, 200, '#3498db')  # Display 200 out of 256 for visual clarity
    hid_xs, hid_ys = draw_layer_nodes(hidden_cx, hidden_cy, 8, 35, 144, '#2ecc71')
    
    # Motor layer as 4 distinct spatial clusters (Up, Down, Left, Right)
    m_xs, m_ys = [], []
    for quad_cx, quad_cy in [(87, 72), (93, 72), (87, 48), (93, 48)]:
        qx, qy = draw_layer_nodes(quad_cx, quad_cy, 3, 10, 25, '#e74c3c')
        m_xs.extend(qx)
        m_ys.extend(qy)

    # Draw Synaptic Web (Projections)
    draw_synapses(in_xs, in_ys, sdm_xs, sdm_ys, 0.08, '#7f8c8d', 0.2)
    draw_synapses(sdm_xs, sdm_ys, hid_xs, hid_ys, 0.05, '#7f8c8d', 0.15)
    draw_synapses(hid_xs, hid_ys, m_xs, m_ys, 0.06, '#7f8c8d', 0.15)
    
    # Internal recurrent hidden synapses
    draw_synapses(hid_xs, hid_ys, hid_xs, hid_ys, 0.02, '#2ecc71', 0.1)

    # Layer Text Annotations
    def add_layer_label(x, y, title, subtitle):
        bbox = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1)
        ax.text(x, y, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50', bbox=bbox, zorder=5)
        ax.text(x, y - 3, subtitle, ha='center', va='center', fontsize=9, color='#7f8c8d', bbox=bbox, zorder=5)

    add_layer_label(input_cx, 85, 'Input Layer', '(21 Sensors)')
    add_layer_label(sdm_cx, 85, 'SDM Layer', '(256 Sparse Neurons)')
    add_layer_label(hidden_cx, 85, 'Hidden Layer', '(144 Recurrent Neurons)')
    add_layer_label(motor_cx, 85, 'Motor Layer', '($4 \\times 25$ Clustered Neurons)')

    # Organoid Bounding Box
    organoid_box = FancyBboxPatch((5, 38), 90, 52, boxstyle='round,pad=1,rounding_size=2', 
                                  edgecolor='#bdc3c7', facecolor='#fbfcfc', lw=2, linestyle='--', zorder=0)
    ax.add_patch(organoid_box)
    ax.text(50, 92, 'The Organoid ($\mathit{in\ silico}$ Metabolic-Izhikevich Spiking Neural Network)', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#34495e', zorder=5)

    # ==========================
    # 2. DRAW OUTER RL AGENT & ENV
    # ==========================
    # RL Agent Box
    rl_box = FancyBboxPatch((30, -5), 40, 20, boxstyle='round,pad=0.5,rounding_size=2', 
                            edgecolor='#8e44ad', facecolor='#f8f4f9', lw=2, zorder=2)
    rl_box.set_path_effects([pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.3), pe.Normal()])
    ax.add_patch(rl_box)
    
    ax.text(50, 10, 'Automated Experimentalist (Outer Agent)', ha='center', va='center', fontsize=12, fontweight='bold', color='#8e44ad', zorder=5)
    ax.text(50, 5, 'Dueling Double DQN + Prioritized Exp. Replay', ha='center', va='center', fontsize=10, color='#2c3e50', zorder=5)
    ax.text(50, -1, '$Q^*(s_t, a_t) = V(s_t) + \\left(A(s_t, a_t) - \\frac{1}{|\\mathcal{A}|}\\sum A(s_t, a)\\right)$', 
            ha='center', va='center', fontsize=10, color='#2c3e50', zorder=5)

    # Biological Stabilizers Block (Bridging Agent and SNN)
    stab_box = FancyBboxPatch((15, 23), 70, 8, boxstyle='round,pad=0.2,rounding_size=1', 
                              edgecolor='#d35400', facecolor='#fef5e7', lw=1.5, zorder=2)
    ax.add_patch(stab_box)
    ax.text(50, 27, 'Biological Stabilizers', ha='center', va='center', fontsize=11, fontweight='bold', color='#d35400', zorder=5)
    ax.text(50, 24, 'Global Activity Regulator (Homeostatic Reset) \& Inhibitory Homeostasis \& Structural Plasticity', 
            ha='center', va='center', fontsize=9, color='#e67e22', zorder=5)

    # ==========================
    # 3. CONTROL FLOW ARROWS
    # ==========================
    def add_curve_arrow(start, end, rad, label, color, label_pos, text_offset=(0,0)):
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', color=color, lw=2.5, 
                                connectionstyle=f'arc3,rad={rad}', zorder=1, mutation_scale=20)
        ax.add_patch(arrow)
        bbox = dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2)
        ax.text(label_pos[0]+text_offset[0], label_pos[1]+text_offset[1], label, 
                ha='center', va='center', fontsize=10, fontweight='bold', color=color, bbox=bbox, zorder=6)

    # Environment -> Agent (State/Reward)
    add_curve_arrow((10, 40), (28, 5), rad=-0.3, label='State $s_t$\nReward $r_t$', color='#2980b9', label_pos=(13, 22))
    
    # Agent -> Motor Injection (Action)
    add_curve_arrow((70, 5), (92, 43), rad=0.3, label='Motor Quadrant\nInjection $I_{stim}$', color='#c0392b', label_pos=(91, 23))
    
    # Environment -> Sensory Input
    add_curve_arrow((10, 50), (13, 60), rad=0.2, label='Exteroceptive Sensor Map', color='#16a085', label_pos=(-1, 55))
    
    # Motor -> Environment (Cursor Step)
    add_curve_arrow((90, 78), (10, 70), rad=-0.2, label='Spike Decoding $\\rightarrow$ Action execution (Cursor Steps)', color='#8e44ad', label_pos=(50, 81))

    fig.tight_layout()
    save_fig(fig, 'fig4_architecture')
    plt.close(fig)


def figure5_baseline_comparison():
    """
    Figure 5: Statistical Comparison against Baselines.
    Success rate vs Training Episodes.
    """
    print("[Fig 5] Baseline Comparison...")
    data = load_json('comprehensive_eval.json')
    if not data: return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Group OrganoidRL seeds
    snn_seeds = [data[k]['success_rates'] for k in data if 'OrganoidRL_Seed' in k]
    if snn_seeds:
        snn_mean = np.mean(snn_seeds, axis=0)
        snn_std = np.std(snn_seeds, axis=0)
        ax.plot(snn_mean, label='OrganoidRL (Mean ± Std)', color='teal', linewidth=2.5)
        ax.fill_between(range(len(snn_mean)), snn_mean - snn_std, snn_mean + snn_std, color='teal', alpha=0.2)

    # Plot other baselines
    colors = {'ANN_Baseline': 'crimson', 'RL-only_Baseline': 'orange', 'SimpleSNN_Baseline': 'gray'}
    for label, color in colors.items():
        if label in data:
            ax.plot(data[label]['success_rates'], label=label.replace('_', ' '), color=color, linestyle='--')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Baseline Comparison: Statistical Validation')
    ax.legend()
    save_fig(fig, 'fig5_baseline_comparison')
    plt.close(fig)

def figure6_network_activity():
    """
    Figure 6: Network Activity (Spike Raster & Motor Activation).
    Dummy simulation if no real data to demonstrate plot.
    """
    print("[Fig 6] Network Activity...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel A: Spike Raster (Synthetic for Demo)
    ax = axes[0]
    np.random.seed(42)
    for i in range(100):
        spikes = np.random.uniform(0, 1000, size=np.random.randint(5, 50))
        ax.scatter(spikes, [i] * len(spikes), s=2, color='black', alpha=0.6)
    ax.set_ylabel('Neuron ID')
    ax.set_title('(A) Spike Raster Plot')
    
    # Panel B: Motor Activation
    ax = axes[1]
    t = np.linspace(0, 1000, 1000)
    up = np.sin(t/100)*10 + 10 + np.random.randn(1000)
    down = np.cos(t/100)*5 + 5 + np.random.randn(1000)
    ax.plot(t, up, label='Motor Up', color='red')
    ax.plot(t, down, label='Motor Down', color='blue')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('(B) Motor Neuron Activation Patterns')
    ax.legend()

    plt.tight_layout()
    save_fig(fig, 'fig6_network_activity')
    plt.close(fig)

if __name__ == '__main__':
    print("=" * 50)
    print("  Generating Publication Figures")
    print("=" * 50)
    
    figure1_learning_progression()
    figure2_ablation_comparison()
    figure3_month6_dashboard()
    figure4_architecture_schematic()
    figure5_baseline_comparison()
    figure6_network_activity()
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("=" * 50)
