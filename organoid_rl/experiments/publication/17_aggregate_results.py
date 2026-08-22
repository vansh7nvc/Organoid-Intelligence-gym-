"""
Aggregate and plot results from Colab V2 training runs.

Handles both the multiseed training logs and the ablation logs.

Usage:
    python 17_aggregate_results.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
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
})

def aggregate_and_plot(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Aggregate Multiseed Data
    multiseed_dir = os.path.join(base_dir, "colab_results_v2")
    multiseed_data = {}
    
    if os.path.exists(multiseed_dir):
        for f in os.listdir(multiseed_dir):
            if f.startswith("training_log_seed_") and f.endswith(".csv"):
                df = pd.read_csv(os.path.join(multiseed_dir, f))
                seed = df['seed'].iloc[0]
                multiseed_data[seed] = df
    
    # 2. Aggregate Ablation Data
    ablation_dir = os.path.join(base_dir, "colab_ablations_v2")
    ablation_data = {"No_GAR": {}, "No_SDM": {}}
    
    if os.path.exists(ablation_dir):
        for f in os.listdir(ablation_dir):
            if f.startswith("training_log_") and f.endswith(".csv"):
                df = pd.read_csv(os.path.join(ablation_dir, f))
                ablation = df['ablation'].iloc[0]
                seed = df['seed'].iloc[0]
                if ablation in ablation_data:
                    ablation_data[ablation][seed] = df

    # Plot 1: Learning Curves
    if multiseed_data:
        print(f"Plotting multiseed learning curves from {len(multiseed_data)} seeds...")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        max_ep = max(df['episode'].max() for df in multiseed_data.values())
        all_rewards = np.full((len(multiseed_data), max_ep + 1), np.nan)
        
        for i, (seed, df) in enumerate(multiseed_data.items()):
            all_rewards[i, df['episode'].values] = df['reward'].values
            
        mean_rewards = np.nanmean(all_rewards, axis=0)
        std_rewards = np.nanstd(all_rewards, axis=0)
        x = np.arange(len(mean_rewards))
        
        ax.plot(x, mean_rewards, 'b-', label='Mean Reward (Full Config)')
        ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color='blue')
        
        if len(mean_rewards) >= 20:
            ma = np.convolve(mean_rewards[~np.isnan(mean_rewards)], np.ones(20)/20, mode='valid')
            ax.plot(np.arange(19, 19 + len(ma)), ma, 'navy', linewidth=2, label='20-ep MA')
            
        # Add ablation means if available
        colors = {'No_GAR': 'red', 'No_SDM': 'orange'}
        for ab_name, ab_seeds in ablation_data.items():
            if not ab_seeds:
                continue
            ab_max_ep = max(df['episode'].max() for df in ab_seeds.values())
            ab_rewards = np.full((len(ab_seeds), ab_max_ep + 1), np.nan)
            for i, df in enumerate(ab_seeds.values()):
                ab_rewards[i, df['episode'].values] = df['reward'].values
            ab_mean = np.nanmean(ab_rewards, axis=0)
            ax.plot(np.arange(len(ab_mean)), ab_mean, color=colors[ab_name], linestyle='--', label=ab_name)
            
        ax.set_title(f"Training Performance Across Seeds (N={len(multiseed_data)})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        
        out_path = os.path.join(output_dir, "v2_learning_curves.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")
        
    # Plot 2: Stage Success Rates
    if multiseed_data:
        print("Plotting stage success rates...")
        # Calculate overall success by stage
        stages = [0, 1, 2] # Basic, Obstacles, Multi-goal
        stage_names = ['Stage 1\n(Basic)', 'Stage 2\n(Obstacles)', 'Stage 3\n(Multi-goal)']
        
        stage_success = {s: [] for s in stages}
        
        for df in multiseed_data.values():
            for s in stages:
                stage_df = df[df['stage'] == s]
                if not stage_df.empty:
                    success_rate = stage_df['success'].mean() * 100
                    stage_success[s].append(success_rate)
                    
        means = [np.mean(stage_success[s]) if stage_success[s] else 0 for s in stages]
        stds = [np.std(stage_success[s]) if stage_success[s] else 0 for s in stages]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(stages))
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='seagreen', alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Curriculum Stage")
        ax.set_ylim(0, 110)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'{height:.1f}%', ha='center', fontweight='bold')
                    
        out_path = os.path.join(output_dir, "v2_stage_success.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")
        
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "aggregated_v2")
    aggregate_and_plot(base_dir, output_dir)
