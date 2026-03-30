import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ablation_results():
    results_dir = "experiments/results"
    tags = ["full", "no_sdm_morph", "no_homeostasis"]
    
    plt.figure(figsize=(12, 6))
    
    for tag in tags:
        path = f"{results_dir}/results_{tag}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            rewards = data['rewards']
            window = 5
            smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            plt.plot(smoothed, label=f"{tag} (Success: {data['goals_reached']}/{data['episodes']})")
    
    plt.title("Ablation Study: Phase 4 Path to 6% Ceiling Break")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "experiments/09_phase4_ablation_results.png"
    plt.savefig(plot_path)
    print(f"\nFinal Ablation Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_ablation_results()
