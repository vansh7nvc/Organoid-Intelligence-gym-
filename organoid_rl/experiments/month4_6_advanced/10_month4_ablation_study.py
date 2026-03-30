import subprocess
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np

def run_training(tag, args=[]):
    results_dir = "experiments/results"
    path = f"{results_dir}/results_{tag}.json"
    if os.path.exists(path):
        print(f"\n>>> Results for {tag} already exist. Skipping...")
        return

    cmd = [sys.executable, "organoid_rl/experiments/09_month4_training.py", "--tag", tag, "--episodes", "100"] + args
    print(f"\n>>> Running Ablation Case: {tag}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

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
            # Smooth for plotting
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
    # Run cases
    # 1. Full Architecture
    run_training("full", [])
    
    # 2. No SDM / No Morphology (Random Soup)
    run_training("no_sdm_morph", ["--no_sdm", "--no_morphology"])
    
    # 3. No Homeostasis
    run_training("no_homeostasis", ["--no_homeostasis"])
    
    # Plot
    plot_ablation_results()
