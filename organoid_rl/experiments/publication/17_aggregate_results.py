import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def aggregate_and_plot_results(results_dir, output_dir):
    """
    Parses Ray Tune results across multiple seeds and generates mean ± std plots.
    """
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all progress.csv files from trial directories
    trial_data = []
    
    for root, dirs, files in os.walk(results_dir):
        if "progress.csv" in files:
            csv_path = os.path.join(root, "progress.csv")
            try:
                df = pd.read_csv(csv_path)
                trial_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                
    if not trial_data:
        print("No valid progress.csv files found in the results directory.")
        return
        
    print(f"Aggregating data from {len(trial_data)} trials/seeds...")
    
    # Align data by episode
    max_episodes = max(df['episode'].max() for df in trial_data) if trial_data else 0
    all_rewards = np.zeros((len(trial_data), int(max_episodes) + 1))
    
    for i, df in enumerate(trial_data):
        episodes = df['episode'].values.astype(int)
        rewards = df['reward'].values
        # Assign to corresponding indices
        for ep, rew in zip(episodes, rewards):
            all_rewards[i, ep] = rew
            
    # Calculate mean and std
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes_x = np.arange(len(mean_rewards))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_x, mean_rewards, 'b-', label='Mean Reward')
    plt.fill_between(episodes_x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Add a moving average
    if len(mean_rewards) >= 20:
        ma = np.convolve(mean_rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(mean_rewards)), ma, 'r-', linewidth=2, label='20-ep MA')
        
    plt.title(f"Multi-Seed Training Performance (N={len(trial_data)} seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(output_dir, "multiseed_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # Save aggregated data as CSV for easy access
    agg_df = pd.DataFrame({
        "episode": episodes_x,
        "mean_reward": mean_rewards,
        "std_reward": std_rewards
    })
    csv_path = os.path.join(output_dir, "aggregated_rewards.csv")
    agg_df.to_csv(csv_path, index=False)
    print(f"Aggregated data saved to {csv_path}")

if __name__ == "__main__":
    import sys
    
    ray_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ray_results", "organoid_multiseed"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "aggregated_results"))
    
    if len(sys.argv) > 1:
        ray_results_dir = sys.argv[1]
        
    aggregate_and_plot_results(ray_results_dir, output_dir)
