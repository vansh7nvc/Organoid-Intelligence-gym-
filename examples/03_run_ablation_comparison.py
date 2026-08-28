"""
Example 03: Run Ablation Comparison
Demonstrates comparing the full biological model against key architectural ablations (No-GAR and No-SDM).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from organoid_rl.environment.core import OrganoidEnv

def run_short_benchmark(name, env_kwargs, n_steps=15):
    print(f"\n--- Testing Configuration: {name} ---")
    env = OrganoidEnv(**env_kwargs)
    obs, info = env.reset(seed=123)
    
    total_reward = 0.0
    spikes_recorded = 0

    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        spikes_recorded = info["total_spikes"]

    print(f"  Result -> Mean Reward / Step: {total_reward / n_steps:6.2f} | "
          f"Final Distance: {info['distance']:.3f} | Total Spikes: {spikes_recorded}")
    return total_reward

def main():
    print("=" * 65)
    print("OrganoidEnv: Component Ablation Demonstration")
    print("=" * 65)

    configs = {
        "Full OrganoidEnv (GAR + SDM + D3)": {
            "use_sdm": True, "use_stabilizer": True, "use_dual_trace": True
        },
        "No-GAR Ablation (No Seizure/Coma Homeostasis)": {
            "use_sdm": True, "use_stabilizer": False, "use_dual_trace": True
        },
        "No-SDM Ablation (No Sparse Memory Expansion)": {
            "use_sdm": False, "use_stabilizer": True, "use_dual_trace": True
        }
    }

    for name, kwargs in configs.items():
        run_short_benchmark(name, kwargs)

    print("\n" + "=" * 65)
    print("Ablation comparison demonstration completed!")
    print("=" * 65)

if __name__ == "__main__":
    main()
