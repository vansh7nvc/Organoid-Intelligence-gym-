"""
Example 01: Quickstart with OrganoidEnv
Demonstrates initializing the environment, inspecting observations, and taking simulation steps.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from organoid_rl.environment.core import OrganoidEnv

def main():
    print("=" * 60)
    print("OrganoidEnv: Quickstart Example")
    print("=" * 60)

    # Initialize the biologically-constrained SNN environment
    env = OrganoidEnv(
        use_sdm=True,           # 256-neuron Sparse Distributed Memory
        use_morphology=True,    # Clustered motor quadrants
        use_dual_trace=True,    # Fast (100ms) + Slow (2.5s) Dopamine Plasticity
        use_stabilizer=True,    # Global Activity Regulator (GAR)
        use_motor_mapping=True  # Directional Motor Stimulation
    )

    obs, info = env.reset(seed=42)
    print(f"\nEnvironment initialized successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation vector dimension: {obs.shape[0]} (21D)")
    print(f"Action space: {env.action_space} (8 discrete stimulation patterns)")

    total_reward = 0.0
    num_steps = 10

    print(f"\nRunning {num_steps} simulation steps...")
    for step in range(1, num_steps + 1):
        # Sample random action (0: Up, 1: Down, 2: Left, 3: Right, 4-7: Diagonals)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step:2d} | Action: {action} | Step Reward: {reward:6.2f} | "
              f"Distance: {info['distance']:.3f} | Total Spikes: {info['total_spikes']}")

        if terminated or truncated:
            print(f"\nEpisode completed at step {step}!")
            obs, info = env.reset()

    print(f"\nTotal accumulated reward across {num_steps} steps: {total_reward:.2f}")
    print("Quickstart completed successfully!")

if __name__ == "__main__":
    main()
