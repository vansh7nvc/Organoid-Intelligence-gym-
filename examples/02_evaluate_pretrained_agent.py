"""
Example 02: Evaluate Pre-trained Agent
Demonstrates loading a trained Dueling Double DQN checkpoint and evaluating navigation performance in OrganoidEnv.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent

def main():
    print("=" * 60)
    print("OrganoidEnv: Evaluate Pre-trained Agent")
    print("=" * 60)

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "organoid_rl", "experiments", "results")
    checkpoint_file = os.path.join(checkpoint_dir, "brain_month6_final_ep50.pth")

    env = OrganoidEnv()
    agent = DQNAgent(obs_dim=21, n_actions=8)

    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location=agent.device)
        agent.online_net.load_state_dict(state_dict)
        agent.online_net.eval()
        print("Checkpoint loaded successfully!")
    else:
        print(f"Checkpoint not found at {checkpoint_file}. Running with initialized policy.")

    num_eval_episodes = 2
    for ep in range(1, num_eval_episodes + 1):
        obs, info = env.reset(seed=ep * 10)
        episode_reward = 0.0
        done = False
        step = 0

        print(f"\n--- Starting Episode {ep} ---")
        while not done and step < 30:
            step += 1
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if step % 5 == 0 or done:
                print(f"Step {step:2d} | Action: {action} | Distance: {info['distance']:.3f} | "
                      f"Cumulative Reward: {episode_reward:.2f}")

        print(f"Episode {ep} finished: {'SUCCESS' if terminated else 'TRUNCATED'} | "
              f"Total Reward: {episode_reward:.2f} | Final Distance: {info['distance']:.3f}")

    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
