import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from organoid_rl.environment.baseline_envs import SkeletonEnv
from organoid_rl.agents.sb3_baselines import create_dqn_agent, create_ppo_agent
from stable_baselines3.common.callbacks import BaseCallback

class TrackMetricsCallback(BaseCallback):
    """
    Custom callback for tracking episode rewards and success rates.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.current_reward = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1
        
        # Check if episode is done
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            # Check success (distance < goal_radius)
            # stable-baselines3 stores info dicts in self.locals["infos"]
            info = self.locals["infos"][0]
            success = 1.0 if info.get("distance", 1.0) < 0.05 else 0.0
            self.successes.append(success)
            
            self.current_reward = 0
            self.current_length = 0
            
        return True

def run_baseline(agent_name, agent_creator, env, total_timesteps=40000, seed=42):
    print(f"\n--- Training {agent_name} Baseline (Seed {seed}) ---")
    
    # We use DummyVecEnv implicitly via SB3
    model = agent_creator(env, seed=seed)
    callback = TrackMetricsCallback()
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Compute metrics mimicking Table IV
    success_rate = np.mean(callback.successes) * 100
    final_reward = np.mean(callback.episode_rewards[-50:]) if len(callback.episode_rewards) > 0 else 0
    
    # Find episodes to 60%
    ep_to_60 = -1
    if len(callback.successes) >= 50:
        for i in range(50, len(callback.successes)):
            if np.mean(callback.successes[i-50:i]) >= 0.6:
                ep_to_60 = i
                break
                
    print(f"Results for {agent_name}:")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Episodes to 60%: {ep_to_60}")
    print(f"  Final Reward: {final_reward:.1f}")
    
    return callback.episode_rewards, callback.successes

if __name__ == "__main__":
    # Create the environment with a TimeLimit of 80 steps matching the paper
    base_env = SkeletonEnv(obs_mode='full')
    env = TimeLimit(base_env, max_episode_steps=80)
    
    seeds = [42, 123, 456]
    results = {'DQN': {'rewards': [], 'successes': []}, 'PPO': {'rewards': [], 'successes': []}}
    
    # Run 1 seed for testing (change to all seeds for full evaluation)
    for seed in [42]: 
        dqn_rewards, dqn_successes = run_baseline("DQN", create_dqn_agent, env, total_timesteps=10000, seed=seed)
        results['DQN']['rewards'].append(dqn_rewards)
        results['DQN']['successes'].append(dqn_successes)
        
        ppo_rewards, ppo_successes = run_baseline("PPO", create_ppo_agent, env, total_timesteps=10000, seed=seed)
        results['PPO']['rewards'].append(ppo_rewards)
        results['PPO']['successes'].append(ppo_successes)
        
    print("\nPhase 2 implementation complete. Script runs successfully!")
