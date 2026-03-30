import gymnasium as gym
import numpy as np
import torch
import json
import os
import time
from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.environment.baseline_envs import SkeletonEnv
from organoid_rl.agents.dqn_agent import DQNAgent
from organoid_rl.agents.baseline_agent import MLPBaselineAgent

def run_experiment(env, agent, n_episodes=100, max_steps=100, label="exp"):
    print(f"--- Running {label} ---")
    results = {
        'rewards': [],
        'success_rates': [],
        'steps': [],
        'spikes': [],
        'episodes_to_60': None
    }
    
    success_count = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_spikes = 0
        # For HER support in DQNAgent
        episode_transitions = []
        
        for step in range(max_steps):
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            if hasattr(agent, 'store_transition'):
                agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
                episode_transitions.append((obs, action, reward, next_obs, terminated or truncated))
            
            obs = next_obs
            ep_reward += reward
            ep_spikes += info.get('total_spikes', 0)
            
            if terminated or truncated:
                if terminated: success_count += 1
                break
        
        # Post-episode learning
        if hasattr(agent, 'learn'):
            agent.learn()
            if hasattr(agent, 'apply_her') and not (label.startswith("ANN") or label.startswith("RL-only")):
                agent.apply_her(episode_transitions)
        
        results['rewards'].append(float(ep_reward))
        results['success_rates'].append(float(success_count / (ep + 1)))
        results['steps'].append(step + 1)
        results['spikes'].append(int(ep_spikes))
        
        # Track sample efficiency (Episodes to 60%)
        rolling_sr = np.mean(results['success_rates'][-10:]) if len(results['success_rates']) >= 10 else results['success_rates'][-1]
        if results['episodes_to_60'] is None and rolling_sr >= 0.6:
            results['episodes_to_60'] = ep + 1
            print(f"--- {label} reached 60% success at episode {ep + 1} ---")

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1}/{n_episodes} | Avg Reward: {np.mean(results['rewards'][-10:]):.2f} | Success: {results['success_rates'][-1]:.2f}")
            
    return results

def main():
    # results relative to this script: ../results/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    
    n_seeds = 5
    n_episodes = 500 
    max_steps = 80
    
    out_path = os.path.join(res_dir, "comprehensive_eval.json")
    all_results = {}
    
    if os.path.exists(out_path):
        print(f"Loading existing progress from {out_path}")
        try:
            with open(out_path, "r") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print("Failed to load JSON (might be empty/corrupt). Starting fresh.")
            
    def run_if_needed(label, env, agent):
        if label in all_results:
            print(f"Skipping {label} (already completed)")
            return
        all_results[label] = run_experiment(env, agent, n_episodes, max_steps, label)
        with open(out_path, "w") as f:
            json.dump(all_results, f)
        print(f"Saved progress for {label}.")

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Main OrganoidRL
        print(f"\n=== SEED {seed} ===")
        env = OrganoidEnv()
        agent = DQNAgent(obs_dim=21, n_actions=8)
        label = f"OrganoidRL_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

        # 2. ANN Baseline (MLP on SkeletonEnv)
        env = SkeletonEnv(obs_mode='core') # 11D obs
        agent = MLPBaselineAgent(obs_dim=11, n_actions=8)
        label = f"ANN_Baseline_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

        # 3. RL-only Baseline (Dueling DQN on SkeletonEnv)
        env = SkeletonEnv(obs_mode='full') # 21D obs with dummy neurons
        agent = DQNAgent(obs_dim=21, n_actions=8)
        label = f"RL-only_Baseline_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

        # 4. No Dual-Trace Ablation
        env = OrganoidEnv(use_dual_trace=False)
        agent = DQNAgent(obs_dim=21, n_actions=8)
        label = f"No_DualTrace_Ablation_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

        # 5. No Motor Mapping (Diffuse) Ablation
        env = OrganoidEnv(use_motor_mapping=False)
        agent = DQNAgent(obs_dim=21, n_actions=8)
        label = f"No_MotorMapping_Ablation_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

        # 6. Simple SNN (No dual-trace, No GAR)
        env = OrganoidEnv(use_dual_trace=False, use_stabilizer=False)
        agent = DQNAgent(obs_dim=21, n_actions=8)
        label = f"SimpleSNN_Baseline_Seed{seed}"
        run_if_needed(label, env, agent)
        env.close()

    print(f"\nAll results saved to {out_path}")

if __name__ == "__main__":
    main()
