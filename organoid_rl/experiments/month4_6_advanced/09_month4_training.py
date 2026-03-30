import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from organoid_rl.environment.core import OrganoidEnv

class QLearningAgent:
    def __init__(self, n_actions, n_obs_bins=8):
        self.n_actions = n_actions
        self.n_obs_bins = n_obs_bins
        self.q_table = {} 
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

    def get_state(self, obs):
        # We still use relative distance as the state for the Q-table
        # The organoid's internal SDM handles the "neural" representation
        rel_x = obs[12] - obs[10]
        rel_y = obs[13] - obs[11]
        
        dist = np.sqrt(rel_x**2 + rel_y**2)
        angle = np.arctan2(rel_y, rel_x)
        
        dist_bin = int(np.digitize(dist, np.linspace(0, 1, self.n_obs_bins)))
        angle_bin = int(np.digitize(angle, np.linspace(-np.pi, np.pi, self.n_obs_bins)))
        
        return (dist_bin, angle_bin)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
            
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def apply_inhibitory_homeostasis(env):
    """
    Phase 4: Updated homeostasis for structured networks.
    """
    # Mean weight of all excitatory synapses (assume index 0 to 3 are E-E/E-I etc)
    # For structured: index 0 is SDM->Hidden, index 1 is Hidden->Motor
    ee_weights = []
    for i in range(len(env.synapses) - 1): # Last one is inhibitory
         ee_weights.append(np.mean(np.array(env.synapses[i].w)))
    
    mean_ee_w = np.mean(ee_weights)
    ideal_ratio = 1.5
    
    # Last synapse is inhibitory (index -1)
    S_INH = env.synapses[-1]
    current_ie_w = np.array(S_INH.w)
    target_ie_w = mean_ee_w * ideal_ratio
    
    updated_ie_w = current_ie_w + 0.1 * (target_ie_w - current_ie_w)
    S_INH.w = np.clip(updated_ie_w, 0, 100.0)

def train_organoid(episodes=100, use_sdm=True, use_morphology=True, use_dual_trace=True, use_homeostasis=True, tag="full"):
    print(f"Starting Month 4 Training [{tag}]...")
    env = OrganoidEnv(use_sdm=use_sdm, use_morphology=use_morphology, use_dual_trace=use_dual_trace)
    agent = QLearningAgent(n_actions=env.action_space.n, n_obs_bins=8)
    
    steps_per_episode = 40
    rewards_history = []
    goals_reached = 0
    efficiency_history = []
    
    results = {
        'tag': tag,
        'episodes': episodes,
        'config': {
            'use_sdm': use_sdm,
            'use_morphology': use_morphology,
            'use_dual_trace': use_dual_trace,
            'use_homeostasis': use_homeostasis
        }
    }
    
    start_time = time.time()
    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.get_state(obs)
        total_reward = 0
        steps_taken = 0
        
        for s in range(steps_per_episode):
            action = agent.choose_action(state)
            next_obs, reward, done, trunc, info = env.step(action)
            steps_taken += 1
            
            next_state = agent.get_state(next_obs)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if done:
                goals_reached += 1
                efficiency_history.append(steps_taken)
                break
        
        if not done:
            efficiency_history.append(steps_per_episode)
            
        if use_homeostasis:
            apply_inhibitory_homeostasis(env)
            
        rewards_history.append(total_reward)
        
        if ep % 5 == 0:
             print(f"[{tag}] Ep {ep} | Rwd: {total_reward:.2f} | Goals: {goals_reached} | Dist: {info['distance']:.3f} | Eps: {agent.epsilon:.2f}", flush=True)

        if ep % 10 == 0:
            # Temporary save for recovery/plotting
            results.update({
                'rewards': rewards_history,
                'efficiency': efficiency_history,
                'goals_reached': goals_reached,
                'current_ep': ep + 1
            })
            with open(f"experiments/results/results_{tag}.json", 'w') as f:
                json.dump(results, f)
    
    end_time = time.time()
    results.update({
        'rewards': rewards_history,
        'efficiency': efficiency_history,
        'goals_reached': goals_reached,
        'duration': end_time - start_time
    })
    
    save_dir = "experiments/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/results_{tag}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Training Complete. Success Rate: {goals_reached/episodes:.1%}. Results saved to {save_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--no_sdm", action="store_true")
    parser.add_argument("--no_morphology", action="store_true")
    parser.add_argument("--no_dual_trace", action="store_true")
    parser.add_argument("--no_homeostasis", action="store_true")
    parser.add_argument("--tag", type=str, default="full")
    args = parser.parse_args()
    
    train_organoid(
        episodes=args.episodes,
        use_sdm=not args.no_sdm,
        use_morphology=not args.no_morphology,
        use_dual_trace=not args.no_dual_trace,
        use_homeostasis=not args.no_homeostasis,
        tag=args.tag
    )
