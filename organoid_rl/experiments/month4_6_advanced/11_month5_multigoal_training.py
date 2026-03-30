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
        self.epsilon_decay = 0.98 # Decays per episode now
        self.min_epsilon = 0.05

    def get_state(self, obs):
        # State: [Rel_X, Rel_Y, Context]
        # Context is the last element of obs
        rel_x = obs[12] - obs[10]
        rel_y = obs[13] - obs[11]
        context = int(obs[14])
        
        dist = np.sqrt(rel_x**2 + rel_y**2)
        angle = np.arctan2(rel_y, rel_x)
        
        dist_bin = int(np.digitize(dist, np.linspace(0, 1, self.n_obs_bins)))
        angle_bin = int(np.digitize(angle, np.linspace(-np.pi, np.pi, self.n_obs_bins)))
        
        return (dist_bin, angle_bin, context)

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

def apply_inhibitory_homeostasis(env):
    """Phase 4/5: Updated homeostasis for structured networks."""
    ee_weights = []
    for i in range(len(env.synapses) - 1): # Last one is inhibitory
         ee_weights.append(np.mean(np.array(env.synapses[i].w)))
    
    mean_ee_w = np.mean(ee_weights)
    ideal_ratio = 1.5
    
    S_INH = env.synapses[-1]
    current_ie_w = np.array(S_INH.w)
    target_ie_w = mean_ee_w * ideal_ratio
    
    updated_ie_w = current_ie_w + 0.1 * (target_ie_w - current_ie_w)
    S_INH.w = np.clip(updated_ie_w, 0, 100.0)

def train_organoid(episodes=150, tag="month5_multigoal"):
    print(f"Starting Month 5 Training: Multi-Goal [{tag}]...")
    env = OrganoidEnv()
    agent = QLearningAgent(n_actions=env.action_space.n, n_obs_bins=8)
    
    steps_per_episode = 100 # Increased for Phase 5 flexibility
    rewards_history = []
    goals_reached = 0
    context_success = {0: 0, 1: 0} # Track per goal
    
    start_time = time.time()
    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.get_state(obs)
        context = int(obs[14])
        total_reward = 0
        
        for s in range(steps_per_episode):
            action = agent.choose_action(state)
            next_obs, reward, done, trunc, info = env.step(action)
            
            next_state = agent.get_state(next_obs)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if done:
                goals_reached += 1
                context_success[context] += 1
                break
        
        # Apply homeostasis
        apply_inhibitory_homeostasis(env)
        rewards_history.append(total_reward)
        
        # Epsilon decay per episode
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        
        if ep % 10 == 0:
            print(f"[{tag}] Ep {ep} | Rwd: {total_reward:.2f} | Goals: {goals_reached} (C0:{context_success[0]} C1:{context_success[1]}) | Eps: {agent.epsilon:.3f}", flush=True)

    end_time = time.time()
    
    results = {
        'tag': tag,
        'episodes': episodes,
        'goals_reached': goals_reached,
        'context_success': context_success,
        'rewards': rewards_history,
        'duration': end_time - start_time
    }
    
    save_dir = "experiments/results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/results_{tag}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(rewards_history)
    plt.title(f"Month 5: Multi-Goal Learning Trace [{tag}]")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plot_path = f"experiments/results/month5_results.png"
    plt.savefig(plot_path)
    
    print(f"Training Complete. Success Rate: {goals_reached/episodes:.1%}. Results saved to {save_path}")
    return results

if __name__ == "__main__":
    train_organoid(episodes=250)
