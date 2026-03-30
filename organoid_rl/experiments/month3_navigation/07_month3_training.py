import time
from environment.core import OrganoidEnv
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class QLearningAgent:
    def __init__(self, n_actions, n_obs_bins=5):
        self.n_actions = n_actions
        self.n_obs_bins = n_obs_bins
        self.q_table = {} 
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

    def get_state(self, obs):
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
        
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.998)

def apply_inhibitory_homeostasis(env):
    """
    Month 3 Objective: Implement inhibitory homeostasis to balance excitatory strengthening.
    This scales the baseline of I-E synapses relative to the mean E-E weights
    to prevent runaway excitation.
    """
    mean_ee_w = np.mean(np.array(env.synapses[0].w))
    # We dynamically adjust baseline w0 of I-E (synapses[2]) based on E-E weights
    ideal_ratio = 1.5 # I-E should be 1.5x stronger than E-E average to maintain balance
    
    current_ie_w = np.array(env.synapses[2].w)
    target_ie_w = mean_ee_w * ideal_ratio
    
    # Soft update towards target
    updated_ie_w = current_ie_w + 0.1 * (target_ie_w - current_ie_w)
    env.synapses[2].w = np.clip(updated_ie_w, 0, 100.0)

def train_organoid():
    print("Initializing Month 3 Training Curriculum...")
    env = OrganoidEnv()
    agent = QLearningAgent(n_actions=env.action_space.n, n_obs_bins=8)
    
    episodes = 100
    steps_per_episode = 40
    rewards_history = []
    weight_history = []
    efficiency_history = [] # Steps taken per goal reached
    
    goals_reached = 0
    
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
            efficiency_history.append(steps_per_episode) # Failed to reach
            
        # Apply homeostasis
        apply_inhibitory_homeostasis(env)
        
        mean_w = np.mean(np.array(env.synapses[0].w))
        weight_history.append(mean_w)
        rewards_history.append(total_reward)
        
        if ep % 5 == 0:
            print(f"Episode {ep}/{episodes} | Rwd: {total_reward:.2f} | Eps: {agent.epsilon:.2f} | Dist: {info['distance']:.3f} | Steps: {steps_taken} | MeanW: {mean_w:.2f}")

    end_time = time.time()
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(rewards_history)
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 4, 2)
    plt.plot(weight_history, color='orange')
    plt.title("Mean E-E Weight")
    plt.xlabel("Episode")
    
    plt.subplot(1, 4, 3)
    plt.plot(efficiency_history, color='green')
    plt.title("Navigation Efficiency")
    plt.xlabel("Episode")
    plt.ylabel("Steps to target")
    
    plt.subplot(1, 4, 4)
    plt.scatter(info['target_pos'][0], info['target_pos'][1], marker='*', s=200, color='gold', label='Target')
    plt.scatter(info['cursor_pos'][0], info['cursor_pos'][1], marker='o', color='blue', label='Final Cursor')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Final State")
    plt.legend()
    
    plt.tight_layout()
    plot_path = "experiments/07_month3_results.png"
    plt.savefig(plot_path)
    
    print(f"Training Complete in {end_time - start_time:.2f}s.")
    print(f"Goals Reached: {goals_reached}/{episodes}")
    print(f"Results saved to {plot_path}")

if __name__ == "__main__":
    train_organoid()
