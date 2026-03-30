from environment.core import OrganoidEnv
import numpy as np
import matplotlib.pyplot as plt
import os

class QLearningAgent:
    def __init__(self, n_actions, n_obs_bins=5):
        self.n_actions = n_actions
        self.n_obs_bins = n_obs_bins
        # Simplified state discretization for the 14-dim observation
        # We'll focus on the relative x, y to target
        self.q_table = {} 
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

    def get_state(self, obs):
        # obs[10:12] = cursor, obs[12:14] = target
        rel_x = obs[12] - obs[10]
        rel_y = obs[13] - obs[11]
        
        # Polar Coordinates for better navigation state
        dist = np.sqrt(rel_x**2 + rel_y**2)
        angle = np.arctan2(rel_y, rel_x)
        
        # Binning
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
        
        # Slower epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.998)

def train_organoid():
    print("Initializing Training Session...")
    env = OrganoidEnv()
    agent = QLearningAgent(n_actions=env.action_space.n, n_obs_bins=8)
    
    episodes = 100
    steps_per_episode = 40
    rewards_history = []
    weight_history = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.get_state(obs)
        total_reward = 0
        
        for s in range(steps_per_episode):
            action = agent.choose_action(state)
            next_obs, reward, done, trunc, info = env.step(action)
            
            next_state = agent.get_state(next_obs)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Telemetry: Track mean E-E weight
        mean_w = np.mean(np.array(env.synapses[0].w))
        weight_history.append(mean_w)
        rewards_history.append(total_reward)
        
        if ep % 5 == 0:
            print(f"Episode {ep}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Dist: {info['distance']:.3f} | MeanW: {mean_w:.2f}")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 3, 2)
    plt.plot(weight_history, color='orange')
    plt.title("Synaptic Weight Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Mean w (E-E)")
    
    plt.subplot(1, 3, 3)
    plt.scatter(info['target_pos'][0], info['target_pos'][1], marker='*', s=200, color='gold', label='Target')
    plt.scatter(info['cursor_pos'][0], info['cursor_pos'][1], marker='o', color='blue', label='Final Cursor')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Final State")
    plt.legend()
    
    plt.tight_layout()
    plot_path = "experiments/06_training_results.png"
    plt.savefig(plot_path)
    print(f"Training Complete. Results saved to {plot_path}")

if __name__ == "__main__":
    train_organoid()
