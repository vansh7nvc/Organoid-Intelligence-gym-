"""
Month 6: Dueling Double DQN Agent with Prioritized Experience Replay,
Noisy Networks, N-step Returns, and Hindsight Experience Replay (HER).

Target: 85-95% success rate on multi-goal organoid navigation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """Factorized Gaussian noise for exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# --- Dueling DQN Network ---
class DuelingDQN(nn.Module):
    """Dueling architecture with noisy layers for exploration."""
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value stream (V)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )
        
        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, n_actions)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- Prioritized Experience Replay ---
class SumTree:
    """Binary tree for efficient priority-based sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """Prioritized experience replay with proportional sampling."""
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0
    
    @property
    def beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, transition)
    
    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data == 0:  # Uninitialized
                continue
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        if len(batch) == 0:
            return None, None, None
        
        # Importance sampling weights
        total = self.tree.total()
        probs = np.array(priorities) / total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()
        
        self.frame += 1
        return batch, idxs, torch.FloatTensor(weights)
    
    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


# --- DQN Agent ---
class DQNAgent:
    """
    Dueling Double DQN with PER, Noisy Nets, N-step returns, and HER.
    """
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99, tau=0.005,
                 n_step=3, batch_size=64, buffer_size=50000,
                 her_k=4, goal_dim=2):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.batch_size = batch_size
        self.her_k = her_k
        self.goal_dim = goal_dim  # Dimensions of goal in obs (target_x, target_y)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        # PER
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Metrics
        self.train_step = 0
        self.losses = []
    
    def choose_action(self, state):
        """Select action using noisy network (no epsilon needed)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)
        self.online_net.train()
        return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store with N-step return calculation."""
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Calculate n-step return
        reward_n = sum(self.gamma ** i * t.reward for i, t in enumerate(self.n_step_buffer))
        state_0 = self.n_step_buffer[0].state
        action_0 = self.n_step_buffer[0].action
        next_state_n = self.n_step_buffer[-1].next_state
        done_n = self.n_step_buffer[-1].done
        
        self.replay_buffer.push(state_0, action_0, reward_n, next_state_n, done_n)
    
    def flush_n_step_buffer(self):
        """Flush remaining transitions at episode end."""
        while len(self.n_step_buffer) > 0:
            reward_n = sum(self.gamma ** i * t.reward for i, t in enumerate(self.n_step_buffer))
            state_0 = self.n_step_buffer[0].state
            action_0 = self.n_step_buffer[0].action
            next_state_n = self.n_step_buffer[-1].next_state
            done_n = self.n_step_buffer[-1].done
            self.replay_buffer.push(state_0, action_0, reward_n, next_state_n, done_n)
            self.n_step_buffer.popleft()
    
    def apply_her(self, episode_transitions):
        """
        Hindsight Experience Replay: Replay failed episodes
        with achieved positions as substitute goals.
        """
        if len(episode_transitions) < 2:
            return
        
        for t_idx, (state, action, _, next_state, done) in enumerate(episode_transitions):
            # Sample k future achieved positions as hindsight goals
            future_indices = list(range(t_idx + 1, len(episode_transitions)))
            if len(future_indices) == 0:
                continue
            
            k = min(self.her_k, len(future_indices))
            sampled = random.sample(future_indices, k)
            
            for future_idx in sampled:
                # The achieved goal = cursor position in the future state
                future_state = episode_transitions[future_idx][3]  # next_state
                achieved_goal = future_state[10:12]  # cursor_pos from obs
                
                # Substitute the target in the observation
                her_state = state.copy()
                her_state[12:14] = achieved_goal  # Replace target_pos
                
                her_next_state = next_state.copy()
                her_next_state[12:14] = achieved_goal
                
                # Recompute reward: distance from cursor to the new "goal"
                cursor_pos = her_next_state[10:12]
                dist = np.linalg.norm(cursor_pos - achieved_goal)
                her_reward = 10.0 if dist < 0.1 else -0.05
                her_done = dist < 0.05
                
                self.replay_buffer.push(her_state, action, her_reward, her_next_state, her_done)
    
    def learn(self):
        """Train on a batch from PER."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        result = self.replay_buffer.sample(self.batch_size)
        if result[0] is None:
            return None
        batch, idxs, weights = result
        weights = weights.to(self.device)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        
        # Double DQN: Online network selects action, target evaluates
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)
        
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Weighted Huber loss
        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(idxs, td_errors)
        
        # Soft update target network
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)
        
        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        self.train_step += 1
        self.losses.append(loss.item())
        return loss.item()

    def save_checkpoint(self, path):
        """Save network weights."""
        torch.save({
            'online_state_dict': self.online_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """Load network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

