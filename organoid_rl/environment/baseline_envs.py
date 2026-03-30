import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SkeletonEnv(gym.Env):
    """
    A direct-control version of OrganoidEnv that bypasses the SNN.
    Used for RL-only and ANN baselines.
    """
    def __init__(self, obs_mode='full'):
        super(SkeletonEnv, self).__init__()
        
        self.obs_mode = obs_mode # 'full' (21D) or 'core' (11D)
        
        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)
        
        # Observation space
        if obs_mode == 'full':
            self.obs_dim = 21
        else:
            self.obs_dim = 11
            
        self.observation_space = spaces.Box(low=-1, high=2, shape=(self.obs_dim,), dtype=np.float32)
        
        # Task state
        self.cursor_pos = np.array([0.1, 0.9])
        self.target_pos = np.array([0.8, 0.8])
        self.targets = [np.array([0.8, 0.8]), np.array([0.2, 0.2])]
        self.active_target_idx = 0
        self.prev_cursor_pos = self.cursor_pos.copy()
        self.prev_dist = 0.0
        self.goal_radius = 0.05
        self.difficulty_stage = 4 # Default to full task
        
        self.obstacles = [
            {'type': 'circle', 'pos': np.array([0.5, 0.5]), 'radius': 0.15},
            {'type': 'box', 'bounds': [0.2, 0.3, 0.4, 0.8]}
        ]
        
        self.visited_cells = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor_pos = np.array([0.1, 0.9])
        self.prev_cursor_pos = self.cursor_pos.copy()
        self.active_target_idx = np.random.choice([0, 1]) if self.difficulty_stage > 2 else 0
        self.target_pos = self.targets[self.active_target_idx]
        self.prev_dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        self.visited_cells = set()
        return self._get_obs(), {}

    def step(self, action):
        # Move cursor directly
        # Mapping same as _apply_action in core.py but with speed instead of stimulation
        speed = 0.05
        dx, dy = 0, 0
        if action == 0: dy = speed
        elif action == 1: dy = -speed
        elif action == 2: dx = -speed
        elif action == 3: dx = speed
        elif action == 4: dy, dx = speed, speed
        elif action == 5: dy, dx = speed, -speed
        elif action == 6: dy, dx = -speed, speed
        elif action == 7: dy, dx = -speed, -speed
        
        self.prev_cursor_pos = self.cursor_pos.copy()
        self.cursor_pos += np.array([dx, dy])
        self.cursor_pos = np.clip(self.cursor_pos, 0, 1)
        
        reward = self._calculate_reward()
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        terminated = dist < self.goal_radius
        truncated = False
        
        self.prev_dist = dist
        
        return self._get_obs(), reward, terminated, truncated, {'distance': float(dist)}

    def _get_obs(self):
        # Core features: cursor (2), target (2), context (1), proximity (4), velocity (2) = 11D
        prox = self._compute_obstacle_proximity()
        velocity = (self.cursor_pos - self.prev_cursor_pos).astype(np.float32)
        spatial = np.concatenate([self.cursor_pos, self.target_pos, [float(self.active_target_idx)]]).astype(np.float32)
        core_obs = np.concatenate([spatial, prox, velocity])
        
        if self.obs_mode == 'full':
            # Add 10D dummy neural activity for consistency with DQNAgent
            neural_obs = np.zeros(10, dtype=np.float32)
            return np.concatenate([neural_obs, core_obs])
        return core_obs

    def _calculate_reward(self):
        # Mirroring the logic from core.py
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        reward = -0.05
        if dist < self.goal_radius: reward += 50.0
        elif dist < 0.15: reward += 5.0
        elif dist < 0.3: reward += 1.0
        
        gamma = 0.99
        phi_new = -dist
        phi_old = -self.prev_dist
        reward += 10.0 * (gamma * phi_new - phi_old)
        
        # Obstacle penalties
        is_in_obstacle = False
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                if np.linalg.norm(self.cursor_pos - obs['pos']) < obs['radius']: is_in_obstacle = True
            elif obs['type'] == 'box':
                b = obs['bounds']
                if b[0] <= self.cursor_pos[0] <= b[1] and b[2] <= self.cursor_pos[1] <= b[3]: is_in_obstacle = True
        
        if is_in_obstacle: reward -= 15.0
        
        # Wall penalty
        if np.any(self.cursor_pos <= 0.01) or np.any(self.cursor_pos >= 0.99): reward -= 3.0
            
        return reward

    def _compute_obstacle_proximity(self):
        # Simplified proximity calculation
        prox = np.ones(4, dtype=np.float32)
        directions = [np.array([0, 0.05]), np.array([0, -0.05]), np.array([0.05, 0]), np.array([-0.05, 0])]
        for i, d in enumerate(directions):
            for step in range(1, 10):
                probe = self.cursor_pos + d * step
                for obs in self.obstacles:
                    if obs['type'] == 'circle' and np.linalg.norm(probe - obs['pos']) < obs['radius']:
                        prox[i] = min(prox[i], step * 0.05)
                    elif obs['type'] == 'box':
                        b = obs['bounds']
                        if b[0] <= probe[0] <= b[1] and b[2] <= probe[1] <= b[3]:
                            prox[i] = min(prox[i], step * 0.05)
        return prox
