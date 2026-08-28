"""
OrganoidEnv: A Gymnasium-compatible RL environment backed by a Brian2
spiking neural network.  The environment simulates an organoid with
Metabolic-Izhikevich neurons, Sparse Distributed Memory, clustered
motor output, and curriculum-based obstacle navigation.

Author: Vansh Sharma
License: MIT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from brian2 import *
from .neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
from .stimulator import stabilize_network_activity
from .rewards import apply_dopamine

# High-performance Cython configuration: use fast Cython/GCC on Linux (Colab), fallback to numpy on Windows
import platform
if platform.system() == "Linux":
    prefs.codegen.target = 'cython'
else:
    prefs.codegen.target = 'numpy'

class OrganoidEnv(gym.Env):
    """
    Gymnasium integration for the Organoid RL project.
    Simulates a network of neurons and exposes an interface for an agent to interact.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, use_sdm=True, use_morphology=True, use_dual_trace=True, use_stabilizer=True, use_motor_mapping=True):
        super(OrganoidEnv, self).__init__()
        
        # --- StudySlate / Ablation Flags ---
        self.use_sdm = use_sdm
        self.use_morphology = use_morphology
        self.use_dual_trace = use_dual_trace
        self.use_stabilizer = use_stabilizer
        self.use_motor_mapping = use_motor_mapping
        
        # --- Environment Configuration ---
        self.dt = 0.1 * ms  # Faster timestep (Phase 5 Optimization)
        self.step_duration = 100 * ms
        self.n_neurons = 500
        self.n_obs_groups = 10
        self.n_act_groups = 8
        self.tau_trace = 50*ms

        # --- Action Space ---
        self.action_space = spaces.Discrete(self.n_act_groups)

        # --- Observation Space (Phase 6: 21D) ---
        # 10 (spike rates) + 2 (cursor) + 2 (target) + 1 (context) + 4 (obstacle proximity) + 2 (velocity)
        self.obs_dim = 21
        self.observation_space = spaces.Box(low=-1, high=2, shape=(self.obs_dim,), dtype=np.float32)

        # --- Curriculum Learning (Phase 6) ---
        self.difficulty_stage = 1  # 1=easy, 2=obstacles, 3=multi-goal, 4=full
        self.goal_radius = 0.15   # Relaxed initially
        
        # --- Target Task Setup ---
        self.targets = [np.array([0.8, 0.8]), np.array([0.2, 0.2])]
        self.active_target_idx = 0
        self.target_pos = self.targets[self.active_target_idx]
        self.obstacles_full = [
            {'type': 'circle', 'pos': np.array([0.5, 0.5]), 'radius': 0.15},
            {'type': 'box', 'bounds': [0.2, 0.3, 0.4, 0.8]}
        ]
        self.obstacles = []  # Empty until stage 2
        
        # --- Velocity Tracking ---
        self.prev_cursor_pos = np.array([0.1, 0.9])
        self._cached_prox = np.ones(4, dtype=np.float32)  # Cache for proximity sensor
        
        # --- Structural Plasticity Tracking ---
        self.synapse_counts = []
        self.visited_cells = set()

        # --- Brian2 Network Setup ---
        self.network = None # Will be initialized in reset()
        self.neurons = None
        self.synapses = []
        self.monitors = {}
        
        # --- Physical Morphology (Phase 4/5) ---
        self.n_sdm = 256
        self.output_neuron_start = 400 # Last 100 neurons are "motor"
        self.n_motor = 100
        self.n_hidden = self.n_neurons - self.n_sdm - self.n_motor
        
        # Motor quadrants (Clustered) - Adjusted for 500 neurons
        self.motor_up = (400, 425)
        self.motor_down = (425, 450)
        self.motor_left = (450, 475)
        self.motor_right = (475, 500)

        # --- Virtual Motor System ---
        self.cursor_pos = np.array([0.1, 0.9])
        self.prev_dist = 0.0
        
        # --- SDM Fixed Projection ---
        np.random.seed(42)
        self.W_sdm = np.random.choice([0, 1], size=(15, self.n_sdm), p=[0.9, 0.1])  # SDM uses 15D core obs
        
        # Build the initial network structure
        self._build_network()

    def _build_network(self):
        """Constructs the Brian2 network objects."""
        start_scope()
        
        # 1. Neurons (Metabolic Izhikevich Model)
        eqs = get_metabolic_izhikevich_eqs()
        threshold = 'v >= 30 and E > 0'
        reset = 'v = c; u += d; E -= spike_cost'
        
        self.neurons = NeuronGroup(self.n_neurons, eqs, threshold=threshold, reset=reset, method='euler', refractory=2*ms)
        
        # Tuned parameters
        N_exc = int(0.8 * self.n_neurons)
        P_exc = self.neurons[:N_exc]
        P_inh = self.neurons[N_exc:]
        
        # Excitatory (RS)
        P_exc.a = 0.02; P_exc.b = 0.2; P_exc.c = -65.0; P_exc.d = 8.0
        P_exc.tau_recovery = 1000*ms
        
        # Inhibitory (FS)
        P_inh.a = 0.1; P_inh.b = 0.2; P_inh.c = -65.0; P_inh.d = 2.0
        P_inh.tau_recovery = 500*ms
        
        self.neurons.u = 'b * v'
        self.neurons.v = -65.0
        self.neurons.E = 1.0
        self.neurons.spike_cost = 0.1
        self.neurons.I = 0.0 

        # 2. Synapses (Ablation Capable)
        if self.use_dual_trace:
            syn_eqs = '''
            dw/dt = (w0 - w) / tau_decay : 1 (clock-driven)
            dTrace1/dt = -Trace1 / tau_trace1 : 1 (clock-driven)
            dTrace2/dt = -Trace2 / tau_trace2 : 1 (clock-driven)
            w0 : 1 (shared)
            '''
            on_pre_plus = 'v_post += w; Trace1 += 1; Trace2 += 1'
            on_pre_minus = 'v_post -= w; Trace1 += 1; Trace2 += 1'
            ns = {'tau_trace1': 100*ms, 'tau_trace2': 2500*ms, 'tau_decay': 10000*ms}
        else:
            syn_eqs = '''
            dw/dt = (w0 - w) / tau_decay : 1 (clock-driven)
            dTrace/dt = -Trace / tau_trace : 1 (clock-driven)
            w0 : 1 (shared)
            '''
            on_pre_plus = 'v_post += w; Trace += 1'
            on_pre_minus = 'v_post -= w; Trace += 1'
            ns = {'tau_trace': 50*ms, 'tau_decay': 10000*ms}

        # Morphology Logic
        def create_syn(pre, post, p, w=15.0, is_inhibitory=False):
            op = on_pre_minus if is_inhibitory else on_pre_plus
            S = Synapses(pre, post, model=syn_eqs, on_pre=op, method='euler', namespace=ns)
            S.connect(p=p)
            S.w = w; S.w0 = w
            if self.use_dual_trace:
                S.Trace1 = 0.0; S.Trace2 = 0.0
            else:
                S.Trace = 0.0
            return S

        if self.use_morphology:
             # Structured Subgroups
             SDM_grp = self.neurons[:self.n_sdm]
             Hidden_grp = self.neurons[self.n_sdm:self.output_neuron_start]
             Motor_grp = self.neurons[self.output_neuron_start:]
             
             self.synapses.append(create_syn(SDM_grp, Hidden_grp, p=0.15, w=20.0))
             self.synapses.append(create_syn(Hidden_grp, Motor_grp, p=0.15, w=20.0))
             self.synapses.append(create_syn(Hidden_grp, Hidden_grp, p=0.05, w=10.0))
             self.synapses.append(create_syn(Motor_grp, Motor_grp, p=0.1, w=10.0))
             self.synapses.append(create_syn(P_inh, self.neurons, p=0.1, w=25.0, is_inhibitory=True))
        else:
             # Random "Soup" Mesh (Month 1-3 style)
             self.synapses.append(create_syn(P_exc, P_exc, p=0.1))
             self.synapses.append(create_syn(P_exc, P_inh, p=0.1))
             self.synapses.append(create_syn(P_inh, P_exc, p=0.1, is_inhibitory=True))
             self.synapses.append(create_syn(P_inh, P_inh, p=0.1, is_inhibitory=True))

        # 3. Background Noise
        self.bg_noise = PoissonInput(self.neurons, 'v', 1, 10*Hz, weight=4) # Lowered for clarity
        
        # 4. Initialize Network
        self.network = Network(self.neurons, *self.synapses, self.bg_noise)
        self.network.store('initial')
        
        self.spike_mon = SpikeMonitor(self.neurons, name='spike_mon')
        self.network.add(self.spike_mon)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # Reset monitors to prevent memory growth and slowdown
        # We MUST remove it before restore() to avoid KeyError
        if self.spike_mon in self.network.objects:
            self.network.remove(self.spike_mon)
            
        # Restore simulation to stored initial state (t=0, v=-65, etc)
        self.network.restore('initial')
        
        # Add a fresh monitor
        self.spike_mon = SpikeMonitor(self.neurons, name='spike_monitor')
        self.monitors['spikes'] = self.spike_mon
        self.network.add(self.spike_mon)
        
        # Jitter initial voltage for diversity
        self.neurons.v = '-65 + 10*rand()' 
        self.neurons.u = 'b * v'
        self.neurons.E = 1.0
        self.neurons.I = 0
        
        # Run initialization
        self.network.run(10*ms)

        # Reset motor and task state
        self.cursor_pos = np.array([0.1, 0.9])
        self.prev_cursor_pos = np.array([0.1, 0.9])
        self.visited_cells = set()
        
        # Curriculum-aware target selection
        if self.difficulty_stage <= 2:
            self.active_target_idx = 0  # Single target in early stages
        else:
            self.active_target_idx = np.random.choice([0, 1])
        self.target_pos = self.targets[self.active_target_idx]
        self.prev_dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        
        # Curriculum-aware obstacles
        if self.difficulty_stage >= 2:
            self.obstacles = self.obstacles_full
        else:
            self.obstacles = []
        
        # Curriculum-aware goal radius
        if self.difficulty_stage == 1:
            self.goal_radius = 0.15
        elif self.difficulty_stage == 2:
            self.goal_radius = 0.10
        elif self.difficulty_stage == 3:
            self.goal_radius = 0.10
        else:
            self.goal_radius = 0.05
        
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        info = {
            'cursor_pos': self.cursor_pos.tolist(),
            'target_pos': self.target_pos.tolist(),
            'distance': float(dist),
            'total_spikes': len(self.spike_mon.t),
        }
        return self._get_obs(), info

    def step(self, action):
        """Executes one time step within the environment."""
        
        # 1. Reset Current
        self.neurons.I = 0
        
        # Stabilization (GAR)
        if self.use_stabilizer:
            stabilize_network_activity(self)
        
        # 3. Apply Action (Stimulation)
        self._apply_action(action)
        
        # 4. SDM Expansion (Sensory Drive) — uses 15D core obs only
        if self.use_sdm:
            self._stimulate_sdm(self._get_obs())

        # 5. Run Simulation
        self.network.run(self.step_duration)
        
        # 6. Update Motor Output FIRST (before reading obs)
        self._update_motor_output()
        
        # 7. Recompute proximity cache after movement
        self._cached_prox = self._compute_obstacle_proximity()
        
        # 8. Get Observation (uses cached proximity)
        obs = self._get_obs()
        
        # 10. Calculate Reward
        reward = self._calculate_reward()
        
        # 11. Update state
        self.prev_dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        
        # 12. Apply Dopamine Learning
        apply_dopamine(self.synapses, reward)
        
        # 13. Check Done (uses curriculum goal_radius)
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        terminated = dist < self.goal_radius
        truncated = False
        
        info = {
            'cursor_pos': self.cursor_pos.tolist(),
            'target_pos': self.target_pos.tolist(),
            'distance': float(dist),
            'total_spikes': len(self.spike_mon.t),
            'last_reward': reward,
            'success': terminated
        }
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        """
        Phase 6: Potential-Based Reward Shaping + Waypoints + Exploration Bonus.
        """
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        reward = -0.05  # Small step cost
        
        # --- Goal Reward ---
        if dist < self.goal_radius:
            reward += 50.0  # Big bonus for reaching goal
        elif dist < 0.15:
            reward += 5.0
        elif dist < 0.3:
            reward += 1.0
            
        # --- Potential-Based Shaping: F = γ·Φ(s') - Φ(s) ---
        # Φ(s) = -distance_to_goal (closer = higher potential)
        gamma_shape = 0.99
        phi_new = -dist
        phi_old = -self.prev_dist
        shaping = gamma_shape * phi_new - phi_old
        reward += 10.0 * shaping
        
        # --- Waypoint Bonus ---
        initial_dist = np.linalg.norm(np.array([0.1, 0.9]) - self.target_pos)
        if dist < initial_dist * 0.5 and self.prev_dist >= initial_dist * 0.5:
            reward += 5.0  # Crossed the halfway mark
            
        # --- Exploration Bonus ---
        cell = (int(self.cursor_pos[0] * 10), int(self.cursor_pos[1] * 10))
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            reward += 0.5
            
        # --- Obstacle Penalties ---
        is_in_obstacle = False
        min_obstacle_dist = 1.0
        for obs_item in self.obstacles:
            if obs_item['type'] == 'circle':
                d_obs = np.linalg.norm(self.cursor_pos - np.array(obs_item['pos'])) - obs_item['radius']
                min_obstacle_dist = min(min_obstacle_dist, d_obs)
                if d_obs < 0:
                    is_in_obstacle = True
            elif obs_item['type'] == 'box':
                b = obs_item['bounds']
                if b[0] <= self.cursor_pos[0] <= b[1] and b[2] <= self.cursor_pos[1] <= b[3]:
                    is_in_obstacle = True
                    min_obstacle_dist = 0.0
        
        if is_in_obstacle:
            reward -= 15.0
        elif min_obstacle_dist < 0.05:
            reward += 1.0  # Near-miss bonus for steering away
            
        # --- Wall Penalty ---
        if np.any(self.cursor_pos <= 0.01) or np.any(self.cursor_pos >= 0.99):
            reward -= 3.0
            
        return reward

    def _get_obs(self):
        """
        Phase 6: 21D observation = spike rates + spatial + context + obstacle proximity + velocity.
        """
        current_t = self.network.t
        window = 100*ms
        
        # 1. Neural Activity Features (10D)
        if len(self.spike_mon.t) > 0:
            recent_spikes_idx = self.spike_mon.i[self.spike_mon.t > current_t - window]
        else:
            recent_spikes_idx = []
        
        obs_group_size = self.n_neurons // self.n_obs_groups
        counts, _ = np.histogram(recent_spikes_idx, bins=self.n_obs_groups, range=(0, self.n_neurons))
        max_spikes = obs_group_size * 20 
        neural_obs = (counts / max_spikes).astype(np.float32)
        neural_obs = np.clip(neural_obs, 0, 1)
        
        # 2. Spatial Features (5D: cursor_x, cursor_y, target_x, target_y, context)
        spatial = np.concatenate([self.cursor_pos, self.target_pos, [float(self.active_target_idx)]]).astype(np.float32)
        
        # 3. Obstacle Proximity Sensors (4D: N, S, E, W) — use cached value
        prox = self._cached_prox
        
        # 4. Velocity Estimate (2D)
        velocity = (self.cursor_pos - self.prev_cursor_pos).astype(np.float32)
        
        return np.concatenate([neural_obs, spatial, prox, velocity])
    
    def _compute_obstacle_proximity(self):
        """Returns 4D proximity sensor: [North, South, East, West] distance to nearest obstacle."""
        prox = np.ones(4, dtype=np.float32)  # Default: far away
        directions = [np.array([0, 0.05]), np.array([0, -0.05]),
                      np.array([0.05, 0]), np.array([-0.05, 0])]
        
        for i, d in enumerate(directions):
            for step in range(1, 10):  # Probe up to 0.5 units
                probe = self.cursor_pos + d * step
                probe = np.clip(probe, 0, 1)
                for obs_item in self.obstacles:
                    if obs_item['type'] == 'circle':
                        if np.linalg.norm(probe - np.array(obs_item['pos'])) < obs_item['radius']:
                            prox[i] = min(prox[i], step * 0.05)
                    elif obs_item['type'] == 'box':
                        b = obs_item['bounds']
                        if b[0] <= probe[0] <= b[1] and b[2] <= probe[1] <= b[3]:
                            prox[i] = min(prox[i], step * 0.05)
        return prox
        
    def _update_motor_output(self):
        """
        Translates clustered motor activity into movement.
        """
        current_t = self.network.t
        window = 100*ms
        
        if len(self.spike_mon.t) > 0:
            motor_mask = (self.spike_mon.t > current_t - window) & (self.spike_mon.i >= self.output_neuron_start)
            motor_indices = self.spike_mon.i[motor_mask]
        else:
            motor_indices = []
            
        # Functional clusters
        up = np.sum((motor_indices >= self.motor_up[0]) & (motor_indices < self.motor_up[1]))
        down = np.sum((motor_indices >= self.motor_down[0]) & (motor_indices < self.motor_down[1]))
        left = np.sum((motor_indices >= self.motor_left[0]) & (motor_indices < self.motor_left[1]))
        right = np.sum((motor_indices >= self.motor_right[0]) & (motor_indices < self.motor_right[1]))
        
        sensitivity = 0.05
        dx = (right - left) * sensitivity
        dy = (up - down) * sensitivity
        
        self.prev_cursor_pos = self.cursor_pos.copy()
        self.cursor_pos += np.array([dx, dy])
        self.cursor_pos = np.clip(self.cursor_pos, 0, 1)

    def _stimulate_sdm(self, obs):
        """
        Sparse Distributed Memory Expansion.
        Uses core 15D obs (first 15 dims) for SDM projection.
        """
        core_obs = obs[:15]  # Only use the core features for SDM
        drive = core_obs @ self.W_sdm
        top_k = np.argsort(drive)[-26:]  # ~10% of 256 SDM neurons
        self.neurons.I[top_k] += 20.0
    
    def apply_structural_plasticity(self):
        """
        Phase 6: Prune weak synapses, grow new ones between co-active pairs.
        """
        total_synapses = 0
        for S in self.synapses:
            w = np.array(S.w)
            total_synapses += len(w)
            # Prune: set very weak synapses to 0
            weak_mask = w < 0.1
            if np.any(weak_mask):
                w[weak_mask] = 0.0
                S.w = w
        self.synapse_counts.append(total_synapses)
    
    def set_curriculum_stage(self, stage):
        """Set the difficulty stage (1-4)."""
        self.difficulty_stage = stage

    def _apply_action(self, action):
        """
        Maps action ID (0-7) to directional motor neuron stimulation.
        Actions 0-3: Cardinal directions (Up, Down, Left, Right)
        Actions 4-7: Diagonal combinations
        """
        if not self.use_motor_mapping:
            # Ablation: Diffuse Stimulation
            np.random.seed(action + int(self.network.t / ms))
            diffuse_indices = np.random.choice(np.arange(self.n_neurons), size=25, replace=False)
            self.neurons.I[diffuse_indices] += 25.0
            return

        stim = 25.0
        if action == 0:    # Up
            self.neurons.I[self.motor_up[0]:self.motor_up[1]] += stim
        elif action == 1:  # Down
            self.neurons.I[self.motor_down[0]:self.motor_down[1]] += stim
        elif action == 2:  # Left
            self.neurons.I[self.motor_left[0]:self.motor_left[1]] += stim
        elif action == 3:  # Right
            self.neurons.I[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 4:  # Up-Right
            self.neurons.I[self.motor_up[0]:self.motor_up[1]] += stim
            self.neurons.I[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 5:  # Up-Left
            self.neurons.I[self.motor_up[0]:self.motor_up[1]] += stim
            self.neurons.I[self.motor_left[0]:self.motor_left[1]] += stim
        elif action == 6:  # Down-Right
            self.neurons.I[self.motor_down[0]:self.motor_down[1]] += stim
            self.neurons.I[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 7:  # Down-Left
            self.neurons.I[self.motor_down[0]:self.motor_down[1]] += stim
            self.neurons.I[self.motor_left[0]:self.motor_left[1]] += stim

    def render(self):
        pass
    
    def close(self):
        pass
