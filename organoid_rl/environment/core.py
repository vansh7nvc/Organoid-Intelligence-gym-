"""
OrganoidEnv: A Gymnasium-compatible RL environment backed by a Brian2
spiking neural network.  The environment simulates an organoid with
Metabolic-Izhikevich neurons, Sparse Distributed Memory, clustered
motor output, and curriculum-based obstacle navigation.

Author: Vansh Sharma
License: MIT
"""

import builtins
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from brian2 import *
from .neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs, get_lif_eqs
from .stimulator import stabilize_network_activity
from .rewards import apply_dopamine
from .fast_snn import load_snn_kernel, FastSNNRunner
from organoid_rl.utils.seeding import set_seed

# High-performance native C kernel configuration:
# Uses compiled C _snn_kernel (0.00e+00 numerical diff vs Brian2) when GCC is available.
# Falls back to Cython on Linux or NumPy on Windows if C kernel is unavailable.
import os
import platform
import shutil
if load_snn_kernel() is not None:
    prefs.codegen.target = 'numpy'
elif platform.system() == "Linux":
    prefs.codegen.target = 'cython'
else:
    prefs.codegen.target = 'numpy'


@dataclass
class OrganoidConfig:
    """
    Canonical configuration dataclass for OrganoidEnv.
    Encapsulates all biophysical parameters and experimental ablation toggles.
    """
    n_neurons: int = 500
    n_motor: int = 100
    tau_recovery_exc: float = 1000.0  # ms
    tau_recovery_inh: float = 500.0   # ms
    refractory_period: float = 2.0    # ms
    hidden_motor_density: float = 0.15
    use_sdm: bool = True
    use_morphology: bool = True
    use_dual_trace: bool = True
    use_stabilizer: bool = True
    use_motor_mapping: bool = True
    use_plasticity: bool = True
    random_snn: bool = False
    use_metabolism: bool = True
    use_pruning: bool = True
    difficulty_stage: int = 1
    neuron_model: str = "metabolic_izhikevich"
    tau_trace: float = 50.0
    trace_weight: float = 1.0
    spike_cost: float = 0.1
    gar_coma_threshold: int = 50
    gar_seizure_threshold: int = 5000


class _NeuronSubgroupUnion:
    """Read-only container exposing unified properties across non-contiguous subgroups."""
    def __init__(self, subgroups):
        self._subgroups = subgroups

    def __len__(self):
        return builtins.sum(len(g) for g in self._subgroups)

    def __iter__(self):
        for g in self._subgroups:
            yield from g

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = []
            for g in self._subgroups:
                items.extend(list(g))
            return items[idx]
        for g in self._subgroups:
            if idx < len(g):
                return g[idx]
            idx -= len(g)
        raise IndexError("Index out of range")

    @property
    def a(self):
        return np.concatenate([np.array(g.a) for g in self._subgroups])

    @property
    def b(self):
        return np.concatenate([np.array(g.b) for g in self._subgroups])

    @property
    def c(self):
        return np.concatenate([np.array(g.c) for g in self._subgroups])

    @property
    def d(self):
        return np.concatenate([np.array(g.d) for g in self._subgroups])

    @property
    def tau_recovery(self):
        return np.concatenate([np.array(g.tau_recovery) for g in self._subgroups])

class OrganoidEnv(gym.Env):
    """
    Gymnasium integration for the Organoid RL project.
    Simulates a network of neurons and exposes an interface for an agent to interact.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        use_sdm: bool = True,
        use_morphology: bool = True,
        use_dual_trace: bool = True,
        use_stabilizer: bool = True,
        use_motor_mapping: bool = True,
        config: Optional[OrganoidConfig] = None,
        **kwargs: Any,
    ):
        super(OrganoidEnv, self).__init__()

        # --- Configuration & StudySlate / Ablation Flags ---
        if config is not None:
            self.config = config
        else:
            if "use_gar" in kwargs and "use_stabilizer" not in kwargs:
                use_stabilizer = bool(kwargs.pop("use_gar"))
            self.config = OrganoidConfig(
                use_sdm=use_sdm,
                use_morphology=use_morphology,
                use_dual_trace=use_dual_trace,
                use_stabilizer=use_stabilizer,
                use_motor_mapping=use_motor_mapping,
                **{k: v for k, v in kwargs.items() if hasattr(OrganoidConfig, k)}
            )

        self.use_sdm = self.config.use_sdm
        self.use_morphology = self.config.use_morphology
        self.use_dual_trace = self.config.use_dual_trace
        self.use_stabilizer = self.config.use_stabilizer
        self.use_gar = self.config.use_stabilizer
        self.use_motor_mapping = self.config.use_motor_mapping
        self.use_plasticity = self.config.use_plasticity
        self.random_snn = self.config.random_snn
        self.use_metabolism = self.config.use_metabolism
        self.use_pruning = self.config.use_pruning
        self.neuron_model = str(self.config.neuron_model).lower()

        self.n_neurons = self.config.n_neurons
        self.n_motor = self.config.n_motor
        self.tau_recovery_exc = self.config.tau_recovery_exc
        self.tau_recovery_inh = self.config.tau_recovery_inh
        self.refractory_period = self.config.refractory_period
        self.hidden_motor_density = self.config.hidden_motor_density

        # Seed handling
        if "seed" in kwargs and kwargs["seed"] is not None:
            self._seed = int(kwargs["seed"])
            set_seed(self._seed)
        else:
            self._seed = None

        # --- Environment Configuration ---
        self.dt = 0.1 * ms  # Faster timestep (Phase 5 Optimization)
        self.step_duration = 100 * ms
        self.n_obs_groups = 10
        self.n_act_groups = 8
        tau_trace_ms = float(getattr(self.config, "tau_trace", 50.0))
        self.tau_trace = tau_trace_ms * ms
        self.trace_weight = float(getattr(self.config, "trace_weight", 1.0))

        # --- Action Space ---
        self.action_space = spaces.Discrete(self.n_act_groups)

        # --- Observation Space (Phase 6: 21D) ---
        # 10 (spike rates) + 2 (cursor) + 2 (target) + 1 (context) + 4 (obstacle proximity) + 2 (velocity)
        self.obs_dim = 21
        self.observation_space = spaces.Box(low=-1, high=2, shape=(self.obs_dim,), dtype=np.float32)

        # --- Curriculum Learning (Phase 6) ---
        self.difficulty_stage = int(self.config.difficulty_stage)  # 1=easy, 2=obstacles, 3=multi-goal, 4=full
        self.goal_radius = 0.15 if self.difficulty_stage == 1 else (0.10 if self.difficulty_stage in (2, 3) else 0.05)

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
        self.network = None  # Will be initialized in reset()
        self.neurons = None
        self.synapses = []
        self.monitors = {}

        # --- Physical Morphology (Phase 4/5) ---
        self.n_sdm = 256
        self.output_neuron_start = 400  # Neurons 400..499 are excitatory motor readout
        self.n_hidden = self.n_neurons - self.n_sdm - self.n_motor

        # Motor quadrants (Clustered) - Adjusted for 500 neurons
        self.motor_up = (400, 425)
        self.motor_down = (425, 450)
        self.motor_left = (450, 475)
        self.motor_right = (475, 500)

        # --- Virtual Motor System ---
        self.cursor_pos = np.array([0.1, 0.9])
        self.prev_dist = 0.0
        self.gar_state_counters = {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0}
        self._cached_recent_spikes_i = np.array([], dtype=np.int32)
        self._cached_step_spike_count = 0
        self._use_cached_spike_history = False
        self._step_counter = 0

        # --- SDM Fixed Projection (isolated local RNG) ---
        rng_sdm = np.random.RandomState(42)
        self.W_sdm = rng_sdm.choice([0, 1], size=(15, self.n_sdm), p=[0.9, 0.1])  # SDM uses 15D core obs

        # Build the initial network structure without disturbing caller RNG streams
        np_state_saved = np.random.get_state()
        try:
            np.random.seed(42)
            brian_seed_target = 42
            seed(brian_seed_target)
            self._build_network()
        finally:
            np.random.set_state(np_state_saved)
            if self._seed is not None:
                set_seed(self._seed)

    @property
    def P_exc(self):
        """Union view of all 400 excitatory neurons (SDM, Hidden-Exc, Motor)."""
        return _NeuronSubgroupUnion([self.P_exc_sdm, self.P_exc_hidden, self.P_exc_motor])

    def _build_network(self):
        """Constructs the Brian2 network objects."""
        start_scope()

        # 1. Neurons (Metabolic Izhikevich or Canonical LIF Baseline)
        if self.neuron_model == "lif":
            eqs = get_lif_eqs()
            threshold = 'v >= -50.0 and E > 0'
            reset = 'v = c'
        else:
            eqs = get_metabolic_izhikevich_eqs()
            threshold = 'v >= 30 and E > 0'
            reset = 'v = c; u += d; E -= spike_cost'

        self.neurons = NeuronGroup(
            self.n_neurons,
            eqs,
            threshold=threshold,
            reset=reset,
            method='euler',
            refractory=self.refractory_period * ms
        )
        self.neurons.add_attribute('refractory')
        self.neurons.refractory = self.refractory_period * ms

        # Neocortical Anatomy & Dale's Principle:
        # Excitatory (RS Pyramidal, 80% = 400 total):
        #   - SDM Sensory Expansion: 0..255 (256 units)
        #   - Hidden Excitatory Reservoir: 256..299 (44 units)
        #   - Clustered Motor Readout: 400..499 (100 units)
        # Inhibitory (FS Interneurons, 20% = 100 total):
        #   - Recurrent Interneurons: 300..399 (100 units)
        self.P_exc_sdm = self.neurons[:self.n_sdm]
        self.P_exc_hidden = self.neurons[self.n_sdm:300]
        self.P_inh = self.neurons[300:self.output_neuron_start]
        self.P_exc_motor = self.neurons[self.output_neuron_start:]

        # Configure Regular Spiking (RS) Excitatory parameters
        for exc_grp in (self.P_exc_sdm, self.P_exc_hidden, self.P_exc_motor):
            exc_grp.a = 0.02
            exc_grp.b = 0.2
            exc_grp.c = -65.0
            exc_grp.d = 8.0
            exc_grp.tau_recovery = self.tau_recovery_exc * ms  # 1000*ms

        # Configure Fast Spiking (FS) Inhibitory parameters
        self.P_inh.a = 0.1
        self.P_inh.b = 0.2
        self.P_inh.c = -65.0
        self.P_inh.d = 2.0
        self.P_inh.tau_recovery = self.tau_recovery_inh * ms  # 500*ms

        self.neurons.u = 'b * v'
        self.neurons.v = -65.0
        self.neurons.E = 1.0
        self.neurons.spike_cost = float(getattr(self.config, "spike_cost", 0.1)) if (self.use_metabolism and self.neuron_model != "lif") else 0.0
        self.gar_coma_threshold = int(getattr(self.config, "gar_coma_threshold", 50))
        self.gar_seizure_threshold = int(getattr(self.config, "gar_seizure_threshold", 5000))
        self.neurons.I = 0.0

        # 2. Synapses (Ablation Capable)
        # Note: When plasticity is disabled or random_snn is active, keep w0 per-synapse or freeze dw/dt.
        w_eq = "dw/dt = 0 / ms : 1 (clock-driven)" if (not self.use_plasticity or self.random_snn) else "dw/dt = (w0 - w) / tau_decay : 1 (clock-driven)"
        w0_decl = "w0 : 1" if self.random_snn else "w0 : 1 (shared)"
        if self.use_dual_trace:
            syn_eqs = f'''
            {w_eq}
            dTrace1/dt = -Trace1 / tau_trace1 : 1 (clock-driven)
            dTrace2/dt = -Trace2 / tau_trace2 : 1 (clock-driven)
            {w0_decl}
            '''
            on_pre_plus = 'v_post += w; Trace1 += 1; Trace2 += 1'
            on_pre_minus = 'v_post -= w; Trace1 += 1; Trace2 += 1'
            ns = {'tau_trace1': 100*ms, 'tau_trace2': 2500*ms, 'tau_decay': 10000*ms}
        else:
            syn_eqs = f'''
            {w_eq}
            dTrace/dt = -Trace / tau_trace : 1 (clock-driven)
            {w0_decl}
            '''
            on_pre_plus = 'v_post += w; Trace += 1'
            on_pre_minus = 'v_post -= w; Trace += 1'
            ns = {'tau_trace': self.tau_trace, 'tau_decay': 10000*ms}

        rng_rand = np.random.RandomState(42 if self._seed is None else self._seed)

        # Morphology Logic
        def create_syn(pre, post, p, w=15.0, is_inhibitory=False):
            op = on_pre_minus if is_inhibitory else on_pre_plus
            syn_method = 'exact' if self.use_dual_trace else 'euler'
            S = Synapses(pre, post, model=syn_eqs, on_pre=op, method=syn_method, namespace=ns)
            S.connect(p=p)
            if self.random_snn:
                w_init = rng_rand.uniform(0.5 * w, 1.5 * w, size=len(S.w))
                S.w = w_init
                S.w0 = w_init
            else:
                S.w = w
                S.w0 = w
            S._is_inhibitory = is_inhibitory
            if self.use_dual_trace:
                S.Trace1 = 0.0; S.Trace2 = 0.0
            else:
                S.Trace = 0.0
            return S

        if self.use_morphology:
             # Structured Subgroups
             SDM_grp = self.P_exc_sdm
             Hidden_exc = self.P_exc_hidden
             Hidden_grp = self.neurons[self.n_sdm:self.output_neuron_start]
             Motor_grp = self.P_exc_motor

             self.SDM_grp = SDM_grp
             self.Hidden_grp = Hidden_grp
             self.Motor_grp = Motor_grp
             self.Hidden_exc = Hidden_exc

             # Feedforward sensory drive: SDM (excitatory) -> Hidden
             self.synapses.append(create_syn(SDM_grp, Hidden_grp, p=0.15, w=20.0, is_inhibitory=False))
             # Feedforward motor actuation: Hidden Excitatory -> Motor (p=0.15)
             self.synapses.append(create_syn(Hidden_exc, Motor_grp, p=self.hidden_motor_density, w=20.0, is_inhibitory=False))
             # Sparse recurrent reservoir: Hidden Excitatory -> Hidden
             self.synapses.append(create_syn(Hidden_exc, Hidden_grp, p=0.05, w=10.0, is_inhibitory=False))
             # Lateral motor coordination: Motor (excitatory) -> Motor
             self.synapses.append(create_syn(Motor_grp, Motor_grp, p=0.1, w=10.0, is_inhibitory=False))
             # Global feedback inhibition: P_inh (inhibitory) -> all neurons
             self.synapses.append(create_syn(self.P_inh, self.neurons, p=0.1, w=25.0, is_inhibitory=True))
        else:
             # Random "Soup" Mesh (strictly adhering to Dale's Principle)
             for exc_grp in (self.P_exc_sdm, self.P_exc_hidden, self.P_exc_motor):
                 self.synapses.append(create_syn(exc_grp, self.neurons, p=0.1, is_inhibitory=False))
             self.synapses.append(create_syn(self.P_inh, self.neurons, p=0.1, is_inhibitory=True))

        # 3. Background Noise
        self.bg_noise = PoissonInput(self.neurons, 'v', 1, 10*Hz, weight=4)

        # 4. Initialize Network
        self.network = Network(self.neurons, *self.synapses, self.bg_noise)
        self.network.store('initial')

        self.spike_mon = SpikeMonitor(self.neurons, name='spike_mon')
        self.network.add(self.spike_mon)

        kernel_fn = load_snn_kernel()
        self._fast_runner = FastSNNRunner(self, kernel_fn) if kernel_fn is not None else None
        if self._fast_runner is not None:
            self._fast_runner.seed(42 if self._seed is None else self._seed)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            set_seed(seed)
            self._seed = seed
            if getattr(self, "_fast_runner", None) is not None:
                self._fast_runner.seed(seed)

        # Reset monitors to prevent memory growth and slowdown
        if self.spike_mon in self.network.objects:
            self.network.remove(self.spike_mon)

        preserve_weights = bool(options and options.get("preserve_weights", False)) and len(getattr(self, "synapses", [])) > 0
        saved_weights = [np.array(S.w, dtype=np.float64).copy() for S in self.synapses] if preserve_weights else None

        # Restore simulation to stored initial state (t=0, v=-65, etc)
        self.network.restore('initial')

        if saved_weights is not None:
            for S, w_arr in zip(self.synapses, saved_weights):
                S.w = w_arr

        # Add a fresh monitor
        self.spike_mon = SpikeMonitor(self.neurons, name='spike_monitor')
        self.monitors['spikes'] = self.spike_mon
        self.network.add(self.spike_mon)

        # Jitter initial voltage deterministically
        self.neurons.v = -65.0 + 10.0 * np.random.rand(self.n_neurons)
        self.neurons.u = self.neurons.b * self.neurons.v
        self.neurons.E = 1.0
        self.neurons.spike_cost = float(getattr(self.config, "spike_cost", 0.1)) if (self.use_metabolism and self.neuron_model != "lif") else 0.0
        self.gar_coma_threshold = int(getattr(self.config, "gar_coma_threshold", 50))
        self.gar_seizure_threshold = int(getattr(self.config, "gar_seizure_threshold", 5000))
        self.neurons.I = 0.0

        # Run initialization
        if getattr(self, "_fast_runner", None) is not None:
            p_rate = 10.0 if getattr(self.bg_noise, "active", True) else 0.0
            self._fast_runner.run_duration(10 * ms, poisson_rate_hz=p_rate)
        else:
            self.network.run(10*ms)

        # Cache recent spike history from initialization window for checkpoint resume parity
        window_sec = 0.1
        t_now_sec = float(self.network.t_)
        n_tot = int(self.spike_mon.variables["N"].get_value()[0])
        if n_tot > 0:
            t_arr = self.spike_mon.variables["t"].get_value()[:n_tot]
            i_arr = self.spike_mon.variables["i"].get_value()[:n_tot]
            mask = t_arr > (t_now_sec - window_sec)
            self._cached_recent_spikes_i = np.array(i_arr[mask], dtype=np.int32)
        else:
            self._cached_recent_spikes_i = np.array([], dtype=np.int32)
        self._cached_step_spike_count = int(len(self._cached_recent_spikes_i))
        self._use_cached_spike_history = False
        self._step_counter = 0
        self.gar_state_counters = {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0}

        # Reset motor and task state
        self.cursor_pos = np.array([0.1, 0.9])
        self.prev_cursor_pos = np.array([0.1, 0.9])
        self.visited_cells = set()

        # Curriculum-aware target selection
        if self.difficulty_stage <= 2:
            self.active_target_idx = 0  # Single target in early stages
        else:
            self.active_target_idx = int(np.random.choice([0, 1]))
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

        # Update obstacle proximity sensor cache
        self._cached_prox = self._compute_obstacle_proximity()

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
        self.neurons.variables["I"].get_value()[:] = 0.0
        
        # Stabilization (GAR)
        if self.use_stabilizer:
            stabilize_network_activity(self)
        
        # 3. Apply Action (Stimulation)
        self._apply_action(action)
        
        # 4. SDM Expansion (Sensory Drive) — uses 15D core obs only
        if self.use_sdm:
            self._stimulate_sdm(self._get_obs())

        # 5. Run Simulation
        if getattr(self, "_fast_runner", None) is not None:
            p_rate = 10.0 if getattr(self.bg_noise, "active", True) else 0.0
            self._fast_runner.run_duration(self.step_duration, poisson_rate_hz=p_rate)
        else:
            self.network.run(self.step_duration)
        self._use_cached_spike_history = False
        window_sec = 0.1
        t_now_sec = float(self.network.t_)
        n_tot = int(self.spike_mon.variables["N"].get_value()[0])
        if n_tot > 0:
            t_arr = self.spike_mon.variables["t"].get_value()[:n_tot]
            i_arr = self.spike_mon.variables["i"].get_value()[:n_tot]
            mask = t_arr > (t_now_sec - window_sec)
            self._cached_recent_spikes_i = np.array(i_arr[mask], dtype=np.int32)
        else:
            self._cached_recent_spikes_i = np.array([], dtype=np.int32)
        self._cached_step_spike_count = int(len(self._cached_recent_spikes_i))
        self._step_counter += 1
        
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
        if self.use_plasticity and not self.random_snn:
            apply_dopamine(self.synapses, reward, trace_scale=self.trace_weight)
        
        # 13. Check Done (uses curriculum goal_radius)
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        terminated = dist < self.goal_radius
        truncated = False
        
        info = {
            'cursor_pos': self.cursor_pos.tolist(),
            'target_pos': self.target_pos.tolist(),
            'distance': float(dist),
            'total_spikes': int(self.spike_mon.variables["N"].get_value()[0]),
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
        current_t_sec = float(self.network.t_)
        window_sec = 0.1
        n_tot = int(self.spike_mon.variables["N"].get_value()[0])

        # 1. Neural Activity Features (10D)
        if getattr(self, "_use_cached_spike_history", False):
            recent_spikes_idx = self._cached_recent_spikes_i
        elif n_tot > 0:
            t_arr = self.spike_mon.variables["t"].get_value()[:n_tot]
            i_arr = self.spike_mon.variables["i"].get_value()[:n_tot]
            recent_spikes_idx = i_arr[t_arr > (current_t_sec - window_sec)]
        else:
            recent_spikes_idx = np.empty(0, dtype=np.int32)
        
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
        current_t_sec = float(self.network.t_)
        window_sec = 0.1
        n_tot = int(self.spike_mon.variables["N"].get_value()[0])

        if n_tot > 0:
            t_arr = self.spike_mon.variables["t"].get_value()[:n_tot]
            i_arr = self.spike_mon.variables["i"].get_value()[:n_tot]
            motor_mask = (t_arr > current_t_sec - window_sec) & (i_arr >= self.output_neuron_start)
            motor_indices = i_arr[motor_mask]
        else:
            motor_indices = np.empty(0, dtype=np.int32)
            
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
        I_arr = self.neurons.variables["I"].get_value()
        I_arr[top_k] += 20.0
    
    def apply_structural_plasticity(self):
        """
        Phase 6: Prune weak synapses, grow new ones between co-active pairs.
        """
        if not self.use_pruning or self.random_snn:
            return
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
        I_arr = self.neurons.variables["I"].get_value()
        if not self.use_motor_mapping:
            # Ablation: Diffuse Stimulation (local RNG avoids perturbing global RNG streams)
            rng_diffuse = np.random.RandomState(int(action) + int(self._step_counter) * 100)
            diffuse_indices = rng_diffuse.choice(np.arange(self.n_neurons), size=25, replace=False)
            I_arr[diffuse_indices] += 25.0
            return

        stim = 25.0
        if action == 0:    # Up
            I_arr[self.motor_up[0]:self.motor_up[1]] += stim
        elif action == 1:  # Down
            I_arr[self.motor_down[0]:self.motor_down[1]] += stim
        elif action == 2:  # Left
            I_arr[self.motor_left[0]:self.motor_left[1]] += stim
        elif action == 3:  # Right
            I_arr[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 4:  # Up-Right
            I_arr[self.motor_up[0]:self.motor_up[1]] += stim
            I_arr[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 5:  # Up-Left
            I_arr[self.motor_up[0]:self.motor_up[1]] += stim
            I_arr[self.motor_left[0]:self.motor_left[1]] += stim
        elif action == 6:  # Down-Right
            I_arr[self.motor_down[0]:self.motor_down[1]] += stim
            I_arr[self.motor_right[0]:self.motor_right[1]] += stim
        elif action == 7:  # Down-Left
            I_arr[self.motor_down[0]:self.motor_down[1]] += stim
            I_arr[self.motor_left[0]:self.motor_left[1]] += stim

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Extracts canonical SNN, GAR, and environment task states for atomic checkpointing.
        Satisfies OrganoidEnv_Master_Verifier_Plan.md §2.1 environment state contract.
        """
        weights = [np.array(S.w).copy() for S in self.synapses]
        if self.use_dual_trace:
            traces_fast = [np.array(S.Trace1).copy() for S in self.synapses]
            traces_slow = [np.array(S.Trace2).copy() for S in self.synapses]
        else:
            traces_fast = [np.array(S.Trace).copy() for S in self.synapses]
            traces_slow = [np.zeros_like(t) for t in traces_fast]

        atp = np.array(self.neurons.E).copy()
        v = np.array(self.neurons.v).copy()
        u = np.array(self.neurons.u).copy()
        pruning_mask = [np.array(S.w > 0.0).copy() for S in self.synapses]

        gar_counters = dict(
            getattr(self, "gar_state_counters", {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0})
        )

        lastspike_offset_ms = np.array(self.neurons.lastspike / ms, dtype=np.float64) - float(self.network.t / ms)

        task_state = {
            "cursor_pos": np.array(self.cursor_pos).copy(),
            "prev_cursor_pos": np.array(self.prev_cursor_pos).copy(),
            "target_pos": np.array(self.target_pos).copy(),
            "active_target_idx": int(self.active_target_idx),
            "prev_dist": float(self.prev_dist),
            "visited_cells": list(self.visited_cells),
            "difficulty_stage": int(self.difficulty_stage),
            "goal_radius": float(self.goal_radius),
            "cached_prox": np.array(self._cached_prox).copy(),
            "cached_recent_spikes_i": np.array(self._cached_recent_spikes_i, dtype=np.int32).copy(),
            "cached_step_spike_count": int(self._cached_step_spike_count),
            "step_counter": int(self._step_counter),
            "lastspike_offset_ms": lastspike_offset_ms,
            "not_refractory": np.array(self.neurons.not_refractory).copy(),
            "fast_runner_rng": self._fast_runner.get_rng_state() if getattr(self, "_fast_runner", None) is not None else None,
        }

        return {
            "snn_weights": weights,
            "snn_traces_fast": traces_fast,
            "snn_traces_slow": traces_slow,
            "snn_metabolic_atp": atp,
            "snn_membrane_v": v,
            "snn_recovery_u": u,
            "snn_pruning_mask": pruning_mask,
            "gar_state_counters": gar_counters,
            "env_task_state": task_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores canonical SNN, GAR, and environment task states from checkpoint.
        """
        for S, w, mask in zip(self.synapses, state_dict["snn_weights"], state_dict["snn_pruning_mask"]):
            w_arr = np.array(w, copy=True)
            # Re-apply pruning mask cleanly
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape == w_arr.shape:
                w_arr = np.where(mask_arr, w_arr, 0.0)
            S.w = w_arr

        if self.use_dual_trace:
            for S, t1, t2 in zip(self.synapses, state_dict["snn_traces_fast"], state_dict["snn_traces_slow"]):
                S.Trace1 = t1
                S.Trace2 = t2
        else:
            for S, t in zip(self.synapses, state_dict["snn_traces_fast"]):
                S.Trace = t

        self.neurons.E = state_dict["snn_metabolic_atp"]
        self.neurons.v = state_dict["snn_membrane_v"]
        self.neurons.u = state_dict["snn_recovery_u"]

        if "gar_state_counters" in state_dict:
            self.gar_state_counters = dict(state_dict["gar_state_counters"])

        if "env_task_state" in state_dict:
            ts = state_dict["env_task_state"]
            self.cursor_pos = np.array(ts["cursor_pos"]).copy()
            self.prev_cursor_pos = np.array(ts["prev_cursor_pos"]).copy()
            self.target_pos = np.array(ts["target_pos"]).copy()
            if "active_target_idx" in ts:
                self.active_target_idx = int(ts["active_target_idx"])
            if "prev_dist" in ts:
                self.prev_dist = float(ts["prev_dist"])
            if "visited_cells" in ts:
                self.visited_cells = set(
                    tuple(c) if isinstance(c, (list, tuple)) else c for c in ts["visited_cells"]
                )
            if "difficulty_stage" in ts:
                self.difficulty_stage = int(ts["difficulty_stage"])
                self.obstacles = self.obstacles_full if self.difficulty_stage >= 2 else []
            if "goal_radius" in ts:
                self.goal_radius = float(ts["goal_radius"])
            if "cached_prox" in ts:
                self._cached_prox = np.array(ts["cached_prox"]).copy()
            if "cached_recent_spikes_i" in ts:
                self._cached_recent_spikes_i = np.array(ts["cached_recent_spikes_i"], dtype=np.int32).copy()
                self._cached_step_spike_count = int(ts.get("cached_step_spike_count", len(self._cached_recent_spikes_i)))
                self._use_cached_spike_history = True
            if "step_counter" in ts:
                self._step_counter = int(ts["step_counter"])
            if "lastspike_offset_ms" in ts:
                self.neurons.lastspike = (float(self.network.t / ms) + np.asarray(ts["lastspike_offset_ms"])) * ms
            if "not_refractory" in ts:
                self.neurons.not_refractory = np.asarray(ts["not_refractory"], dtype=bool)
            if ts.get("fast_runner_rng") is not None and getattr(self, "_fast_runner", None) is not None:
                self._fast_runner.set_rng_state(ts["fast_runner_rng"])

    def render(self):
        pass
    
    def close(self):
        pass
