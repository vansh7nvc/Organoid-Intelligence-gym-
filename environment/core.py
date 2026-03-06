import gymnasium as gym
from gymnasium import spaces
import numpy as np
from brian2 import *
from .neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
from .stimulator import check_and_stabilize
from .rewards import apply_dopamine

# High-performance Cython configuration
# High-performance Cython configuration
# prefs.codegen.target = 'cython'
# prefs.codegen.cpp.compiler = 'mingw32'
# prefs.codegen.cpp.extra_compile_args = ['-DMS_WIN64']
prefs.codegen.target = 'numpy'

class OrganoidEnv(gym.Env):
    """
    Gymnasium integration for the Organoid RL project.
    Simulates a network of neurons and exposes an interface for an agent to interact.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(OrganoidEnv, self).__init__()
        
        # --- Environment Configuration ---
        self.dt = 0.05 * ms  # Simulation timestep
        self.step_duration = 100 * ms # How long the simulation runs per step
        self.n_neurons = 1000
        self.n_obs_groups = 10 # Number of groups for observation (spike rate coding)
        self.n_act_groups = 8  # Number of groups for action (stimulation)
        self.tau_trace = 50*ms # Decay constant for eligibility trace

        # --- Action Space (Day 2) ---
        # Discrete(8): Stimulate one of 8 specific neuron subgroups
        self.action_space = spaces.Discrete(self.n_act_groups)

        # --- Observation Space ---
        # 10 (Fire rates) + 2 (Current Cursor [x,y]) + 2 (Target [x,y]) = 14 dimensions
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_groups + 4,), dtype=np.float32)

        # --- Target Task Setup ---
        self.target_pos = np.array([0.8, 0.8]) # The goal for the organoid

        # --- Brian2 Network Setup ---
        self.network = None # Will be initialized in reset()
        self.neurons = None
        self.synapses = []
        self.monitors = {}
        
        # --- Virtual Motor System (New for Task 3) ---
        self.cursor_pos = np.array([0.5, 0.5]) # Normalized [x, y]
        self.output_neuron_start = 900 # Last 100 neurons are "motor" neurons
        
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

        # 2. Synapses (Plasticity Enabled)
        # Model for plasticity: w is weight, Trace tracks eligibility
        # Weight Homeostasis: w decays back to baseline
        syn_eqs = '''
        dw/dt = (w0 - w) / tau_decay : 1
        dTrace/dt = -Trace / tau_trace : 1
        w0 : 1 (shared)
        '''
        
        # E-E: Plastic
        # On spike, v_post increases by w, and Trace increases
        S_EE = Synapses(P_exc, P_exc, model=syn_eqs, on_pre='v_post += w; Trace += 1', 
                        method='euler',
                        namespace={'tau_trace': self.tau_trace, 'tau_decay': 10000*ms})
        S_EE.connect(p=0.1)
        S_EE.w0 = 15.0 # Baseline
        S_EE.w = 15.0 # Initial weight
        S_EE.Trace = 0.0
        self.synapses.append(S_EE)
        
        # E-I: Plastic
        S_EI = Synapses(P_exc, P_inh, model=syn_eqs, on_pre='v_post += w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace, 'tau_decay': 10000*ms})
        S_EI.connect(p=0.1)
        S_EI.w0 = 15.0
        S_EI.w = 15.0
        S_EI.Trace = 0.0
        self.synapses.append(S_EI)
        
        # I-E: Inhibitory
        S_IE = Synapses(P_inh, P_exc, model=syn_eqs, on_pre='v_post -= w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace, 'tau_decay': 10000*ms})
        S_IE.connect(p=0.1)
        S_IE.w0 = 25.0
        S_IE.w = 25.0
        S_IE.Trace = 0.0
        self.synapses.append(S_IE)
        
        # I-I: Inhibitory
        S_II = Synapses(P_inh, P_inh, model=syn_eqs, on_pre='v_post -= w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace, 'tau_decay': 10000*ms})
        S_II.connect(p=0.1)
        S_II.w0 = 25.0
        S_II.w = 25.0
        S_II.Trace = 0.0
        self.synapses.append(S_II)
        
        # 3. Background Noise
        self.bg_noise = PoissonInput(self.neurons, 'v', 1, 20*Hz, weight=5)
        
        # 4. Monitors
        self.spike_mon = SpikeMonitor(self.neurons)
        self.monitors['spikes'] = self.spike_mon

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # Reset variables
        self.neurons.v = '-65 + 10*rand()' 
        self.neurons.u = 'b * v'
        self.neurons.E = 1.0
        self.neurons.I = 0
        
        # Reset Simulation Clock by moving to new Network instance
        device.reinit()
        device.activate()
        
        # Re-inject components into a new Network
        # IMPORTANT: Unpack the synapses list
        self.network = Network(self.neurons, *self.synapses, self.bg_noise, self.spike_mon)
        self.network.store() 
        
        # Run initialization
        self.network.run(10*ms)
        
        # Reset motor and task state
        self.cursor_pos = np.array([0.5, 0.5])
        self.target_pos = np.random.rand(2) # Randomized target per episode
        
        return self._get_obs(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        
        # 1. Reset Current
        self.neurons.I = 0
        
        # 2. Stabilizer ("God Mode")
        check_and_stabilize(self)
        
        # 3. Apply Action (Stimulation)
        self._apply_action(action)
        
        # 4. Run Simulation
        self.network.run(self.step_duration)
        
        # 5. Get Observation
        obs = self._get_obs()
        
        # 6. Calculate Reward (Distance-based for Target Task)
        reward = self._calculate_reward()
        
        # 7. Update Motor Output (Move Cursor based on activity)
        self._update_motor_output()
        
        # 8. Apply Dopamine Learning
        # We only apply learning if there is a reward (or penalty)
        apply_dopamine(self.synapses, reward)
        
        # 9. Check Done
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        terminated = dist < 0.05 # Goal reached
        truncated = False
        info = {
            'cursor_pos': self.cursor_pos.tolist(),
            'target_pos': self.target_pos.tolist(),
            'distance': float(dist),
            'total_spikes': len(self.spike_mon.t)
        }
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        """
        Rewards the organoid for moving closer to the target.
        """
        dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        
        # Sparse or continuous reward
        if dist < 0.1:
            return 10.0 # High reward for being close
        elif dist < 0.3:
            return 1.0 # Moderate reward
        
        return 0.0 # No reward far away

    def _get_obs(self):
        """
        Returns normalized spike rates + cursor pos + target pos.
        """
        current_t = self.network.t
        window = 100*ms
        
        # 1. Neural Activity Features
        if len(self.spike_mon.t) > 0:
            recent_spikes_idx = self.spike_mon.i[self.spike_mon.t > current_t - window]
        else:
            recent_spikes_idx = []
        
        obs_group_size = self.n_neurons // self.n_obs_groups
        counts, _ = np.histogram(recent_spikes_idx, bins=self.n_obs_groups, range=(0, self.n_neurons))
        
        max_spikes = obs_group_size * 20 
        neural_obs = (counts / max_spikes).astype(np.float32)
        neural_obs = np.clip(neural_obs, 0, 1) 
        
        # 2. task State Features
        task_obs = np.concatenate([self.cursor_pos, self.target_pos]).astype(np.float32)
        
        return np.concatenate([neural_obs, task_obs])
        
    def _update_motor_output(self):
        """
        Translates activity in motor neurons (900-1000) into cursor movement.
        """
        current_t = self.network.t
        window = 100*ms
        
        # Filter spikes in motor window
        if len(self.spike_mon.t) > 0:
            motor_mask = (self.spike_mon.t > current_t - window) & (self.spike_mon.i >= self.output_neuron_start)
            motor_indices = self.spike_mon.i[motor_mask]
        else:
            motor_indices = []
            
        # Count activity per quadrant
        up = np.sum((motor_indices >= 900) & (motor_indices < 925))
        down = np.sum((motor_indices >= 925) & (motor_indices < 950))
        left = np.sum((motor_indices >= 950) & (motor_indices < 975))
        right = np.sum((motor_indices >= 975) & (motor_indices < 1000))
        
        # Calculate velocity (normalized)
        sensitivity = 0.01
        dx = (right - left) * sensitivity
        dy = (up - down) * sensitivity
        
        # Update and clip position
        self.cursor_pos += np.array([dx, dy])
        self.cursor_pos = np.clip(self.cursor_pos, 0, 1)

    def _apply_action(self, action):
        """
        Maps action ID (0-7) to a specific neuron subgroup stimulation.
        """
        group_size = 50
        start_idx = action * group_size
        end_idx = start_idx + group_size
        
        # Check bounds
        if end_idx <= self.n_neurons:
             # Stimulate this subgroup
             self.neurons.I[start_idx:end_idx] += 20.0 # Strong current injection

    def render(self):
        pass
    
    def close(self):
        pass
