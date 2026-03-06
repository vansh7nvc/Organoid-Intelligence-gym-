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

        # --- Observation Space (Day 3) ---
        # Box(0, 1, shape=(10,)): Normalized firing rates of 10 subgroups
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_obs_groups,), dtype=np.float32)

        # --- Brian2 Network Setup ---
        self.network = None # Will be initialized in reset()
        self.neurons = None
        self.synapses = []
        self.pre_synapses = []
        self.monitors = {}
        
        # Build the initial network structure (but don't start simulation yet)
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
        # Trace decays exponentially
        syn_eqs = '''
        w : 1
        dTrace/dt = -Trace / tau_trace : 1
        '''
        
        # E-E: Plastic
        # On spike, v_post increases by w, and Trace increases
        S_EE = Synapses(P_exc, P_exc, model=syn_eqs, on_pre='v_post += w; Trace += 1', 
                        method='euler',
                        namespace={'tau_trace': self.tau_trace})
        S_EE.connect(p=0.1)
        S_EE.w = 15.0 # Initial weight
        S_EE.Trace = 0.0
        self.synapses.append(S_EE)
        
        # E-I: Plastic
        S_EI = Synapses(P_exc, P_inh, model=syn_eqs, on_pre='v_post += w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace})
        S_EI.connect(p=0.1)
        S_EI.w = 15.0
        S_EI.Trace = 0.0
        self.synapses.append(S_EI)
        
        # I-E: Inhibitory (subtracted w) - Also has traces if we want symmetric learning
        S_IE = Synapses(P_inh, P_exc, model=syn_eqs, on_pre='v_post -= w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace})
        S_IE.connect(p=0.1)
        S_IE.w = 25.0
        S_IE.Trace = 0.0
        self.synapses.append(S_IE)
        
        # I-I: Inhibitory
        S_II = Synapses(P_inh, P_inh, model=syn_eqs, on_pre='v_post -= w; Trace += 1',
                        method='euler',
                        namespace={'tau_trace': self.tau_trace})
        S_II.connect(p=0.1)
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
        
        # 6. Calculate Reward (Placeholder)
        reward = 0.0 
        
        # 7. Apply Dopamine Learning
        apply_dopamine(self.synapses, reward)
        
        # 8. Check Done
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

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
             
    def _get_obs(self):
        """
        Returns normalized spike rates for observations.
        """
        current_t = self.network.t
        window = 100*ms
        
        if len(self.spike_mon.t) > 0:
            recent_spikes_idx = self.spike_mon.i[self.spike_mon.t > current_t - window]
        else:
            recent_spikes_idx = []
        
        obs_group_size = self.n_neurons // self.n_obs_groups
        counts, _ = np.histogram(recent_spikes_idx, bins=self.n_obs_groups, range=(0, self.n_neurons))
        
        max_spikes = obs_group_size * 20 
        obs = counts / max_spikes
        obs = np.clip(obs, 0, 1) 
        
        return obs.astype(np.float32)

    def render(self):
        pass
    
    def close(self):
        pass
