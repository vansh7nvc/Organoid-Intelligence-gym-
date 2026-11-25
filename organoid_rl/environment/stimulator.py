"""
Global Activity Regulator (GAR) for OrganoidEnv.

Monitors global spike rate each step and intervenes when the network
is too quiet (kickstart) or too active (emergency inhibition).  Also
handles NaN recovery from numerical instability.

Author: Vansh Sharma
License: MIT
"""

from brian2 import *
import numpy as np

def stabilize_network_activity(env):
    """
    Implements the Global Activity Regulator (GAR).
    Checks global activity and intervenes if too low or too high.
    
    Parameters:
    env (OrganoidEnv): The environment instance containing the network.
    """
    if not hasattr(env, "gar_state_counters") or env.gar_state_counters is None:
        env.gar_state_counters = {"seizure_clamps": 0, "coma_kicks": 0, "nan_resets": 0}
    
    # 1. Check for NaN (Numerical Instability)
    if np.any(np.isnan(env.neurons.v)) or np.any(np.isinf(env.neurons.v)):
        # print("GAR: NaNs detected! Resetting network state.")
        env.neurons.v = -65.0
        env.neurons.u = 0.0 
        env.neurons.E = 1.0
        env.neurons.I = 0.0
        env.gar_state_counters["nan_resets"] += 1
        return

    # 2. Get Spike Count in the last step (use cached count on first step after checkpoint resume)
    if getattr(env, "_use_cached_spike_history", False):
        spike_count = int(getattr(env, "_cached_step_spike_count", 0))
    else:
        current_t = env.network.t
        last_step_t = current_t - env.step_duration
        
        if len(env.spike_mon.t) > 0:
            recent_spikes = env.spike_mon.t > last_step_t
            spike_count = int(np.sum(recent_spikes))
        else:
            spike_count = 0
    
    # 3. Logic (Homeostatic Activity Regulation)
    coma_thresh = getattr(env, "gar_coma_threshold", 50)
    seizure_thresh = getattr(env, "gar_seizure_threshold", 5000)

    if spike_count < coma_thresh:
        # Low Activity: Kickstart (Biological Analogy: Neuromodulatory Reset)
        env.neurons.v += 2.0 
        env.neurons.I = 5.0 * np.random.rand(env.n_neurons)
        env.gar_state_counters["coma_kicks"] += 1
        
    elif spike_count > seizure_thresh:
        # High Activity: Seizure Calming (Biological Analogy: Global Inhibition Burst)
        env.neurons.v = -85.0 
        env.neurons.I = -100.0 
        env.gar_state_counters["seizure_clamps"] += 1
        
    else:
        # Normal Activity
        pass
