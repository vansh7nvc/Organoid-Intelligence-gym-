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
    
    # 1. Check for NaN (Numerical Instability)
    if np.any(np.isnan(env.neurons.v)) or np.any(np.isinf(env.neurons.v)):
        # print("GAR: NaNs detected! Resetting network state.")
        env.neurons.v = -65.0
        env.neurons.u = 0.0 
        env.neurons.E = 1.0
        env.neurons.I = 0.0
        return

    # 2. Get Spike Count in the last step
    current_t = env.network.t
    last_step_t = current_t - env.step_duration
    
    if len(env.spike_mon.t) > 0:
        recent_spikes = env.spike_mon.t > last_step_t
        spike_count = np.sum(recent_spikes)
    else:
        spike_count = 0
    
    # 3. Logic (Homeostatic Activity Regulation)
    if spike_count < 50:
        # Low Activity: Kickstart (Biological Analogy: Neuromodulatory Reset)
        env.neurons.v += 2.0 
        env.neurons.I = '5 * rand()' 
        
    elif spike_count > 5000:
        # High Activity: Seizure Calming (Biological Analogy: Global Inhibition Burst)
        env.neurons.v = -85.0 
        env.neurons.I = -100.0 
        
    else:
        # Normal Activity
        pass
