from brian2 import *
import numpy as np

def check_and_stabilize(env):
    """
    Implements the "God Mode" stabilizer.
    Checks global activity and intervenes if too low or too high.
    
    Parameters:
    env (OrganoidEnv): The environment instance containing the network.
    """
    
    # 1. Check for NaN (Numerical Instability)
    if np.any(np.isnan(env.neurons.v)) or np.any(np.isinf(env.neurons.v)):
        print("God Mode: NaNs detected! Resetting network state.")
        env.neurons.v = -65.0
        env.neurons.u = 0.0 # reset u?
        env.neurons.E = 1.0
        env.neurons.I = 0.0
        # Should we assume 0 spikes for this step?
        return

    # 2. Get Spike Count in the last step
    current_t = env.network.t
    last_step_t = current_t - env.step_duration
    
    # Simple check: Count spikes in [last_step_t, current_t]
    # Note: accessing spike_mon.t might be slow if history is huge.
    # For now we assume regular clearing or short episodes.
    if len(env.spike_mon.t) > 0:
        recent_spikes = env.spike_mon.t > last_step_t
        spike_count = np.sum(recent_spikes)
    else:
        spike_count = 0
    
    # 3. Logic
    if spike_count < 50:
        # Low Activity: Kickstart
        # print(f"God Mode: Activity too low ({spike_count} spikes). Injecting Noise.")
        env.neurons.v += 2 # Gentle bump
        env.neurons.I = '5 * rand()' # Gentle noise injection
        
    elif spike_count > 5000:
        # High Activity: Seizure Calming ( > 50Hz average)
        # print(f"God Mode: Activity too high ({spike_count} spikes). Inhibiting.")
        env.neurons.v = -85 # Reset voltage
        env.neurons.I = -100 # Inhibitory current
        
    else:
        # Normal Activity
        pass
