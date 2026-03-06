from environment.core import OrganoidEnv
import numpy as np
from brian2 import *

def diagnostic():
    env = OrganoidEnv()
    env.reset()
    
    # 1. Check neurons
    print(f"Number of neurons: {len(env.neurons)}")
    
    # 2. Check Synapses
    S_EE = env.synapses[0]
    print(f"S_EE connections: {len(S_EE)}")
    
    # 3. Stimulate Group 0
    print("Stimulating group 0 (0-50)...")
    env.step(0)
    
    # 4. Check spikes of neurons 0-50 vs 50-100
    spikes = env.spike_mon
    # Get spikes from the LAST 100ms
    current_t = env.network.t
    recent_mask = spikes.t > (current_t - 100.1*ms)
    recent_spikes = spikes.i[recent_mask]
    
    g0_spikes = np.sum((recent_spikes >= 0) & (recent_spikes < 50))
    g1_spikes = np.sum((recent_spikes >= 50) & (recent_spikes < 100))
    
    print(f"G0 spikes (Stimulated): {g0_spikes}")
    print(f"G1 spikes (Control): {g1_spikes}")
    
    # 5. Check Traces
    indices_g0 = np.where(S_EE.i < 50)[0]
    indices_g1 = np.where((S_EE.i >= 50) & (S_EE.i < 100))[0]
    
    print(f"Num synapses G0: {len(indices_g0)}")
    print(f"Num synapses G1: {len(indices_g1)}")
    
    t0_vals = np.array(S_EE.Trace[indices_g0])
    t1_vals = np.array(S_EE.Trace[indices_g1])
    
    print(f"Mean Trace G0: {np.mean(t0_vals)}")
    print(f"Mean Trace G1: {np.mean(t1_vals)}")
    
    print(f"Sample Traces G0: {t0_vals[:5]}")
    print(f"Sample Traces G1: {t1_vals[:5]}")
    
    if np.array_equal(t0_vals[:100], t1_vals[:100]):
         print("CRITICAL: Traces are IDENTICAL across groups! Something is shared.")

if __name__ == "__main__":
    diagnostic()
