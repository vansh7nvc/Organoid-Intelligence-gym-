"""
Dopamine-modulated reward signal for OrganoidEnv.

Implements the D3 (Dual-trace Dopamine-modulated) STDP rule:
fast eligibility traces capture recent activity while slow traces
provide long-horizon temporal credit assignment.

Rule:  Δw = lr × Reward × (Trace1 + 0.5 × Trace2)

Author: Vansh Sharma
License: MIT
"""

from brian2 import *

def apply_dopamine(synapses, reward, learning_rate=0.01, trace_scale=1.0):
    """
    Applies the Dopamine-modulated STDP rule with Dual-Trace support.
    Updates weights based on eligibility traces (Fast & Slow) and reward.
    
    Rule: w += learning_rate * Reward * (Trace1 + 0.5 * Trace2)
    For single-trace mode: w += learning_rate * Reward * trace_scale * Trace
    
    Parameters:
    synapses (list of Synapses): The Synapses objects to update.
    reward (float): The current reward signal.
    learning_rate (float): The learning rate.
    trace_scale (float): Weight multiplier for single trace mode (default: 1.0).
    """
    if reward == 0:
        return
        
    for S in synapses:
        w_arr = S.variables["w"].get_value()
        vdict = S.variables
        if "Trace1" in vdict and "Trace2" in vdict:
            t1 = vdict["Trace1"].get_value()
            t2 = vdict["Trace2"].get_value()
            w_arr += learning_rate * reward * (t1 + 0.5 * t2)
            np.clip(w_arr, 0.0, 100.0, out=w_arr)
        elif "Trace" in vdict:
            t = vdict["Trace"].get_value()
            w_arr += learning_rate * reward * trace_scale * t
            np.clip(w_arr, 0.0, 100.0, out=w_arr)
