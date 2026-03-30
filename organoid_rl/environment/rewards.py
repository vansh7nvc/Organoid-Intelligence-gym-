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

def apply_dopamine(synapses, reward, learning_rate=0.01):
    """
    Applies the Dopamine-modulated STDP rule with Dual-Trace support.
    Updates weights based on eligibility traces (Fast & Slow) and reward.
    
    Rule: w += learning_rate * Reward * (Trace1 + 0.5 * Trace2)
    
    Parameters:
    synapses (list of Synapses): The Synapses objects to update.
    reward (float): The current reward signal.
    learning_rate (float): The learning rate.
    """
    if reward == 0:
        return
        
    for S in synapses:
        # Support for Dual-Trace (Phase 4)
        if 'Trace1' in S.equations.names and 'Trace2' in S.equations.names:
            # Combined trace: Fast (1.0) + Slow (0.5 weight for long-term credit)
            S.w += learning_rate * reward * (S.Trace1 + 0.5 * S.Trace2)
            S.w = np.clip(S.w, 0, 100.0)
        elif 'Trace' in S.equations.names:
            # Legacy Single Trace support
            S.w += learning_rate * reward * S.Trace
            S.w = np.clip(S.w, 0, 100.0)
