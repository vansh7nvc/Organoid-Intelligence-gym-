from brian2 import *

def apply_dopamine(synapses, reward, learning_rate=0.01):
    """
    Applies the Dopamine-modulated STDP rule.
    Updates weights based on eligibility traces and reward.
    
    Rule: w += learning_rate * Reward * Trace
    
    Parameters:
    synapses (list of Synapses): The Synapses objects to update.
    reward (float): The current reward signal.
    learning_rate (float): The learning rate.
    """
    if reward == 0:
        return
        
    for S in synapses:
        # Check if Synapses object has 'Trace' variable
        # In Brian2, variables are accessed via getattr or dictionary-like access.
        # Safest is to check if 'Trace' is in S.equations.names
        if 'Trace' in S.equations.names:
            # Apply update
            # We assume w and Trace are dimensionless or consistent units.
            # S.w is a StateView.
            # Modifying state variables is efficient in Brian2.
            
            # w += lr * R * T
            # We clamp weights to reasonable bounds if needed, but for now just update.
            S.w += learning_rate * reward * S.Trace
