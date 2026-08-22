"""
Metabolic Izhikevich neuron model for OrganoidEnv.

Adds an energy state variable (E) that acts as a biological governor:
neurons can only fire when E > 0, and each spike costs a fixed amount
of energy that recovers exponentially.

Author: Vansh Sharma
License: MIT
"""

from brian2 import *

# codegen target is set in core.py (numpy)

def get_metabolic_izhikevich_eqs():
    """
    Returns the equation string for the Metabolic Izhikevich neuron model.
    Includes the 'Metabolic Governor' state variable E (Energy).
    """
    return '''
    dv/dt = (0.04*v_clipped**2 + 5*v_clipped + 140 - u + I) / ms : 1 (unless refractory)
    v_clipped = clip(v, -100, 60) : 1
    du/dt = a*(b*v - u) / ms : 1
    dE/dt = (1 - E) / tau_recovery : 1
    
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    tau_recovery : second
    spike_cost : 1
    '''

def create_metabolic_neurons(N, model_type='RS'):
    """
    Creates a NeuronGroup of N Metabolic Izhikevich neurons.
    
    Parameters:
    N (int): Number of neurons.
    model_type (str): Type of Izhikevich neuron ('RS' for Regular Spiking).
    """
    eqs = get_metabolic_izhikevich_eqs()
    
    # The Constraint: Firing only if v >= 30mV AND E > 0
    threshold = 'v >= 30 and E > 0'
    
    # The Cost: Reset v, u AND decrement energy E
    reset = 'v = c; u += d; E -= spike_cost'
    
    G = NeuronGroup(N, eqs, threshold=threshold, reset=reset, method='euler')
    
    # Initial Parameters setup
    G.v = -65.0
    G.E = 1.0  # Full ATP reserves
    G.tau_recovery = 100*ms
    G.spike_cost = 0.1
    G.I = 0.0
    
    if model_type == 'RS':
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
    elif model_type == 'FS':
        G.a = 0.1
        G.b = 0.2
        G.c = -65.0
        G.d = 2.0
    
    G.u = 'b * v'
    
    return G
