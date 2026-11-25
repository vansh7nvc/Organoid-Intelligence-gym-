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
    Includes the 'Metabolic Governor' state variable E (Energy) and K_ATP-like
    metabolic hyperpolarization clamp during ATP exhaustion (E <= 0) to keep
    membrane voltage physiologically bounded without distorting healthy spiking.
    """
    return '''
    dv/dt = (int(E > 0) * (0.04*v_clipped**2 + 5*v_clipped + 140 - u + I) + int(E <= 0) * ((-80.0 - v_clipped) / 0.5)) / ms : 1 (unless refractory)
    v_clipped = clip(v, -100, 60) : 1
    du/dt = a*(b*v_clipped - u) / ms : 1
    dE/dt = (1 - E) / tau_recovery : 1
    
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    tau_recovery : second
    spike_cost : 1
    '''


def get_lif_eqs():
    """
    Returns canonical Leaky Integrate-and-Fire (LIF) equations for the SimpleSNN baseline
    (no quadratic upstroke, no adaptation variable u dynamics, no metabolic fatigue).
    """
    return '''
    dv/dt = (-(v + 65.0) + I) / (20*ms) : 1 (unless refractory)
    du/dt = 0 / ms : 1
    dE/dt = 0 / tau_recovery : 1

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
    
    G = NeuronGroup(N, eqs, threshold=threshold, reset=reset, method='euler', refractory=2*ms)
    G.add_attribute('refractory')
    G.refractory = 2*ms
    
    # Initial Parameters setup
    G.v = -65.0
    G.E = 1.0  # Full ATP reserves
    G.tau_recovery = 1000*ms
    G.spike_cost = 0.1
    G.I = 0.0
    
    if model_type == 'RS':
        G.a = 0.02
        G.b = 0.2
        G.c = -65.0
        G.d = 8.0
        G.tau_recovery = 1000*ms
    elif model_type == 'FS':
        G.a = 0.1
        G.b = 0.2
        G.c = -65.0
        G.d = 2.0
        G.tau_recovery = 500*ms
    
    G.u = 'b * v'
    
    return G
