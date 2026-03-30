from brian2 import *
import matplotlib.pyplot as plt
from environment.neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
import os

# High-performance Cython configuration
prefs.codegen.target = 'cython'
prefs.codegen.cpp.compiler = 'mingw32'
prefs.codegen.cpp.extra_compile_args = ['-DMS_WIN64']

def run_seizure_test(duration=60*second, plot_output='experiments/02_seizure_test_result.png'):
    start_scope()
    
    # 1. Scale Population (Day 1)
    N_exc = 800
    N_inh = 200
    N_total = N_exc + N_inh
    
    # Use a single group and slicings for E/I populations
    eqs = get_metabolic_izhikevich_eqs()
    threshold = 'v >= 30 and E > 0'
    reset = 'v = c; u += d; E -= spike_cost'
    
    neurons = NeuronGroup(N_total, eqs, threshold=threshold, reset=reset, method='euler', refractory=2*ms)
    
    # Subgroups
    P_exc = neurons[:N_exc]
    P_inh = neurons[N_exc:]
    
    # Initial Parameters setup
    neurons.v = '-65 + 20*rand()' # Randomized initial voltages closer to threshold
    neurons.E = 1.0
    neurons.spike_cost = 0.1
    neurons.I = 0.0
    
    # Excitatory (RS)
    P_exc.a = 0.02
    P_exc.b = 0.2
    P_exc.c = -65.0
    P_exc.d = 8.0
    P_exc.tau_recovery = 1000*ms # Tuning (Day 4 candidate) - much slower recovery
    
    # Inhibitory (FS)
    P_inh.a = 0.1
    P_inh.b = 0.2
    P_inh.c = -65.0
    P_inh.d = 2.0
    P_inh.tau_recovery = 500*ms
    
    neurons.u = 'b * v'
    
    # 2. Wiring (Day 2)
    S_EE = Synapses(P_exc, P_exc, on_pre='v += 15; E -= 0.01') # Increased w_exc
    S_EE.connect(p=0.1)
    
    S_EI = Synapses(P_exc, P_inh, on_pre='v += 15; E -= 0.01')
    S_EI.connect(p=0.1)
    
    S_IE = Synapses(P_inh, P_exc, on_pre='v -= 25') # Increased w_inh
    S_IE.connect(p=0.1)
    
    S_II = Synapses(P_inh, P_inh, on_pre='v -= 25')
    S_II.connect(p=0.1)
    
    # 3. Random Noise (Day 3)
    noise = PoissonInput(neurons, 'v', 1, 100*Hz, weight=5) # Balanced noise
    
    # 4. Monitors
    spike_mon = SpikeMonitor(neurons)
    pop_mon = PopulationRateMonitor(neurons)
    
    print(f"Starting simulation for {duration}...")
    run(duration, report='text')
    
    # 5. Visualization (Day 5)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(spike_mon.t/second, spike_mon.i, '.k', ms=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot (Seizure Test)')
    
    plt.subplot(212)
    plt.plot(pop_mon.t/second, pop_mon.smooth_rate(window='flat', width=100*ms)/Hz)
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Network Firing Rate')
    
    plt.tight_layout()
    plt.savefig(plot_output)
    print(f"Result saved to {plot_output}")
    
    avg_rate = len(spike_mon.i) / (N_total * duration / second)
    print(f"Average Firing Rate: {avg_rate:.2f} Hz")
    
    return avg_rate

if __name__ == "__main__":
    run_seizure_test()
