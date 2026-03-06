from brian2 import *
from organoid_rl.environment.neurons import create_metabolic_neurons, get_metabolic_izhikevich_eqs
import matplotlib.pyplot as plt

# Day 4-5: Synaptic Transmission & Transmission Delays
# Proving that delays are necessary for causal stability and STDP setup.

# 1. Configuration
prefs.codegen.target = 'cython'
prefs.codegen.cpp.compiler = 'mingw32'
prefs.codegen.cpp.extra_compile_args = ['-DMS_WIN64']

# 2. Create 2 Neurons (Source and Target)
G = create_metabolic_neurons(2, model_type='RS')
G.I = [15.0, 0] # Stimulate only the first neuron

# 3. Synapse Model with Delay Factor
S = Synapses(G, G, model='w : 1', on_pre='v_post += w')
S.connect(i=0, j=1)

# Set weight and the crucial Transmission Delay
S.w = 20.0 # Increase weight to ensure post-spike
S.delay = '1*ms + rand()*2*ms'

# 4. Monitors - MUST be initialized before run()
v_mon = StateMonitor(G, 'v', record=True)
s_mon = SpikeMonitor(G)

# 5. Run
print("Simulating synaptic transmission with axonal delays...")
run(200*ms)

# 6. Verification
plt.figure(figsize=(10, 6))
plt.plot(v_mon.t/ms, v_mon.v[0], label='Pre-neuron (N0)')
plt.plot(v_mon.t/ms, v_mon.v[1], label='Post-neuron (N1)')

# Highlights causal gap
pre_spikes = s_mon.t[s_mon.i == 0]
post_spikes = s_mon.t[s_mon.i == 1]
if len(pre_spikes) > 0 and len(post_spikes) > 0:
    plt.axvspan(pre_spikes[0]/ms, post_spikes[0]/ms, color='gray', alpha=0.2, label='Transmission Delay')

plt.title('Day 4-5: Synaptic Transmission & Axonal Delays')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (v)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('organoid_rl/experiments/02_synapse_delay_test.png')
print("Verification complete. Results saved to 'organoid_rl/experiments/02_synapse_delay_test.png'")
