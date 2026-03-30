from brian2 import *
from organoid_rl.environment.neurons import create_metabolic_neurons
import matplotlib.pyplot as plt

# Day 3 Stress Test: Proving the Metabolic Stabilizer
# Standard Izhikevich would fire forever at high current. 
# Ours should self-regulate as energy E depletes.

# 1. Setup high-performance backend
prefs.codegen.target = 'cython'
prefs.codegen.cpp.compiler = 'mingw32'
prefs.codegen.cpp.extra_compile_args = ['-DMS_WIN64']

# 2. Create single neuron with aggressive firing parameters
N = 1
# Using slightly higher cost to see depletion faster in a 5s window
G = create_metabolic_neurons(N, model_type='RS')
G.spike_cost = 0.05
G.tau_recovery = 150*ms # Slower recovery to emphasize exhaustion

# 3. Inject "dangerously high" current (normalized for our unitless equations)
# In Izhikevich 2003, I=10 is high, I=20 is extreme. 
# We'll use 25.0 to force Phase 1 (Gamma) -> Phase 2 (Slowing) -> Phase 3 (Exhaustion)
G.I = 25.0 

# 4. Monitors
state_mon = StateMonitor(G, ['v', 'E'], record=True)
spike_mon = SpikeMonitor(G)

# 5. Run for 5 seconds
print("Starting 5s Stress Test...")
run(5000*ms)

# 6. Quantitative Analysis
n_spikes = spike_mon.count[0]
avg_rate = n_spikes / 5.0
print(f"Total spikes in 5s: {n_spikes}")
print(f"Average firing rate: {avg_rate:.2f} Hz")

# 7. Visualization
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(state_mon.t/second, state_mon.v[0])
plt.title('Day 3 Stress Test: Single Neuron Metabolic Regulation')
plt.ylabel('Membrane (v)')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(state_mon.t/second, state_mon.E[0], 'r', label='Energy (E)')
plt.axhline(0, color='k', linestyle='--', label='Death Zone')
plt.ylabel('ATP Reserves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
# Sliding window firing rate to see Phase 1 -> 3
window = 100*ms
bin_edges = np.arange(0, 5.0, window/second)
spike_times = spike_mon.t/second
counts, _ = np.histogram(spike_times, bins=bin_edges)
rates = counts / (window/second)
plt.plot(bin_edges[:-1], rates, color='g', drawstyle='steps-post')
plt.ylabel('Firing Rate (Hz)')
plt.xlabel('Time (s)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('organoid_rl/experiments/01_stress_test_result.png')
print("Stress test complete. Result saved to 'experiments/01_stress_test_result.png'")
