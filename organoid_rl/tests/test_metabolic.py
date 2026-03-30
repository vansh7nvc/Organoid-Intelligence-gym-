from brian2 import *
from organoid_rl.environment.neurons import create_metabolic_neurons
import matplotlib.pyplot as plt

# 1. Setup
N = 1
G = create_metabolic_neurons(N, model_type='RS')

# Apply high constant input to force rapid firing
G.I = 15.0 

# 2. Monitor v and E
statemon = StateMonitor(G, ['v', 'E'], record=True)
spikemon = SpikeMonitor(G)

# 3. Run
run(500*ms)

# 4. Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(statemon.t/ms, statemon.v[0])
plt.ylabel('v (Membrane)')
plt.title('Metabolic Izhikevich: Firing and Energy Depletion')

plt.subplot(2, 1, 2)
plt.plot(statemon.t/ms, statemon.E[0], 'r')
plt.axhline(0, color='k', linestyle='--')
plt.ylabel('E (Energy)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.savefig('metabolic_test.png')
print("Simulation finished. Plot saved as 'metabolic_test.png'")
