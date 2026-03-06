from environment.core import OrganoidEnv
import numpy as np
import matplotlib.pyplot as plt
from environment.rewards import apply_dopamine

def run_delayed_reward_test():
    env = OrganoidEnv()
    env.reset()
    
    # Select a synapse group to monitor
    # S_EE is the first synapse group added in _build_network
    S_EE = env.synapses[0]
    
    # Find synapses where pre-synaptic neuron is in 0-50 (Action 0 target)
    target_indices = np.where(S_EE.i < 50)[0]
    # Control group: 50-100
    control_indices = np.where((S_EE.i >= 50) & (S_EE.i < 100))[0]
    
    # Store initial mean weights
    w_target_init = np.mean(np.array(S_EE.w[target_indices]))
    w_control_init = np.mean(np.array(S_EE.w[control_indices]))
    
    print(f"Initial Weights: Target={w_target_init:.4f}, Control={w_control_init:.4f}")
    
    # Step 1: Stimulate (Action 0)
    print("Step 1: Stimulation (Action 0)")
    # Action 0 injects current into neurons 0-50
    obs, _, _, _, _ = env.step(0)
    
    # Step 2: Delay (Action 7 - neurons 350-400)
    # This serves as a delay where target trace should decay but stay positive
    print("Step 2: Delay (Action 7)")
    obs, _, _, _, _ = env.step(7)
    
    # Step 3: Apply Reward
    print("Step 3: Applying Reward (100.0)")
    # Manual dopamine injection
    apply_dopamine(env.synapses, reward=100.0) 
    
    # Measure new weights
    w_target_final = np.mean(np.array(S_EE.w[target_indices]))
    w_control_final = np.mean(np.array(S_EE.w[control_indices]))
    
    print(f"Final Weights: Target={w_target_final:.4f}, Control={w_control_final:.4f}")
    
    # Trace values
    t_target = np.mean(np.array(S_EE.Trace[target_indices]))
    t_control = np.mean(np.array(S_EE.Trace[control_indices]))
    print(f"Traces: Target={t_target:.4f}, Control={t_control:.4f}")
    
    # Validation
    delta_target = w_target_final - w_target_init
    delta_control = w_control_final - w_control_init
    
    print(f"Delta: Target={delta_target:.4f}, Control={delta_control:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Target (Stimulated)', 'Control (Unstimulated)'], [delta_target, delta_control], color=['green', 'gray'])
    plt.ylabel('Weight Change')
    plt.title('Effect of Delayed Reward on Synaptic Weights')
    plt.text(0, delta_target, f"{delta_target:.4f}", ha='center', va='bottom')
    plt.text(1, delta_control, f"{delta_control:.4f}", ha='center', va='bottom')
    plt.ylim(bottom=min(0, delta_control, delta_target) * 1.1)
    
    output_file = 'experiments/04_delayed_reward_result.png'
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
    
    if delta_target > delta_control and delta_target > 0:
        print("SUCCESS: Target synapses strengthened significantly more than control.")
    else:
        print("FAILURE: Hypothesis not supported.")

if __name__ == "__main__":
    run_delayed_reward_test()
