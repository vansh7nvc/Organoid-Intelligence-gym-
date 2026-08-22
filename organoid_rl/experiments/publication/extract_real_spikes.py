import gymnasium as gym
import numpy as np
import torch
import os
import pickle
from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent

def extract():
    print("Initializing environment for spike extraction...")
    env = OrganoidEnv()
    agent = DQNAgent(obs_dim=21, n_actions=8)
    
    # Try to load trained weights
    weight_path = r'c:\Users\Acer\OneDrive\Desktop\Organoid Intelligence\organoid_rl\experiments\results\brain_month6_final_ep450.pth'
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        try:
            # Depending on how DQNAgent is implemented, weight loading might vary
            # Assuming agent has a load method or we can load state_dict
            state_dict = torch.load(weight_path, map_location='cpu')
            if 'model' in state_dict:
                agent.policy_net.load_state_dict(state_dict['model'])
            else:
                agent.policy_net.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}. Proceeding with random agent.")
    else:
        print("No weights found. Using random agent.")

    obs, _ = env.reset()
    all_spikes_i = []
    all_spikes_t = []
    motor_up_rates = []
    motor_down_rates = []
    
    print("Running episode to capture spike data...")
    from brian2 import ms
    for step in range(100): # 100 steps * 50ms = 5000ms of data
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access the spike monitor directly from the environment's Brian2 network
        # OrganoidEnv should have a spike_mon attribute based on core.py
        if hasattr(env, 'spike_mon'):
            # Fetch spikes from the last 50ms window
            current_t = env.network.t
            mask = env.spike_mon.t > (current_t - 50*ms) # Brian2 units (ms)
            all_spikes_i.extend(env.spike_mon.i[mask].astype(int).tolist())
            # Store time relative to episode start
            all_spikes_t.extend(np.array(env.spike_mon.t[mask]).tolist())
            
            # Extract motor activation for Fig 6B
            # Quadrants defined in core.py: Up (400-425), Down (425-450)
            motor_indices = env.spike_mon.i[mask]
            up_count = np.sum((motor_indices >= 400) & (motor_indices < 425))
            down_count = np.sum((motor_indices >= 425) & (motor_indices < 450))
            # Convert counts to Hz (50ms window)
            motor_up_rates.append(float(up_count / 0.05 / 25)) # Hz per neuron in quadrant
            motor_down_rates.append(float(down_count / 0.05 / 25))
            
        if terminated or truncated:
            break

    # Save data
    output_path = r'c:\Users\Acer\OneDrive\Desktop\Organoid Intelligence\organoid_rl\experiments\results\real_spikes.npz'
    np.savez(output_path, 
             i=np.array(all_spikes_i), 
             t=np.array(all_spikes_t),
             motor_up=np.array(motor_up_rates),
             motor_down=np.array(motor_down_rates))
    
    print(f"Saved {len(all_spikes_i)} spikes to {output_path}")

if __name__ == "__main__":
    # Ensure relative imports work by calling from root if needed
    # But here we use absolute paths for simplicity in this scratch script
    extract()
