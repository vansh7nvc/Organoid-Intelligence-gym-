import gymnasium as gym
import numpy as np
import sys
import os

# Add parent directory to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from organoid_rl.environment.core import OrganoidEnv

def test_sdm_sparsity():
    print("Testing SDM Sparsity and Uniqueness...")
    env = OrganoidEnv()
    env.reset()
    
    # 1. State 1: Cursor at [0.1, 0.1]
    env.cursor_pos = np.array([0.1, 0.1])
    obs1 = env._get_obs()
    env.neurons.I[:] = 0 # CLEAR PROPERLY
    env._stimulate_sdm(obs1)
    driven_units1 = np.array(np.where(env.neurons.I[:512] > 0)[0])
    
    # 2. State 2: Cursor at [0.9, 0.9]
    env.cursor_pos = np.array([0.9, 0.9])
    obs2 = env._get_obs()
    env.neurons.I[:] = 0 # CLEAR PROPERLY
    env._stimulate_sdm(obs2)
    driven_units2 = np.array(np.where(env.neurons.I[:512] > 0)[0])
    
    sparsity1 = len(driven_units1) / 512
    sparsity2 = len(driven_units2) / 512
    
    # Convert to sets for intersection
    set1 = set(driven_units1)
    set2 = set(driven_units2)
    overlap = len(set1.intersection(set2))
    
    print(f"State 1 Sparsity: {sparsity1:.2%}")
    print(f"State 2 Sparsity: {sparsity2:.2%}")
    print(f"Overlap between distinct states: {overlap} neurons")
    
    if sparsity1 == 0 or sparsity1 > 0.4:
         print(f"FAILED: Sparsity out of expected range ({sparsity1:.2%})")
         return False
         
    if overlap > len(set1) * 0.7:
         print(f"FAILED: Too much overlap ({overlap} / {len(set1)})")
         return False
         
    print("SDM Sparsity Check PASSED.")
    return True

def test_obstacle_penalties():
    print("\nTesting Obstacle Penalties...")
    env = OrganoidEnv()
    env.reset()
    
    # 1. Clear area
    env.cursor_pos = np.array([0.1, 0.1])
    reward_clear = env._calculate_reward()
    
    # 2. Inside circular obstacle at [0.5, 0.5]
    env.cursor_pos = np.array([0.5, 0.5])
    reward_obs = env._calculate_reward()
    
    print(f"Clear Reward: {reward_clear}")
    print(f"Obstacle Reward: {reward_obs}")
    
    if reward_obs > reward_clear - 15.0:
        print("FAILED: Obstacle penalty not strong enough.")
        return False
        
    print("Obstacle Penalty Check PASSED.")
    return True

def run_stability_check():
    print("\nRunning Stability Check (Dual-Trace)...")
    env = OrganoidEnv()
    obs, _ = env.reset()
    
    for i in range(10):
        # Sample random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Spikes={info['total_spikes']}")
        
        if np.any(np.isnan(env.neurons.v)):
            print("FAILURE: NaNs detected in voltage!")
            return False
            
    print("Stability Check PASSED.")
    return True

if __name__ == "__main__":
    s1 = test_sdm_sparsity()
    s2 = test_obstacle_penalties()
    s3 = run_stability_check()
    
    if s1 and s2 and s3:
        print("\nALL MONTH 4 DIAGNOSTICS PASSED.")
        sys.exit(0)
    else:
        print("\nDIAGNOSTICS FAILED.")
        sys.exit(1)
