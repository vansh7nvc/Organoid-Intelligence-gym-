"""
Quick diagnostic: Run 1 episode with random actions and verify cursor movement.
Tests the fixed motor neuron targeting.
"""
import sys, os, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from organoid_rl.environment.core import OrganoidEnv

env = OrganoidEnv()
obs, _ = env.reset()

print(f"Target: {env.target_pos}")
print(f"Start:  {env.cursor_pos}")
print(f"Goal Radius: {env.goal_radius}")
init_dist = np.linalg.norm(env.cursor_pos - env.target_pos)
print(f"Initial Distance: {init_dist:.4f}")
print(f"\n{'Step':>4} | {'X':>7} | {'Y':>7} | {'Dist':>6} | {'Act':>3} | {'dX':>7} | {'dY':>7} | {'Rwd':>7}")
print("-" * 70)

prev_pos = env.cursor_pos.copy()
total_reward = 0
for step in range(80):
    action = np.random.randint(env.action_space.n)
    obs, reward, done, trunc, info = env.step(action)
    
    dx = env.cursor_pos[0] - prev_pos[0]
    dy = env.cursor_pos[1] - prev_pos[1]
    dist = np.linalg.norm(env.cursor_pos - env.target_pos)
    total_reward += reward
    
    print(f"{step:4d} | {env.cursor_pos[0]:7.4f} | {env.cursor_pos[1]:7.4f} | {dist:6.4f} | {action:3d} | {dx:+7.4f} | {dy:+7.4f} | {reward:7.2f}")
    
    prev_pos = env.cursor_pos.copy()
    
    if done:
        print(f"\nGOAL REACHED at step {step}!")
        break

final_dist = np.linalg.norm(env.cursor_pos - env.target_pos)
displacement = np.linalg.norm(env.cursor_pos - np.array([0.1, 0.9]))
print(f"\nFinal Pos: {env.cursor_pos}, Final Dist: {final_dist:.4f}")
print(f"Net Displacement from start: {displacement:.4f}")
print(f"Total Reward: {total_reward:.2f}")
print(f"\nVERDICT: {'PASS - cursor moves meaningfully' if displacement > 0.15 else 'FAIL - cursor barely moved'}")
