from environment.core import OrganoidEnv
import numpy as np
import matplotlib.pyplot as plt

def run_closed_loop_test():
    env = OrganoidEnv()
    obs, _ = env.reset()
    
    positions = []
    positions.append(env.cursor_pos.copy())
    
    print("Starting closed-loop cursor test (20 steps)...")
    print(f"Initial Position: {env.cursor_pos}")
    
    for i in range(20):
        # Action: Random stimulation to see resultant movement
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        
        pos = np.array(info['cursor_pos'])
        positions.append(pos)
        
        if i % 5 == 0:
            print(f"Step {i}: Position {pos}, Spikes {info['total_spikes']}")

    positions = np.array(positions)
    
    # Plot Trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], '-o', markersize=4, label='Cursor Path')
    plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Closed-Loop Organoid Cursor Movement")
    plt.legend()
    
    output_file = 'experiments/05_cursor_test_result.png'
    plt.savefig(output_file)
    print(f"Success! Trajectory saved to {output_file}")

if __name__ == "__main__":
    run_closed_loop_test()
