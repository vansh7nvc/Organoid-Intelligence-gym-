import time
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from organoid_rl.environment.core import OrganoidEnv

def benchmark(episodes=5):
    print(f"Starting Speed Benchmark ({episodes} episodes)...")
    env = OrganoidEnv()
    
    start_time = time.time()
    for ep in range(episodes):
        env.reset()
        for _ in range(40): # 40 steps
            action = env.action_space.sample()
            env.step(action)
        print(f"Ep {ep} complete")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Time for {episodes} eps: {total_time:.2f}s")
    print(f"Avg Time per episode: {total_time/episodes:.2f}s")
    
    # Comparison note: Previous baseline (1000 neurons, 0.05ms dt) was ~36s/episode (3 min per 5 eps).
    # Expected: ~5-10s per episode.

if __name__ == "__main__":
    benchmark()
