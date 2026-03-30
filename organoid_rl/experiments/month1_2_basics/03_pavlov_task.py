from environment.core import OrganoidEnv
import time
import numpy as np

def run_pavlov_task():
    env = OrganoidEnv()
    obs, _ = env.reset()
    
    print("Starting Pavlov Task (Week 3 Check)...")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    steps = 20
    print(f"Running for {steps} steps...")
    
    start_time = time.time()
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check observation validity
        valid = np.all((obs >= 0) & (obs <= 1))
        status = "VALID" if valid else "INVALID"
        
        mean_activity = obs.mean()
        print(f"Step {i}: Action={action}, Obs Mean={mean_activity:.4f} [{status}]")
        
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f}s")
    print("Success! Environment loop works.")

if __name__ == "__main__":
    run_pavlov_task()
