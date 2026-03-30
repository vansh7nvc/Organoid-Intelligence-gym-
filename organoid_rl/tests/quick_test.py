import sys
import os
import numpy as np
from brian2 import *

# Correct the path to import environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from organoid_rl.environment.core import OrganoidEnv

def test_single_step():
    print("Initializing environment...")
    prefs.codegen.target = 'numpy'
    env = OrganoidEnv()
    print("Resetting environment...")
    env.reset()
    print("Running a single step...")
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step complete. Reward: {reward}")
    print("Success!")

if __name__ == "__main__":
    test_single_step()
