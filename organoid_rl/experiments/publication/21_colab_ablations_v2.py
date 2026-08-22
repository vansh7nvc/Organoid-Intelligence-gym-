"""
OrganoidEnv Colab Ablation Training (v2)
Optimized for Google Colab to evaluate the two most critical ablation claims:
1. No GAR (No Global Activity Regulator)
2. No SDM (No Spatial Distributed Memory layer)

Configuration:
- 3 seeds (42, 123, 456)
- 100 episodes per config (evaluates base learning capability)
- Google Drive checkpointing
- Per-episode CSV logging
"""

import os
import sys
import time
import json
import signal
import pandas as pd
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent

# --- CONFIGURATION ---
SEEDS = [42, 123, 456]
EPISODES_PER_RUN = 100
CHECKPOINT_FREQ = 25
GDRIVE_DIR = "/content/drive/MyDrive/OrganoidEnv_Results/ablations_v2"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "colab_results_v2")

# Determine save directory (Drive if on Colab, Local otherwise)
try:
    import google.colab
    IN_COLAB = True
    SAVE_DIR = GDRIVE_DIR
    print("Running in Google Colab.")
    
    # Check if drive is mounted
    if not os.path.exists("/content/drive/MyDrive"):
        print("WARNING: Google Drive not mounted! Results will be lost when instance terminates.")
        print("Please mount drive: from google.colab import drive; drive.mount('/content/drive')")
        SAVE_DIR = "/content/colab_ablations_v2"
except ImportError:
    IN_COLAB = False
    SAVE_DIR = LOCAL_DIR
    print("Running locally.")

os.makedirs(SAVE_DIR, exist_ok=True)

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    global SHUTDOWN_REQUESTED
    print("\n\n[WARNING] Interrupt received! Gracefully saving and shutting down...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

ABLATIONS = {
    "No_GAR": {"use_sdm": True, "use_morphology": True, "use_dual_trace": True, "use_stabilizer": False},
    "No_SDM": {"use_sdm": False, "use_morphology": True, "use_dual_trace": True, "use_stabilizer": True}
}

def run_ablation(ablation_name, config, seed):
    global SHUTDOWN_REQUESTED
    run_id = f"{ablation_name}_seed_{seed}"
    print(f"\n{'='*50}\nStarting Training for {run_id}\n{'='*50}")
    
    np.random.seed(seed)
    env = OrganoidEnv(**config)
    
    # Initialize Agent
    agent = DQNAgent(
        obs_dim=21,
        n_actions=8,
        batch_size=64,
        buffer_size=100000,
        lr=1e-4,
        gamma=0.99,
        tau=0.005
    )
    
    # CSV Log file setup
    log_file = os.path.join(SAVE_DIR, f"training_log_{run_id}.csv")
    start_episode = 0
    cumulative_goals = 0
    
    if os.path.exists(log_file) and os.path.getsize(log_file) > 100:
        # Load existing progress
        try:
            df = pd.read_csv(log_file)
            if not df.empty and 'episode' in df.columns:
                max_ep = df['episode'].max()
                if not pd.isna(max_ep):
                    start_episode = int(max_ep) + 1
                    cumulative_goals = df['success'].sum()
            print(f"Resuming {run_id} from episode {start_episode}")
        except Exception as e:
            print(f"Could not read existing log, starting from 0: {e}")
            
        # Load agent weights if available
        weight_file = os.path.join(SAVE_DIR, f"agent_weights_{run_id}.pth")
        if os.path.exists(weight_file):
            agent.load_checkpoint(weight_file)
            print("Loaded saved agent weights.")
    else:
        # Initialize CSV
        with open(log_file, "w") as f:
            f.write("ablation,seed,episode,reward,success,stage,loss,duration_sec,cumulative_goals\n")
            
    if start_episode >= EPISODES_PER_RUN:
        print(f"Run {run_id} already completed {EPISODES_PER_RUN} episodes. Skipping.")
        return
        
    start_time = time.time()
    
    for episode in range(start_episode, EPISODES_PER_RUN):
        if SHUTDOWN_REQUESTED:
            break
            
        ep_start_time = time.time()
        
        # 1. Reset Environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Determine curriculum stage (1=Basic, 2=Obstacles, 3=Multi-goal)
        stage = 1
        if episode >= 50:
            stage = 2
        if episode >= 75:
            stage = 3
            
        env.set_curriculum_stage(stage)
        
        # 2. Episode Loop
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count >= 80:
                done = True
                
        # 3. Structural Plasticity (every 10 episodes)
        if episode % 10 == 0:
            env.apply_structural_plasticity()
            
        # 4. Metrics & Logging
        ep_duration = time.time() - ep_start_time
        success = 1 if info.get('success', False) else 0
        cumulative_goals += success
        loss = agent.losses[-1] if len(agent.losses) > 0 else 0.0
        
        # Format log line
        log_line = f"{ablation_name},{seed},{episode},{total_reward:.4f},{success},{stage},{loss:.4f},{ep_duration:.2f},{cumulative_goals}\n"
        with open(log_file, "a") as f:
            f.write(log_line)
            
        # Console output
        print(f"[{ablation_name}] Seed {seed} | Ep {episode:3d}/{EPISODES_PER_RUN} | Reward: {total_reward:6.1f} | Success: {success} | Loss: {loss:.4f} | Time: {ep_duration:.1f}s")
        
        # 5. Checkpointing
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            weight_file = os.path.join(SAVE_DIR, f"agent_weights_{run_id}.pth")
            agent.save_checkpoint(weight_file)
            print(f"--> Saved checkpoint to {weight_file}")
            
    total_time = time.time() - start_time
    print(f"\nFinished {run_id} in {total_time/3600:.2f} hours.")
    
    
if __name__ == "__main__":
    print(f"Colab Ablation Runner V2. Saving to {SAVE_DIR}")
    for name, config in ABLATIONS.items():
        for s in SEEDS:
            if SHUTDOWN_REQUESTED:
                break
            run_ablation(name, config, s)
            
    print("\nTraining script completed successfully.")
