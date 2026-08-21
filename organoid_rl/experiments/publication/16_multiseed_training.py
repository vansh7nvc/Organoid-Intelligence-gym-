import os
import sys
import numpy as np
import time
import ray
from ray import train, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent
def get_curriculum_stage(episode):
    """Returns the difficulty stage based on episode number."""
    if episode < 100:
        return 1   # Easy: no obstacles, single target, large radius
    elif episode < 200:
        return 2   # Medium: obstacles enabled, single target
    elif episode < 350:
        return 3   # Hard: obstacles + multi-goal
    else:
        return 4   # Full: obstacles + multi-goal + tight radius

def apply_inhibitory_homeostasis(env):
    """Maintain E/I balance in structured networks."""
    ee_weights = []
    for i in range(len(env.synapses) - 1):
        ee_weights.append(np.mean(np.array(env.synapses[i].w)))
    
    mean_ee_w = np.mean(ee_weights)
    ideal_ratio = 1.5
    
    S_INH = env.synapses[-1]
    current_ie_w = np.array(S_INH.w)
    target_ie_w = mean_ee_w * ideal_ratio
    
    updated_ie_w = current_ie_w + 0.1 * (target_ie_w - current_ie_w)
    S_INH.w = np.clip(updated_ie_w, 0, 100.0)

def train_organoid_seed(config):
    """
    Ray Tune Trainable function for OrganoidEnv.
    """
    seed = config.get("seed", 42)
    total_episodes = config.get("total_episodes", 500)
    steps_per_episode = config.get("steps_per_episode", 80)
    
    np.random.seed(seed)
    
    # Initialize environment and agent
    env = OrganoidEnv()
    
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        n_step=3,
        batch_size=64,
        buffer_size=50000,
        her_k=4
    )
    
    # Metrics
    goals_reached = 0
    context_success = {0: 0, 1: 0}
    stage_goals = {1: 0, 2: 0, 3: 0, 4: 0}
    stage_episodes = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for ep in range(total_episodes):
        stage = get_curriculum_stage(ep)
        env.set_curriculum_stage(stage)
        stage_episodes[stage] += 1
        
        obs, _ = env.reset()
        context = env.active_target_idx
        total_reward = 0
        episode_transitions = []
        loss_history = []
        
        for s in range(steps_per_episode):
            action = agent.choose_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_transitions.append((obs.copy(), action, reward, next_obs.copy(), done))
            
            if s % 4 == 0:
                loss = agent.learn()
                if loss is not None:
                    loss_history.append(loss)
            
            obs = next_obs
            total_reward += reward
            
            if done:
                goals_reached += 1
                context_success[context] += 1
                stage_goals[stage] += 1
                break
                
        agent.flush_n_step_buffer()
        
        if not done and len(episode_transitions) > 5:
            agent.apply_her(episode_transitions)
            
        apply_inhibitory_homeostasis(env)
        if ep % 10 == 0 and ep > 0:
            env.apply_structural_plasticity()
            
        success = 1.0 if total_reward > 30 else 0.0
        avg_loss = float(np.mean(loss_history)) if loss_history else 0.0
        
        # Report metrics to Ray Tune
        train.report({
            "episode": ep,
            "reward": total_reward,
            "success": success,
            "stage": stage,
            "loss": avg_loss,
            "cumulative_goals": goals_reached
        })

if __name__ == "__main__":
    ray.init()
    
    # Run 5 seeds in parallel
    num_samples = 5
    seeds = [42, 123, 456, 789, 1024]
    
    search_space = {
        "seed": tune.grid_search(seeds),
        "total_episodes": 500,
        "steps_per_episode": 80
    }
    
    tuner = tune.Tuner(
        train_organoid_seed,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="reward",
            mode="max",
            num_samples=1 # since grid_search does the 5 samples
        ),
        run_config=tune.RunConfig(
            name="organoid_multiseed",
            storage_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "ray_results"))
        )
    )
    
    results = tuner.fit()
    print("Multi-seed training completed!")
