"""
Month 6: Grand Unification Training Script.
Dueling Double DQN + Curriculum Learning + HER + Structural Plasticity.
Target: 85-95% success rate.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent

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

def train_month6(total_episodes=500, steps_per_episode=80, tag="month6_final"):
    print(f"=" * 60)
    print(f"  MONTH 6: GRAND UNIFICATION [{tag}]")
    print(f"  Target: 85-95% success rate")
    print(f"  Episodes: {total_episodes} | Steps/ep: {steps_per_episode}")
    print(f"=" * 60, flush=True)
    
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
    rewards_history = []
    goals_reached = 0
    context_success = {0: 0, 1: 0}
    stage_goals = {1: 0, 2: 0, 3: 0, 4: 0}
    stage_episodes = {1: 0, 2: 0, 3: 0, 4: 0}
    losses = []
    
    save_dir = "experiments/results"
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    
    for ep in range(total_episodes):
        # Curriculum: set difficulty stage
        stage = get_curriculum_stage(ep)
        env.set_curriculum_stage(stage)
        stage_episodes[stage] += 1
        
        obs, _ = env.reset()
        context = env.active_target_idx
        total_reward = 0
        episode_transitions = []
        
        for s in range(steps_per_episode):
            action = agent.choose_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            
            # Store for DQN
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_transitions.append((obs.copy(), action, reward, next_obs.copy(), done))
            
            # Learn every 4 steps
            if s % 4 == 0:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            
            obs = next_obs
            total_reward += reward
            
            if done:
                goals_reached += 1
                context_success[context] += 1
                stage_goals[stage] += 1
                break
        
        # Flush remaining n-step transitions
        agent.flush_n_step_buffer()
        
        # Hindsight Experience Replay
        if not done and len(episode_transitions) > 5:
            agent.apply_her(episode_transitions)
        
        # Homeostasis + Structural Plasticity
        apply_inhibitory_homeostasis(env)
        if ep % 10 == 0 and ep > 0:
            env.apply_structural_plasticity()
        
        rewards_history.append(total_reward)
        
        # Logging
        if ep % 10 == 0:
            recent_rate = sum(1 for r in rewards_history[-20:] if r > 30) / min(20, len(rewards_history))
            avg_loss = np.mean(losses[-100:]) if losses else 0
            elapsed = time.time() - start_time
            print(f"[Ep {ep:>3d}] Stage {stage} | Rwd: {total_reward:>8.2f} | "
                  f"Goals: {goals_reached} (C0:{context_success[0]} C1:{context_success[1]}) | "
                  f"Recent%: {recent_rate:.0%} | Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.0f}s", flush=True)
        
        # Periodic save (JSON + Brain)
        if ep % 50 == 0 and ep > 0:
            _save_results(tag, ep, total_episodes, goals_reached, context_success,
                         stage_goals, stage_episodes, rewards_history, losses, start_time, save_dir)
            agent.save_checkpoint(f"{save_dir}/brain_{tag}_ep{ep}.pth")
            print(f"Checkpoint saved: brain_{tag}_ep{ep}.pth", flush=True)
    
    end_time = time.time()
    
    # Final save
    results = _save_results(tag, total_episodes, total_episodes, goals_reached, context_success,
                           stage_goals, stage_episodes, rewards_history, losses, start_time, save_dir)
    
    # Generate plots
    _generate_plots(rewards_history, losses, stage_goals, stage_episodes,
                   context_success, save_dir)
    
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total Goals: {goals_reached}/{total_episodes} ({goals_reached/total_episodes:.1%})")
    print(f"  Duration: {end_time - start_time:.0f}s")
    print(f"  Stage Success Rates:")
    for s in [1, 2, 3, 4]:
        if stage_episodes[s] > 0:
            print(f"    Stage {s}: {stage_goals[s]}/{stage_episodes[s]} ({stage_goals[s]/stage_episodes[s]:.1%})")
    print(f"{'=' * 60}", flush=True)
    
    return results

def _save_results(tag, ep, total_episodes, goals_reached, context_success,
                 stage_goals, stage_episodes, rewards_history, losses, start_time, save_dir):
    results = {
        'tag': tag,
        'episodes_completed': ep,
        'total_episodes': total_episodes,
        'goals_reached': goals_reached,
        'context_success': context_success,
        'stage_goals': stage_goals,
        'stage_episodes': stage_episodes,
        'rewards': rewards_history,
        'avg_loss': float(np.mean(losses[-100:])) if losses else 0,
        'duration': time.time() - start_time
    }
    
    save_path = f"{save_dir}/results_{tag}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)
    return results

def _generate_plots(rewards, losses, stage_goals, stage_episodes,
                   context_success, save_dir):
    """Generate comprehensive Month 6 dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Month 6: Grand Unification — Results Dashboard", fontsize=16, fontweight='bold')
    
    # 1. Reward Curve with Curriculum Markers
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, color='steelblue')
    # Moving average
    if len(rewards) > 20:
        ma = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax.plot(range(19, len(rewards)), ma, color='navy', linewidth=2, label='20-ep MA')
    # Stage transitions
    for ep, label in [(100, 'Stage 2'), (200, 'Stage 3'), (350, 'Stage 4')]:
        if ep < len(rewards):
            ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
            ax.text(ep+2, ax.get_ylim()[1]*0.9, label, fontsize=8, color='red')
    ax.set_title("Training Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Loss Curve
    ax = axes[0, 1]
    if losses:
        ax.plot(losses, alpha=0.2, color='coral')
        if len(losses) > 50:
            ma = np.convolve(losses, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(losses)), ma, color='darkred', linewidth=2)
    ax.set_title("DQN Training Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Huber Loss")
    ax.grid(True, alpha=0.3)
    
    # 3. Stage Success Rates
    ax = axes[0, 2]
    stages = [1, 2, 3, 4]
    rates = [stage_goals[s]/max(1, stage_episodes[s]) for s in stages]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar([f"Stage {s}" for s in stages], [r*100 for r in rates], color=colors)
    ax.set_title("Success Rate by Curriculum Stage")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{rate:.0%}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Context Success Split
    ax = axes[1, 0]
    ctx_labels = ['Context 0\n(0.8, 0.8)', 'Context 1\n(0.2, 0.2)']
    ctx_values = [context_success[0], context_success[1]]
    ax.bar(ctx_labels, ctx_values, color=['#3498db', '#e67e22'])
    ax.set_title("Goals by Context")
    ax.set_ylabel("Goals Reached")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Reward Distribution
    ax = axes[1, 1]
    ax.hist(rewards, bins=30, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axvline(x=np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Cumulative Success Rate
    ax = axes[1, 2]
    cum_goals = np.cumsum([1 if r > 30 else 0 for r in rewards])
    cum_rate = cum_goals / np.arange(1, len(rewards) + 1)
    ax.plot(cum_rate * 100, color='green', linewidth=2)
    ax.set_title("Cumulative Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{save_dir}/month6_dashboard.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Dashboard saved to {plot_path}", flush=True)

if __name__ == "__main__":
    train_month6(total_episodes=500, steps_per_episode=80)
