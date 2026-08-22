from stable_baselines3 import DQN, PPO
import os

def create_dqn_agent(env, learning_rate=1e-4, buffer_size=100000, batch_size=64, seed=None):
    """
    Creates a DQN agent matching the paper's D3QN settings as closely as possible.
    """
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        seed=seed,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0
    )
    return model

def create_ppo_agent(env, learning_rate=3e-4, batch_size=64, n_steps=2048, seed=None):
    """
    Creates a PPO agent baseline.
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0
    )
    return model
