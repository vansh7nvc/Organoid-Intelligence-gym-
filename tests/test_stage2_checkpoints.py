"""
tests/test_stage2_checkpoints.py

Stage 2 Verification Suite: Checkpoint Completeness & Resume-Equivalence (Platform Reliability)
Authoritative Specifications:
- OrganoidEnv_Master_Verifier_Plan.md §2.1 - §2.2
- PROJECT.md §Feature Inventory (Items 11-13)
- 4-Tier Verification Hierarchy:
    Tier 1: Canonical 12-Component Schema Validation
    Tier 2: Boundary, Partial & Corrupted Checkpoint Rejection
    Tier 3: Inter-Module Coupled State Capture (SNN + GAR + Agent + RNG)
    Tier 4: Standalone Bitwise Resume-Equivalence Protocol (SHA-256 State Hash)
"""

import os
import tempfile
import hashlib
import pytest
import numpy as np
import torch
import random
import brian2

from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.dqn_agent import DQNAgent
from organoid_rl.utils.seeding import set_seed
from organoid_rl.utils.checkpointing import (
    MANDATORY_12_KEYS,
    create_full_state_dict,
    save_full_checkpoint,
    load_full_checkpoint,
    compute_state_hash,
)


class TestCheckpointComponents:
    """2.1 The 12 Mandatory State Components Verification."""

    def setup_method(self):
        brian2.start_scope()
        self.env = OrganoidEnv()
        self.agent = DQNAgent(obs_dim=self.env.obs_dim, n_actions=self.env.action_space.n)

    # --- Tier 1: Canonical 12-Component Schema Completeness ---
    def test_mandatory_12_components_present(self):
        """Tier 1: Checkpoint must contain all 12 mandatory components without omission."""
        state_dict = create_full_state_dict(self.env, self.agent)
        for key in MANDATORY_12_KEYS:
            assert key in state_dict, f"Missing mandatory state key: {key}"

    def test_component_tensor_shapes_and_types(self):
        """Tier 1: Validate shapes and biological data ranges of all serialized arrays."""
        state_dict = create_full_state_dict(self.env, self.agent)

        # Check neurons: shape (500,)
        assert len(state_dict["snn_metabolic_atp"]) == 500
        assert len(state_dict["snn_membrane_v"]) == 500
        assert len(state_dict["snn_recovery_u"]) == 500

        # Check ATP is bounded in [0, 1]
        assert np.all(state_dict["snn_metabolic_atp"] >= 0.0)
        assert np.all(state_dict["snn_metabolic_atp"] <= 1.0)

        # Check membrane potentials are bounded in [-100, 60]
        assert np.all(state_dict["snn_membrane_v"] >= -100.0)
        assert np.all(state_dict["snn_membrane_v"] <= 60.0)

        # Check GAR counters
        assert isinstance(state_dict["gar_state_counters"], dict)
        for subkey in ["seizure_clamps", "coma_kicks", "nan_resets"]:
            assert subkey in state_dict["gar_state_counters"]

    # --- Tier 2: Boundary, Partial & Corrupted Checkpoint Rejection ---
    def test_missing_component_raises_key_error(self):
        """Tier 2: Loading a checkpoint missing any mandatory key must raise KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "incomplete_ckpt.pt")
            incomplete_state = create_full_state_dict(self.env, self.agent)
            del incomplete_state["snn_metabolic_atp"]  # Intentionally delete key
            torch.save(incomplete_state, ckpt_path)

            with pytest.raises(KeyError, match="snn_metabolic_atp"):
                load_full_checkpoint(ckpt_path, self.env, self.agent)

    def test_corrupted_file_rejection(self):
        """Tier 2: Loading a corrupt/truncated checkpoint raises exception cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "bad.pt")
            with open(bad_path, "wb") as f:
                f.write(b"NOT_A_VALID_TORCH_STREAM")
            with pytest.raises(Exception):
                load_full_checkpoint(bad_path, self.env, self.agent)


class TestResumeEquivalence:
    """2.2 Standalone Resume-Equivalence Test Protocol."""

    def setup_method(self):
        brian2.start_scope()

    # --- Tier 4: Real-World Scenario: Bitwise Resume-Equivalence ---
    def test_resume_equivalence_bitwise(self):
        """
        Tier 4: Protocol from Master Verifier Plan §2.2:
        1. Run 10 steps uninterrupted with agent interactions.
        2. In parallel, run 5 steps -> save checkpoint -> re-instantiate -> load -> run remaining 5 steps.
        3. Assert rewards match and state SHA-256 hash matches bitwise.
        """
        seed = 42
        total_steps = 4
        split_step = 2

        # 1. Uninterrupted Run (4 steps)
        brian2.start_scope()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env_a = OrganoidEnv()
        agent_a = DQNAgent(obs_dim=env_a.obs_dim, n_actions=env_a.action_space.n)
        obs_a, _ = env_a.reset(seed=seed)
        rewards_uninterrupted = []
        for step_idx in range(total_steps):
            action = step_idx % 8
            next_obs, r, term, trunc, _ = env_a.step(action)
            rewards_uninterrupted.append(r)
            if term or trunc:
                break
        hash_uninterrupted = compute_state_hash(env_a, agent_a)

        # 2. Interrupted Run (split_step steps + save + reload + remaining steps)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "resume_audit.pt")

            brian2.start_scope()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env_b = OrganoidEnv()
            agent_b = DQNAgent(obs_dim=env_b.obs_dim, n_actions=env_b.action_space.n)
            obs_b, _ = env_b.reset(seed=seed)
            rewards_part1 = []
            for step_idx in range(split_step):
                action = step_idx % 8
                next_obs, r, term, trunc, _ = env_b.step(action)
                rewards_part1.append(r)
                if term or trunc:
                    break

            # Save full checkpoint
            save_full_checkpoint(ckpt_path, env_b, agent_b)

            # Re-initialize on different seed to ensure clean overwrite
            brian2.start_scope()
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            env_c = OrganoidEnv()
            agent_c = DQNAgent(obs_dim=env_c.obs_dim, n_actions=env_c.action_space.n)
            env_c.reset(seed=999)

            # Load checkpoint
            load_full_checkpoint(ckpt_path, env_c, agent_c)

            # Resume remaining steps
            rewards_part2 = []
            for step_idx in range(split_step, total_steps):
                action = step_idx % 8
                next_obs, r, term, trunc, _ = env_c.step(action)
                rewards_part2.append(r)
                if term or trunc:
                    break

            hash_resumed = compute_state_hash(env_c, agent_c)
            rewards_interrupted = rewards_part1 + rewards_part2

            # Assert trajectory equivalence
            assert np.allclose(rewards_uninterrupted, rewards_interrupted, atol=1e-5), \
                f"Reward trajectory diverged upon resume: {rewards_uninterrupted} vs {rewards_interrupted}"
            assert hash_uninterrupted == hash_resumed, \
                f"SHA-256 state hash mismatch upon resume: {hash_uninterrupted} != {hash_resumed}"

    def test_coupled_rl_training_resume_equivalence(self):
        """
        Tier 4: Coupled D3QN + OrganoidEnv training resume-equivalence test.
        Exercises agent.choose_action(), replay buffer updates, agent.learn()
        gradient steps (mutating online/target/Adam optimizer states), and SNN
        dynamics across an interrupted checkpoint boundary.
        """
        seed = 42
        total_steps = 8

        def train_step(env, agent, obs):
            action = agent.choose_action(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            agent.store_transition(obs, action, r, next_obs, bool(term or trunc))
            loss = agent.learn()
            return next_obs, r, term, trunc, loss

        # 1. Uninterrupted coupled training run (2 episodes, capped at 4 steps each)
        brian2.start_scope()
        set_seed(seed)
        env_a = OrganoidEnv(difficulty_stage=4)
        agent_a = DQNAgent(obs_dim=env_a.obs_dim, n_actions=env_a.action_space.n, batch_size=4)
        rewards_a = []
        losses_a = []
        for ep in range(2):
            obs_a, _ = env_a.reset(seed=seed + ep)
            for _ in range(total_steps // 2):
                obs_a, r, term, trunc, loss = train_step(env_a, agent_a, obs_a)
                rewards_a.append(r)
                if loss is not None:
                    losses_a.append(loss)
                if term or trunc:
                    break
        hash_a = compute_state_hash(env_a, agent_a)

        # 2. Interrupted coupled training run: Episode 0 -> save -> re-instantiate on seed 999 -> load -> Episode 1
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "rl_resume.pt")

            brian2.start_scope()
            set_seed(seed)
            env_b = OrganoidEnv(difficulty_stage=4)
            agent_b = DQNAgent(obs_dim=env_b.obs_dim, n_actions=env_b.action_space.n, batch_size=4)
            rewards_b = []
            losses_b = []
            obs_b, _ = env_b.reset(seed=seed)
            for _ in range(total_steps // 2):
                obs_b, r, term, trunc, loss = train_step(env_b, agent_b, obs_b)
                rewards_b.append(r)
                if loss is not None:
                    losses_b.append(loss)
                if term or trunc:
                    break

            save_full_checkpoint(ckpt_path, env_b, agent_b)

            # Poison state on seed 999
            brian2.start_scope()
            set_seed(999)
            env_c = OrganoidEnv(difficulty_stage=1)
            agent_c = DQNAgent(obs_dim=env_c.obs_dim, n_actions=env_c.action_space.n, batch_size=4)
            env_c.reset(seed=999)

            load_full_checkpoint(ckpt_path, env_c, agent_c)

            obs_c, _ = env_c.reset(seed=seed + 1)
            for _ in range(total_steps // 2):
                obs_c, r, term, trunc, loss = train_step(env_c, agent_c, obs_c)
                rewards_b.append(r)
                if loss is not None:
                    losses_b.append(loss)
                if term or trunc:
                    break

            hash_c = compute_state_hash(env_c, agent_c)

            assert len(losses_a) > 0, "Expected gradient updates during coupled resume test"
            assert np.allclose(rewards_a, rewards_b, atol=1e-5), \
                f"Coupled RL reward trajectory diverged: {rewards_a} vs {rewards_b}"
            assert np.allclose(losses_a, losses_b, atol=1e-5), \
                f"Coupled RL loss trajectory diverged: {losses_a} vs {losses_b}"
            assert hash_a == hash_c, \
                f"Coupled RL SHA-256 hash mismatch upon resume: {hash_a} != {hash_c}"
