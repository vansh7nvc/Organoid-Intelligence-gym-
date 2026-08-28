"""
Unit test suite for OrganoidEnv and Optimization Agents.
Compatible with standard unittest and pytest.
"""

import unittest
import numpy as np
import torch
from organoid_rl.environment.core import OrganoidEnv
from organoid_rl.agents.baseline_agent import MLPBaselineAgent
from organoid_rl.agents.dqn_agent import DQNAgent


class TestOrganoidEnv(unittest.TestCase):
    """Test suite for OrganoidEnv Gymnasium environment."""

    @classmethod
    def setUpClass(cls):
        cls.env = OrganoidEnv()

    def test_observation_and_action_spaces(self):
        """Verify observation and action spaces."""
        self.assertEqual(self.env.observation_space.shape, (21,))
        self.assertEqual(self.env.action_space.n, 8)
        self.assertEqual(self.env.obs_dim, 21)

    def test_reset(self):
        """Verify environment reset output and info."""
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (21,))
        self.assertIn("distance", info)
        self.assertIn("target_pos", info)
        self.assertIn("cursor_pos", info)
        self.assertIn("total_spikes", info)

    def test_step(self):
        """Verify step execution."""
        obs, info = self.env.reset(seed=42)
        action = 0  # Up
        next_obs, reward, terminated, truncated, step_info = self.env.step(action)
        
        self.assertEqual(next_obs.shape, (21,))
        self.assertIsInstance(reward, (float, np.floating, int))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIn("distance", step_info)
        self.assertIn("total_spikes", step_info)

    def test_all_actions(self):
        """Verify each of the 8 discrete actions executes cleanly."""
        self.env.reset(seed=100)
        for act in range(8):
            obs, reward, term, trunc, info = self.env.step(act)
            self.assertEqual(obs.shape, (21,))
            if term or trunc:
                self.env.reset()

    def test_ablations(self):
        """Verify that all ablation configurations initialize and step without errors."""
        ablation_configs = [
            {"use_sdm": False},
            {"use_stabilizer": False},
            {"use_dual_trace": False},
            {"use_morphology": False},
            {"use_motor_mapping": False},
            {
                "use_sdm": False,
                "use_stabilizer": False,
                "use_dual_trace": False,
                "use_morphology": False,
                "use_motor_mapping": False,
            },
        ]
        for cfg in ablation_configs:
            with self.subTest(config=cfg):
                env = OrganoidEnv(**cfg)
                obs, info = env.reset(seed=42)
                self.assertEqual(obs.shape, (21,))
                next_obs, reward, term, trunc, step_info = env.step(1)
                self.assertEqual(next_obs.shape, (21,))

    def test_curriculum_stages(self):
        """Verify setting difficulty stages."""
        self.env.set_curriculum_stage(2)
        self.assertEqual(self.env.difficulty_stage, 2)
        self.env.set_curriculum_stage(1)
        self.assertEqual(self.env.difficulty_stage, 1)


class TestOptimizationAgents(unittest.TestCase):
    """Test suite for DQN and MLP baseline agents."""

    def test_mlp_baseline_agent(self):
        """Test MLPBaselineAgent initialization, action selection, and learning."""
        agent = MLPBaselineAgent(obs_dim=21, n_actions=8)
        state = np.zeros(21, dtype=np.float32)
        action = agent.choose_action(state)
        self.assertIn(action, list(range(8)))

        # Store transitions and verify learning step
        for i in range(70):
            agent.store_transition(state, action, 1.0, state, False)
        agent.learn()

    def test_dqn_agent(self):
        """Test DQNAgent initialization, action selection, and replay buffer."""
        agent = DQNAgent(obs_dim=21, n_actions=8, batch_size=32)
        state = np.zeros(21, dtype=np.float32)
        action = agent.choose_action(state)
        self.assertIn(action, list(range(8)))

        # Store transitions
        agent.store_transition(state, action, 1.0, state, False)
        self.assertGreaterEqual(len(agent.n_step_buffer), 1)


if __name__ == "__main__":
    unittest.main()
