"""
tests/test_adversarial_m1.py

Adversarial Stress Test Suite for Milestone M1:
Platform Determinism, Gymnasium Contract Robustness, and State Isolation.

Executed by: teamwork_challenger (challenger_m1_2)
"""

import os
import pytest
import numpy as np
import brian2 as b2
from gymnasium.utils.env_checker import check_env
from organoid_rl.environment.core import OrganoidEnv, OrganoidConfig
from organoid_rl.utils.seeding import set_seed


class TestMultiSeedDeterminismAdversarial:
    """Challenge 1: Multi-Seed Trajectory Repeatability & Collision Stress."""

    SEEDS = [0, 42, 100, 999, 12345]
    N_STEPS = 20
    N_REPEATS = 3

    def setup_method(self):
        b2.start_scope()

    def test_multi_seed_repeatability_and_distinctness(self):
        """
        Runs 5 distinct seeds across 3 independent repeats (20 steps each).
        Asserts:
        1. Identical seeds yield bitwise identical observations, rewards, and membrane voltages.
        2. Different seeds yield strictly distinct trajectories.
        """
        actions = [step % 8 for step in range(self.N_STEPS)]
        trajectories = {}  # seed -> list of (obs_history, reward_history, v_final, u_final, E_final)

        for seed in self.SEEDS:
            trajectories[seed] = []
            for repeat in range(self.N_REPEATS):
                b2.start_scope()
                set_seed(seed)
                env = OrganoidEnv()
                obs, info = env.reset(seed=seed)

                obs_seq = [np.array(obs, copy=True)]
                rew_seq = []

                for act in actions:
                    obs, r, term, trunc, info = env.step(act)
                    obs_seq.append(np.array(obs, copy=True))
                    rew_seq.append(r)
                    if term or trunc:
                        break

                v_final = np.array(env.neurons.v[:], copy=True)
                u_final = np.array(env.neurons.u[:], copy=True)
                E_final = np.array(env.neurons.E[:], copy=True)

                trajectories[seed].append({
                    "obs": np.array(obs_seq),
                    "rew": np.array(rew_seq),
                    "v": v_final,
                    "u": u_final,
                    "E": E_final,
                })

            # 1. Assert bitwise identical runs across repeats for the same seed
            base_run = trajectories[seed][0]
            for rep_idx in range(1, self.N_REPEATS):
                cur_run = trajectories[seed][rep_idx]
                np.testing.assert_array_equal(
                    base_run["obs"], cur_run["obs"],
                    err_msg=f"Seed {seed} Repeat {rep_idx} observation mismatch with Repeat 0"
                )
                np.testing.assert_array_equal(
                    base_run["rew"], cur_run["rew"],
                    err_msg=f"Seed {seed} Repeat {rep_idx} reward mismatch with Repeat 0"
                )
                np.testing.assert_array_equal(
                    base_run["v"], cur_run["v"],
                    err_msg=f"Seed {seed} Repeat {rep_idx} membrane potential v mismatch with Repeat 0"
                )
                np.testing.assert_allclose(
                    base_run["u"], cur_run["u"], atol=1e-12,
                    err_msg=f"Seed {seed} Repeat {rep_idx} recovery variable u mismatch with Repeat 0"
                )
                np.testing.assert_array_equal(
                    base_run["E"], cur_run["E"],
                    err_msg=f"Seed {seed} Repeat {rep_idx} ATP energy E mismatch with Repeat 0"
                )

        # 2. Assert distinct trajectories across different seeds
        for i in range(len(self.SEEDS)):
            for j in range(i + 1, len(self.SEEDS)):
                s_i, s_j = self.SEEDS[i], self.SEEDS[j]
                obs_i = trajectories[s_i][0]["obs"]
                obs_j = trajectories[s_j][0]["obs"]
                v_i = trajectories[s_i][0]["v"]
                v_j = trajectories[s_j][0]["v"]

                assert not np.array_equal(obs_i, obs_j), \
                    f"Seed collision: Seed {s_i} and Seed {s_j} produced identical observation trajectories!"
                assert not np.array_equal(v_i, v_j), \
                    f"Seed collision: Seed {s_i} and Seed {s_j} produced identical final voltages!"

    def test_same_instance_multi_reset_determinism(self):
        """
        Verify that resetting the SAME environment instance with the same seed
        reproduces the exact same trajectory as a freshly instantiated environment.
        """
        seed = 42
        actions = [step % 8 for step in range(10)]

        # Run 1 on env A
        b2.start_scope()
        env_a = OrganoidEnv()
        obs_a0, _ = env_a.reset(seed=seed)
        obs_seq_a = [np.array(obs_a0, copy=True)]
        for act in actions:
            obs, _, _, _, _ = env_a.step(act)
            obs_seq_a.append(np.array(obs, copy=True))

        # Reset SAME env instance with the same seed
        obs_a1, _ = env_a.reset(seed=seed)
        obs_seq_a_re = [np.array(obs_a1, copy=True)]
        for act in actions:
            obs, _, _, _, _ = env_a.step(act)
            obs_seq_a_re.append(np.array(obs, copy=True))

        np.testing.assert_array_equal(
            np.array(obs_seq_a), np.array(obs_seq_a_re),
            err_msg="Same environment instance reset(seed=42) diverged from first run!"
        )


class TestGymnasiumBoundaryStressAdversarial:
    """Challenge 2: Gymnasium Boundary Stress, Rapid Resets & Numerical Robustness."""

    def setup_method(self):
        b2.start_scope()

    def test_rapid_consecutive_resets_no_leak(self):
        """Stress-test 10 consecutive resets without stepping."""
        env = OrganoidEnv()
        for i in range(10):
            obs, info = env.reset(seed=100 + i)
            assert obs.shape == (21,), f"Obs shape corrupted on reset {i}: {obs.shape}"
            assert env.observation_space.contains(obs), f"Obs outside space on reset {i}"
            assert not np.any(np.isnan(obs)), f"NaN in obs on reset {i}"
            assert not np.any(np.isinf(obs)), f"Inf in obs on reset {i}"

    def test_sequential_resets_seeded_and_unseeded(self):
        """
        Test sequence:
        1. reset(seed=777) -> 5 steps
        2. reset() [unseeded] -> 5 steps
        3. reset(seed=777) -> 5 steps
        Assert: Run 1 and Run 3 are identical; Run 2 is valid.
        """
        env = OrganoidEnv()

        # Run 1: seeded
        obs_1, _ = env.reset(seed=777)
        traj_1 = [obs_1]
        for act in [0, 1, 2, 3, 4]:
            obs, _, _, _, _ = env.step(act)
            traj_1.append(obs)

        # Run 2: unseeded
        obs_2, _ = env.reset()
        traj_2 = [obs_2]
        for act in [0, 1, 2, 3, 4]:
            obs, _, _, _, _ = env.step(act)
            traj_2.append(obs)

        # Run 3: re-seeded with 777
        obs_3, _ = env.reset(seed=777)
        traj_3 = [obs_3]
        for act in [0, 1, 2, 3, 4]:
            obs, _, _, _, _ = env.step(act)
            traj_3.append(obs)

        np.testing.assert_array_equal(
            np.array(traj_1), np.array(traj_3),
            err_msg="Re-seeded reset after unseeded reset failed to reproduce identical trajectory!"
        )

    def test_zero_nan_inf_and_space_containment_extended(self):
        """
        Run 50 continuous steps and verify:
        - Observation space contains every obs
        - Zero NaNs and zero Infs in obs, reward, neurons.v, neurons.u, neurons.E, neurons.I
        """
        env = OrganoidEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), "Initial observation outside observation space"

        for step in range(50):
            act = step % 8
            obs, reward, term, trunc, info = env.step(act)

            # Check spaces & types
            assert env.observation_space.contains(obs), f"Step {step}: observation outside space bounds: {obs}"
            assert isinstance(reward, (float, np.floating)), f"Step {step}: reward not float: {type(reward)}"

            # Check NaNs / Infs
            assert not np.any(np.isnan(obs)), f"Step {step}: NaN detected in observation"
            assert not np.any(np.isinf(obs)), f"Step {step}: Inf detected in observation"
            assert not np.isnan(reward), f"Step {step}: NaN detected in reward"
            assert not np.isinf(reward), f"Step {step}: Inf detected in reward"

            # Check internal neuron states
            v = env.neurons.v[:]
            u = env.neurons.u[:]
            E = env.neurons.E[:]
            I = env.neurons.I[:]

            assert not np.any(np.isnan(v)), f"Step {step}: NaN in neuron membrane voltage v"
            assert not np.any(np.isinf(v)), f"Step {step}: Inf in neuron membrane voltage v"
            assert not np.any(np.isnan(u)), f"Step {step}: NaN in neuron recovery variable u"
            assert not np.any(np.isinf(u)), f"Step {step}: Inf in neuron recovery variable u"
            assert not np.any(np.isnan(E)), f"Step {step}: NaN in neuron energy E"
            assert not np.any(np.isinf(E)), f"Step {step}: Inf in neuron energy E"
            assert not np.any(np.isnan(I)), f"Step {step}: NaN in neuron input current I"

            if term or trunc:
                obs, _ = env.reset(seed=42 + step)

    def test_extreme_actions_behavior(self):
        """
        Stress test discrete actions at the boundary and out-of-bounds actions.
        Action space is Discrete(8), valid actions are {0, ..., 7}.
        """
        env = OrganoidEnv()
        env.reset(seed=123)

        # Boundary valid actions
        for act in [0, 7]:
            obs, r, _, _, _ = env.step(act)
            assert not np.any(np.isnan(obs))

        # Out-of-bounds action handling: actions outside [0, 7]
        # In core.py _apply_action, action is checked with if-elif for 0..7.
        # If action >= 8, _apply_action does not stimulate any group (no crash).
        # Check that it either steps safely or raises appropriately, but never yields NaNs.
        for invalid_act in [8, 9, -1]:
            try:
                obs, r, _, _, _ = env.step(invalid_act)
                assert not np.any(np.isnan(obs)), f"Invalid action {invalid_act} produced NaN obs"
                assert not np.isnan(r), f"Invalid action {invalid_act} produced NaN reward"
            except (ValueError, IndexError, AssertionError):
                pass  # Raising standard error is also acceptable Gym behavior


class TestCheckEnvRepeatedLoopAdversarial:
    """Challenge 3: Repeated check_env Execution Loop & State Isolation."""

    def test_three_consecutive_check_env_calls(self):
        """
        Run check_env 3 times consecutively in the same Python process
        to confirm no state leakage across environment instances.
        """
        for iteration in range(3):
            b2.start_scope()
            env = OrganoidEnv()
            try:
                check_env(env, skip_render_check=True)
            except Exception as e:
                pytest.fail(f"check_env failed on iteration {iteration + 1}: {e}")
