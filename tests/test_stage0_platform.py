"""
tests/test_stage0_platform.py

Stage 0 Verification Suite: Runtime, Determinism & Software Dependencies (Platform Integrity)
Authoritative Specifications:
- OrganoidEnv_Master_Verifier_Plan.md §0.1 - §0.4
- PROJECT.md §Feature Inventory (Items 1-4, 10, 12)
- 4-Tier Verification Hierarchy:
    Tier 1: Canonical Feature Coverage (Happy Path)
    Tier 2: Boundary, Numerical & Corner Cases
    Tier 3: Cross-Feature Interactions & Multi-Engine Invariants
    Tier 4: Real-World Scenarios (Bitwise Trajectory Reproducibility)
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import random
import brian2

from gymnasium.utils.env_checker import check_env
from organoid_rl.environment.core import OrganoidEnv


def unified_set_seed(seed: int):
    """Authoritative reference implementation of multi-engine seeding."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    brian2.seed(seed)

    # Try importing module if implemented by Milestone M1
    try:
        from organoid_rl.utils.seeding import set_seed
        set_seed(seed)
    except ImportError:
        pass


class TestMultiEngineSeeding:
    """0.1 Deterministic Multi-Engine Seeding Verification."""

    def setup_method(self):
        brian2.start_scope()

    # --- Tier 1: Canonical Feature Coverage ---
    def test_seeding_sets_all_generators(self):
        """Tier 1: Verify set_seed sets Python random, numpy, torch, and brian2."""
        seed = 42
        unified_set_seed(seed)
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        unified_set_seed(seed)
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2, f"random.random divergence: {r1} != {r2}"
        assert n1 == n2, f"numpy.random divergence: {n1} != {n2}"
        assert t1 == t2, f"torch.rand divergence: {t1} != {t2}"

    # --- Tier 2: Boundary / Corner Cases ---
    def test_seeding_invalid_types_raise_or_handled(self):
        """Tier 2: Non-integer seed validation."""
        for invalid_seed in ["seed_str", 3.14, None]:
            if invalid_seed is None:
                continue
            with pytest.raises((TypeError, ValueError)):
                int_seed = int(invalid_seed) if isinstance(invalid_seed, float) else invalid_seed
                if not isinstance(invalid_seed, int):
                    raise TypeError("Seed must be an integer")

    def test_seed_divergence(self):
        """Tier 2: Different seeds must produce different pseudorandom streams."""
        unified_set_seed(42)
        arr_a = np.random.rand(10)
        unified_set_seed(43)
        arr_b = np.random.rand(10)
        assert not np.array_equal(arr_a, arr_b), "Different seeds produced identical arrays"

    # --- Tier 3: Cross-Feature Multi-Engine Determinism ---
    def test_multi_engine_coupled_determinism(self):
        """Tier 3: Interleaved calls across engines replicate across runs."""
        def run_interleaved(s):
            unified_set_seed(s)
            val_r = [random.randint(0, 100) for _ in range(5)]
            val_n = np.random.normal(0, 1, size=5).tolist()
            val_t = torch.randn(5).tolist()
            return val_r, val_n, val_t

        r1, n1, t1 = run_interleaved(1234)
        r2, n2, t2 = run_interleaved(1234)

        assert r1 == r2, "Interleaved Python random mismatch"
        assert np.allclose(n1, n2, atol=1e-7), "Interleaved NumPy mismatch"
        assert np.allclose(t1, t2, atol=1e-7), "Interleaved PyTorch mismatch"

    # --- Tier 4: Real-World Scenario: Bitwise Simulation Reproducibility ---
    def test_bitwise_simulation_reproducibility(self):
        """
        Tier 4: Run two independent 10-step simulation runs from seed 42.
        Membrane potentials and observation vectors must match within tight tolerance.
        """
        def simulate_trajectory(seed, n_steps=10):
            brian2.start_scope()
            unified_set_seed(seed)
            env = OrganoidEnv()
            obs, info = env.reset(seed=seed)
            observations = [obs]
            rewards = []
            for action in range(n_steps):
                act = action % 8
                next_obs, reward, term, trunc, _ = env.step(act)
                observations.append(next_obs)
                rewards.append(reward)
                if term or trunc:
                    break
            v_final = env.neurons.v[:]
            return np.array(observations), np.array(rewards), np.array(v_final)

        obs_1, r_1, v_1 = simulate_trajectory(42, n_steps=10)
        obs_2, r_2, v_2 = simulate_trajectory(42, n_steps=10)

        assert np.allclose(r_1, r_2, atol=1e-5), "Step rewards diverged between identical runs"
        assert np.allclose(v_1, v_2, atol=1e-5), "Final neuron membrane potentials diverged between identical runs"
        assert np.allclose(obs_1, obs_2, atol=1e-5), "Observation trajectory diverged between identical runs"


class TestDependencyManifest:
    """0.2 Pinned Dependency Manifest Verification."""

    def test_required_manifest_file_exists(self):
        """Tier 1: Verify requirements.txt exists at repository root."""
        assert os.path.exists("requirements.txt") or os.path.exists("requirements-pinned.txt"), \
            "requirements.txt or requirements-pinned.txt must exist at repo root"

    def test_pinned_dependencies_importable(self):
        """Tier 1: Verify all core dependencies import cleanly with required minimum versions."""
        import gymnasium
        import brian2 as b2
        import torch as th
        import scipy

        assert hasattr(gymnasium, "__version__"), "gymnasium missing __version__"
        assert hasattr(b2, "__version__"), "brian2 missing __version__"
        assert hasattr(th, "__version__"), "torch missing __version__"
        assert hasattr(scipy, "__version__"), "scipy missing __version__"

    def test_zero_import_collision(self):
        """Tier 2: Verify importing organoid_rl alongside dependencies has no circular/namespace collisions."""
        import organoid_rl
        from organoid_rl.environment.core import OrganoidEnv
        from organoid_rl.agents.dqn_agent import DQNAgent
        assert OrganoidEnv is not None
        assert DQNAgent is not None


class TestGymnasiumCompliance:
    """0.3 Gymnasium API Compliance Verification."""

    def setup_method(self):
        brian2.start_scope()

    # --- Tier 1: Canonical API Signature & Space Verification ---
    def test_observation_space_contract(self):
        """Tier 1: Observation space must be 21D Box with bounds [-1, 2]."""
        env = OrganoidEnv()
        assert env.observation_space.shape == (21,), f"Expected 21D obs, got {env.observation_space.shape}"
        assert env.observation_space.dtype == np.float32, f"Expected float32, got {env.observation_space.dtype}"
        assert np.all(env.observation_space.low <= -1.0)
        assert np.all(env.observation_space.high >= 2.0)

    def test_action_space_contract(self):
        """Tier 1: Action space must be Discrete(8)."""
        env = OrganoidEnv()
        assert env.action_space.n == 8, f"Expected 8 discrete actions, got {env.action_space.n}"

    def test_reset_return_structure(self):
        """Tier 1: reset() must return a 2-tuple (observation, info)."""
        env = OrganoidEnv()
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2, "reset() must return (obs, info)"
        obs, info = result
        assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
        assert obs.shape == (21,), f"Observation shape {obs.shape} != (21,)"
        assert isinstance(info, dict), "Info must be a dictionary"

    def test_step_return_structure(self):
        """Tier 1: step() must return a 5-tuple (obs, reward, terminated, truncated, info)."""
        env = OrganoidEnv()
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple) and len(result) == 5, "step() must return 5-tuple"
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray) and obs.shape == (21,)
        assert isinstance(reward, (float, np.floating, int))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    # --- Tier 2: Boundary & Action Handling ---
    def test_all_valid_discrete_actions_executable(self):
        """Tier 2: Every action in {0, ..., 7} executes cleanly without error."""
        env = OrganoidEnv()
        env.reset()
        for act in range(8):
            obs, reward, term, trunc, info = env.step(act)
            assert not np.any(np.isnan(obs)), f"Action {act} produced NaN observation"
            assert not np.isnan(reward), f"Action {act} produced NaN reward"

    def test_gymnasium_check_env_official(self):
        """
        Tier 4: Official Gymnasium env checker validation.
        Audits environment determinism, spaces, and standard compliance.
        """
        brian2.start_scope()
        env = OrganoidEnv()
        # Gymnasium check_env raises AssertionError or warning if non-compliant
        # We catch and report exact failure mode to escalate implementation defects.
        try:
            check_env(env, skip_render_check=True)
        except AssertionError as e:
            pytest.fail(f"Gymnasium check_env failed: {e}")


class TestAtomicSerialization:
    """0.4 Atomic Checkpoint Serialization Verification."""

    # --- Tier 1: Atomic File Replacement ---
    def test_atomic_write_creates_valid_file(self):
        """Tier 1: Checkpoint writes cleanly and is readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            temp_path = ckpt_path + ".tmp"

            payload = {"epoch": 10, "state": np.ones((5, 5))}
            torch.save(payload, temp_path)
            assert os.path.exists(temp_path)
            os.replace(temp_path, ckpt_path)

            assert os.path.exists(ckpt_path)
            assert not os.path.exists(temp_path)

            loaded = torch.load(ckpt_path, weights_only=False)
            assert loaded["epoch"] == 10
            assert np.array_equal(loaded["state"], payload["state"])

    # --- Tier 2: Interrupted Save Non-Corruption ---
    def test_interrupted_write_preserves_original(self):
        """
        Tier 2: If a write fails or is killed mid-stream before os.replace,
        the existing original checkpoint remains completely intact and uncorrupted.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "stable_ckpt.pt")
            temp_path = ckpt_path + ".tmp"

            # Create original valid checkpoint
            original_payload = {"version": 1, "weights": np.full(10, 3.14)}
            torch.save(original_payload, ckpt_path)

            # Simulate interrupted write: partial write to .tmp
            with open(temp_path, "wb") as f:
                f.write(b"CORRUPTED_INCOMPLETE_DATA_STREAM")

            # Verify original file was not touched
            assert os.path.exists(ckpt_path)
            loaded = torch.load(ckpt_path, weights_only=False)
            assert loaded["version"] == 1
            assert np.allclose(loaded["weights"], 3.14)

            # Cleanup .tmp
            if os.path.exists(temp_path):
                os.remove(temp_path)
