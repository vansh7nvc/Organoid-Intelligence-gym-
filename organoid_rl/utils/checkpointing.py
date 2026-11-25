"""
Atomic 12-component checkpointing utilities for OrganoidEnv and DQNAgent.

Implements the canonical checkpoint specification from:
OrganoidEnv_Master_Verifier_Plan.md §2.1 - §2.2
"""

import hashlib
import os
import numpy as np
import torch


MANDATORY_12_KEYS = [
    "snn_weights",
    "snn_traces_fast",
    "snn_traces_slow",
    "snn_metabolic_atp",
    "snn_membrane_v",
    "snn_recovery_u",
    "snn_pruning_mask",
    "gar_state_counters",
    "agent_online_net",
    "agent_target_net",
    "agent_optimizer",
    "rng_states",
]


def create_full_state_dict(env, agent):
    """
    Extracts all 12 mandatory components (plus auxiliary task & replay states)
    from OrganoidEnv and DQNAgent.
    """
    state_dict = {}
    state_dict.update(env.get_state_dict())
    state_dict.update(agent.get_state_dict())
    return state_dict


def save_full_checkpoint(filepath, env, agent):
    """
    Atomically serializes full coupled state to `<filepath>.tmp` and
    promotes it via `os.replace()` to guarantee crash safety.
    """
    filepath_str = str(filepath)
    parent_dir = os.path.dirname(filepath_str)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    state_dict = create_full_state_dict(env, agent)
    tmp_path = filepath_str + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, filepath_str)


def load_full_checkpoint(filepath, env, agent):
    """
    Loads checkpoint, strictly verifies all 12 mandatory components are present,
    and restores OrganoidEnv, DQNAgent, and all RNG states.
    """
    state_dict = torch.load(filepath, map_location=agent.device, weights_only=False)
    for key in MANDATORY_12_KEYS:
        if key not in state_dict:
            raise KeyError(f"Mandatory checkpoint component missing: '{key}'")

    env.load_state_dict(state_dict)
    agent.load_state_dict(state_dict)
    return state_dict


def compute_state_hash(env, agent) -> str:
    """
    Computes deterministic SHA-256 hash over SNN weights, membrane states,
    ATP reserves, and agent neural parameters for bitwise resume verification.
    """
    h = hashlib.sha256()
    for S in env.synapses:
        h.update(np.ascontiguousarray(S.w[:]).tobytes())
    h.update(np.ascontiguousarray(env.neurons.v[:]).tobytes())
    h.update(np.ascontiguousarray(env.neurons.E[:]).tobytes())
    h.update(np.ascontiguousarray(env.neurons.u[:]).tobytes())
    for p in agent.online_net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    for p in agent.target_net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
