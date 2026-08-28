"""
Agents module for OrganoidRL.
Includes Dueling DDQN and baseline MLP agents.
"""

from .dqn_agent import DQNAgent
from .baseline_agent import MLPBaselineAgent

__all__ = [
    "DQNAgent",
    "MLPBaselineAgent",
]
