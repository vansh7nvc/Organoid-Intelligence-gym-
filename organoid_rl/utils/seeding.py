"""
Unified multi-engine seeding utility for OrganoidEnv.
Sets random seeds across Python random, NumPy, PyTorch, and Brian2.
"""

import random
import numpy as np
import torch
import brian2


def set_seed(seed: int) -> None:
    """
    Sets random seeds across Python random, NumPy, PyTorch, and Brian2.

    Parameters:
    seed (int): The integer seed to set across all engines.
    """
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    brian2.seed(seed)
