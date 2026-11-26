"""
utils.seed
==========
SciNet++ (2026) -- reproducibility helpers.

Seeds Python, NumPy and PyTorch (CPU + CUDA) and sets deterministic cuDNN flags
so that repeated runs of the whole pipeline are comparable.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used across the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
