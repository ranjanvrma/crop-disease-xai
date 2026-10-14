"""Seeding so every run is reproducible from its config alone."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    With ``deterministic=True`` cuDNN autotuning is turned off and PyTorch is
    asked for deterministic kernels. Ops without a deterministic version only
    warn instead of crashing, so GPU runs are reproducible as far as the
    hardware allows.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker a distinct, derived seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
