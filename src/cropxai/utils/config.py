"""Config loading: one YAML per experiment, composed from shared pieces.

An experiment file lists the files it builds on under ``inherits`` (paths
relative to ``configs/``). They are merged left to right, then the
experiment's own keys, then any ``key=value`` command-line overrides::

    # configs/experiment/prepare_data.yaml
    inherits: [base.yaml, data/default.yaml]
    run_name: prepare_data
"""

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from cropxai import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load an experiment config, resolving ``inherits`` and CLI overrides."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    own = OmegaConf.load(path)

    parents = [OmegaConf.load(CONFIG_DIR / p) for p in own.pop("inherits", [])]
    cfg = OmegaConf.merge(*parents, own, OmegaConf.from_dotlist(overrides or []))
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)  # configs are inputs, never mutated mid-run
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Absolute paths pass through; relative ones are taken from the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# Keys that change what a training run produces. Paths, logging and the
# explanation settings don't, so editing them never forces a retrain.
TRAINING_KEYS = ["seed", "deterministic", "image_size", "batch_size", "data", "augment", "model", "train"]


def training_fingerprint(cfg: DictConfig) -> str:
    """Short hash of everything that affects a training run's result."""
    subset = OmegaConf.create({k: cfg[k] for k in TRAINING_KEYS if k in cfg})
    return hashlib.md5(OmegaConf.to_yaml(subset, sort_keys=True).encode()).hexdigest()[:12]
