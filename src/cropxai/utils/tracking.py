"""Experiment tracking.

Two sinks, on purpose:
  * Weights & Biases — curves, system stats, artefacts; for looking at runs.
  * ``results/metrics/*.json`` — the source of truth that the results
    notebook, README table, paper tables and web dashboard are built from.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from cropxai.utils.config import resolve_path


def init_wandb(cfg: DictConfig, job_type: str, run_id: str | None = None):
    """Start a W&B run with the full resolved config attached.

    ``cfg.wandb.mode`` may be ``online``, ``offline`` or ``disabled`` (tests
    and quick local checks use ``disabled``). Pass ``run_id`` to continue an
    interrupted run's curves in the same W&B run.
    """
    import wandb  # imported lazily so data-only tooling doesn't need it

    return wandb.init(
        project=cfg.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=f"{cfg.run_name}-seed{cfg.seed}",
        group=cfg.run_name,  # the 3 seeds of one experiment share a group
        job_type=job_type,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=run_id,
        resume="allow" if run_id else None,
    )


def git_commit() -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_metrics(cfg: DictConfig, metrics: dict[str, Any], kind: str) -> Path:
    """Write one run's metrics to ``results/metrics/<kind>/<run>_seed<n>.json``.

    ``kind`` groups files by table (e.g. ``accuracy``, ``explanation``,
    ``stability``, ``ondevice``). Every file carries the config and
    environment it came from, so a number can always be traced to its run.
    """
    import torch

    out_dir = resolve_path(cfg.paths.results_dir) / "metrics" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_name}_seed{cfg.seed}.json"

    record = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "kind": kind,
        "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path
