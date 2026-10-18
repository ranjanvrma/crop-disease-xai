"""Compressed variants of a trained model, described by a ``compress:`` config section.

    compress:
      parent: mnv3_fe     # experiment whose checkpoint (same seed) is the starting point
      prune_ratio: 0.5    # structured pruning, then the short fine-tune in ``train:``
      int8: false         # true: post-training INT8 quantization of the parent, no training
      calib_images: 512   # training images used to calibrate INT8 activation ranges
"""

import torch
from omegaconf import DictConfig, OmegaConf

from cropxai.utils.config import load_config


def compress_cfg(cfg: DictConfig) -> DictConfig | None:
    return cfg.get("compress")


def is_int8(cfg: DictConfig) -> bool:
    c = compress_cfg(cfg)
    return bool(c and c.int8)


def parent_config(cfg: DictConfig) -> DictConfig:
    """The parent experiment's config for the same seed, sharing this run's paths and data."""
    parent = load_config(f"configs/experiment/{cfg.compress.parent}.yaml", [f"seed={cfg.seed}"])
    shared = {k: cfg[k] for k in ("paths", "data", "wandb", "num_workers") if k in cfg}
    return OmegaConf.merge(parent, shared)


def root_config(cfg: DictConfig) -> DictConfig:
    """Follow the parent chain back to the uncompressed model."""
    while compress_cfg(cfg):
        cfg = parent_config(cfg)
    return cfg


def pruned_from_parent(cfg: DictConfig) -> torch.nn.Module:
    """Parent weights, pruned: the starting point of a pruned variant's fine-tune."""
    from cropxai.compress.prune import prune_structured
    from cropxai.train.loop import load_trained

    model, _, _ = load_trained(parent_config(cfg), torch.device("cpu"))
    return prune_structured(model, cfg.compress.prune_ratio, cfg.image_size)


def quantize_variant(cfg: DictConfig) -> dict:
    """Post-training INT8 quantization of the parent; writes the checkpoint and completion marker."""
    from cropxai.compress.quantize import quantize_int8
    from cropxai.data.dataset import load_examples
    from cropxai.eval.efficiency import count_params
    from cropxai.train.loop import checkpoint_path, load_trained, mark_complete, save_atomic
    from cropxai.utils.seed import seed_everything

    seed_everything(cfg.seed, cfg.deterministic)
    model, classes, _ = load_trained(parent_config(cfg), torch.device("cpu"))
    calib, _, _, _, _ = load_examples(cfg, "train", cfg.compress.calib_images)   # train split only
    program, quantizer = quantize_int8(model, calib, cfg.image_size)

    ckpt = checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_atomic({
        "program": program,
        "classifier": model.get_classifier().state_dict(),
        "classes": classes,
        "params": count_params(model),
        "quantizer": quantizer,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "best": {},
    }, ckpt)
    info = {"quantizer": quantizer, "calib_images": len(calib)}
    mark_complete(cfg, info)
    print(f"[{cfg.run_name} seed{cfg.seed}] INT8 ({quantizer}) written to {ckpt}")
    return info | {"checkpoint": str(ckpt)}
