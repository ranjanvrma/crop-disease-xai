"""Supervised training: feature extraction or fine-tuning, one seed per call."""

import math
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from cropxai.data.dataset import load_classes, make_loader
from cropxai.eval.classification import classification_metrics, predict
from cropxai.models import build_model, freeze_frozen_batchnorm, set_trainable
from cropxai.utils.config import resolve_path
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import init_wandb


def checkpoint_path(cfg: DictConfig) -> Path:
    return resolve_path(cfg.paths.checkpoints_dir) / cfg.run_name / f"seed{cfg.seed}" / "best.pt"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: nn.Module, t: DictConfig) -> torch.optim.Optimizer:
    """AdamW; the backbone gets ``lr * backbone_lr_mult`` so fine-tuning
    doesn't wreck the pretrained features."""
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": t.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": t.lr * t.backbone_lr_mult})
    return torch.optim.AdamW(groups, weight_decay=t.weight_decay)


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up, then cosine decay to zero, stepped once per batch."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def memory_format(device: torch.device) -> torch.memory_format:
    """channels_last lets cuDNN use tensor cores for convs under AMP (GPU only)."""
    return torch.channels_last if device.type == "cuda" else torch.contiguous_format


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, scaler, loss_fn, device, amp: bool) -> float:
    model.train()
    freeze_frozen_batchnorm(model)
    total, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=memory_format(device))
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / n


def fit(cfg: DictConfig) -> dict:
    """Train one seed; keep the checkpoint with the best val macro-F1.

    Only ``train`` and ``val`` are touched here. Test and field evaluation
    happen afterwards in ``scripts/evaluate.py``.
    """
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    t = cfg.train
    amp = t.amp and device.type == "cuda"

    classes = load_classes(cfg)
    model = build_model(cfg.model, num_classes=len(classes)).to(device, memory_format=memory_format(device))
    set_trainable(model, t.mode)

    train_loader, val_loader = make_loader(cfg, "train"), make_loader(cfg, "val")
    optimizer = make_optimizer(model, t)
    steps = t.epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, t.warmup_epochs * len(train_loader), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)

    ckpt = checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    run = init_wandb(cfg, job_type="train")
    best = {"val_macro_f1": -1.0}

    for epoch in range(t.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, device, amp)
        val = classification_metrics(*predict(model, val_loader, device))
        log = {"epoch": epoch, "train/loss": train_loss, **{f"val/{k}": v for k, v in val.items()}}
        run.log(log)
        print(f"[{cfg.run_name} seed{cfg.seed}] epoch {epoch}: loss {train_loss:.4f} "
              f"val top1 {val['top1']:.4f} f1 {val['macro_f1']:.4f} ece {val['ece']:.4f}")

        if val["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": val["macro_f1"], "val_top1": val["top1"], "val_ece": val["ece"]}
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "best": best,
            }, ckpt)

    run.summary.update({f"best/{k}": v for k, v in best.items()})
    run.finish()
    return best | {"checkpoint": str(ckpt)}


def load_trained(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, list[str], dict]:
    """Rebuild the model for ``cfg`` and load its best checkpoint (eval mode)."""
    ckpt = torch.load(checkpoint_path(cfg), map_location="cpu", weights_only=False)
    # Weights come from the checkpoint, so skip downloading ImageNet weights.
    model = build_model(OmegaConf.merge(cfg.model, {"pretrained": False}), num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], ckpt
