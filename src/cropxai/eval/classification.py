"""Classification metrics: top-1, macro-F1, NLL and calibration (ECE)."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All logits and labels for a loader, on CPU."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE: confidence-weighted gap between confidence and accuracy, over equal-width bins.

    The app's "unsure" threshold assumes confidence means something, so this
    is reported alongside accuracy.
    """
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (conf[in_bin].mean() - correct[in_bin].mean()).abs()
    return float(ece)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Top-1, macro-F1 (over classes present in ``labels``), NLL and ECE."""
    probs = logits.softmax(dim=1)
    preds = probs.argmax(dim=1)
    present = np.unique(labels.numpy())
    return {
        "top1": float(preds.eq(labels).float().mean()),
        "macro_f1": float(f1_score(labels.numpy(), preds.numpy(), labels=present, average="macro", zero_division=0)),
        "nll": float(F.cross_entropy(logits, labels)),
        "ece": expected_calibration_error(probs, labels),
        "n": int(len(labels)),
    }


def field_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Field-set metrics, reported two ways (both go in the paper, clearly named).

    * ``strict``: argmax over all 38 PlantVillage classes, which is what the app does.
    * ``restricted``: argmax over only the classes that occur in the field
      set, which is the usual way to compare with PlantDoc-style results.
    """
    present = torch.unique(labels)
    masked = torch.full_like(logits, float("-inf"))
    masked[:, present] = logits[:, present]

    strict = classification_metrics(logits, labels)
    restricted = classification_metrics(masked.clamp(min=-1e4), labels)
    return {f"strict_{k}": v for k, v in strict.items()} | {
        "restricted_top1": restricted["top1"],
        "restricted_macro_f1": restricted["macro_f1"],
        "n_field_classes": int(len(present)),
    }
