"""Structured (channel) pruning with Torch-Pruning's dependency graph."""

import torch
import torch_pruning as tp
from torch import nn


def prune_structured(model: nn.Module, ratio: float, image_size: int) -> nn.Module:
    """Remove ``ratio`` of the channels in every prunable layer group, in place.

    L2-magnitude importance, uniform ratio per group. The classifier's output
    (the classes) is never pruned; its input shrinks with the last conv layer,
    so the head stays global-average-pool + one linear layer and closed-form
    Grad-CAM++ remains exact. With a uniform ratio the resulting layer shapes
    depend only on the architecture and ratio, not on the weights, so a pruned
    checkpoint can be reloaded by pruning a fresh model the same way.
    """
    example = torch.randn(1, 3, image_size, image_size, device=next(model.parameters()).device)
    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[model.get_classifier()],
    )
    pruner.step()
    return model
