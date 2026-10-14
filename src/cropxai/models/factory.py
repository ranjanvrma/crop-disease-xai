"""Build any model in the study from ``cfg.model``.

Every CNN here ends in global-average-pool -> one linear layer. That is what
makes the closed-form, gradient-free Grad-CAM++ exact (Phase 6), and
``tests/test_models.py`` checks it for each architecture.
"""

import timm
from omegaconf import DictConfig
from torch import nn

from cropxai.models.scratch_cnn import ScratchCNN

CNN_ARCHS = {"scratch_cnn", "resnet50", "efficientnet_b0", "efficientnet_b3", "mobilenetv3_large_100"}
VIT_ARCHS = {"deit_small_patch16_224"}  # desktop comparison only; not on the mobile track


def _single_linear_mobilenetv3(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace MobileNetV3's ``GAP -> 1x1 conv(960->1280) -> hardswish -> linear``
    head with ``GAP -> linear(960 -> num_classes)``.

    The hidden 1x1 conv makes the class score non-linear in the pooled
    features, which breaks the closed-form Grad-CAM++ derivation.
    """
    model.conv_head = nn.Identity()
    model.norm_head = nn.Identity()
    model.act2 = nn.Identity()
    model.classifier = nn.Linear(model.num_features, num_classes)
    model.head_hidden_size = model.num_features
    return model


def build_model(model_cfg: DictConfig, num_classes: int) -> nn.Module:
    """Create the model named by ``model_cfg.arch`` with a fresh classifier."""
    arch = model_cfg.arch
    if arch == "scratch_cnn":
        return ScratchCNN(num_classes, drop_rate=model_cfg.drop_rate)
    if arch not in CNN_ARCHS | VIT_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Options: {sorted(CNN_ARCHS | VIT_ARCHS)}")

    model = timm.create_model(
        arch, pretrained=model_cfg.pretrained, num_classes=num_classes, drop_rate=model_cfg.drop_rate
    )
    if arch == "mobilenetv3_large_100":
        model = _single_linear_mobilenetv3(model, num_classes)
    return model


def set_trainable(model: nn.Module, mode: str) -> None:
    """``feature_extract``: only the classifier trains. ``finetune``: everything."""
    if mode not in {"feature_extract", "finetune"}:
        raise ValueError(f"Unknown train mode '{mode}'")
    for p in model.parameters():
        p.requires_grad = mode == "finetune"
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """Keep BatchNorm running stats fixed in layers that aren't training.

    Call after ``model.train()``. Without this, feature extraction would
    still drift the frozen backbone through its BN statistics.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in m.parameters()
        ):
            m.eval()
