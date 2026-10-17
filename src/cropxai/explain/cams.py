"""Library CAMs (pytorch-grad-cam): Grad-CAM++, Grad-CAM, Score-CAM.

The target layer is always the layer whose output feeds global average
pooling, i.e. exactly what ``model.forward_features`` returns. The closed-form
version (``closed_form.py``) uses the same feature maps, so the two are
directly comparable.
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

METHODS = {"gradcampp": GradCAMPlusPlus, "gradcam": GradCAM, "scorecam": ScoreCAM}


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """The module whose output is the final feature map (input to GAP)."""
    if arch == "scratch_cnn":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4
    if arch in {"efficientnet_b0", "efficientnet_b3"}:
        return model.bn2  # conv_head -> BN -> SiLU, fused in timm's BatchNormAct2d
    if arch == "mobilenetv3_large_100":
        return model.blocks[-1]  # our single-linear head pools the block output directly
    raise ValueError(f"No CAM target layer for '{arch}' (CAMs are for CNNs only).")


def compute_cams(
    model: nn.Module, arch: str, method: str, images: torch.Tensor, classes: list[int], batch_size: int = 32
) -> np.ndarray:
    """Heatmaps in [0, 1] at input resolution, shape (N, H, W), for the given class per image."""
    cam_cls = METHODS[method]
    out = []
    with cam_cls(model=model, target_layers=[target_layer(model, arch)]) as cam:
        if method == "scorecam":
            cam.batch_size = batch_size  # forward passes per batch of masked inputs
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size]
            targets = [ClassifierOutputTarget(int(c)) for c in classes[start:start + batch_size]]
            out.append(cam(input_tensor=x, targets=targets))
    return np.concatenate(out)
