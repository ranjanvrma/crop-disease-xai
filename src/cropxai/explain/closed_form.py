"""Gradient-free Grad-CAM++ for a GAP + single-linear-layer head.

Derivation (notation as in pytorch-grad-cam / Chattopadhay et al. eq. 19):
for feature maps A^k of size H x W and a head ``y_c = sum_k W[c,k] * mean(A^k) + b_c``,

    g_k = d y_c / d A^k_ij = W[c,k] / (H*W)            (constant in i, j and input)

so the Grad-CAM++ pixel weights collapse to

    a_k = g_k^2 / (2 g_k^2 + S_k g_k^3 + eps),         S_k = sum_ij A^k_ij
    w_k = sum_ij relu(g_k) * a_k = H*W * relu(g_k) * a_k
    cam = relu( sum_k w_k A^k )

Only the feature maps (one forward pass) and the classifier weights are
needed: no backward pass, so it runs on mobile runtimes that are
forward-only. ``eps`` matches the library so the two agree to float precision.
"""

import numpy as np
import torch
from torch.nn import functional as F

EPS = 1e-6  # same constant pytorch-grad-cam uses


def closed_form_gradcampp(features: torch.Tensor, weight: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """Low-resolution Grad-CAM++ maps, shape (N, H, W), before normalisation.

    Args:
        features: final conv feature maps, (N, K, H, W).
        weight: classifier weight matrix, (num_classes, K).
        classes: target class per image, (N,).
    """
    n, k, h, w = features.shape
    g = weight[classes] / (h * w)                       # (N, K)
    s = features.sum(dim=(2, 3))                        # (N, K)
    a = g**2 / (2 * g**2 + s * g**3 + EPS)
    a = torch.where(g != 0, a, torch.zeros_like(a))
    cam_weights = (h * w) * g.clamp(min=0) * a          # (N, K)
    cam = (cam_weights[:, :, None, None] * features).sum(dim=1)
    return cam.clamp(min=0)


def to_heatmap(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Normalise to [0, 1] and upsample to (H, W) the way pytorch-grad-cam does.

    min-max -> bilinear resize (half-pixel centres, same as cv2.INTER_LINEAR)
    -> min-max again (the library re-normalises after aggregating layers).
    """
    def minmax(x: torch.Tensor) -> torch.Tensor:
        lo = x.amin(dim=(1, 2), keepdim=True)
        x = x - lo
        return x / (1e-7 + x.amax(dim=(1, 2), keepdim=True))

    up = F.interpolate(minmax(cam)[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    return minmax(up).cpu().numpy().astype(np.float32)


@torch.no_grad()
def explain_closed_form(model: torch.nn.Module, images: torch.Tensor, classes: torch.Tensor) -> np.ndarray:
    """One forward pass -> heatmaps in [0, 1] at input resolution, (N, H, W)."""
    features = model.forward_features(images)
    weight = model.get_classifier().weight.float().to(features.device)
    cam = closed_form_gradcampp(features.float(), weight, classes.to(features.device))
    return to_heatmap(cam, images.shape[-2:])
