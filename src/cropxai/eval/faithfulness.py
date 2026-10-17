"""Faithfulness metrics: does the heatmap reflect what drove the prediction?

All scores use the softmax probability of the explained class. Images are
ImageNet-normalised tensors; "removing" a pixel means setting it to the
dataset mean (0 in normalised space), which avoids out-of-range colours.
"""

import numpy as np
import torch
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def class_prob(model: nn.Module, x: torch.Tensor, cls: int | torch.Tensor) -> torch.Tensor:
    probs = model(x).float().softmax(dim=1)
    if isinstance(cls, int):
        return probs[:, cls]
    return probs.gather(1, cls.view(-1, 1)).squeeze(1)


def gaussian_blur(x: torch.Tensor, kernel: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded (no torchvision dependency)."""
    coords = torch.arange(kernel, dtype=x.dtype, device=x.device) - kernel // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    pad = kernel // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), g.transpose(2, 3), groups=x.shape[1])


@torch.no_grad()
def deletion_insertion(
    model: nn.Module, image: torch.Tensor, cam: np.ndarray, cls: int, n_steps: int, blur_kernel: int, blur_sigma: float
) -> tuple[float, float]:
    """Deletion and insertion AUC for one image (Petsiuk et al., RISE).

    Deletion: remove pixels most-salient-first (to the mean colour) and track
    the class probability; a faithful map makes it fall fast (lower AUC is
    better). Insertion: start from a blurred image and restore pixels
    most-salient-first; a faithful map makes it rise fast (higher is better).
    """
    _, h, w = image.shape
    order = torch.from_numpy(cam).flatten().argsort(descending=True).to(image.device)
    fractions = torch.linspace(0, 1, n_steps + 1)
    counts = (fractions * h * w).round().long()

    # rank[p] = position of pixel p in the saliency order
    rank = torch.empty_like(order)
    rank[order] = torch.arange(h * w, device=image.device)
    keep = rank.view(1, h, w) < counts.view(-1, 1, 1).to(image.device)  # (S+1, H, W): top-t pixels
    keep = keep[:, None].float()

    x = image[None]
    blurred = gaussian_blur(x, blur_kernel, blur_sigma)
    deleted = x * (1 - keep)                       # mean colour == 0 in normalised space
    inserted = blurred * (1 - keep) + x * keep

    p_del = class_prob(model, deleted, cls).cpu().numpy()
    p_ins = class_prob(model, inserted, cls).cpu().numpy()
    dx = 1.0 / n_steps
    return float(np.trapezoid(p_del, dx=dx)), float(np.trapezoid(p_ins, dx=dx))


@torch.no_grad()
def average_drop_and_increase(
    model: nn.Module, images: torch.Tensor, cams: np.ndarray, classes: torch.Tensor
) -> tuple[float, float]:
    """Average Drop (%) and Increase in Confidence (%) (Chattopadhay et al.).

    The explanation map is used as a soft mask on the image (in pixel space).
    Average Drop = mean(max(0, Y - O) / Y) * 100 (lower is better);
    Increase in Confidence = % of images where O > Y (higher is better).
    """
    mean, std = IMAGENET_MEAN.to(images.device), IMAGENET_STD.to(images.device)
    m = torch.from_numpy(cams).to(images.device)[:, None]
    masked = ((images * std + mean) * m - mean) / std
    y = class_prob(model, images, classes)
    o = class_prob(model, masked, classes)
    drop = ((y - o).clamp(min=0) / y.clamp(min=1e-8)).mean().item() * 100
    increase = (o > y).float().mean().item() * 100
    return drop, increase


def road_score(
    model: nn.Module, images: torch.Tensor, cams: np.ndarray, classes: torch.Tensor, percentiles: list[int]
) -> np.ndarray:
    """ROAD combined score per image (Rong et al.): (LeRF - MoRF) / 2, higher is better.

    Uses noisy linear imputation instead of a constant baseline, so removed
    pixels don't leak their location through artefacts.
    """
    targets = [ClassifierOutputSoftmaxTarget(int(c)) for c in classes]
    with torch.no_grad():
        return np.asarray(ROADCombined(percentiles=list(percentiles))(images, cams, targets, model), dtype=np.float64)
