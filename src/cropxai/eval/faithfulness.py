"""Faithfulness metrics: does the heatmap reflect what drove the prediction?

All scores use the softmax probability of the explained class. Images are
ImageNet-normalised tensors; "removing" a pixel means setting it to the
dataset mean (0 in normalised space), which avoids out-of-range colours.
"""

import numpy as np
import torch
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


# Neighbour weights of ROAD's noisy linear imputation (Rong et al., 2022):
# 1/6 for edge neighbours, 1/12 for corner neighbours.
ROAD_KERNEL = torch.tensor([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0.0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])


@torch.no_grad()
def linear_impute(images: torch.Tensor, keep: torch.Tensor, tol: float = 1e-6, max_iter: int = 2000) -> torch.Tensor:
    """Fill removed pixels with ROAD's linear interpolation, batched on the image's device.

    Every removed pixel must equal the weighted mean of its in-image 8
    neighbours (known or removed), which is the same sparse system
    pytorch-grad-cam solves per image with scipy. Here it is solved for the
    whole batch at once by conjugate gradient, using convolutions.

    Args:
        images: (N, C, H, W).
        keep: (N, H, W) bool, True where the pixel is kept.
    Returns the images with removed pixels interpolated (no noise added).
    """
    n, c, h, w = images.shape
    kernel = ROAD_KERNEL.to(images).view(1, 1, 3, 3)
    known = keep.to(images).unsqueeze(1).expand(n, c, h, w).reshape(n * c, 1, h, w)
    miss = 1 - known
    x = images.reshape(n * c, 1, h, w)

    def neighbours(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=1)  # zero padding: off-image neighbours don't count

    weight_sum = neighbours(torch.ones(1, 1, h, w, device=images.device, dtype=images.dtype))

    def system(u: torch.Tensor) -> torch.Tensor:  # SPD operator on the removed pixels
        return (weight_sum * u - neighbours(u)) * miss

    rhs = neighbours(x * known) * miss
    u = torch.zeros_like(x)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum(dim=(1, 2, 3), keepdim=True)
    rhs_norm = rs.sqrt().clamp(min=1e-12)
    for i in range(max_iter):
        ap = system(p)
        alpha = rs / (p * ap).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-30)
        u = u + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum(dim=(1, 2, 3), keepdim=True)
        if i % 10 == 0 and (rs_new.sqrt() / rhs_norm).max() < tol:
            break
        p = r + (rs_new / rs.clamp(min=1e-30)) * p
        rs = rs_new
    return (x * known + u * miss).reshape(n, c, h, w)


@torch.no_grad()
def road_score(
    model: nn.Module,
    images: torch.Tensor,
    cams: np.ndarray,
    classes: torch.Tensor,
    percentiles: list[int],
    noise: float = 0.01,
    seed: int = 0,
    batch_size: int = 64,
) -> np.ndarray:
    """ROAD combined score per image (Rong et al.): mean over percentiles of (LeRF - MoRF) / 2.

    At percentile p, MoRF removes the pixels whose heatmap value is at or
    above the p-th percentile (most relevant first); LeRF does the same on
    1 - heatmap. Removed pixels are filled by noisy linear imputation, so they
    don't leak their location through artefacts. Higher is better. Same
    definition as pytorch-grad-cam's ROADCombined, computed on the GPU.
    """
    cam = torch.from_numpy(cams).to(images.device).flatten(1)            # (N, H*W)
    gen = torch.Generator(device=images.device).manual_seed(seed)
    n, h, w = cams.shape
    scores = torch.zeros(n, device=images.device)
    for pct in percentiles:
        for sign, relevance in ((-1.0, cam), (1.0, 1 - cam)):             # MoRF, LeRF
            threshold = torch.quantile(relevance, pct / 100, dim=1, keepdim=True)
            keep = (relevance < threshold).view(n, h, w)
            for start in range(0, n, batch_size):
                sl = slice(start, start + batch_size)
                filled = linear_impute(images[sl], keep[sl])
                removed = (~keep[sl]).unsqueeze(1).to(images)
                filled = filled + noise * removed * torch.randn(
                    filled.shape, generator=gen, device=images.device, dtype=images.dtype)
                scores[sl] += sign * class_prob(model, filled, classes[sl]) / 2
    return (scores / len(percentiles)).cpu().numpy().astype(np.float64)
