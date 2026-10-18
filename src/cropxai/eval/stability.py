"""Explanation stability under compression: how similar are two models' heatmaps?

Compares a compressed model's heatmaps with its uncompressed parent's on the
same images and the same target class. All maps are in [0, 1], shape (H, W).
"""

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F


def ssim(a: np.ndarray, b: np.ndarray, window: int = 11, sigma: float = 1.5) -> float:
    """Structural similarity of two maps in [0, 1] (Gaussian window, standard constants)."""
    x = torch.from_numpy(a).float()[None, None]
    y = torch.from_numpy(b).float()[None, None]
    coords = torch.arange(window, dtype=torch.float32) - window // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :])[None, None]

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=window // 2)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(s.mean())


def topk_iou(a: np.ndarray, b: np.ndarray, fraction: float = 0.1) -> float:
    """IoU of the top ``fraction`` most-highlighted pixels of each map."""
    k = max(1, int(round(fraction * a.size)))
    top_a = np.zeros(a.size, dtype=bool)
    top_b = np.zeros(b.size, dtype=bool)
    top_a[np.argsort(a.ravel())[-k:]] = True
    top_b[np.argsort(b.ravel())[-k:]] = True
    return float((top_a & top_b).sum() / (top_a | top_b).sum())


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of pixel values (0 if either map is constant)."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def stability_metrics(parent_cams: np.ndarray, variant_cams: np.ndarray) -> dict[str, float]:
    """Mean Spearman, SSIM and top-10% IoU over images."""
    pairs = list(zip(parent_cams, variant_cams))
    return {
        "spearman": float(np.mean([rank_correlation(p, v) for p, v in pairs])),
        "ssim": float(np.mean([ssim(p, v) for p, v in pairs])),
        "top10_iou": float(np.mean([topk_iou(p, v) for p, v in pairs])),
        "n_images": len(pairs),
    }
