"""Localisation metrics: does the heatmap point at the actual lesions?

Needs lesion masks (binary, lesion = 1), resized/cropped exactly like the image.
"""

import numpy as np


def pointing_game_hit(cam: np.ndarray, mask: np.ndarray) -> bool:
    """True if the heatmap's maximum falls inside a lesion."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[y, x] > 0)


def iou(cam: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """IoU between the thresholded heatmap (cam >= threshold) and the lesion mask."""
    pred, gt = cam >= threshold, mask > 0
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 0.0


def localisation_metrics(cams: np.ndarray, masks: np.ndarray, threshold: float) -> dict[str, float]:
    """Pointing-game accuracy and mean IoU over images that have a mask."""
    hits = [pointing_game_hit(c, m) for c, m in zip(cams, masks)]
    ious = [iou(c, m, threshold) for c, m in zip(cams, masks)]
    return {"pointing_game": float(np.mean(hits)), "iou": float(np.mean(ious)), "n_masks": len(masks)}
