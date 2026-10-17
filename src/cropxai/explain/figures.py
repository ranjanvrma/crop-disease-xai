"""Example heatmap figures: one row per image, one column per method."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cropxai.eval.faithfulness import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

METHOD_TITLES = {"gradcampp": "Grad-CAM++", "gradcam": "Grad-CAM", "scorecam": "Score-CAM", "closed_form": "Closed-form"}


def to_rgb(images: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> (N, H, W, 3) in [0, 1]."""
    return (images * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def save_examples_figure(
    images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, cams: dict[str, np.ndarray], path: Path
) -> None:
    """Grid of image | heatmap overlays. Title marks correct (✓) / wrong (✗)."""
    rgb = to_rgb(images)
    methods = list(cams)
    n = len(rgb)
    fig, axes = plt.subplots(n, 1 + len(methods), figsize=(2.2 * (1 + len(methods)), 2.2 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(rgb[i])
        axes[i, 0].set_title("✓" if preds[i] == labels[i] else "✗", fontsize=9)
        for j, m in enumerate(methods, start=1):
            axes[i, j].imshow(rgb[i])
            if i < len(cams[m]):
                axes[i, j].imshow(cams[m][i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(METHOD_TITLES.get(m, m), fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
