"""Check closed-form (gradient-free) Grad-CAM++ against pytorch-grad-cam.

    python scripts/verify_closed_form.py --config configs/experiment/mnv3_fe.yaml --seed 0

On a fixed test sample, compares heatmaps at input resolution (max / mean
absolute difference, Pearson r) and times both on the current device. If the
derivation is right, differences are at float-rounding level. Writes
``results/metrics/closed_form/<run>_seed<n>.json`` (a paper subsection).
"""

import argparse
import time

import numpy as np
import torch

from cropxai.data.dataset import load_examples
from cropxai.explain.cams import compute_cams
from cropxai.explain.closed_form import explain_closed_form
from cropxai.train.loop import get_device, load_trained
from cropxai.utils.config import load_config
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import save_metrics


def timed(fn, device: torch.device) -> tuple[object, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def verify(cfg) -> dict:
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    model, _, _ = load_trained(cfg, device)
    images, _, _, _, _ = load_examples(cfg, "test", cfg.explain.verify_n_images)
    images = images.to(device)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    lib, lib_ms = timed(lambda: compute_cams(model, cfg.model.arch, "gradcampp", images, preds.tolist(), 1), device)
    ours, ours_ms = timed(lambda: np.concatenate(
        [explain_closed_form(model, images[i:i + 1], preds[i:i + 1]) for i in range(len(images))]), device)

    diff = np.abs(lib - ours)
    r = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(lib, ours) if a.std() > 0 and b.std() > 0]
    metrics = {
        "n_images": len(images),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pearson_r_min": float(np.min(r)),
        "pearson_r_mean": float(np.mean(r)),
        "ms_per_image_library": lib_ms / len(images),
        "ms_per_image_closed_form": ours_ms / len(images),
        "device": str(device),
    }
    print(metrics)
    save_metrics(cfg, metrics, kind="closed_form")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", help="repeatable; default: cfg.seeds")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    seeds = args.seeds or list(load_config(args.config, args.overrides).seeds)
    for seed in seeds:
        verify(load_config(args.config, [*args.overrides, f"seed={seed}"]))


if __name__ == "__main__":
    main()
