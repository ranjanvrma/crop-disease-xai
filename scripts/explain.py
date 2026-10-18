"""Explanation quality of trained checkpoints: faithfulness + localisation.

    python scripts/explain.py --config configs/experiment/mnv3_fe.yaml --seed 0

For each split in ``cfg.explain.splits`` and each CAM method, explains the
model's *predicted* class on a fixed image sample and records deletion /
insertion AUC, Average Drop, Increase in Confidence, ROAD and (where lesion
masks exist) pointing game + IoU. Writes
``results/metrics/explanation/<run>_seed<n>.json`` and an example figure.
"""

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # Score-CAM's per-batch progress bars flood the log

import numpy as np
import torch

from cropxai.compress.variants import compress_cfg, root_config
from cropxai.data.dataset import load_examples
from cropxai.eval.faithfulness import average_drop_and_increase, deletion_insertion, road_score
from cropxai.eval.localisation import localisation_metrics
from cropxai.eval.stability import stability_metrics
from cropxai.explain.cams import compute_cams
from cropxai.explain.figures import save_examples_figure
from cropxai.train.loop import get_device, load_trained
from cropxai.utils.config import load_config, resolve_path
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import save_metrics


def method_metrics(model, images, cams, preds, masks, has_mask, e) -> dict:
    """All metrics for one method on one split (means over images)."""
    d = e.deletion
    del_ins = [deletion_insertion(model, img, cam, int(c), d.n_steps, d.blur_kernel, d.blur_sigma)
               for img, cam, c in zip(images, cams, preds)]
    drop, increase = average_drop_and_increase(model, images, cams, preds)
    n_road = min(e.road_n_images, len(images))
    road = road_score(model, images[:n_road], cams[:n_road], preds[:n_road], e.road_percentiles)
    out = {
        "deletion_auc": float(np.mean([x[0] for x in del_ins])),
        "insertion_auc": float(np.mean([x[1] for x in del_ins])),
        "average_drop": drop,
        "increase_in_confidence": increase,
        "road": float(road.mean()),
        "road_n_images": n_road,
        "n_images": len(images),
    }
    if masks is not None and has_mask[: len(images)].any():
        sel = has_mask[: len(images)]
        out |= localisation_metrics(cams[sel], masks[: len(images)][sel], e.iou_threshold)
    return out


def explain(cfg) -> dict:
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    model, _, _ = load_trained(cfg, device)
    e = cfg.explain
    root = load_trained(root_config(cfg), device)[0] if compress_cfg(cfg) else None

    results = {}
    for split in e.splits:
        images, labels, paths, masks, has_mask = load_examples(
            cfg, split, e.n_images, allow_heldout=split == "field_test"
        )
        images = images.to(device)
        with torch.no_grad():
            preds = model(images).argmax(dim=1)
        split_out = {"sample_accuracy": float((preds.cpu() == labels).float().mean()), "n_masks": int(has_mask.sum())}

        figure_cams = {}
        for method in e.methods:
            n = e.scorecam_n_images if method == "scorecam" else len(images)
            t0 = time.perf_counter()
            bs = e.scorecam_batch_size if method == "scorecam" else e.batch_size
            cams = compute_cams(model, cfg.model.arch, method, images[:n], preds[:n].tolist(), bs)
            t1 = time.perf_counter()
            split_out[method] = method_metrics(model, images[:n], cams, preds[:n], masks, has_mask, e)
            figure_cams[method] = cams[: e.n_figure_examples]
            print(f"[{cfg.run_name} seed{cfg.seed}] {split} {method} "
                  f"(cams {t1 - t0:.0f}s, metrics {time.perf_counter() - t1:.0f}s): {split_out[method]}", flush=True)
        if root is not None:
            # Stability: both models' closed-form Grad-CAM++ maps for the class the
            # uncompressed parent predicts, on the same images.
            with torch.no_grad():
                target = root(images).argmax(dim=1)
            parent_cams = compute_cams(root, root_config(cfg).model.arch, "closed_form", images, target.tolist(), e.batch_size)
            variant_cams = compute_cams(model, cfg.model.arch, "closed_form", images, target.tolist(), e.batch_size)
            split_out["stability"] = stability_metrics(parent_cams, variant_cams)
            print(f"[{cfg.run_name} seed{cfg.seed}] {split} stability vs parent: {split_out['stability']}", flush=True)
        results[split] = split_out

        fig_path = resolve_path(cfg.paths.results_dir) / "figures/heatmaps" / f"{cfg.run_name}_seed{cfg.seed}_{split}.png"
        k = e.n_figure_examples
        save_examples_figure(images[:k].cpu(), labels[:k], preds[:k].cpu(), figure_cams, fig_path)

    path = save_metrics(cfg, results, kind="explanation")
    print(f"wrote {path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", help="repeatable; default: cfg.seeds")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    seeds = args.seeds or list(load_config(args.config, args.overrides).seeds)
    for seed in seeds:
        explain(load_config(args.config, [*args.overrides, f"seed={seed}"]))


if __name__ == "__main__":
    main()
