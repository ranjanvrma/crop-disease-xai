"""Final evaluation of trained checkpoints: in-domain test, field test, efficiency.

    python scripts/evaluate.py --config configs/experiment/effnet_b0_ft.yaml

This is the only place the field set is read (``allow_heldout=True``). Run it
once per finished experiment, never to choose hyperparameters. Writes
``results/metrics/accuracy/<run>_seed<n>.json``.
"""

import argparse

import torch
from omegaconf import OmegaConf

from cropxai.data.dataset import make_loader
from cropxai.eval.classification import classification_metrics, field_metrics, predict
from cropxai.eval.efficiency import count_params, measure_latency, state_dict_size_mb
from cropxai.models import build_model
from cropxai.train.loop import checkpoint_path, get_device
from cropxai.utils.config import load_config
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import save_metrics


def evaluate(cfg) -> dict:
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    ckpt = torch.load(checkpoint_path(cfg), map_location="cpu", weights_only=False)

    # Weights come from the checkpoint, so skip downloading ImageNet weights.
    model_cfg = OmegaConf.merge(cfg.model, {"pretrained": False})
    model = build_model(model_cfg, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    test = classification_metrics(*predict(model, make_loader(cfg, "test"), device))
    field = field_metrics(*predict(model, make_loader(cfg, "field_test", allow_heldout=True), device))

    metrics = {
        "arch": cfg.model.arch,
        "train_mode": cfg.train.mode,
        "params": count_params(model),
        "size_mb_fp32": state_dict_size_mb(model),
        "best_epoch": ckpt["best"]["epoch"],
        "val_macro_f1": ckpt["best"]["val_macro_f1"],
        "test": test,
        "field": field,
        "latency_desktop_gpu": measure_latency(model, cfg.image_size, device) if device.type == "cuda" else None,
        "latency_desktop_cpu": measure_latency(model, cfg.image_size, torch.device("cpu")),
    }
    path = save_metrics(cfg, metrics, kind="accuracy")
    print(f"wrote {path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", help="repeatable; default: cfg.seeds")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    seeds = args.seeds or list(load_config(args.config, args.overrides).seeds)
    for seed in seeds:
        evaluate(load_config(args.config, [*args.overrides, f"seed={seed}"]))


if __name__ == "__main__":
    main()
