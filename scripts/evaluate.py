"""Final evaluation of trained checkpoints: in-domain test, field test, efficiency.

    python scripts/evaluate.py --config configs/experiment/effnet_b0_ft.yaml

This is the only place the field set is read (``allow_heldout=True``). Run it
once per finished experiment, never to choose hyperparameters. Writes
``results/metrics/accuracy/<run>_seed<n>.json``.
"""

import argparse

import torch

from cropxai.compress.variants import is_int8
from cropxai.data.dataset import make_loader
from cropxai.eval.classification import classification_metrics, field_metrics, predict
from cropxai.eval.efficiency import count_params, measure_latency, state_dict_size_mb
from cropxai.train.loop import get_device, load_trained
from cropxai.utils.config import load_config
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import save_metrics


def evaluate(cfg) -> dict:
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    model, _, ckpt = load_trained(cfg, device)
    int8 = is_int8(cfg)

    test = classification_metrics(*predict(model, make_loader(cfg, "test"), device))
    field = field_metrics(*predict(model, make_loader(cfg, "field_test", allow_heldout=True), device))

    metrics = {
        "arch": cfg.model.arch,
        "train_mode": cfg.train.mode,
        "int8": int8,
        "quantizer": ckpt.get("quantizer"),
        "params": ckpt["params"] if int8 else count_params(model),
        "size_mb_fp32": None if int8 else state_dict_size_mb(model),
        "best_epoch": ckpt["best"].get("epoch"),
        "val_macro_f1": ckpt["best"].get("val_macro_f1"),
        "test": test,
        "field": field,
        # Desktop latency of the INT8 graph runs reference (de)quantize ops and says nothing
        # about real INT8 speed; INT8 latency is measured on the phone instead.
        "latency_desktop_gpu": None if int8 or device.type != "cuda" else measure_latency(model, cfg.image_size, device),
        "latency_desktop_cpu": None if int8 else measure_latency(model, cfg.image_size, torch.device("cpu")),
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
