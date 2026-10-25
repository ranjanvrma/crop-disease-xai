"""Export one trained seed as the offline web app's model bundle (CPU only, about a minute).

    python scripts/export_web.py --config configs/experiment/mnv3_fe.yaml --seed 0
    python scripts/export_web.py --config configs/experiment/mnv3_fe.yaml --demo   # untrained, for app development

Writes ``web/public/model/`` (see ``src/cropxai/export/web.py``). The unsure
threshold is tuned on the validation split when the PlantVillage images are
available; accuracy numbers shown in the app come only from
``results/metrics/accuracy/<run>_seed<n>.json`` (never typed in by hand).
INT8 variants have no ONNX path yet: export their float parent.
"""

import argparse
import json

import torch
from omegaconf import OmegaConf

from cropxai.compress.variants import is_int8
from cropxai.data.dataset import load_classes, make_loader
from cropxai.eval.classification import predict
from cropxai.export.web import export_web_bundle, tune_unsure_threshold
from cropxai.models import build_model
from cropxai.train.loop import load_trained
from cropxai.utils.config import CONFIG_DIR, load_config, resolve_path
from cropxai.utils.seed import seed_everything
from cropxai.utils.tracking import git_commit


def measured_accuracy(cfg) -> dict | None:
    """Test / field accuracy of this exact run, if its results JSON exists."""
    path = resolve_path(cfg.paths.results_dir) / "metrics" / "accuracy" / f"{cfg.run_name}_seed{cfg.seed}.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return {
        "test_top1": m["test"]["top1"],
        "field_strict_top1": m["field"]["strict_top1"],
        "field_n": m["field"]["strict_n"],
        "source": str(path.relative_to(resolve_path("."))) if path.is_relative_to(resolve_path(".")) else path.name,
    }


def unsure_threshold(cfg, model) -> tuple[float, str]:
    u = cfg.app.unsure
    if not (resolve_path(cfg.paths.data_root) / cfg.data.plantvillage_dir).is_dir():
        return float(u.fallback_threshold), "config fallback (validation images not available)"
    logits, labels = predict(model, make_loader(cfg, "val"), torch.device("cpu"))
    t =tune_unsure_threshold(logits.softmax(dim=1), labels, float(u.target_val_accuracy))
    return t, f"tuned on val ({len(labels)} images) for {u.target_val_accuracy:.0%} accepted accuracy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="default: cfg.seed")
    parser.add_argument("--demo", action="store_true", help="untrained random weights (predictions are meaningless)")
    parser.add_argument("--out", default=None, help="default: cfg.app.out_dir")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    overrides = [*args.overrides, *([f"seed={args.seed}"] if args.seed is not None else [])]
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_DIR / "app/default.yaml"), load_config(args.config, overrides))
    if is_int8(cfg):
        raise SystemExit("INT8 variants have no ONNX export yet; export the float parent instead.")
    seed_everything(cfg.seed, cfg.deterministic)

    if args.demo:
        classes = load_classes(cfg)
        model = build_model(OmegaConf.merge(cfg.model, {"pretrained": False}), len(classes)).eval()
        threshold, threshold_source = float(cfg.app.unsure.fallback_threshold), "config fallback (demo model)"
        accuracy = None
    else:
        model, classes, _ = load_trained(cfg, torch.device("cpu"))
        threshold, threshold_source = unsure_threshold(cfg, model)
        accuracy = measured_accuracy(cfg)

    out = resolve_path(args.out or cfg.app.out_dir)
    meta = export_web_bundle(
        model, classes, out, cfg.image_size, resolve_path(cfg.app.remedies), threshold,
        provenance={
            "demo": args.demo,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "arch": cfg.model.arch,
            "git_commit": git_commit(),
            "unsure_threshold_source": threshold_source,
            "measured_accuracy": accuracy,
        },
    )
    print(f"wrote {out} ({meta['feature_shape']} features, unsure below {threshold:.3f}: {threshold_source})")


if __name__ == "__main__":
    main()
