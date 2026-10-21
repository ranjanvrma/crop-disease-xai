"""Export a trained model as a bundle for the offline web app (``web/``).

The app runs on any phone or desktop browser with ONNX Runtime Web. Like the
INT8 graph, the exported model has two outputs, logits and the final feature
maps, so the app computes the closed-form Grad-CAM++ itself from one forward
pass (``src/cropxai/explain/closed_form.py``; the JavaScript port is
``web/src/xai.js``, checked against this file by the parity test).

Bundle layout (``<out>/``):

    model.onnx       float32, input "image" (1, 3, S, S), outputs "logits", "features"
    classifier.bin   float32 little-endian: weight (num_classes x K, row-major), then bias
    model.json       classes, preprocessing, feature shape, unsure threshold, provenance
    remedies.json    knowledge/remedies.yaml, as JSON
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from cropxai.compress.quantize import TwoOutputs
from cropxai.data.transforms import IMAGENET_MEAN, IMAGENET_STD

FORMAT_VERSION = 1
RESIZE_RATIO = 1.14  # eval_transform: resize short side to int(S * 1.14), then centre-crop S


def export_onnx(model: nn.Module, path: Path, image_size: int) -> tuple[int, int, int]:
    """Write the two-output ONNX graph (batch size 1). Returns the feature shape (K, H, W)."""
    module = TwoOutputs(model.cpu().eval()).eval()
    example = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        _, features = module(example)
        torch.onnx.export(
            module, (example,), str(path),
            input_names=["image"], output_names=["logits", "features"],
            opset_version=17, dynamo=False,
        )
    return tuple(int(d) for d in features.shape[1:])


def write_classifier(classifier: nn.Linear, path: Path) -> None:
    w = classifier.weight.detach().float().cpu().numpy()
    b = classifier.bias.detach().float().cpu().numpy()
    np.concatenate([w.ravel(), b]).astype("<f4").tofile(path)


def tune_unsure_threshold(probs: torch.Tensor, labels: torch.Tensor, target_accuracy: float) -> float:
    """Smallest confidence threshold whose accepted VALIDATION predictions reach ``target_accuracy``.

    Tuned on val only (HANDOFF rule 3). Val images are lab photos, so this does
    not protect against the lab-to-field shift; the app says so.
    """
    conf, pred = probs.max(dim=1)
    order = conf.argsort(descending=True)
    correct = pred[order].eq(labels[order]).float()
    running = correct.cumsum(0) / torch.arange(1, len(correct) + 1)
    ok = (running >= target_accuracy).nonzero()
    if len(ok) == 0:
        return 1.0
    return float(conf[order][ok[-1, 0]])


def remedies_json(src: Path, classes: list[str]) -> dict:
    remedies = yaml.safe_load(src.read_text(encoding="utf-8"))
    missing = sorted(set(classes) - set(remedies["classes"]))
    if missing:
        raise ValueError(f"knowledge base has no entry for: {missing}")
    return remedies


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_web_bundle(
    model: nn.Module,
    classes: list[str],
    out_dir: Path,
    image_size: int,
    remedies_src: Path,
    unsure_threshold: float,
    provenance: dict,
) -> dict:
    """Write the full bundle and return the contents of ``model.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_shape = export_onnx(model, out_dir / "model.onnx", image_size)
    classifier = model.get_classifier()
    if classifier.weight.shape != (len(classes), feature_shape[0]):
        raise ValueError("classifier does not match the feature maps; the closed form needs GAP + one Linear")
    write_classifier(classifier, out_dir / "classifier.bin")

    meta = {
        "format_version": FORMAT_VERSION,
        "classes": classes,
        "image_size": image_size,
        "resize_short": int(image_size * RESIZE_RATIO),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "feature_shape": list(feature_shape),
        "unsure_threshold": unsure_threshold,
        "onnx_sha256": sha256(out_dir / "model.onnx"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance,
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    remedies = remedies_json(remedies_src, classes)
    (out_dir / "remedies.json").write_text(json.dumps(remedies, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return meta
