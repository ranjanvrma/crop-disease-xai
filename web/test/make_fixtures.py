"""Reference outputs from the Python pipeline for the JavaScript parity tests.

    python web/test/make_fixtures.py        # then: cd web && npm test

Writes ``web/test/fixtures/`` (git-ignored): a seeded MobileNetV3 bundle, a
synthetic photo, and what the research code makes of it (albumentations
preprocessing, PyTorch logits and feature maps, closed-form Grad-CAM++ heatmaps).
"""

import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cropxai.data.transforms import eval_transform
from cropxai.explain.closed_form import closed_form_gradcampp, to_heatmap
from cropxai.export.web import export_web_bundle
from cropxai.models import build_model

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fixtures"
SIZE = 224


def synthetic_photo(h: int, w: int) -> np.ndarray:
    """A leaf-like test image: green gradient, brown spots, noise (uint8 RGB)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([40 + 60 * xx / w, 90 + 120 * yy / h, 30 + 40 * (xx + yy) / (h + w)], axis=-1)
    for cy, cx, r in rng.integers([0, 0, 10], [h, w, 40], size=(12, 3)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = [120, 70, 30]
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write(name: str, array: np.ndarray, dtype: str) -> None:
    np.ascontiguousarray(array, dtype=dtype).tofile(OUT / name)


def main() -> None:
    torch.manual_seed(0)
    classes = json.loads((ROOT / "splits/classes_v1.json").read_text(encoding="utf-8"))
    model = build_model(OmegaConf.create({"arch": "mobilenetv3_large_100", "pretrained": False, "drop_rate": 0.0}), len(classes)).eval()
    with torch.no_grad():  # random init gives near-zero logits; spread them so heatmaps differ per class
        model.get_classifier().weight.mul_(20)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = export_web_bundle(model, classes, OUT / "bundle", SIZE, ROOT / "knowledge/remedies.yaml", 0.5, {"demo": True})

    cfg = OmegaConf.create({"image_size": SIZE})
    photo = synthetic_photo(301, 403)  # odd sizes exercise rounding in resize and crop
    x = eval_transform(cfg)(image=photo)["image"][None]
    crop = A.Compose(eval_transform(cfg).transforms[:2])(image=photo)["image"]  # resize + crop, no normalise

    with torch.no_grad():
        features = model.forward_features(x)
        logits = model.forward_head(features)
        explained = [int(c) for c in logits[0].argsort(descending=True)[:3]]
        heat = to_heatmap(closed_form_gradcampp(features.repeat(3, 1, 1, 1), model.get_classifier().weight, torch.tensor(explained)), (SIZE, SIZE))

    rgba = np.concatenate([photo, np.full(photo.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    write("photo_rgba.bin", rgba, "u1")
    Image.fromarray(photo).save(OUT / "photo.png")  # for the browser tests
    write("crop_rgb.bin", crop, "u1")
    write("input.bin", x.numpy(), "<f4")
    write("logits.bin", logits.numpy(), "<f4")
    write("features.bin", features.numpy(), "<f4")
    write("heatmaps.bin", heat, "<f4")
    (OUT / "fixtures.json").write_text(json.dumps({
        "photo": {"width": photo.shape[1], "height": photo.shape[0]},
        "explained_classes": explained,
        "feature_shape": meta["feature_shape"],
        "image_size": SIZE,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
