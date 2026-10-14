"""Shared fixtures: a tiny synthetic PlantVillage / PlantDoc / field dataset."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PV_CLASSES = ["Potato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]
PER_CLASS = 20


def _smooth_image(rng: np.random.Generator) -> Image.Image:
    """Low-frequency random image: distinct per seed, stable under re-encoding."""
    tiny = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    return Image.fromarray(tiny).resize((64, 64), Image.Resampling.BICUBIC)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for cls in PV_CLASSES:
        d = tmp_path / "plantvillage/color" / cls
        d.mkdir(parents=True)
        for i in range(PER_CLASS):
            _smooth_image(rng).save(d / f"{i}.jpg")

    # An exact duplicate inside PlantVillage.
    pv = tmp_path / "plantvillage/color"
    (pv / "Tomato___healthy/dupe.jpg").write_bytes((pv / "Tomato___healthy/0.jpg").read_bytes())

    # PlantDoc: one mapped class, one explicitly excluded class.
    for name in ["Tomato leaf", "Weird class"]:
        d = tmp_path / "plantdoc/test" / name
        d.mkdir(parents=True)
        for i in range(3):
            _smooth_image(rng).save(d / f"pd{i}.jpg")

    # Own field photo that is a resized, re-encoded copy of a PV training image.
    d = tmp_path / "field/images/Tomato___Early_blight"
    d.mkdir(parents=True)
    src = Image.open(pv / "Tomato___Early_blight/5.jpg")
    src.resize((48, 48)).save(d / "my_leaf.jpg", quality=70)
    (tmp_path / "field/masks/Tomato___Early_blight").mkdir(parents=True)
    Image.new("L", (48, 48)).save(tmp_path / "field/masks/Tomato___Early_blight/my_leaf.png")

    (tmp_path / "map.yaml").write_text(
        "plantdoc_to_plantvillage:\n  Tomato leaf: Tomato___healthy\n  Weird class: null\n"
    )
    return tmp_path
