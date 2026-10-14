"""Manifest building and the held-out field-set guard, on a synthetic dataset."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cropxai.data.dataset import LeafDataset
from cropxai.data.guard import FieldLeakError, assert_no_field_leak, check_split_access
from cropxai.data.hashing import dhash, min_hamming
from cropxai.data.manifest import build_manifest
from cropxai.utils.config import load_config

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


def _cfg(root: Path):
    return load_config(
        "configs/experiment/prepare_data.yaml",
        [f"paths.data_root={root.as_posix()}",
         f"data.class_map={(root / 'map.yaml').as_posix()}",
         "data.plantdoc_dirs=[plantdoc/test]"],
    )


def test_build_manifest(dataset_root):
    manifest, summary = build_manifest(_cfg(dataset_root))

    field = manifest[manifest["split"] == "field_test"]
    assert set(field["source"]) == {"plantdoc", "field"}
    assert len(field) == 3 + 1                         # 'Weird class' excluded
    assert set(field["label"]) == {"Tomato___healthy", "Tomato___Early_blight"}
    assert summary["n_lesion_masks"] == 1

    assert summary["dropped_exact_duplicates"] == 1     # dupe.jpg
    assert summary["dropped_near_duplicates_of_heldout"] == 1
    assert "plantvillage/color/Tomato___Early_blight/5.jpg" not in set(manifest["path"])

    pv = manifest[manifest["source"] == "plantvillage"]
    assert set(pv["split"]) == {"train", "val", "test"}
    for split in ["val", "test"]:                      # stratified: every class present
        assert set(pv.loc[pv["split"] == split, "label"]) == set(PV_CLASSES)


def test_split_is_reproducible(dataset_root):
    a, _ = build_manifest(_cfg(dataset_root))
    b, _ = build_manifest(_cfg(dataset_root))
    assert a.equals(b)


def test_guard_rejects_field_row_in_train(dataset_root):
    manifest, _ = build_manifest(_cfg(dataset_root))
    manifest.loc[manifest["source"] == "field", "split"] = "train"
    with pytest.raises(FieldLeakError):
        assert_no_field_leak(manifest, cross_hamming=4)


def test_guard_rejects_near_duplicate_in_train(dataset_root):
    manifest, _ = build_manifest(_cfg(dataset_root))
    field_hash = manifest.loc[manifest["source"] == "field", "dhash"].iloc[0]
    train_idx = manifest.index[manifest["split"] == "train"][0]
    manifest.loc[train_idx, "dhash"] = field_hash
    with pytest.raises(FieldLeakError):
        assert_no_field_leak(manifest, cross_hamming=4)


def test_field_split_needs_explicit_opt_in():
    with pytest.raises(FieldLeakError):
        check_split_access("field_test", allow_heldout=False)
    check_split_access("field_test", allow_heldout=True)
    check_split_access("train", allow_heldout=False)


def test_unmapped_plantdoc_class_fails_loudly(dataset_root):
    (dataset_root / "map.yaml").write_text("plantdoc_to_plantvillage:\n  Tomato leaf: Tomato___healthy\n")
    with pytest.raises(ValueError, match="Weird class"):
        build_manifest(_cfg(dataset_root))


def test_dataset_loads_items(dataset_root):
    manifest, _ = build_manifest(_cfg(dataset_root))
    ds = LeafDataset(manifest, "train", dataset_root, {c: i for i, c in enumerate(PV_CLASSES)})
    image, label = ds[0]
    assert image.ndim == 3 and image.shape[2] == 3
    assert 0 <= label < len(PV_CLASSES)
    with pytest.raises(FieldLeakError):
        LeafDataset(manifest, "field_test", dataset_root, {})


def test_min_hamming():
    q = np.array([0b1111, 0], dtype=np.uint64)
    ref = np.array([0b0111, 0b1000_0000], dtype=np.uint64)
    assert min_hamming(q, ref).tolist() == [1, 1]
    assert min_hamming(q, np.array([], dtype=np.uint64)).tolist() == [64, 64]


def test_dhash_stable_under_reencoding(tmp_path):
    img = _smooth_image(np.random.default_rng(1))
    img.save(tmp_path / "a.png")
    img.resize((40, 40)).save(tmp_path / "b.jpg", quality=60)
    a, b = dhash(tmp_path / "a.png"), dhash(tmp_path / "b.jpg")
    assert bin(a ^ b).count("1") <= 4
