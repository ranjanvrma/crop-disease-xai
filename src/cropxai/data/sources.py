"""Scan each image source into a table of (path, label, source, mask_path).

Labels are always PlantVillage class names (the folder names, e.g.
``Tomato___Early_blight``), so every source shares one label space.
Paths are stored relative to the data root so manifests work on any machine.
"""

from pathlib import Path

import pandas as pd
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COLUMNS = ["path", "label", "source", "mask_path"]


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _rel(p: Path, data_root: Path) -> str:
    return p.relative_to(data_root).as_posix()


def scan_plantvillage(data_root: Path, pv_dir: str) -> pd.DataFrame:
    """PlantVillage: ``<pv_dir>/<Class___name>/*.jpg``."""
    root = data_root / pv_dir
    rows = [
        (_rel(img, data_root), class_dir.name, "plantvillage", "")
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir())
        for img in _images_in(class_dir)
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def load_class_map(path: Path) -> dict[str, str | None]:
    """PlantDoc folder name -> PlantVillage class (``None`` = excluded)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))["plantdoc_to_plantvillage"]


def scan_plantdoc(
    data_root: Path, pd_dirs: list[str], class_map: dict[str, str | None], pv_classes: set[str]
) -> pd.DataFrame:
    """PlantDoc classification folders, relabelled to PlantVillage classes.

    Fails loudly on any folder name missing from the class map, so a renamed
    or new PlantDoc class can never be silently dropped or mislabelled.
    """
    rows = []
    for d in pd_dirs:
        for class_dir in sorted(p for p in (data_root / d).iterdir() if p.is_dir()):
            if class_dir.name not in class_map:
                raise ValueError(
                    f"PlantDoc class '{class_dir.name}' (in {d}) is not in the class map. "
                    "Add it to configs/data/plantdoc_to_plantvillage.yaml (or map it to null)."
                )
            target = class_map[class_dir.name]
            if target is None:
                continue
            if target not in pv_classes:
                raise ValueError(f"Class map target '{target}' is not a PlantVillage class.")
            rows += [(_rel(img, data_root), target, "plantdoc", "") for img in _images_in(class_dir)]
    return pd.DataFrame(rows, columns=COLUMNS)


def scan_field(data_root: Path, field_dir: str, pv_classes: set[str]) -> pd.DataFrame:
    """Own field photos: ``<field_dir>/images/<PV class>/*.jpg``.

    Optional lesion masks: ``<field_dir>/masks/<PV class>/<same stem>.png``.
    Returns an empty table if no field photos exist yet.
    """
    images_root = data_root / field_dir / "images"
    masks_root = data_root / field_dir / "masks"
    if not images_root.exists():
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for class_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        if class_dir.name not in pv_classes:
            raise ValueError(f"Field folder '{class_dir.name}' is not a PlantVillage class name.")
        for img in _images_in(class_dir):
            mask = masks_root / class_dir.name / f"{img.stem}.png"
            rows.append((_rel(img, data_root), class_dir.name, "field",
                         _rel(mask, data_root) if mask.exists() else ""))
    return pd.DataFrame(rows, columns=COLUMNS)
