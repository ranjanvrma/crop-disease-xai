"""Build the split manifest: one CSV listing every image, its label and split.

Pipeline: scan sources -> hash -> drop unreadable -> exact dedup ->
drop PlantVillage images that near-duplicate held-out images ->
stratified train/val/test split of PlantVillage -> guard check.

The split seed is fixed in the data config and independent of the training
seed, so all three training seeds see exactly the same split.
"""

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cropxai.data.guard import HELDOUT_SOURCES, HELDOUT_SPLIT, assert_no_field_leak
from cropxai.data.hashing import dhash, from_hex, min_hamming, to_hex
from cropxai.data.sources import load_class_map, scan_field, scan_plantdoc, scan_plantvillage
from cropxai.utils.config import resolve_path


def add_hashes(df: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Add a ``dhash`` column; drop and report files that can't be decoded."""
    hashes, unreadable = [], []
    for rel in tqdm(df["path"], desc="hashing", unit="img"):
        try:
            hashes.append(to_hex(dhash(data_root / rel)))
        except OSError:
            hashes.append(None)
            unreadable.append(rel)
    df = df.assign(dhash=hashes)
    return df[df["dhash"].notna()].reset_index(drop=True), unreadable


def stratified_split(df: pd.DataFrame, val: float, test: float, seed: int) -> pd.Series:
    """Label-stratified train/val/test assignment (fractions of the whole)."""
    rest, test_idx = train_test_split(df.index, test_size=test, stratify=df["label"], random_state=seed)
    train_idx, val_idx = train_test_split(
        rest, test_size=val / (1 - test), stratify=df.loc[rest, "label"], random_state=seed
    )
    split = pd.Series("train", index=df.index)
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def build_manifest(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    """Return the full manifest and a summary of everything that was dropped."""
    d = cfg.data
    data_root = resolve_path(cfg.paths.data_root)

    pv = scan_plantvillage(data_root, d.plantvillage_dir)
    pv_classes = set(pv["label"])
    class_map = load_class_map(resolve_path(d.class_map))
    heldout = pd.concat(
        [scan_plantdoc(data_root, list(d.plantdoc_dirs), class_map, pv_classes),
         scan_field(data_root, d.field_dir, pv_classes)],
        ignore_index=True,
    )

    df, unreadable = add_hashes(pd.concat([pv, heldout], ignore_index=True), data_root)

    # Exact duplicates (same dHash) within the training pool and within the
    # held-out pool: keep the first copy of each.
    is_heldout = df["source"].isin(HELDOUT_SOURCES)
    before = len(df)
    df = pd.concat([df[~is_heldout].drop_duplicates("dhash"), df[is_heldout].drop_duplicates("dhash")])
    exact_dupes = before - len(df)

    # PlantVillage images that near-duplicate a held-out image leave the pool.
    pool = df[~df["source"].isin(HELDOUT_SOURCES)].reset_index(drop=True)
    field = df[df["source"].isin(HELDOUT_SOURCES)].reset_index(drop=True)
    near = min_hamming(from_hex(pool["dhash"]), from_hex(field["dhash"])) <= d.dedup.cross_hamming
    pool = pool[~near].reset_index(drop=True)

    pool["split"] = stratified_split(pool, d.split.val, d.split.test, d.split.seed)
    field["split"] = HELDOUT_SPLIT
    manifest = pd.concat([pool, field], ignore_index=True)

    assert_no_field_leak(manifest, d.dedup.cross_hamming)

    summary = {
        "n_images": len(manifest),
        "n_classes": len(pv_classes),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "source_counts": manifest["source"].value_counts().to_dict(),
        "field_test_classes": sorted(field["label"].unique().tolist()),
        "dropped_unreadable": len(unreadable),
        "dropped_exact_duplicates": exact_dupes,
        "dropped_near_duplicates_of_heldout": int(near.sum()),
        "n_lesion_masks": int((manifest["mask_path"] != "").sum()),
    }
    return manifest, summary


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV with stable dtypes (hashes stay strings)."""
    return pd.read_csv(resolve_path(path), dtype=str, keep_default_na=False)
