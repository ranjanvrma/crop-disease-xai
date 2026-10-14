"""The held-out field-set guard.

PlantDoc and own field photos measure real-world generalisation. If any of
them (or a near-duplicate) reaches training or tuning, the field numbers are
meaningless. These checks run when the manifest is built AND every time a
dataset is loaded.
"""

import pandas as pd

from cropxai.data.hashing import from_hex, min_hamming

HELDOUT_SOURCES = {"plantdoc", "field"}
HELDOUT_SPLIT = "field_test"
TUNING_SPLITS = {"train", "val"}


class FieldLeakError(RuntimeError):
    """Held-out field data reached a training/tuning split."""


def assert_no_field_leak(manifest: pd.DataFrame, cross_hamming: int) -> None:
    """Raise ``FieldLeakError`` if the manifest leaks held-out data into tuning.

    Checks:
      1. every PlantDoc / field image is in ``field_test`` and nothing else is;
      2. no train/val image is a near-duplicate (dHash distance <=
         ``cross_hamming``) of any field_test image;
      3. no file appears twice.
    """
    heldout = manifest["source"].isin(HELDOUT_SOURCES)
    in_field_split = manifest["split"] == HELDOUT_SPLIT
    if (heldout != in_field_split).any():
        bad = manifest[heldout != in_field_split].head(5)
        raise FieldLeakError(f"Held-out source/split mismatch, e.g.:\n{bad}")

    dupes = manifest["path"].duplicated()
    if dupes.any():
        raise FieldLeakError(f"Duplicate paths in manifest, e.g. {manifest.loc[dupes, 'path'].head(3).tolist()}")

    tuning = manifest[manifest["split"].isin(TUNING_SPLITS)]
    field = manifest[in_field_split]
    dist = min_hamming(from_hex(tuning["dhash"]), from_hex(field["dhash"]))
    leaks = dist <= cross_hamming
    if leaks.any():
        raise FieldLeakError(
            f"{int(leaks.sum())} train/val images are near-duplicates of field_test images, "
            f"e.g. {tuning['path'][leaks].head(3).tolist()}"
        )


def check_split_access(split: str, allow_heldout: bool) -> None:
    """Refuse to load the field set unless the caller explicitly opts in.

    Only final evaluation scripts pass ``allow_heldout=True``. Training,
    model selection and threshold tuning never do.
    """
    if split == HELDOUT_SPLIT and not allow_heldout:
        raise FieldLeakError(
            "Refusing to load 'field_test' without allow_heldout=True. "
            "Only final evaluation may read the field set."
        )
