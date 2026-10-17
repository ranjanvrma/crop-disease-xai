"""PyTorch dataset and loaders built on the split manifest."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cropxai.data.guard import assert_no_field_leak, check_split_access
from cropxai.data.manifest import load_manifest
from cropxai.utils.config import resolve_path
from cropxai.utils.seed import make_generator, seed_worker


class LeafDataset(Dataset):
    """Images of one split, returned as ``(tensor, class_index)``."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        data_root: Path,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
        allow_heldout: bool = False,
    ):
        check_split_access(split, allow_heldout)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Split '{split}' is empty in the manifest.")
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[i]
        with Image.open(self.data_root / row["path"]) as img:
            image = np.asarray(img.convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, self.class_to_idx[row["label"]]


def load_classes(cfg: DictConfig) -> list[str]:
    """The ordered class list written alongside the manifest."""
    return json.loads(resolve_path(cfg.data.classes_file).read_text(encoding="utf-8"))


def make_loader(cfg: DictConfig, split: str, allow_heldout: bool = False) -> DataLoader:
    """DataLoader for one split; train split is shuffled and augmented.

    The guard runs on every load, so a hand-edited manifest can't leak the
    field set into training.
    """
    from cropxai.data.transforms import eval_transform, train_transform

    manifest = load_manifest(cfg.data.manifest)
    assert_no_field_leak(manifest, cfg.data.dedup.cross_hamming)
    classes = load_classes(cfg)

    is_train = split == "train"
    dataset = LeafDataset(
        manifest, split,
        data_root=resolve_path(cfg.paths.data_root),
        class_to_idx={c: i for i, c in enumerate(classes)},
        transform=train_transform(cfg) if is_train else eval_transform(cfg),
        allow_heldout=allow_heldout,
    )
    workers = cfg.num_workers
    return DataLoader(
        dataset,
        # No gradients at eval time, so larger batches fit and run faster.
        batch_size=cfg.batch_size if is_train else cfg.batch_size * 2,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(cfg.seed),
        # Keep workers alive between epochs instead of re-spawning them.
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def load_examples(
    cfg: DictConfig, split: str, n: int, allow_heldout: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[str], np.ndarray | None, np.ndarray]:
    """A fixed sample of ``n`` images (eval transform) for explanation metrics.

    Images with lesion masks come first, so every mask is used; the rest is a
    seeded random sample (same images for every model and compression level).

    Returns images (N,3,H,W), labels (N,), paths, masks (N,H,W) or None,
    and a boolean array saying which rows have a real mask.
    """
    from cropxai.data.transforms import eval_transform

    check_split_access(split, allow_heldout)
    manifest = load_manifest(cfg.data.manifest)
    assert_no_field_leak(manifest, cfg.data.dedup.cross_hamming)
    rows = manifest[manifest["split"] == split]
    with_mask = rows[rows["mask_path"] != ""]
    rest = rows[rows["mask_path"] == ""]
    n_rest = max(0, min(n - len(with_mask), len(rest)))
    rows = pd.concat([with_mask.head(n), rest.sample(n_rest, random_state=cfg.explain.sample_seed)]).head(n)

    class_to_idx = {c: i for i, c in enumerate(load_classes(cfg))}
    data_root = resolve_path(cfg.paths.data_root)
    transform = eval_transform(cfg)
    images, masks = [], []
    for row in rows.itertuples():
        with Image.open(data_root / row.path) as img:
            image = np.asarray(img.convert("RGB"))
        if row.mask_path:
            with Image.open(data_root / row.mask_path) as m:
                mask = (np.asarray(m.convert("L").resize(img.size)) > 127).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        out = transform(image=image, mask=mask)
        images.append(out["image"])
        masks.append(np.asarray(out["mask"]))

    has_mask = (rows["mask_path"] != "").to_numpy()
    labels = torch.tensor([class_to_idx[label] for label in rows["label"]])
    return torch.stack(images), labels, rows["path"].tolist(), (np.stack(masks) if has_mask.any() else None), has_mask
