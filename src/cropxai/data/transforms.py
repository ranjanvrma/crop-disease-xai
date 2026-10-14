"""Albumentations pipelines. Parameters come from ``cfg.augment``."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(cfg: DictConfig) -> A.Compose:
    s, a = cfg.image_size, cfg.augment
    return A.Compose([
        A.RandomResizedCrop(size=(s, s), scale=tuple(a.crop_scale)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=a.vflip_p),
        A.ColorJitter(brightness=a.jitter, contrast=a.jitter, saturation=a.jitter, hue=0.02, p=a.jitter_p),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(cfg: DictConfig) -> A.Compose:
    """Deterministic resize + centre crop (resize short side to 1.14 x crop)."""
    s = cfg.image_size
    return A.Compose([
        A.SmallestMaxSize(max_size=int(s * 1.14)),
        A.CenterCrop(height=s, width=s),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
