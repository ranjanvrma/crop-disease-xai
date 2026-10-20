"""Albumentations pipelines. Parameters come from ``cfg.augment``."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def field_augmentations(f: DictConfig) -> list:
    """Extra corruptions that imitate phone photos taken in a field (``augment.field``).

    PlantVillage leaves are centred on plain backgrounds under even light, which
    lets a model learn shortcuts that fail on field photos. These make the
    training images look more like the field: rotation and perspective, blur,
    sensor noise, shadows, strong lighting changes, JPEG artefacts and occlusions.
    """
    return [
        A.Affine(rotate=(-f.rotate, f.rotate), shear=(-f.shear, f.shear), p=f.affine_p),
        A.Perspective(scale=(0.02, 0.08), p=f.perspective_p),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9)), A.Defocus(radius=(1, 3))], p=f.blur_p),
        A.GaussNoise(std_range=(0.02, 0.08), p=f.noise_p),
        A.RandomShadow(p=f.shadow_p),
        A.RandomBrightnessContrast(brightness_limit=f.brightness, contrast_limit=f.brightness, p=f.brightness_p),
        A.ImageCompression(quality_range=(40, 95), p=f.jpeg_p),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), p=f.dropout_p),
    ]


def train_transform(cfg: DictConfig) -> A.Compose:
    s, a = cfg.image_size, cfg.augment
    field = field_augmentations(a.field) if a.get("field") else []  # absent in the original experiments
    return A.Compose([
        A.RandomResizedCrop(size=(s, s), scale=tuple(a.crop_scale)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=a.vflip_p),
        A.ColorJitter(brightness=a.jitter, contrast=a.jitter, saturation=a.jitter, hue=0.02, p=a.jitter_p),
        *field,
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
