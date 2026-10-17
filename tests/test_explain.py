"""Closed-form Grad-CAM++ vs pytorch-grad-cam, and metric sanity checks."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cropxai.eval.faithfulness import average_drop_and_increase, deletion_insertion
from cropxai.eval.localisation import iou, localisation_metrics, pointing_game_hit
from cropxai.explain.cams import compute_cams
from cropxai.explain.closed_form import explain_closed_form
from cropxai.models import CNN_ARCHS, build_model


@pytest.mark.parametrize("arch", sorted(CNN_ARCHS))
def test_closed_form_matches_library(arch):
    """The gradient-free form must reproduce the library's Grad-CAM++ exactly."""
    torch.manual_seed(0)
    model = build_model(OmegaConf.create({"arch": arch, "pretrained": False, "drop_rate": 0.0}), 5).eval()
    images = torch.randn(4, 3, 96, 96)
    classes = torch.tensor([0, 1, 2, 3])

    library = compute_cams(model, arch, "gradcampp", images, classes.tolist())
    ours = explain_closed_form(model, images, classes)

    assert library.shape == ours.shape == (4, 96, 96)
    np.testing.assert_allclose(ours, library, atol=1e-4)


class TopLeftModel(nn.Module):
    """Class-0 logit = mean brightness of the top-left quadrant; class 1 fixed at 0."""

    def forward(self, x):
        h, w = x.shape[-2:]
        score = x[:, :, : h // 2, : w // 2].mean(dim=(1, 2, 3))
        return torch.stack([score, torch.zeros_like(score)], dim=1)


def _quadrant_cam(top_left: bool, size: int = 32) -> np.ndarray:
    cam = np.zeros((size, size), dtype=np.float32)
    if top_left:
        cam[: size // 2, : size // 2] = 1.0
    else:
        cam[size // 2:, size // 2:] = 1.0
    return cam + np.random.default_rng(0).random((size, size), dtype=np.float32) * 1e-3  # break ties


def test_deletion_insertion_rewards_the_right_region():
    model = TopLeftModel()
    image = -torch.ones(3, 32, 32)
    image[:, :16, :16] = 1.0  # the evidence; a strong blur washes it out
    good_del, good_ins = deletion_insertion(model, image, _quadrant_cam(True), 0, 16, 31, 10.0)
    bad_del, bad_ins = deletion_insertion(model, image, _quadrant_cam(False), 0, 16, 31, 10.0)
    assert good_del < bad_del   # removing the evidence first drops confidence faster
    assert good_ins > bad_ins   # restoring the evidence first raises it faster


def test_average_drop_zero_for_all_ones_mask():
    model = TopLeftModel()
    images = torch.rand(2, 3, 32, 32)
    drop, increase = average_drop_and_increase(model, images, np.ones((2, 32, 32), np.float32), torch.tensor([0, 0]))
    assert drop == pytest.approx(0.0, abs=1e-4)
    assert increase == 0.0


def test_localisation():
    mask = np.zeros((10, 10), np.uint8)
    mask[:5, :5] = 1
    cam = np.zeros((10, 10), np.float32)
    cam[2, 2] = 1.0
    assert pointing_game_hit(cam, mask)
    assert iou(cam, mask, threshold=0.5) == pytest.approx(1 / 25)
    miss = np.zeros((10, 10), np.float32)
    miss[8, 8] = 1.0
    m = localisation_metrics(np.stack([cam, miss]), np.stack([mask, mask]), 0.5)
    assert m["pointing_game"] == 0.5 and m["n_masks"] == 2
