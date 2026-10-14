"""Model builders, and the GAP + single-linear head the closed-form CAM relies on."""

import pytest
import torch
from omegaconf import OmegaConf

from cropxai.models import CNN_ARCHS, build_model, set_trainable

NUM_CLASSES = 7


def _model(arch: str):
    cfg = OmegaConf.create({"arch": arch, "pretrained": False, "drop_rate": 0.2})
    return build_model(cfg, NUM_CLASSES).eval()


@pytest.mark.parametrize("arch", sorted(CNN_ARCHS))
def test_cnn_head_is_gap_plus_single_linear(arch):
    """logits == W · mean_hw(feature_maps) + b, exactly what Phase 6 needs.

    If this holds, d(logit_c)/d(A^k_ij) = W[c, k] / (H*W), a constant, so
    Grad-CAM++ has a gradient-free closed form.
    """
    torch.manual_seed(0)
    model = _model(arch)
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        feats = model.forward_features(x)
        assert feats.ndim == 4, "final features must be spatial conv maps"
        fc = model.get_classifier()
        assert isinstance(fc, torch.nn.Linear) and fc.in_features == feats.shape[1]
        expected = fc(feats.mean(dim=(2, 3)))
        assert torch.allclose(model(x), expected, atol=1e-5)
        assert model(x).shape == (2, NUM_CLASSES)


def test_deit_builds():
    model = _model("deit_small_patch16_224")
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, NUM_CLASSES)


def test_feature_extract_trains_only_the_head():
    model = _model("efficientnet_b0")
    set_trainable(model, "feature_extract")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"classifier.weight", "classifier.bias"}

    set_trainable(model, "finetune")
    assert all(p.requires_grad for p in model.parameters())


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        _model("not_a_model")
