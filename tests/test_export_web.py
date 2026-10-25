"""Web-app export: the ONNX graph must reproduce the PyTorch model and its feature maps."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cropxai.export.web import export_web_bundle, tune_unsure_threshold
from cropxai.models import build_model

ROOT = Path(__file__).resolve().parents[1]
ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("arch", ["scratch_cnn", "mobilenetv3_large_100"])
def test_bundle_matches_pytorch(tmp_path, arch):
    torch.manual_seed(0)
    classes = json.loads((ROOT / "splits/classes_v1.json").read_text(encoding="utf-8"))
    model = build_model(OmegaConf.create({"arch": arch, "pretrained": False, "drop_rate": 0.0}), len(classes)).eval()
    meta = export_web_bundle(model, classes, tmp_path, 96, ROOT / "knowledge/remedies.yaml", 0.5, {"demo": True})

    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        features = model.forward_features(x)
        logits = model.forward_head(features)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    got_logits, got_features = session.run(["logits", "features"], {"image": x.numpy()})
    np.testing.assert_allclose(got_logits, logits.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(got_features, features.numpy(), atol=1e-4, rtol=1e-4)

    assert meta["feature_shape"] == list(features.shape[1:])
    assert meta["resize_short"] == 109
    flat = np.fromfile(tmp_path / "classifier.bin", dtype="<f4")
    k = features.shape[1]
    classifier = model.get_classifier()
    np.testing.assert_array_equal(flat[: len(classes) * k].reshape(len(classes), k), classifier.weight.detach().numpy())
    np.testing.assert_array_equal(flat[len(classes) * k:], classifier.bias.detach().numpy())
    remedies = json.loads((tmp_path / "remedies.json").read_text(encoding="utf-8"))
    assert set(remedies["classes"]) == set(classes)


def test_unsure_threshold_reaches_target_on_accepted_predictions():
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
    labels = torch.tensor([0, 0, 1, 0, 1])  # the 0.7 and 0.55 predictions are wrong
    assert tune_unsure_threshold(probs, labels, 1.0) == pytest.approx(0.8)
    assert tune_unsure_threshold(probs, labels, 0.75) == pytest.approx(0.6)
    assert tune_unsure_threshold(probs, torch.tensor([1, 1, 0, 1, 0]), 0.99) == 1.0  # never good enough
