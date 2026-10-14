"""Classification metrics and table aggregation."""

import torch

from cropxai.eval.classification import classification_metrics, expected_calibration_error, field_metrics
from cropxai.eval.tables import format_mean_std, replace_between_markers


def test_ece_zero_when_confidence_matches_accuracy():
    # 4 samples at confidence 0.75, 3 of them correct -> perfectly calibrated.
    probs = torch.tensor([[0.75, 0.25]] * 4)
    labels = torch.tensor([0, 0, 0, 1])
    assert expected_calibration_error(probs, labels) < 1e-6


def test_ece_overconfident():
    probs = torch.tensor([[0.99, 0.01]] * 2)
    labels = torch.tensor([0, 1])  # 50 % accurate at 99 % confidence
    assert abs(expected_calibration_error(probs, labels) - 0.49) < 1e-5


def test_classification_metrics():
    logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    labels = torch.tensor([0, 1, 2, 2])
    m = classification_metrics(logits, labels)
    assert m["top1"] == 0.75 and m["n"] == 4


def test_field_restricted_ignores_absent_classes():
    # Class 2 never appears in the field labels; the model's top logit is class 2.
    logits = torch.tensor([[1.0, 0.0, 9.0], [0.0, 1.0, 9.0]])
    labels = torch.tensor([0, 1])
    m = field_metrics(logits, labels)
    assert m["strict_top1"] == 0.0
    assert m["restricted_top1"] == 1.0
    assert m["n_field_classes"] == 2


def test_format_mean_std():
    assert format_mean_std([1.0, 2.0, 3.0], 1, expected_n=3) == "2.0 ± 1.0"
    assert format_mean_std([1.0], 1, expected_n=3) == "1.0 (n=1)"
    assert format_mean_std([None], 1, expected_n=3) == "—"


def test_replace_between_markers():
    text = "a\n<!-- t:start -->\nold\n<!-- t:end -->\nb"
    assert replace_between_markers(text, "t", "new") == "a\n<!-- t:start -->\nnew\n<!-- t:end -->\nb"
