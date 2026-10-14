"""Seeding and config loading."""

import torch

from cropxai.utils.config import load_config
from cropxai.utils.seed import seed_everything


def test_seed_everything_is_repeatable():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_config_inherits_and_overrides(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("inherits: [base.yaml]\nrun_name: demo\nimage_size: 256\n")

    cfg = load_config(exp, overrides=["seed=2"])

    assert cfg.project == "crop-disease-xai"   # from base.yaml
    assert cfg.run_name == "demo"              # experiment overrides base
    assert cfg.image_size == 256
    assert cfg.seed == 2                       # CLI overrides everything
    assert list(cfg.seeds) == [0, 1, 2]
