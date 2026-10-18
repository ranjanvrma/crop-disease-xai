"""Pruning, INT8 quantization, stability metrics and the compressed-variant pipeline."""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cropxai.compress.prune import prune_structured
from cropxai.compress.quantize import QuantizedModel, quantize_int8
from cropxai.data.manifest import build_manifest
from cropxai.eval.stability import stability_metrics
from cropxai.explain.cams import compute_cams
from cropxai.explain.closed_form import explain_closed_form
from cropxai.models import build_model
from cropxai.utils.config import load_config

from conftest import PV_CLASSES


def _mnv3():
    torch.manual_seed(0)
    cfg = OmegaConf.create({"arch": "mobilenetv3_large_100", "pretrained": False, "drop_rate": 0.0})
    return build_model(cfg, 5).eval()


def test_pruning_keeps_closed_form_exact():
    model = prune_structured(_mnv3(), 0.5, image_size=64).eval()
    x = torch.randn(3, 64, 64).unsqueeze(0).repeat(2, 1, 1, 1) + torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = model.forward_features(x)
        assert torch.allclose(model(x), model.get_classifier()(feats.mean(dim=(2, 3))), atol=1e-5)
    classes = torch.tensor([0, 3])
    library = compute_cams(model, "mobilenetv3_large_100", "gradcampp", x, classes.tolist())
    np.testing.assert_allclose(explain_closed_form(model, x, classes), library, atol=1e-4)


def test_pruned_checkpoint_reloads_into_a_freshly_pruned_model():
    pruned = prune_structured(_mnv3(), 0.5, image_size=64)
    fresh = prune_structured(build_model(OmegaConf.create(
        {"arch": "mobilenetv3_large_100", "pretrained": False, "drop_rate": 0.0}), 5), 0.5, image_size=64)
    fresh.load_state_dict(pruned.state_dict())   # same shapes regardless of the weights


def test_int8_model_runs_and_explains_without_gradients():
    model = _mnv3()
    calib = torch.randn(16, 3, 64, 64)
    program, quantizer = quantize_int8(model, calib, image_size=64, batch_size=8)
    q = QuantizedModel(program, model.get_classifier(), torch.device("cpu"))
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        logits, float_logits = q(x), model(x)
    assert logits.shape == (4, 5) and quantizer
    assert np.corrcoef(logits.flatten(), float_logits.flatten())[0, 1] > 0.9   # INT8 stays close to float
    cams = compute_cams(q, "mobilenetv3_large_100", "closed_form", x, [0, 1, 2, 3])
    assert cams.shape == (4, 64, 64) and np.isfinite(cams).all()


def test_stability_identical_maps():
    maps = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
    s = stability_metrics(maps, maps)
    assert s["spearman"] > 0.999 and s["ssim"] > 0.999 and s["top10_iou"] == 1.0
    other = stability_metrics(maps, np.random.default_rng(1).random((3, 32, 32)).astype(np.float32))
    assert other["spearman"] < 0.2 and other["top10_iou"] < 0.3


def test_compressed_variants_end_to_end(dataset_root: Path, tmp_path: Path):
    """Parent -> pruned variant (fine-tuned) and INT8 variant -> evaluate + explain with stability."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate import evaluate
    from explain import explain

    from cropxai.compress.variants import quantize_variant
    from cropxai.train.loop import fit, is_trained

    work = tmp_path / "work"
    shared = [
        f"paths.data_root={dataset_root.as_posix()}",
        f"paths.checkpoints_dir={(work / 'ckpt').as_posix()}",
        f"paths.results_dir={(work / 'results').as_posix()}",
        f"data.class_map={(dataset_root / 'map.yaml').as_posix()}",
        "data.plantdoc_dirs=[plantdoc/test]",
        f"data.manifest={(work / 'manifest.csv').as_posix()}",
        f"data.classes_file={(work / 'classes.json').as_posix()}",
        "image_size=32", "batch_size=8", "num_workers=0", "wandb.mode=disabled", "train.epochs=1",
        "explain.n_images=6", "explain.road_n_images=3", "explain.n_figure_examples=2", "explain.batch_size=4",
    ]
    parent = load_config("configs/experiment/scratch_cnn.yaml", [*shared, "seed=0"])
    manifest, _ = build_manifest(parent)
    work.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(parent.data.manifest, index=False)
    Path(parent.data.classes_file).write_text(json.dumps(PV_CLASSES))
    fit(parent)

    as_scratch = ["model.arch=scratch_cnn", "model.pretrained=false", "compress.parent=scratch_cnn", "seed=0"]
    pruned = load_config("configs/experiment/mnv3_fe_p50.yaml", [*shared, *as_scratch, "run_name=scratch_p50"])
    fit(pruned)
    assert is_trained(pruned)
    acc = evaluate(pruned)
    assert acc["params"] < evaluate(parent)["params"]                 # pruning shrank the model
    expl = explain(pruned)
    assert set(expl["test"]["stability"]) >= {"spearman", "ssim", "top10_iou"}

    int8 = load_config("configs/experiment/mnv3_fe_int8.yaml",
                       [*shared, *as_scratch, "run_name=scratch_int8", "compress.calib_images=8"])
    quantize_variant(int8)
    assert is_trained(int8)
    acc8 = evaluate(int8)
    assert acc8["int8"] and acc8["latency_desktop_cpu"] is None
    expl8 = explain(int8)
    assert "closed_form" in expl8["test"] and "stability" in expl8["field_test"]
