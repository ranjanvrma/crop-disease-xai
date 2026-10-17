"""End-to-end smoke test: manifest -> train -> evaluate -> explain -> verify -> JSON.

Checks the plumbing only, on random synthetic images. Its numbers mean nothing.
"""

import json
from pathlib import Path

from cropxai.data.manifest import build_manifest
from cropxai.utils.config import load_config

from conftest import PV_CLASSES


def test_train_and_evaluate_scratch_cnn(dataset_root: Path, tmp_path: Path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate import evaluate
    from explain import explain
    from verify_closed_form import verify

    from cropxai.train.loop import fit

    work = tmp_path / "work"
    cfg = load_config("configs/experiment/scratch_cnn.yaml", [
        f"paths.data_root={dataset_root.as_posix()}",
        f"paths.checkpoints_dir={(work / 'ckpt').as_posix()}",
        f"paths.results_dir={(work / 'results').as_posix()}",
        f"data.class_map={(dataset_root / 'map.yaml').as_posix()}",
        "data.plantdoc_dirs=[plantdoc/test]",
        f"data.manifest={(work / 'manifest.csv').as_posix()}",
        f"data.classes_file={(work / 'classes.json').as_posix()}",
        "train.epochs=1", "train.warmup_epochs=1", "image_size=32", "batch_size=8",
        "num_workers=0", "wandb.mode=disabled",
        "explain.n_images=6", "explain.scorecam_n_images=3", "explain.road_n_images=3",
        "explain.n_figure_examples=2", "explain.verify_n_images=4", "explain.batch_size=4",
    ])

    manifest, _ = build_manifest(cfg)
    work.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.data.manifest, index=False)
    Path(cfg.data.classes_file).write_text(json.dumps(PV_CLASSES))

    best = fit(cfg)
    assert Path(best["checkpoint"]).exists()
    assert Path(best["checkpoint"]).with_name("complete.json").exists()

    metrics = evaluate(cfg)
    assert 0.0 <= metrics["test"]["top1"] <= 1.0
    assert metrics["field"]["strict_n"] == 4      # 3 PlantDoc + 1 own field photo
    out = work / "results/metrics/accuracy/scratch_cnn_seed0.json"
    record = json.loads(out.read_text())
    assert record["config"]["run_name"] == "scratch_cnn"

    explanation = explain(cfg)
    for split in ["test", "field_test"]:
        for method in ["gradcampp", "gradcam", "scorecam"]:
            m = explanation[split][method]
            assert 0.0 <= m["deletion_auc"] <= 1.0 and 0.0 <= m["insertion_auc"] <= 1.0
    assert explanation["field_test"]["n_masks"] == 1               # the synthetic field photo has a mask
    assert "pointing_game" in explanation["field_test"]["gradcampp"]
    assert (work / "results/figures/heatmaps/scratch_cnn_seed0_field_test.png").exists()

    check = verify(cfg)
    assert check["max_abs_diff"] < 1e-4                            # closed form == library
