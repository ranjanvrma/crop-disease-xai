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
    overrides = [
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
    ]
    cfg = load_config("configs/experiment/scratch_cnn.yaml", overrides)

    manifest, _ = build_manifest(cfg)
    work.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.data.manifest, index=False)
    Path(cfg.data.classes_file).write_text(json.dumps(PV_CLASSES))

    best = fit(cfg)
    assert Path(best["checkpoint"]).exists()
    assert Path(best["checkpoint"]).with_name("complete.json").exists()
    from cropxai.train.loop import is_trained
    assert is_trained(cfg)
    assert not is_trained(load_config("configs/experiment/scratch_cnn.yaml", [*overrides, "train.epochs=2"]))

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


def test_fingerprint_tracks_training_settings_only():
    from cropxai.utils.config import training_fingerprint

    base = load_config("configs/experiment/mnv3_fe.yaml")
    same = load_config("configs/experiment/mnv3_fe.yaml", ["paths.results_dir=/elsewhere", "explain.n_images=5"])
    changed = load_config("configs/experiment/mnv3_fe.yaml", ["train.epochs=99"])
    assert training_fingerprint(base) == training_fingerprint(same)      # paths/explain don't matter
    assert training_fingerprint(base) != training_fingerprint(changed)   # training settings do


def test_training_resumes_after_interruption(dataset_root: Path, tmp_path: Path, monkeypatch, capsys):
    import cropxai.train.loop as loop

    work = tmp_path / "work"
    cfg = load_config("configs/experiment/scratch_cnn.yaml", [
        f"paths.data_root={dataset_root.as_posix()}",
        f"paths.checkpoints_dir={(work / 'ckpt').as_posix()}",
        f"data.class_map={(dataset_root / 'map.yaml').as_posix()}",
        "data.plantdoc_dirs=[plantdoc/test]",
        f"data.manifest={(work / 'manifest.csv').as_posix()}",
        f"data.classes_file={(work / 'classes.json').as_posix()}",
        "train.epochs=3", "train.warmup_epochs=1", "image_size=32", "batch_size=8",
        "num_workers=0", "wandb.mode=disabled",
    ])
    manifest, _ = build_manifest(cfg)
    work.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.data.manifest, index=False)
    Path(cfg.data.classes_file).write_text(json.dumps(PV_CLASSES))

    # Simulate a Colab disconnect during the second epoch.
    real_epoch, calls = loop.train_one_epoch, {"n": 0}

    def flaky_epoch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("disconnected")
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(loop, "train_one_epoch", flaky_epoch)
    try:
        loop.fit(cfg)
    except KeyboardInterrupt:
        pass
    assert loop.resume_path(cfg).exists() and not loop.is_trained(cfg)

    monkeypatch.setattr(loop, "train_one_epoch", real_epoch)
    capsys.readouterr()
    loop.fit(cfg)
    out = capsys.readouterr().out
    assert "resuming after epoch 0" in out
    assert "epoch 0:" not in out and "epoch 2:" in out       # continued, didn't restart
    assert loop.is_trained(cfg) and not loop.resume_path(cfg).exists()
