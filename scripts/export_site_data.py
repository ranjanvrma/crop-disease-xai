"""Collect the results the demo website shows into one JSON file (CPU only, seconds).

    python scripts/export_site_data.py

Reads only files the pipeline already wrote: ``results/tables/*.csv`` (from
``make_tables.py``), ``results/metrics/closed_form/*.json`` and the model
bundles in ``web/public/models/*/model.json`` (from ``export_web.py``). Writes
``web/public/data/site.json`` and ``web/public/models/index.json``. Nothing is
typed in by hand, so the website can never show a number the experiments did
not produce.
"""

import csv
import json
import re
import statistics
from pathlib import Path

from cropxai import REPO_ROOT

RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "web" / "public" / "models"
OUT = REPO_ROOT / "web" / "public" / "data" / "site.json"

DEFAULT_MODEL = "mnv3_ft_field"
# Order and one-line descriptions for the model picker; only bundles that exist are listed.
MODEL_NOTES = {
    "mnv3_ft_field": "MobileNetV3, fully fine-tuned with field-style augmentation (default)",
    "mnv3_fe": "MobileNetV3, frozen backbone (baseline, parent of the compressed models)",
    "effnet_b0_fe": "EfficientNet-B0, frozen backbone (best field accuracy)",
    "mnv3_fe_p50_ft8": "MobileNetV3 pruned 50%, fine-tuned 8 epochs (4x smaller)",
}

MEAN_STD = re.compile(r"^\s*(-?[\d.]+)\s*(?:±\s*([\d.]+)|\(n=(\d+)\))?")


def number(cell: str) -> dict | None:
    """'96.11 ± 0.27' -> {mean, std}; '99.76 (n=1)' -> {mean, std: None}; '—' -> None."""
    m = MEAN_STD.match(cell)
    if not m:
        return None
    return {"mean": float(m.group(1)), "std": float(m.group(2)) if m.group(2) else None}


def read_table(name: str) -> dict:
    with (RESULTS / "tables" / f"{name}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return {"columns": rows[0], "rows": rows[1:], "source": f"results/tables/{name}.csv"}


def closed_form() -> list[dict]:
    """Closed-form vs library Grad-CAM++, aggregated over seeds per run."""
    by_run: dict[str, list[dict]] = {}
    for path in sorted((RESULTS / "metrics" / "closed_form").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        by_run.setdefault(d["run_name"], []).append(d["metrics"])
    out = []
    for run, ms in by_run.items():
        out.append({
            "run": run,
            "seeds": len(ms),
            "n_images": ms[0]["n_images"],
            "max_abs_diff": max(m["max_abs_diff"] for m in ms),
            "pearson_r_min": min(m["pearson_r_min"] for m in ms),
            "ms_library": statistics.mean(m["ms_per_image_library"] for m in ms),
            "ms_closed_form": statistics.mean(m["ms_per_image_closed_form"] for m in ms),
            "device": ms[0]["device"],
        })
    return out


def field_details() -> dict:
    """Size of the field test set and how many PlantVillage classes it covers."""
    m = json.loads((RESULTS / "metrics" / "accuracy" / f"{DEFAULT_MODEL}_seed0.json").read_text(encoding="utf-8"))["metrics"]
    return {"field_n": m["field"]["strict_n"], "field_classes": m["field"]["n_field_classes"], "test_n": m["test"]["n"]}


def models_index() -> dict:
    models = []
    for run in [*MODEL_NOTES, *sorted(p.name for p in MODELS.iterdir() if p.is_dir())]:
        meta_path = MODELS / run / "model.json"
        if not meta_path.exists() or any(m["id"] == run for m in models):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        models.append({
            "id": run,
            "label": run,
            "note": MODEL_NOTES.get(run, meta.get("arch", "")),
            "arch": meta["arch"],
            "seed": meta["seed"],
            "size_mb": round((MODELS / run / "model.onnx").stat().st_size / 1e6, 1),
            "measured_accuracy": meta.get("measured_accuracy"),
            "demo": meta.get("demo", False),
        })
    return {"default": DEFAULT_MODEL, "models": models}


def main() -> None:
    tables = {name: read_table(name) for name in ("table1_accuracy", "table2_explanation", "table3_stability")}
    for t in tables.values():
        t["numbers"] = [[number(c) for c in row] for row in t["rows"]]
    site = {
        "tables": tables,
        "closed_form": closed_form(),
        "field": field_details(),
        "figures": sorted(p.name for p in (REPO_ROOT / "web" / "public" / "figures").glob("*.jpg")),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(site, indent=1), encoding="utf-8")
    index = models_index()
    (MODELS / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} and models/index.json ({len(index['models'])} models)")


if __name__ == "__main__":
    main()
