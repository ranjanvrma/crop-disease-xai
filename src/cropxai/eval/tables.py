"""Aggregate per-seed results JSON into mean ± std tables.

Every cell is computed from ``results/metrics/**.json``. A row with fewer
seeds than expected is marked ``(n=k)`` rather than presented as final.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Table 1 columns: (header, path into the metrics dict, scale, decimals)
TABLE1 = [
    ("Params (M)", "params", 1e-6, 2),
    ("Weights (MB)", lambda m: m["params"] * (1 if m.get("int8") else 4) / 1e6, 1, 2),
    ("Top-1 PV test (%)", "test.top1", 100, 2),
    ("Macro-F1 PV test (%)", "test.macro_f1", 100, 2),
    ("ECE PV test", "test.ece", 1, 3),
    ("Field top-1 strict (%)", "field.strict_top1", 100, 2),
    ("Field top-1 restricted (%)", "field.restricted_top1", 100, 2),
    ("Desktop GPU latency (ms)", "latency_desktop_gpu.median_ms", 1, 2),
    ("Desktop CPU latency (ms)", "latency_desktop_cpu.median_ms", 1, 1),
]

# Table 2 columns; "{m}" is replaced by the CAM method. Faithfulness is on the
# in-domain test sample; localisation needs lesion masks, which exist only on
# the field set.
TABLE2 = [
    ("Del. AUC ↓ (test)", "test.{m}.deletion_auc", 1, 3),
    ("Ins. AUC ↑ (test)", "test.{m}.insertion_auc", 1, 3),
    ("Avg. Drop % ↓ (test)", "test.{m}.average_drop", 1, 1),
    ("Inc. Conf. % ↑ (test)", "test.{m}.increase_in_confidence", 1, 1),
    ("ROAD ↑ (test)", "test.{m}.road", 1, 3),
    ("Del. AUC ↓ (field)", "field_test.{m}.deletion_auc", 1, 3),
    ("Ins. AUC ↑ (field)", "field_test.{m}.insertion_auc", 1, 3),
    ("Pointing ↑ (field)", "field_test.{m}.pointing_game", 100, 1),
    ("IoU ↑ (field)", "field_test.{m}.iou", 1, 3),
]
METHOD_NAMES = {"gradcampp": "Grad-CAM++", "closed_form": "Grad-CAM++ (closed form)",
                "gradcam": "Grad-CAM", "scorecam": "Score-CAM"}

# Table 3: compressed variant vs its uncompressed parent (closed-form Grad-CAM++ maps).
TABLE3 = [
    ("Spearman ↑ (test)", "test.stability.spearman", 1, 3),
    ("SSIM ↑ (test)", "test.stability.ssim", 1, 3),
    ("Top-10% IoU ↑ (test)", "test.stability.top10_iou", 1, 3),
    ("Spearman ↑ (field)", "field_test.stability.spearman", 1, 3),
    ("SSIM ↑ (field)", "field_test.stability.ssim", 1, 3),
    ("Top-10% IoU ↑ (field)", "field_test.stability.top10_iou", 1, 3),
]


def _get(d: dict, dotted):
    """Value at a dotted path, or computed by ``dotted(d)`` when it is a function."""
    if callable(dotted):
        try:
            return dotted(d)
        except (KeyError, TypeError):
            return None
    for key in dotted.split("."):
        if d is None:
            return None
        d = d.get(key)
    return d


def load_runs(metrics_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(metrics_dir.glob("*.json"))]


def format_mean_std(values: list[float], decimals: int, expected_n: int) -> str:
    """'mean ± std' with sample std (ddof=1); flags incomplete seed sets."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    mean = np.mean(vals)
    cell = f"{mean:.{decimals}f}"
    if len(vals) > 1:
        cell += f" ± {np.std(vals, ddof=1):.{decimals}f}"
    if len(vals) < expected_n:
        cell += f" (n={len(vals)})"
    return cell


def _row(label: dict, group: list[dict], columns: list, expected_n: int, method: str = "") -> dict:
    row = dict(label)
    for header, path, scale, dec in columns:
        path = path.format(m=method) if isinstance(path, str) else path
        vals = [(_get(m, path) * scale) if _get(m, path) is not None else None for m in group]
        row[header] = format_mean_std(vals, dec, expected_n)
    return row


def accuracy_table(runs: list[dict], expected_n: int = 3) -> pd.DataFrame:
    """Table 1: one row per experiment (run_name), seeds aggregated."""
    rows = []
    for run_name in sorted({r["run_name"] for r in runs}):
        group = [r["metrics"] for r in runs if r["run_name"] == run_name]
        rows.append(_row({"Model": run_name}, group, TABLE1, expected_n))
    return pd.DataFrame(rows)


def explanation_table(runs: list[dict], expected_n: int = 3) -> pd.DataFrame:
    """Table 2: one row per (experiment, CAM method), seeds aggregated."""
    rows = []
    for run_name in sorted({r["run_name"] for r in runs}):
        group = [r["metrics"] for r in runs if r["run_name"] == run_name]
        for method, name in METHOD_NAMES.items():
            if any(_get(m, f"test.{method}") for m in group):
                rows.append(_row({"Model": run_name, "Method": name}, group, TABLE2, expected_n, method))
    return pd.DataFrame(rows)


def stability_table(runs: list[dict], expected_n: int = 3) -> pd.DataFrame:
    """Table 3: one row per compressed variant, seeds aggregated."""
    rows = []
    for run_name in sorted({r["run_name"] for r in runs}):
        group = [r["metrics"] for r in runs if r["run_name"] == run_name]
        if any(_get(m, "test.stability") for m in group):
            rows.append(_row({"Variant": run_name}, group, TABLE3, expected_n))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Plain pipe table (no tabulate dependency)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def replace_between_markers(text: str, name: str, content: str) -> str:
    """Swap the block between ``<!-- name:start -->`` and ``<!-- name:end -->``."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{content}\n{end}{after}"
