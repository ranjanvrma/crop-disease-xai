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
    ("Size (MB)", "size_mb_fp32", 1, 1),
    ("Top-1 PV test (%)", "test.top1", 100, 2),
    ("Macro-F1 PV test (%)", "test.macro_f1", 100, 2),
    ("ECE PV test", "test.ece", 1, 3),
    ("Field top-1 strict (%)", "field.strict_top1", 100, 2),
    ("Field top-1 restricted (%)", "field.restricted_top1", 100, 2),
    ("Desktop GPU latency (ms)", "latency_desktop_gpu.median_ms", 1, 2),
    ("Desktop CPU latency (ms)", "latency_desktop_cpu.median_ms", 1, 1),
]


def _get(d: dict, dotted: str):
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


def accuracy_table(runs: list[dict], expected_n: int = 3) -> pd.DataFrame:
    """Table 1: one row per experiment (run_name), seeds aggregated."""
    rows = []
    for run_name in sorted({r["run_name"] for r in runs}):
        group = [r["metrics"] for r in runs if r["run_name"] == run_name]
        row = {"Model": run_name}
        for header, path, scale, dec in TABLE1:
            vals = [(_get(m, path) * scale) if _get(m, path) is not None else None for m in group]
            row[header] = format_mean_std(vals, dec, expected_n)
        rows.append(row)
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
