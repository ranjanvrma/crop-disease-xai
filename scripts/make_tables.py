"""Rebuild results tables from results/metrics and refresh the README.

    python scripts/make_tables.py
    python scripts/make_tables.py --results-dir /content/drive/MyDrive/cropxai/results

Writes ``results/tables/<table>.{csv,md}`` and replaces the matching
``<!-- tableN:start/end -->`` blocks in README.md. Safe to re-run any time.
"""

import argparse
import sys

from cropxai import REPO_ROOT
from cropxai.utils.config import resolve_path
from cropxai.eval.tables import (
    accuracy_table,
    explanation_table,
    load_runs,
    replace_between_markers,
    stability_table,
    to_markdown,
)

TABLES = [
    ("table1", "table1_accuracy", "accuracy", accuracy_table),
    ("table2", "table2_explanation", "explanation", explanation_table),
    ("table3", "table3_stability", "explanation", stability_table),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results", help="folder holding metrics/ (default: repo results/)")
    args = parser.parse_args()
    results = resolve_path(args.results_dir)
    sys.stdout.reconfigure(encoding="utf-8")  # table headers contain arrows; Windows consoles default to cp1252

    out = results / "tables"
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    for marker, name, kind, build in TABLES:
        runs = load_runs(results / "metrics" / kind)
        if not runs:
            print(f"{name}: no results yet, left unchanged.")
            continue
        table = build(runs)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        md = to_markdown(table)
        (out / f"{name}.md").write_text(md + "\n", encoding="utf-8", newline="\n")
        text = replace_between_markers(text, marker, md)
        print(f"{name}:\n{md}\n")

    readme.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
