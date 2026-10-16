"""Rebuild results tables from results/metrics and refresh the README.

    python scripts/make_tables.py

Writes ``results/tables/table1_accuracy.{csv,md}`` and replaces the Table 1
block in README.md. Safe to re-run any time.
"""

from cropxai import REPO_ROOT
from cropxai.eval.tables import accuracy_table, load_runs, replace_between_markers, to_markdown


def main() -> None:
    runs = load_runs(REPO_ROOT / "results/metrics/accuracy")
    out = REPO_ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)

    if not runs:
        print("No accuracy results yet; README left unchanged.")
        return

    table = accuracy_table(runs)
    table.to_csv(out / "table1_accuracy.csv", index=False)
    md = to_markdown(table)
    (out / "table1_accuracy.md").write_text(md + "\n", encoding="utf-8")

    readme = REPO_ROOT / "README.md"
    readme.write_text(replace_between_markers(readme.read_text(encoding="utf-8"), "table1", md), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
