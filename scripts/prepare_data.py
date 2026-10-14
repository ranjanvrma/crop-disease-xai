"""Build the split manifest + class list from the raw datasets.

    python scripts/prepare_data.py --config configs/experiment/prepare_data.yaml

Writes ``splits/manifest_v1.csv``, ``splits/classes_v1.json`` and
``splits/manifest_v1_summary.json``. Commit all three: they pin the split.
"""

import argparse
import json

from cropxai.data.manifest import build_manifest
from cropxai.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    manifest, summary = build_manifest(cfg)

    manifest_path = resolve_path(cfg.data.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    classes = sorted(manifest.loc[manifest["source"] == "plantvillage", "label"].unique())
    resolve_path(cfg.data.classes_file).write_text(json.dumps(classes, indent=2))

    summary_path = manifest_path.with_name(manifest_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
