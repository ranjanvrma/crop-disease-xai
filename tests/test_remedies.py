"""The remedies knowledge base must cover every class the model can predict."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"healthy", "fungal", "oomycete", "bacterial", "viral", "pest", "physiological"}
URGENCY = {"none", "low", "medium", "high"}


def _load():
    remedies = yaml.safe_load((ROOT / "knowledge/remedies.yaml").read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "splits/classes_v1.json").read_text(encoding="utf-8"))
    return remedies, classes


def test_every_class_has_exactly_one_entry():
    remedies, classes = _load()
    assert sorted(remedies["classes"]) == sorted(classes)


def test_entries_are_complete():
    remedies, _ = _load()
    assert remedies["disclaimer"].strip()
    for name, r in remedies["classes"].items():
        assert r["kind"] in KINDS, name
        assert r["urgency"] in URGENCY, name
        assert r["crop"] and r["name"] and r["look_for"] and r["cultural"], name
        healthy = r["kind"] == "healthy"
        assert healthy == name.endswith("___healthy"), name
        assert (r["pathogen"] is None) == healthy, name
        assert (r["urgency"] == "none") == healthy, name
        if not healthy:
            assert r["organic"] or r["chemical"], name
