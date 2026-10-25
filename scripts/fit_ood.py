"""Fit the web app's "is this a supported crop?" check for exported bundles (CPU, a few minutes).

    python scripts/fit_ood.py --bundles web/public/models/mnv3_ft_field web/public/models/mnv3_fe \
        --train-list lists/train.txt --val-list lists/val.txt --field-list lists/field.txt \
        --ood nonplant=lists/nonplant.txt vegetation=lists/vegetation.txt

Each list file holds one image path per line (``#`` starts a comment). ``train``
and ``val`` must be PlantVillage images of the 38 classes (labels come from the
parent folder name); ``field`` is optional (PlantDoc photos of supported crops,
labels from the manifest); each ``--ood`` group holds images that are *not*
supported crops (other plants, non-plants). The report also refits the check
with one crop left out at a time, as a proxy for a leaf of an unseen crop.

The check is a Mahalanobis distance (Lee et al., 2018) on the pooled
penultimate features the bundle already outputs: class means and one shared
covariance fitted on ``train``, kept to the top ``--dims`` principal directions
so the file stays small (about 0.65 MB at 128 dimensions). The score is the distance to the
closest class mean. The threshold is set on ``val`` so that ``--keep`` of real
PlantVillage photos pass; nothing is tuned on the ``ood`` images, which are only
used for the report. Writes ``<bundle>/ood.json`` and prints the report as JSON.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def read_list(path):
    if not path:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def preprocess(path, meta):
    """Same as web/src/preprocess.js (and the Python eval transform)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s, short = meta["image_size"], meta["resize_short"]
    scale = short / min(w, h)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    oy, ox = (rh - s) // 2, (rw - s) // 2
    crop = rgb[oy:oy + s, ox:ox + s].astype(np.float32) / 255
    x = (crop - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return x.transpose(2, 0, 1)


def features(bundle, paths, batch=32):
    meta = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(bundle / "model.onnx"), providers=["CPUExecutionProvider"])
    feats, logits = [], []
    for i in range(0, len(paths), batch):
        x = np.stack([preprocess(p, meta) for p in paths[i:i + batch]])
        # The graph may have a fixed batch of 1.
        for xi in x:
            lg, ft = sess.run(["logits", "features"], {"image": xi[None]})
            logits.append(lg[0])
            feats.append(ft[0].mean(axis=(1, 2)))
    return np.array(feats, np.float64), np.array(logits, np.float64), meta


def label_of(path, classes):
    name = Path(path).parent.name
    if name not in classes:
        raise ValueError(f"{path}: folder {name!r} is not one of the 38 classes")
    return classes.index(name)


def fit(f, y, n_classes, dims):
    means = np.stack([f[y == c].mean(0) for c in range(n_classes)])
    centred = f - means[y]
    cov = centred.T @ centred / len(f)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:dims]
    vals, vecs = vals[order], vecs[:, order]
    # Whitening projection: z = (f - mu) @ proj, distance^2 = ||z||^2.
    proj = vecs / np.sqrt(vals + 1e-6 * vals.max())
    return means, proj


def score(f, means, proj):
    z = f @ proj
    zm = means @ proj
    d2 = ((z[:, None, :] - zm[None]) ** 2).sum(-1)
    return np.sqrt(d2.min(1) / proj.shape[1])


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def auroc(id_scores, ood_scores):
    """P(ood score > id score); higher score = more out-of-distribution."""
    a, b = np.asarray(id_scores), np.asarray(ood_scores)
    if not len(a) or not len(b):
        return None
    return float(((b[:, None] > a[None]).mean() + 0.5 * (b[:, None] == a[None]).mean()))


def holdout_crops(train_f, y, val_f, vy, classes, dims, keep):
    """Proxy for "a leaf of a crop it has never seen": refit without one crop's classes,
    set the threshold on the remaining val photos, and count how many of that crop's
    val photos get flagged. Optimistic or not, it is only a proxy: the network itself
    was still trained on that crop."""
    out = {}
    for crop in sorted({c.split("___")[0] for c in classes}):
        kept = [i for i, c in enumerate(classes) if not c.startswith(crop + "___")]
        tr = np.isin(y, kept)
        means, proj = fit(train_f[tr], np.searchsorted(kept, y[tr]), len(kept), dims)
        sv = score(val_f, means, proj)
        inside = np.isin(vy, kept)
        out[crop] = float((sv[~inside] > np.quantile(sv[inside], keep)).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--field-list", help="PlantDoc photos; labels from --manifest")
    ap.add_argument("--manifest", default="splits/manifest_v1.csv", help="labels for the field photos")
    ap.add_argument("--data-root", default="data", help="manifest paths are relative to this")
    ap.add_argument("--ood", nargs="*", default=[], metavar="NAME=LIST", help="groups of unsupported images, report only")
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--keep", type=float, default=0.99, help="share of val photos that must pass")
    ap.add_argument("--report", help="also write the full report (with per-image scores) here")
    args = ap.parse_args()

    lists = {"train": read_list(args.train_list), "val": read_list(args.val_list), "field": read_list(args.field_list)}
    groups = dict(g.split("=", 1) for g in args.ood)
    lists.update({name: read_list(path) for name, path in groups.items()})
    field_labels = {}
    if lists["field"]:
        import csv
        root = Path(args.data_root).resolve()
        with open(args.manifest, encoding="utf-8") as f:
            by_path = {r["path"]: r["label"] for r in csv.DictReader(f)}
        for p in lists["field"]:
            rel = Path(p).resolve().relative_to(root).as_posix()
            field_labels[p] = by_path.get(rel) or by_path[rel.removesuffix(".jpg")]

    report = {}
    for b in map(Path, args.bundles):
        out = {k: features(b, v) for k, v in lists.items() if v}
        meta = out["train"][2]
        classes = meta["classes"]
        y = np.array([label_of(p, classes) for p in lists["train"]])
        vy = np.array([label_of(p, classes) for p in lists["val"]])
        means, proj = fit(out["train"][0], y, len(classes), args.dims)
        s = {k: score(v[0], means, proj) for k, v in out.items()}
        msp = {k: softmax(v[1]).max(1) for k, v in out.items()}
        threshold = float(np.quantile(s["val"], args.keep))
        flagged = {k: v > threshold for k, v in s.items()}
        rep = {
            "n": {k: len(v) for k, v in lists.items() if v},
            "flagged": {k: round(float(v.mean()), 4) for k, v in flagged.items() if k != "train"},
            "auroc_vs_val": {k: auroc(s["val"], s[k]) for k in s if k not in ("train", "val")},
            "auroc_vs_val_max_softmax": {k: auroc(-msp["val"], -msp[k]) for k in s if k not in ("train", "val")},
            "unsure_flagged": {k: round(float((v < meta["unsure_threshold"]).mean()), 4) for k, v in msp.items() if k != "train"},
            "holdout_crop_flagged": holdout_crops(out["train"][0], y, out["val"][0], vy, classes, args.dims, args.keep),
        }
        rep["holdout_crop_flagged_mean"] = float(np.mean(list(rep["holdout_crop_flagged"].values())))
        if "field" in out:
            fy = np.array([classes.index(field_labels[p]) for p in lists["field"]])
            ok = out["field"][1].argmax(1) == fy
            f = flagged["field"]
            rep["field_accuracy"] = {"passed": float(ok[~f].mean()) if (~f).any() else None, "n_passed": int((~f).sum()),
                                     "flagged": float(ok[f].mean()) if f.any() else None, "n_flagged": int(f.sum())}
        public = dict(rep)
        rep["per_image"] = {
            k: [{"path": Path(p).name, "score": round(float(sc), 3), "max_softmax": round(float(m), 3),
                 "top": classes[int(lg.argmax())], "flagged": bool(sc > threshold)}
                for p, sc, m, lg in zip(lists[k], s[k], msp[k], out[k][1])]
            for k in groups
        }
        report[b.name] = rep
        (b / "ood.json").write_text(json.dumps({
            "method": "mahalanobis_pca",
            "dims": args.dims,
            "threshold": threshold,
            "threshold_source": f"{args.keep:.0%} of {len(lists['val'])} PlantVillage validation photos pass",
            "fitted_on": f"{len(lists['train'])} PlantVillage training photos",
            # float32, little-endian, base64: proj is K x dims (row-major), mean_proj is classes x dims.
            "proj_b64": b64(proj),
            "mean_proj_b64": b64(means @ proj),
            "report": public,
        }, separators=(",", ":")), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
