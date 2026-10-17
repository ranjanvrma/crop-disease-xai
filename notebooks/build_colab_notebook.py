"""Build notebooks/colab_bootstrap.ipynb. Edit this file, not the notebook.

    python notebooks/build_colab_notebook.py
"""

import hashlib
import json
from pathlib import Path

cells = []


def build_fingerprint(path: Path) -> str:
    """Hash of this generator, stamped into the notebook (text mode normalises line endings)."""
    return hashlib.md5(path.read_text(encoding="utf-8").encode()).hexdigest()[:10]


BUILD = build_fingerprint(Path(__file__))


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# crop-disease-xai — Colab runner

**Edit the settings cell below, then `Runtime → Run all`.** Every step skips itself when its result already exists,
so re-running is always safe:

| step | runs when |
|---|---|
| train | this seed has no checkpoint that finished with the *current* training settings |
| evaluate / verify / explain | their result is missing or older than the checkpoint (explain/verify also when the explanation settings changed) |

Checkpoints and all results live on Drive (`MyDrive/cropxai/`), so nothing is lost when Colab resets.

**One-time setup:** Runtime → Change runtime type → **T4 GPU**. Secrets (key icon) → add `WANDB_API_KEY`, notebook access on.
Dataset zips in `MyDrive/cropxai/raw/` (see `data/README.md`).
""")

code("""
# ===================== SETTINGS: edit, then Runtime -> Run all =====================
EXP   = "mnv3_fe"   # any file name in configs/experiment/ (without .yaml)
SEEDS = [0]         # one seed ~20-40 min; [0, 1, 2] for the full set

TRAIN    = True     # train seeds that haven't finished (set False to only re-use existing checkpoints)
EVALUATE = True     # in-domain test + PlantDoc field accuracy, size, latency
VERIFY   = True     # closed-form vs library Grad-CAM++ check   (CNNs only)
EXPLAIN  = True     # explanation metrics + heatmap figures      (CNNs only)

FORCE     = False   # True = redo the selected steps even if results exist
RUN_TESTS = False   # unit tests; only needed after code changes
# ====================================================================================
REPO_URL = "https://github.com/ranjanvrma/crop-disease-xai.git"
BRANCH   = "main"
DRIVE    = "/content/drive/MyDrive/cropxai"
NOTEBOOK_BUILD = "__BUILD__"   # set by notebooks/build_colab_notebook.py; don't edit
""".replace("__BUILD__", BUILD))

md("## Setup (automatic)")

code("""
import hashlib, os, time
from google.colab import drive, userdata

def sh(cmd):
    \"\"\"Run a shell command, streaming output; stop Run-all if it fails; report time.\"\"\"
    print(f"\\n$ {cmd}", flush=True)
    t0 = time.time()
    get_ipython().system(cmd)
    if get_ipython().user_ns.get("_exit_code", 0) != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    print(f"[took {(time.time() - t0) / 60:.1f} min]", flush=True)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'NO GPU: Runtime -> Change runtime type -> T4 GPU'")
drive.mount("/content/drive")

REPO = "/content/crop-disease-xai"
if not os.path.exists(REPO):
    sh(f"git clone -q -b {BRANCH} {REPO_URL} {REPO}")
else:
    # The Colab copy is a disposable mirror of GitHub (checkpoints/results live on Drive),
    # so overwrite any leftovers from earlier sessions instead of merging.
    sh(f"git -C {REPO} fetch -q origin {BRANCH} && git -C {REPO} reset -q --hard origin/{BRANCH}")
os.chdir(REPO)
sh("git log --oneline -1")

# The cells you run live in this open notebook, not in the clone: catch an outdated tab.
repo_build = hashlib.md5(open("notebooks/build_colab_notebook.py", encoding="utf-8").read().encode()).hexdigest()[:10]
if repo_build != NOTEBOOK_BUILD:
    raise RuntimeError("This notebook is outdated. File -> Open notebook -> GitHub tab -> "
                       "reopen colab_bootstrap.ipynb, then Run all again.")

# Install once per Colab machine; again only if the dependency list changed.
deps_hash = hashlib.md5(open("pyproject.toml", "rb").read()).hexdigest()
marker = "/content/.cropxai_installed"
if not os.path.exists(marker) or open(marker).read() != deps_hash:
    sh('pip install -q -e ".[train,dev]"')
    open(marker, "w").write(deps_hash)
else:
    print("packages: already installed on this machine")

os.environ["CROPXAI_DATA"] = "/content/data"           # local disk: fast
os.environ["CROPXAI_CKPT"] = f"{DRIVE}/checkpoints"    # Drive: survives resets
RESULTS = f"{DRIVE}/results"                            # Drive: survives resets
os.makedirs(os.environ["CROPXAI_CKPT"], exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
try:
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    os.environ.pop("WANDB_MODE", None)
except Exception:
    os.environ["WANDB_MODE"] = "offline"
    print("No WANDB_API_KEY secret found: W&B logs stay local (offline).")
""")

code("""
# Data: unzip once per session from Drive to local disk.
for name in ["plantvillage", "plantdoc", "field"]:
    z = f"{DRIVE}/raw/{name}.zip"
    if os.path.exists(f"/content/data/{name}"):
        print(f"{name}: ready")
    elif os.path.exists(z):
        os.makedirs("/content/data", exist_ok=True)
        sh(f"unzip -q '{z}' -d /content/data")
        print(f"{name}: unpacked")
    elif name == "field":
        print("field: no zip yet (optional until you have your own photos)")
    else:
        raise FileNotFoundError(f"Missing {z}. Upload it to Drive first (see data/README.md).")

# Split: committed in the repo; fall back to Drive, then build it.
if os.path.exists("splits/manifest_v1.csv"):
    print("split: from repo")
elif os.path.exists(f"{DRIVE}/splits/manifest_v1.csv"):
    sh(f"cp {DRIVE}/splits/* splits/")
    print("split: restored from Drive")
else:
    sh("python scripts/prepare_data.py --config configs/experiment/prepare_data.yaml")
    sh(f"mkdir -p {DRIVE}/splits && cp splits/* {DRIVE}/splits/")

if RUN_TESTS:
    sh("python -m pytest -q")
""")

md("## Pipeline (train → evaluate → verify → explain, skipping what's done)")

code("""
import json, sys
from omegaconf import OmegaConf
sys.path.insert(0, f"{REPO}/src")   # a fresh editable install isn't importable until a restart
from cropxai.train.loop import checkpoint_path, complete_marker, is_trained
from cropxai.utils.config import load_config

cfg_path = f"configs/experiment/{EXP}.yaml"
assert os.path.exists(cfg_path), f"No such experiment: {cfg_path}"
is_cnn = not EXP.startswith("deit")      # Grad-CAM++ steps are for CNNs only
ovr = f"paths.results_dir={RESULTS}"

def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

for seed in SEEDS:
    cfg = load_config(cfg_path, [f"seed={seed}"])
    ckpt = str(checkpoint_path(cfg))
    trained = is_trained(cfg)            # finished AND trained with the current settings
    print(f"\\n========== {EXP} seed {seed} ==========")

    if TRAIN and (FORCE or not trained):
        if complete_marker(cfg).exists() and not trained:
            print("train: settings changed since this seed was trained -> retraining")
        sh(f"python scripts/train.py --config {cfg_path} --seed {seed} {ovr}")
    elif trained:
        print("train: already finished with these settings, skipping")
    elif os.path.exists(ckpt):
        print("train: skipped; NOTE this checkpoint is outdated or from an unfinished run (preview only)")

    if not os.path.exists(ckpt):
        print("no checkpoint for this seed; nothing else to do")
        continue

    steps = [
        ("evaluate", EVALUATE, "accuracy", "evaluate.py"),
        ("verify", VERIFY and is_cnn, "closed_form", "verify_closed_form.py"),
        ("explain", EXPLAIN and is_cnn, "explanation", "explain.py"),
    ]
    for name, wanted, kind, script in steps:
        result = f"{RESULTS}/metrics/{kind}/{EXP}_seed{seed}.json"
        if not wanted:
            continue
        stale = FORCE or mtime(result) < mtime(ckpt)
        if not stale and kind in ("explanation", "closed_form"):
            # Also redo it if the explanation settings changed since the result was written.
            saved = json.load(open(result))["config"].get("explain")
            stale = saved != OmegaConf.to_container(cfg.explain, resolve=True)
            if stale:
                print(f"{name}: explanation settings changed -> recomputing")
        if stale:
            sh(f"python scripts/{script} --config {cfg_path} --seed {seed} {ovr}")
        else:
            print(f"{name}: up to date, skipping")
""")

md("## Results")

code("""
from IPython.display import Image, display

sh(f"python scripts/make_tables.py --results-dir {RESULTS}")

def load(kind, seed):
    p = f"{RESULTS}/metrics/{kind}/{EXP}_seed{seed}.json"
    return json.load(open(p))["metrics"] if os.path.exists(p) else None

for seed in SEEDS:
    print(f"\\n----- {EXP} seed {seed} -----")
    acc, cf = load("accuracy", seed), load("closed_form", seed)
    if acc:
        print(f"PlantVillage test top-1 {acc['test']['top1']:.4f} | "
              f"field strict {acc['field']['strict_top1']:.4f} | field restricted {acc['field']['restricted_top1']:.4f}")
    if cf:
        print(f"closed-form vs library: max |diff| {cf['max_abs_diff']:.2e}, mean Pearson r {cf['pearson_r_mean']:.6f}")
    for split in ["test", "field_test"]:
        fig = f"{RESULTS}/figures/heatmaps/{EXP}_seed{seed}_{split}.png"
        if os.path.exists(fig):
            print(split)
            display(Image(fig, width=900))
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("colab_bootstrap.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(cells)} cells)")
