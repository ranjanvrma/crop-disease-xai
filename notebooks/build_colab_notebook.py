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

**Just press `Runtime → Run all`. Nothing to edit.** It works through every experiment and seed in `EXPS`, in order.
When a Colab session ends, open a new one and press Run all again: finished work is skipped and it resumes
where it stopped. The last cell shows overall progress. Every step skips itself when its result already exists:

| step | runs when |
|---|---|
| train | this seed has no checkpoint that finished with the *current* training settings |
| evaluate / verify / explain | their result is missing or older than the checkpoint (explain/verify also when the explanation settings changed) |

Checkpoints and all results live on Drive (`MyDrive/cropxai/`), so nothing is lost when Colab resets.

**One-time setup:** Runtime → Change runtime type → **T4 GPU**. Secrets (key icon) → add `WANDB_API_KEY`, notebook access on.
Dataset zips in `MyDrive/cropxai/raw/` (see `data/README.md`).
""")

code("""
# ============ SETTINGS: normally nothing to change, just Runtime -> Run all ============
# The whole queue, in order. Seeds come from each experiment's config. Finished work is skipped.
EXPS = [
    "mnv3_fe", "effnet_b0_fe", "scratch_cnn",          # baselines (done)
    "mnv3_fe_p50", "mnv3_fe_p70",                      # structured pruning + 2-epoch fine-tune
    "mnv3_fe_p50_ft8", "mnv3_fe_p70_ft8",              # same, 8-epoch fine-tune
    "mnv3_fe_int8", "mnv3_fe_p50_int8", "mnv3_fe_p70_int8",  # INT8 post-training quantization
]

TRAIN    = True     # train seeds that haven't finished (set False to only re-use existing checkpoints)
EVALUATE = True     # in-domain test + PlantDoc field accuracy, size, latency
VERIFY   = True     # closed-form vs library Grad-CAM++ check   (CNNs only)
EXPLAIN  = True     # explanation metrics + heatmap figures      (CNNs only)
WEB_MODEL = ("mnv3_fe", 0)   # (experiment, seed) exported for the offline web app; None to skip (CPU, ~3 min)

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
    sh('pip install -q -e ".[train,compress,dev,web]"')
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
from cropxai.compress.variants import is_int8
from cropxai.train.loop import checkpoint_path, complete_marker, is_trained
from cropxai.utils.config import load_config

ovr = f"paths.results_dir={RESULTS}"

def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

def stale_result(kind, exp, seed, cfg, ckpt):
    \"\"\"True if this result must be (re)computed for the current checkpoint and settings.\"\"\"
    result = f"{RESULTS}/metrics/{kind}/{exp}_seed{seed}.json"
    if FORCE or mtime(result) < mtime(ckpt):
        return True
    if kind in ("explanation", "closed_form"):   # explanation settings changed since it was written?
        saved = json.load(open(result))["config"].get("explain")
        return saved != OmegaConf.to_container(cfg.explain, resolve=True)
    return False

def run_seed(exp, seed):
    cfg_path = f"configs/experiment/{exp}.yaml"
    cfg = load_config(cfg_path, [f"seed={seed}"])
    ckpt = str(checkpoint_path(cfg))
    print(f"\\n========== {exp} seed {seed} ==========")

    trained = is_trained(cfg)                # finished AND trained with the current settings
    if TRAIN and (FORCE or not trained):
        if complete_marker(cfg).exists() and not trained:
            print("train: settings changed since this seed was trained -> retraining")
        sh(f"python scripts/train.py --config {cfg_path} --seed {seed} {ovr}")
    elif trained:
        print("train: done, skipping")
    elif os.path.exists(ckpt):
        print("train: skipped; NOTE this checkpoint is outdated or from an unfinished run (preview only)")
    if not os.path.exists(ckpt):
        print("no checkpoint for this seed; nothing else to do")
        return

    steps = [
        ("evaluate", EVALUATE, "accuracy", "evaluate.py"),
        ("verify", VERIFY and not is_int8(cfg), "closed_form", "verify_closed_form.py"),  # INT8: no gradients
        ("explain", EXPLAIN, "explanation", "explain.py"),
    ]
    for name, wanted, kind, script in steps:
        if not wanted:
            continue
        if stale_result(kind, exp, seed, cfg, ckpt):
            sh(f"python scripts/{script} --config {cfg_path} --seed {seed} {ovr}")
        else:
            print(f"{name}: done, skipping")

for exp in EXPS:
    assert os.path.exists(f"configs/experiment/{exp}.yaml"), f"No such experiment: {exp}"
for exp in EXPS:
    for seed in load_config(f"configs/experiment/{exp}.yaml").seeds:
        run_seed(exp, seed)
print("\\nQueue finished: every experiment and seed is done.")
""")

md("## Results")

code("""
# Heatmap figures are saved on Drive: MyDrive/cropxai/results/figures/heatmaps/
sh(f"python scripts/make_tables.py --results-dir {RESULTS}")

print("\\nProgress (seeds fully done per experiment):")
total = done_total = 0
for exp in EXPS:
    seeds = list(load_config(f"configs/experiment/{exp}.yaml").seeds)
    done = 0
    for s in seeds:
        c = load_config(f"configs/experiment/{exp}.yaml", [f"seed={s}"])
        kinds = ["accuracy", "explanation"] + ([] if is_int8(c) else ["closed_form"])
        results_ok = all(os.path.exists(f"{RESULTS}/metrics/{k}/{exp}_seed{s}.json") for k in kinds)
        done += int(is_trained(c) and results_ok)   # results from older settings don't count
    total += len(seeds)
    done_total += done
    print(f"  {exp:14s} {done}/{len(seeds)}  {'done' if done == len(seeds) else ''}")
print(f"  overall: {done_total}/{total} seeds")
""")

md("## Web app model (CPU only)")

code("""
# Exports the chosen seed for the offline web app (web/). The unsure threshold is tuned on val.
# Download MyDrive/cropxai/web_model/ and put its files in web/public/model/ (see web/README.md).
if WEB_MODEL:
    exp, seed = WEB_MODEL
    cfg = load_config(f"configs/experiment/{exp}.yaml", [f"seed={seed}"])
    if is_trained(cfg):
        sh(f"python scripts/export_web.py --config configs/experiment/{exp}.yaml --seed {seed} "
           f"--out {DRIVE}/web_model {ovr}")
    else:
        print(f"web model: {exp} seed {seed} is not trained yet, skipping")
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("colab_bootstrap.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(cells)} cells)")
