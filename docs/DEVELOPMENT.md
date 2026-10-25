# Developer guide: crop-disease-xai

> **For contributors.** Read this before changing anything substantial. It records the current
> state, the research rules, how to run everything, what is left, and every non-obvious decision made
> so far. For how to send a change, see [CONTRIBUTING.md](../CONTRIBUTING.md).
>
> *State as of 25 October 2026: all branches merged into `main`; live demo at https://crop-disease-xai.onrender.com/.*

---

## Contents
1. [What this project is](#1-what-this-project-is)
2. [Current state in one screen](#2-current-state-in-one-screen)
3. [Rules (non-negotiable)](#3-rules-non-negotiable)
4. [Repository map](#4-repository-map)
5. [What you need before you can run anything](#5-what-you-need-before-you-can-run-anything)
6. [Local setup](#6-local-setup)
7. [Running on Google Colab](#7-running-on-google-colab)
8. [How the pipeline works](#8-how-the-pipeline-works)
9. [Results so far](#9-results-so-far)
10. [What is left, in order](#10-what-is-left-in-order)
11. [Gotchas and known issues](#11-gotchas-and-known-issues)
12. [Decision log](#12-decision-log)

---

## 1. What this project is

Crop-leaf disease classification (38 PlantVillage classes) with **Grad-CAM++** heatmap explanations,
**model compression** (structured pruning + INT8 quantization), **remedies** for every class, and an
**offline web app** (`web/`) that runs the model and the heatmap on any phone or desktop browser. An
ExecuTorch Android build is kept only for the research benchmark (Pixel 6a).

**Research question:** can a compressed crop-disease model running on a phone still give faithful
Grad-CAM++ explanations, and how much does compression change what the model looks at?

Sub-questions:
1. Accuracy vs. size for each compression stage (pruning 50/70%, INT8, both stacked).
2. Do Grad-CAM++ maps stay faithful (deletion/insertion, Average Drop, ROAD…) and stable (vs. the uncompressed parent) after compression?
3. Can Grad-CAM++ run on a phone in real time? Yes in principle: see the closed-form trick below.
4. Lab → field gap: models trained on clean lab photos (PlantVillage) are tested on real field photos (PlantDoc).

**The key technical idea (already implemented and verified):** mobile runtimes have no backward pass,
but every CNN here ends in global-average-pool + a single linear layer, so the gradient of class score
$y_c$ w.r.t. feature map $A^k$ ($H\times W$) is the constant $g_k = W_{ck}/(HW)$. Grad-CAM++ then
collapses to a closed form using only the feature maps and the classifier weights:

$$\alpha_k = \frac{g_k^2}{2g_k^2 + g_k^3\sum_{ij}A^k_{ij} + \epsilon},\quad w_k = HW\cdot\mathrm{ReLU}(g_k)\,\alpha_k,\quad L_c=\mathrm{ReLU}\Big(\sum_k w_kA^k\Big)$$

with $\epsilon = 10^{-6}$ (same as `pytorch-grad-cam`). Implementation: `src/cropxai/explain/closed_form.py`.
Verified equal to the library to ~1e-6 on trained checkpoints, and about 2.4× faster on GPU.
The original project brief (a PDF, not in the repo) asked for exactly this.

**Scope is deliberately compute-lean** (the owner has only a free Colab GPU): see §12 for what was cut and why.

---

## 2. Current state in one screen

| Phase | What | Status |
|---|---|---|
| 1 | Data pipeline, dedup, pinned split, field-leak guard | ✅ done |
| 2 | Models, training (with resume), evaluation | ✅ done |
| 3 | Grad-CAM++/Grad-CAM/Score-CAM + faithfulness/localisation metrics, closed-form Grad-CAM++ | ✅ done |
| — | Baselines: `mnv3_fe` (3 seeds), `effnet_b0_fe` (3 seeds), `scratch_cnn` (1 run) | ✅ done, results in `results/metrics/` |
| 4 | Compression: pruning 50/70% (2- and 8-epoch fine-tune), INT8, and both stacked: 7 variants × 3 seeds | ✅ done, results in `results/metrics/` |
| 5 | Core experiment: explanation quality + stability under compression | 🟡 metrics done (README Tables 2–3); centrepiece figure and write-up not done |
| — | Field-robust model `mnv3_ft_field` (full fine-tune + field-style augmentation), 3 seeds | ✅ done |
| 6 | On-device closed-form Grad-CAM++ | ✅ in the web app (`web/src/xai.js`), tested equal to the Python closed form |
| — | **Offline web app + demo site** with 4 real models (`mnv3_fe`, `effnet_b0_fe`, `mnv3_fe_p50_ft8`, `mnv3_ft_field`), "not a supported crop" warning, remedies | ✅ **live at https://crop-disease-xai.onrender.com/** (Render, `render.yaml`) |
| — | Remedies knowledge base, all 38 classes (`knowledge/remedies.yaml`) | ✅ written; **not yet reviewed by a plant pathologist** |
| — | Name check (no AI-assistant attribution) as git hooks + GitHub Action | ✅ enable hooks per clone: `git config core.hooksPath .githooks` |
| 7 | ExecuTorch export + Pixel 6a benchmarks (research Table 4 only) | ⬜ not started |
| 8 | Paper, release (weights, lesion annotations, demo video) | ⬜ not started |

**The very next action:** Phase 5 write-up (centrepiece figure from `results/metrics/`), then Phase 7
benchmarks on the Pixel 6a. No GPU training is needed for either.

Tests: `45 passed` on Linux (CPU torch 2.14, Python 3.13, with the `web` extra) / `42 passed, 1
skipped` on Windows without `onnxruntime`; plus `cd web && npm test` (parity) and `npm run test:e2e`
(browser tests). CI (`.github/workflows/ci.yml`) runs them on pull requests and pushes to `main`;
`.github/workflows/name-check.yml` runs on every push.

All work is on `main`; the earlier branches (`local-gpu-run`, `web-app`, `demo-site`) were merged.

---

## 3. Rules (non-negotiable)

These have been followed since day one. Keep them.

1. **Never fabricate results.** Tables (README, paper, dashboard) are generated only from
   `results/metrics/**.json` written by real runs. Missing → show "—". No hand-typed numbers.
2. **Three seeds per reported number**, mean ± sample std (`configs/base.yaml: seeds`). The only
   exception is the scratch CNN, which is explicitly labelled a single run and shows `(n=1)`.
3. **The field test set is untouchable.** PlantDoc and the owner's own field photos are
   `split=field_test`. `src/cropxai/data/guard.py` enforces this (including perceptual-hash
   near-duplicates). Only `evaluate.py`/`explain.py` may load it with `allow_heldout=True`.
   Never tune anything (epochs, thresholds, the app's "unsure" cut-off) on it; tune on `val`.
   Do not remove "bad" PlantDoc images after seeing results; that would be tuning on the test set.
4. **No hard-coded hyperparameters.** One YAML per experiment in `configs/experiment/`, composed via
   `inherits:`. Scripts take `--config` plus `key=value` overrides.
5. **Don't invent citations**, and don't claim the research gap is still open. The literature search
   must be re-run by a human before writing (`docs/research-gap.md`).
6. **The app is the offline web app in `web/`** (ONNX Runtime Web, WebAssembly), so it runs on any
   phone model and on desktop from one codebase (owner's decision, 20 Oct). Its model comes only from
   `scripts/export_web.py`, and `web/test/parity.test.mjs` must keep passing. **The research Android
   benchmark uses ExecuTorch only** (XNNPACK backend); PyTorch Mobile is deprecated, never use it. The
   ExecuTorch environment must be separate from training (create it in Phase 7); read current
   ExecuTorch docs before writing that export code.
7. **Never hand-edit `notebooks/colab_bootstrap.ipynb`.** Edit `notebooks/build_colab_notebook.py`
   and run it (`python notebooks/build_colab_notebook.py`).
8. **No AI-assistant attribution in the repository:** no `Co-Authored-By` or "generated with" trailers
   in commits, and no such mentions in code or docs. This is enforced automatically:
   `scripts/check_names.py` runs as git hooks (staged changes, file names, commit messages) and as
   the `name-check` GitHub Action on every push. **Turn the hooks on once per clone:**
   `git config core.hooksPath .githooks`
9. **Keep it compute-lean.** Any new GPU work must state its estimated cost first. Don't reintroduce
   the dropped models (§12) without a reason and a budget.
10. Research (`notebooks/`, `scripts/`) and the app (`web/`) stay separate. The app reads only the
    exported bundle; any accuracy it shows comes from `results/metrics/` via the export script.
11. **Remedies** (`knowledge/remedies.yaml`) are general guidance until a plant pathologist reviews
    them; keep `reviewed: false` until then, and never present them as verified.
12. Style: simple over clever, small functions, type hints, docstrings, config-driven. Commit per milestone.

---

## 4. Repository map

```
configs/
  base.yaml                     seeds [0,1,2], paths (env-driven), W&B, image_size 224, batch 64
  data/default.yaml             data paths, split fractions, dedup threshold, augmentation
  data/plantdoc_to_plantvillage.yaml   PlantDoc folder → PlantVillage class (verified against the real download)
  model/{mnv3,effnet_b0,scratch_cnn}.yaml
  train/feature_extract.yaml    frozen backbone, 6 epochs
  train/scratch.yaml            scratch CNN, 30 epochs
  train/prune_finetune.yaml     2-epoch fine-tune after pruning
  explain/default.yaml          explanation sample sizes, ROAD/Score-CAM settings
  app/default.yaml              web export: output folder, unsure-threshold target (tuned on val)
  experiment/*.yaml             one file per run (see §8)
src/cropxai/
  data/       sources.py (scan datasets) · hashing.py (dHash) · manifest.py (dedup + split)
              guard.py (field-leak guard) · dataset.py (loaders, explanation samples) · transforms.py
  models/     factory.py (timm models; MobileNetV3 head → GAP + single Linear) · scratch_cnn.py
  train/      loop.py (fit with per-epoch resume, settings fingerprint, completion marker, load_trained)
  explain/    cams.py (library CAMs + "closed_form" method) · closed_form.py · figures.py
  eval/       classification.py (top-1, macro-F1, ECE, field strict/restricted) · efficiency.py
              faithfulness.py (deletion/insertion, Avg Drop, Inc. Conf, GPU ROAD) · localisation.py
              stability.py (Spearman, SSIM, top-10% IoU) · tables.py (README tables)
  compress/   prune.py (Torch-Pruning) · quantize.py (PT2E INT8, QuantizedModel) · variants.py
  export/     web.py (ONNX bundle for the web app); ExecuTorch export goes here too (Phase 7)
  utils/      config.py (YAML composition, training fingerprint) · seed.py · tracking.py (W&B + results JSON)
scripts/      prepare_data · train · evaluate · verify_closed_form · explain · make_tables · export_web
knowledge/    remedies.yaml: symptoms, cultural/organic/chemical control for all 38 classes
web/          offline web app (Vite + ONNX Runtime Web); see web/README.md
splits/       manifest_v1.csv (every image: relative path, label, source, split, dHash), classes, summary
notebooks/    build_colab_notebook.py → colab_bootstrap.ipynb (the Colab runner)
results/      metrics/<kind>/<run>_seed<n>.json (source of truth) · figures/ (git-ignored PNGs) · tables/
docs/         DEVELOPMENT.md (this guide) · research-gap.md (literature-search log, to be filled by a human)
tests/        synthetic-data unit + end-to-end tests (no datasets needed)
data/         git-ignored datasets (README explains the layout)
```

---

## 5. What you need before you can run anything

| Item | Where it is | Notes |
|---|---|---|
| Code | this GitHub repo | |
| **PlantVillage** (colour, 38 classes, 54,305 images) | GitHub `spMohanty/PlantVillage-Dataset` (`raw/color`) | |
| **PlantDoc** (2,578 images) | GitHub `pratikkayal/PlantDoc-Dataset` (`train/` + `test/`) | ⚠️ see filename note below |
| **The owner's prepared zips** | owner's Drive: `MyDrive/cropxai/raw/plantvillage.zip`, `plantdoc.zip` | **easiest and safest**: ask the owner to share them |
| **Trained baseline checkpoints** | owner's Drive: `MyDrive/cropxai/checkpoints/{mnv3_fe,effnet_b0_fe,scratch_cnn}/seed*/` | Phase 4 needs `mnv3_fe` (all 3 seeds); each folder has `best.pt` + `complete.json` |
| **Finished results JSON + heatmap figures** | owner's Drive: `MyDrive/cropxai/results/{metrics,figures}/` | needed for the tables and the paper |
| Weights & Biases account | wandb.ai (free) | optional; set `WANDB_MODE=disabled` to skip |
| A GPU | Colab T4 (free tier) or any CUDA GPU | the owner's laptop has an RTX 3050 4 GB (untested here) |

**⚠️ PlantDoc filenames.** On Windows, 87 PlantDoc files contain characters Windows forbids (e.g.
`IMG_1629.JPG?1507122477.jpg`). When the owner extracted the data, each path component had
`< > : " | ? * \` replaced by `_` and trailing spaces/dots stripped. The committed
`splits/manifest_v1.csv` uses these **sanitized** names. If you download PlantDoc fresh on
Linux/macOS, those files won't be found. Options, best first:
1. Use the owner's `plantdoc.zip` from Drive (exact files).
2. Rename fresh files with the same rule, then check that every manifest path exists:
   ```python
   import re, pandas as pd, pathlib
   m = pd.read_csv("splits/manifest_v1.csv", dtype=str, keep_default_na=False)
   missing = [p for p in m.path if not (pathlib.Path("data") / p).exists()]
   print(len(missing), missing[:5])
   ```

**If you can't get the checkpoints:** retrain them. `mnv3_fe` is 3 seeds × ~13 min on a T4 (6 epochs,
frozen backbone). Just run the queue; training is skipped only when a checkpoint with the current
settings exists.

---

## 6. Local setup

```bash
git clone https://github.com/ranjanvrma/crop-disease-xai.git
cd crop-disease-xai
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[train,compress,dev,web]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q -p no:warnings                 # expect 45 passed
```

- The tests use tiny synthetic images (`tests/conftest.py`), so they need no datasets.
- Put datasets under `data/` (layout in `data/README.md`) or point `CROPXAI_DATA` elsewhere.
  Checkpoints go to `checkpoints/` or `CROPXAI_CKPT`.
- **You do not need to run `prepare_data.py`.** The split is pinned in `splits/`. Re-running it
  is only needed when adding your own field photos (field photos only ever go to `field_test`).
- For a local GPU run, install the CUDA build of PyTorch first (pytorch.org). The owner's local venv
  was CPU-only.

Run one experiment locally (any experiment name from `configs/experiment/`):
```bash
python scripts/train.py              --config configs/experiment/mnv3_fe_p50.yaml --seed 0
python scripts/evaluate.py           --config configs/experiment/mnv3_fe_p50.yaml --seed 0
python scripts/verify_closed_form.py --config configs/experiment/mnv3_fe_p50.yaml --seed 0   # not for *_int8
python scripts/explain.py            --config configs/experiment/mnv3_fe_p50.yaml --seed 0
python scripts/make_tables.py        # rebuilds README Tables 1–3 from results/metrics
```
There is no local queue runner yet. A `scripts/run_queue.py` mirroring the notebook's pipeline
cell would be a small, useful addition if you train locally.

---

## 7. Running on Google Colab

The notebook is the main runner. **Nothing to edit; just press Run all.**

**One-time setup (your own Google account):**
1. Runtime → Change runtime type → **T4 GPU**.
2. Secrets (key icon) → add `WANDB_API_KEY`, notebook access on (optional; without it W&B runs offline).
3. Put `plantvillage.zip` and `plantdoc.zip` in **your** `MyDrive/cropxai/raw/`. Each zip contains
   the dataset folder at its top level (`plantvillage/color/...`, `plantdoc/{train,test}/...`).
4. Put the baseline checkpoints in **your** `MyDrive/cropxai/checkpoints/` (same folder structure as
   the owner's), and the finished results in `MyDrive/cropxai/results/metrics/`.
5. If you work from a fork, change `REPO_URL` in `notebooks/build_colab_notebook.py`, rebuild the
   notebook and push.

**Every session:** File → Open notebook → GitHub tab → `notebooks/colab_bootstrap.ipynb` → **Run all**.

What the notebook does (`notebooks/build_colab_notebook.py`):
- Mounts Drive, clones the repo, or **hard-resets** an existing copy to GitHub (the Colab copy is
  disposable; everything valuable is on Drive).
- **Outdated-notebook guard:** the notebook is stamped with a hash of the builder script
  (`NOTEBOOK_BUILD`). If GitHub has a newer builder, setup stops and tells you to reopen the notebook.
  Cells live in the open tab, so a stale tab would otherwise run old logic.
- Installs packages once per Colab machine (re-installs only if `pyproject.toml` changed).
- Unzips the datasets to local disk (`/content/data`; reading 55k files from Drive is very slow).
- Runs the **queue** `EXPS`: for every experiment and every seed in its config:
  - **train**, skipped if a `complete.json` with a matching settings fingerprint exists; resumes from
    `last.pt` after a disconnect (loses at most one epoch). INT8 variants are "trained" by calibration.
  - **evaluate / verify / explain**, skipped if the result JSON is newer than the checkpoint and the
    explanation settings are unchanged. `verify` is skipped for INT8 (no gradients).
- Results and figures are written straight to `MyDrive/cropxai/results/`; checkpoints to
  `MyDrive/cropxai/checkpoints/`.
- The last cell rebuilds the tables and prints progress per experiment.

**Free-tier realities:** sessions disconnect when idle, and the GPU quota runs out after a few hours a
day ("Cannot connect to GPU backend"). Never click "Connect without GPU". Wait, or use paid units.

---

## 8. How the pipeline works

### Experiments (`configs/experiment/`)

| Experiment | What | Seeds | State |
|---|---|---|---|
| `mnv3_fe` | MobileNetV3-Large (timm `mobilenetv3_large_100`), head replaced by GAP + Linear(960→38), frozen backbone, 6 epochs | 3 | done |
| `effnet_b0_fe` | EfficientNet-B0, frozen backbone, 6 epochs: comparison row | 3 | done |
| `scratch_cnn` | 4-stage CNN from scratch, 30 epochs: single-run reference | 1 | done |
| `mnv3_fe_p50`, `mnv3_fe_p70` | parent `mnv3_fe`, 50/70% structured channel pruning, 2-epoch fine-tune (lr 1e-4) | 3 | to run |
| `mnv3_fe_int8` | INT8 post-training quantization of `mnv3_fe` | 3 | to run |
| `mnv3_fe_p50_int8`, `mnv3_fe_p70_int8` | INT8 of the pruned variants | 3 | to run |
| `prepare_data` | builds the split (already done) | – | done |

A compressed variant's config has a `compress:` section (`parent`, `prune_ratio`, `int8`,
`calib_images`). `src/cropxai/compress/variants.py` follows the parent chain (same seed) and
`root_config` finds the uncompressed model for stability metrics.

### Data
- Split: PlantVillage 80/10/10 stratified (seed 0, independent of training seed):
  **43,424 train / 5,428 val / 5,428 test**. Field test: **2,522 PlantDoc** images (all of PlantDoc
  train + test, mapped to 28 PlantVillage classes).
- 79 exact duplicates removed (23 PlantVillage, 56 PlantDoc); 2 PlantVillage images dropped as
  near-duplicates of PlantDoc images (dHash distance ≤ 4).
- Field metrics are reported two ways: **strict** = argmax over all 38 classes (what the app does);
  **restricted** = argmax over the classes present in the field set.

### Training (`src/cropxai/train/loop.py`)
AdamW, linear warm-up + cosine decay per step, AMP fp16 on GPU, channels-last on GPU, best checkpoint
by **val macro-F1**. Feature extraction freezes the backbone, including BatchNorm running stats.
`complete.json` stores a fingerprint of the training-relevant config keys
(`utils/config.py: TRAINING_KEYS`). Changing paths or explanation settings never forces a retrain;
changing model/train/data/augment/compress settings does.

### Compression (`src/cropxai/compress/`)
- **Pruning:** Torch-Pruning `MetaPruner`, `GroupMagnitudeImportance(p=2)`, uniform ratio, classifier
  ignored, so the GAP + Linear head survives and the closed form stays exact (tested). With a uniform
  ratio the pruned shapes depend only on architecture and ratio, so `load_trained` rebuilds a pruned
  model by pruning a fresh one and loading the state dict.
- **INT8:** PT2E with the **XNNPACK symmetric per-channel** quantizer (`torch.ao`, the scheme
  ExecuTorch uses on Android). It falls back to torchao's X86 quantizer if `torch.ao` lacks it; the
  quantizer name is saved in each result. The model is exported with **two outputs (logits, final
  feature maps)**, calibrated on 512 training images, and saved as a serialized `torch.export`
  program inside `best.pt`, together with the float classifier weights.
  `QuantizedModel` gives it the same interface as the float models.
- INT8 variants are explained with `closed_form` only (no gradients). Desktop latency is not reported
  for INT8 (reference quantize ops are not representative); INT8 latency is a phone measurement.

### Explanations (`scripts/explain.py`, `configs/explain/default.yaml`)
- Fixed seeded sample: 300 test images and 300 field images (field images with lesion masks first);
  the same images for every model. The predicted class is explained.
- Methods: baselines get Grad-CAM++, Grad-CAM, Score-CAM (Score-CAM on 50 images, one image at a time
  to avoid OOM). Pruned variants get Grad-CAM++ only; INT8 variants get closed-form Grad-CAM++.
- Metrics: deletion/insertion AUC (20 steps; deletion to the mean colour, insertion from a Gaussian
  blur), Average Drop, Increase in Confidence, **ROAD** (100 images, percentiles 20/40/60/80, noisy
  linear imputation solved by batched conjugate gradient on the GPU and tested against the library),
  and pointing game / IoU when lesion masks exist (none yet).
- **Stability (Table 3):** for compressed variants, both models' closed-form maps for the class the
  uncompressed parent predicts: Spearman, SSIM, top-10% IoU.

### Results flow
Each run writes `results/metrics/<kind>/<run>_seed<n>.json` (`kind` ∈ accuracy, explanation,
closed_form), including the full config, git commit and library versions. `scripts/make_tables.py`
aggregates seeds into README Tables 1–3 (between `<!-- tableN:start/end -->` markers) and
`results/tables/*.csv|md`. Table 4 (on-device) is still a placeholder.

---

## 9. Results so far

> From `results/metrics/**.json` (README Tables 1–3, regenerated with `scripts/make_tables.py`).
> All produced on an RTX 5070 Ti laptop GPU (float runs torch 2.14, INT8 runs torch 2.10; recorded
> per result). Mean ± std over 3 seeds unless noted.

| Model | PV test top-1 | Field strict | Field restricted | Weights |
|---|---|---|---|---|
| `mnv3_fe` (parent) | 96.66 ± 0.07% | 21.77 ± 0.14% | 24.56 ± 0.20% | 12.0 MB |
| `effnet_b0_fe` | 96.11 ± 0.27% | 24.27 ± 0.81% | 26.99 ± 1.07% | 16.2 MB |
| `scratch_cnn` (n = 1) | 99.76% | 15.42% | 17.41% | 4.7 MB |
| `mnv3_ft_field` | 99.44 ± 0.03% | 23.43 ± 0.56% | 27.44 ± 0.00% | 12.0 MB |
| `mnv3_fe_int8` | 94.12 ± 0.04% | 21.28 ± 0.30% | 23.67 ± 0.41% | 3.0 MB |
| `mnv3_fe_p50` / `_p50_ft8` | 69.25% / 88.87% | 6.42% / 9.90% | 8.55% / 13.16% | 3.2 MB |
| `mnv3_fe_p70` / `_p70_ft8` | 51.62% / 77.82% | 4.06% / 5.74% | 6.16% / 7.77% | 1.2 MB |
| `mnv3_fe_p50_int8` / `_p70_int8` | 61.05% / 46.54% | 6.05% / 3.13% | 9.01% / 5.00% | 0.8 / 0.3 MB |

**Findings:**
1. **Large lab → field gap** for every model; the from-scratch CNN is the most extreme (99.8% → 15.4%,
   single run), consistent with shortcut learning on PlantVillage.
2. **Generic field-style augmentation closes almost none of the gap** (`mnv3_ft_field`: field strict
   21.8% → 23.4%). The identical restricted accuracy across seeds (692/2522 each) was checked and is a
   coincidence, not a bug (the restricted macro-F1 differs per seed).
3. **INT8 preserves explanations** (stability vs parent: Spearman 0.92, SSIM 0.90, top-10% IoU 0.64 on
   test) at −2.5 points lab accuracy. **Uniform structured pruning of the feature-extraction parent
   does not:** accuracy collapses (2-epoch fine-tune) and is only partly recovered with 8 epochs, and
   heatmap stability falls to Spearman 0.37–0.46 (50%) and 0.21–0.40 (70%); field-set rank
   correlation at 70% is about zero. Stacking INT8 on pruning costs further accuracy.
4. **Grad-CAM was more faithful than Grad-CAM++** on both feature-extraction models on every
   faithfulness metric; on the fully fine-tuned `mnv3_ft_field` they are roughly equal. Report this
   honestly; lesion-mask localisation (needs masks) is where Grad-CAM++'s multi-lesion advantage
   would have to show.
5. **Caveat for Table 2:** deletion AUC of the pruned models looks "better" only because those models
   are far less confident; their insertion AUC, Average Drop and Increase in Confidence are worse.
   Compare faithfulness between models of similar accuracy, or use Table 3.
6. Closed-form Grad-CAM++ matched `pytorch-grad-cam` to ~1e-6 on every trained float model.
7. PlantDoc contains some images that are not single leaves (e.g. fruit, whole fields); kept as-is
   (rule 3) and to be listed as a limitation.

---

## 10. What is left, in order

1. **Phase 5: core experiment write-up (no GPU needed)**
   - `scripts/make_figures.py`: accuracy vs. weights size and explanation quality (ROAD, Table 3
     stability) vs. size on one plot, from `results/metrics/`.
   - Which stage changes explanations most (answer from Table 3: pruning ≫ INT8), and whether
     compression changes field accuracy (shortcut learning).
2. **Phase 7: on-device benchmarks**
   - ExecuTorch export in a **separate** environment (create it then; pin ExecuTorch,
     torch and torchao together after reading the current docs). Export the chosen variant with two
     outputs (logits, feature maps); quantize with ExecuTorch's own XNNPACK quantizer; lower to
     XNNPACK; produce a `.pte`; store the classifier weights as an asset.
   - A minimal Android benchmark harness (the user-facing app is the web app).
   - Pixel 6a: size on disk, cold start, latency median + p95 after warm-up, heatmap latency, peak
     memory; many iterations, fixed device. Fill README Table 4 from a results JSON (`kind=ondevice`).
3. **Phase 8: paper and release:** LaTeX in a new `paper/` folder (tables generated from results);
   weights and lesion annotations on Hugging Face; demo video of the live site.
4. **Optional (state the GPU cost first):** compress a fully fine-tuned parent (e.g. `mnv3_ft_field`)
   or use a gentler pruning schedule, to test whether the pruning collapse is due to the frozen-backbone
   parent.
5. **Needs a human (cannot be automated):**
   - Collect 150–300 own field leaf photos (PlantVillage classes), labels confirmed by a plant expert;
     put them in `data/field/images/<PlantVillage class>/` and re-run `prepare_data.py`.
   - Lesion masks (CVAT or Label Studio, binary PNG, same stem) in `data/field/masks/...`; without
     them pointing game and IoU stay "—".
   - Re-run the literature search and fill `docs/research-gap.md`.
   - Plant pathologist / extension officer review of `knowledge/remedies.yaml`, then `reviewed: true`.

---

## 11. Gotchas and known issues

- **Windows text encoding:** always `read_text(encoding="utf-8")` / `write_text(..., encoding="utf-8")`.
  The default cp1252 crashed on the `⚠️` in the class-map YAML. Write generated files with
  `newline="\n"` (`.gitattributes` enforces LF).
- **Editable install inside Colab** isn't importable until a runtime restart, so the notebook does
  `sys.path.insert(0, "<repo>/src")`.
- **INT8 needs torch ≤ 2.10:** torch 2.14 removed `torch.ao`'s XNNPACKQuantizer. The committed INT8
  results were made in a torch 2.10 environment. On newer torch the code silently falls back to
  torchao's X86 quantizer (recorded as `quantizer` in each result), which is a *different* scheme:
  don't mix the two in one table.
- **INT8 checkpoints in a fresh process** need the `quantized_decomposed` ops registered before
  loading (fixed in `compress/quantize.py`, commit `99839d2`); the unit test quantizes and loads in
  one process, so it can't catch a regression here.
- **Score-CAM memory:** the library upsamples all channels of all batch images at once (~6 GB for
  32 images), so it is run one image at a time with 128 masked inputs per forward pass.
- **ROAD speed:** the library's scipy imputation was the main explain cost (~110 s per method-split);
  replaced by the batched GPU conjugate-gradient solver in `eval/faithfulness.py` (same equations,
  matches within 1e-4). On CPU it is slower than scipy, which is fine.
- **Persistent DataLoader workers:** augmentation randomness continues across epochs rather than being
  re-seeded each epoch. Runs are still reproducible from their seed. A resumed run's augmentation
  stream differs slightly from an uninterrupted one.
- **Changing configs can retrain finished runs:** anything in `TRAINING_KEYS` changes the fingerprint.
  Before touching `configs/{model,train,data}` or `base.yaml`, check that the fingerprints of finished
  experiments stay the same.
- **Explanation results are recomputed** when `configs/explain/default.yaml` changes, because the
  notebook compares the saved explain config. That costs ~5 min per seed.
- **Old PlantVillage zip:** the full GitHub repo zip is 2.5 GB (colour + grayscale + segmented); only
  `raw/color` is used.
- `C:\nonexistent\tables` on the owner's machine is an empty leftover from a test; harmless.

---

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 14 Oct | ExecuTorch (not LiteRT) for mobile export | pure-PyTorch path; the PT2E INT8 graph evaluated on desktop is the same graph that ships |
| 14 Oct | Pixel 6a primary benchmark device; Android only (no iOS) | owner's devices |
| 14 Oct | Training on Google Colab (free T4) | owner's compute |
| 14 Oct | MobileNetV3 head replaced by GAP + single Linear | the stock hidden 1×1 conv breaks the closed form |
| 16 Oct | All of PlantDoc (train + test) is the field test set; class map verified against the real folders | never train on field data |
| 17 Oct | Feature extraction 10 → 6 epochs | val macro-F1 plateaued by epoch 6 on `mnv3_fe` seed 0 (validation only) |
| 17 Oct | ROAD reimplemented on GPU; ROAD sample 100 images × 4 percentiles; Score-CAM 50 images | explain time per seed fell from ~15.5 to ~5 min |
| 17 Oct | Per-epoch resume + settings fingerprint + single Colab queue | free-tier disconnects; no manual re-runs |
| 18 Oct | **Scope cut 1:** dropped ResNet-50, EfficientNet-B3, DeiT-S | over 20 GPU-hours of comparison rows |
| 18 Oct | **Scope cut 2:** dropped knowledge distillation, the teacher and fine-tuned baselines; compression = pruning 50/70% + INT8 on the finished `mnv3_fe`; scratch CNN kept as a single run (original 30-epoch settings, so its fingerprint still matches) | remaining GPU work had to fit ~1–1.5 hours |
| 18 Oct | Score-CAM only for uncompressed models; INT8 explained by closed-form Grad-CAM++ | cost, and INT8 has no gradients (mirrors the phone) |
| 20 Oct | **App = offline PWA with ONNX Runtime Web**, one codebase for every phone and desktop; ExecuTorch kept only for the Pixel 6a research benchmark | the app must run on all mobile devices (Android and iPhone, any model) and on desktop |
| 20 Oct | App preprocessing reproduces albumentations/cv2 exactly (fixed-point bilinear) and the heatmap reproduces `closed_form.py`; both are tested | the app must see and explain exactly what the evaluated model sees |
| 20 Oct | Unsure threshold = smallest confidence whose accepted **val** predictions reach 99% accuracy | rule 3; val is lab data, so the app also shows measured field accuracy |
| 20 Oct | Added `mnv3_ft_field` (full fine-tune, field-style augmentation) as an opt-in experiment; existing fingerprints unchanged (checked) | 22% field accuracy makes the current model unfit for real farms |
| 20 Oct | No AI attribution trailers in commits (rule 8 confirmed by the owner) | owner's choice |
| 20 Oct | Baselines retrained and the Phase 4 queue run on an RTX 5070 Ti laptop (unchanged configs; manifest paths and hashes verified) | free Colab GPU quota exhausted |
| 20 Oct | Added 8-epoch fine-tune variants `mnv3_fe_p50_ft8` / `p70_ft8` (the 2-epoch variants are kept) | the 2-epoch fine-tune was still improving on val when it stopped |
| 21 Oct | Ran `mnv3_ft_field` (3 seeds) | test whether field-style augmentation narrows the lab → field gap (it barely does) |
| 25 Oct | Demo site (model picker, Results, Explainability, About) deployed on Render at https://crop-disease-xai.onrender.com/ | public showcase; the model runs in the visitor's browser |
| 25 Oct | Name check as git hooks + GitHub Action; all branches merged into `main` | keep AI attribution out; one organized branch |

**Limitations this creates (state them in the paper):** compression is studied on one architecture
whose backbone was trained by feature extraction (not full fine-tuning); there is no distillation
stage; the scratch CNN is a single run.
