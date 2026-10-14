# crop-disease-xai

Crop-leaf disease detection with **Grad-CAM++** explanations, a staged
**compression** pipeline (knowledge distillation → structured pruning → INT8),
and **on-device** Android inference via ExecuTorch, including a closed-form,
gradient-free Grad-CAM++ that runs in a single forward pass.

**Research question:** can a compressed crop-disease model running on a phone
still give faithful Grad-CAM++ explanations, and how much does compression
change what the model looks at?

> **Status:** Phase 1 (scaffold + data). No models have been trained yet, so
> every results cell below is empty. Cells fill only from
> `results/metrics/**.json` written by real runs (3 seeds, mean ± std).

## Results

### Table 1 — Accuracy & efficiency
| Model | Params | Size (MB) | Top-1 (PV test) | Macro-F1 | Field acc. | Desktop latency | Pixel 6a latency |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

### Table 2 — Explanation quality
| Model | Deletion AUC ↓ | Insertion AUC ↑ | Avg. Drop ↓ | Inc. in Conf. ↑ | ROAD | Pointing game ↑ | IoU vs masks ↑ |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

### Table 3 — Stability under compression (vs. uncompressed parent)
| Variant | Spearman ↑ | SSIM ↑ | Top-10% IoU ↑ |
|---|---|---|---|
| — | — | — | — |

### Table 4 — On-device (Pixel 6a)
| Variant | Disk (MB) | Cold start | Latency median | Latency p95 | Heatmap latency | Peak memory | Battery / N inf. |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

## Datasets
- **PlantVillage** (training): lab images, 38 classes.
- **PlantDoc** (field test only): real-world photos, mapped onto shared
  PlantVillage classes via `configs/data/plantdoc_to_plantvillage.yaml`.
- **Own field photos + lesion masks** (field test only).

See `data/README.md` for download and layout.

## Quick start
Training runs on Google Colab: open `notebooks/colab_bootstrap.ipynb`.

Local (tests and data tooling):
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

## Repository layout
```
configs/     one YAML per experiment, composed via `inherits:`
src/cropxai/ utils · data · models · train · compress · explain · eval · export
scripts/     command-line entry points (take --config + key=value overrides)
splits/      committed split manifests (relative paths + image hashes)
results/     metrics JSON (source of truth) and figures
notebooks/   Colab bootstrap + results notebook that regenerates every table
research/    paper (LaTeX) and research notes
android/     Kotlin + CameraX app            (Phase 7)
web/         showcase + results dashboard    (Phase 8)
```
