# CropXAI web app

An offline, installable web app (PWA) that checks a crop-leaf photo and explains its answer. One
codebase is built for any recent Android or iPhone browser and for desktop (Chrome, Edge, Firefox,
Safari), using only standard web features (WebAssembly, canvas, service workers). Everything runs on the device: the photo never leaves it, and after the first visit the app
works without a connection.

## What it shows

- **Diagnosis** with confidence, or **"Not sure"** below a threshold tuned on the validation split.
- **"Not a supported crop?" warning** when the photo is unlike every training leaf: a Mahalanobis
  distance on the pooled features the model already outputs (`models/<run>/ood.json`, fitted by
  `scripts/fit_ood.py`, threshold = 99% of PlantVillage validation photos pass). Catches non-plant photos
  well, misses a fair share of leaves of unknown crops, and flags most field photos; the user can still
  ask to see the guess. Measured numbers are on the Explainability page.
- **Supported crops, photo tips and sample photos** (including a non-crop) on the Diagnose page.
- **Where the model looked:** Grad-CAM++ heatmap over the photo, computed on the device from one
  forward pass (the closed form in `src/cropxai/explain/closed_form.py`, ported to `src/xai.js`).
  "Show evidence for" switches the heatmap to the second or third candidate class.
- **Check it yourself:** the visible signs of the predicted disease (from `knowledge/remedies.yaml`),
  so the user can judge whether the highlighted area shows them.
- **Is the heatmap honest?** A live deletion test: the brightest 20% of the image is hidden and the
  model is asked again, then the same with a random 20%. A faithful heatmap hurts the score more.
- **Under the hood:** each preprocessing step with timings, the exact split of the class score into
  bias plus per-channel contributions (with the top channels' feature maps), the Grad-CAM++ formula,
  all top answers, and the model's provenance.
- **What to do:** cultural, organic and chemical options for the predicted class, with a disclaimer.
  The remedies have **not yet been reviewed by a plant pathologist** (`reviewed: false`).

## Demo website (Render)

The site has four pages: **Diagnose** (the app, with a picker for every exported model),
**Explainability** (methods, faithfulness, stability under compression, calibration and the
lab-to-field gap), **Results** (Tables 1–3) and **About** (remedies for all 38 classes).
All numbers come from `web/public/data/site.json`, generated from `results/` by
`scripts/export_site_data.py`.

The model bundles in `web/public/models/<run>/` are committed, so a static host only needs Node:
`render.yaml` at the repo root builds `web/` with `npm ci && npm run build` and publishes `web/dist`.
Only the default model is downloaded on the first visit; the others when first chosen.

To refresh the models and data:

```bash
python scripts/export_web.py --config configs/experiment/mnv3_ft_field.yaml --seed 0 --out web/public/models/mnv3_ft_field
python scripts/export_site_data.py
```

## Run it

```bash
# 1. Model bundle -> web/public/model/   (CPU only, about a minute)
pip install -e ".[train,compress,web]"
python scripts/export_web.py --config configs/experiment/mnv3_fe.yaml --seed 0    # needs the trained checkpoint
python scripts/export_web.py --config configs/experiment/mnv3_fe.yaml --demo      # or: untrained, for development

# 2. App
cd web
npm ci
npm run dev        # http://localhost:5173
npm run build      # static site in web/dist/, deployable to any static host (GitHub Pages, Netlify, ...)
```

The Colab notebook also exports the bundle (`WEB_MODEL` in its settings cell) to
`MyDrive/cropxai/web_model/`; copy those four files into `web/public/model/`.

To install it on a phone, open the deployed site and choose "Add to Home screen" (Android Chrome
offers "Install app"; on iPhone use Safari's Share menu). The site must be served over HTTPS
(or `localhost`) for offline mode and installation to work.

## Tests

```bash
python web/test/make_fixtures.py     # reference outputs from the Python pipeline
cd web
npm test                             # parity: preprocessing, ONNX outputs, heatmaps, score split
npm run build && npm run test:e2e    # browser tests: 4 phone/tablet sizes + desktop, and offline mode
```

`npm run test:e2e` needs a demo or real bundle in `public/model/`. If Playwright's own Chromium is not
installed, point `CHROMIUM_PATH` at a Chromium binary.

What the parity test checks against the research code, on the same seeded model and image:
preprocessing (albumentations resize and crop: within one intensity level on fewer than 0.5% of values,
because OpenCV's optimised kernels round a few values differently), ONNX Runtime Web logits and feature
maps (< 1e-3), closed-form Grad-CAM++ heatmaps (< 1e-4), and the score decomposition (adds up to the logit).

## Limits

- The browser tests run in Chromium with phone-sized screens and touch. Real Safari on iPhone and
  low-end Android phones still need a manual check (speed and memory).
- The model runs in float32 on the CPU through WebAssembly (about 12 MB of weights plus about 14 MB of
  runtime, cached after the first visit). An INT8 web build is not done yet.
- Photos with a short side over 1020 px are first scaled down by the browser, then resized exactly
  like the evaluation pipeline; the evaluation code itself resizes in one step.
- The model was trained on lab photos (PlantVillage). The app shows the measured field accuracy from
  `results/metrics/accuracy/` when it is available, and says so when it is not.
