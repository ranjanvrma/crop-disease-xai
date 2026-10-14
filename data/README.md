# Data

Nothing in this folder is committed except this file. The split is pinned by
`splits/manifest_v1.csv` (relative paths + dHash per image), which **is**
committed.

## Expected layout (under `$CROPXAI_DATA`, default `data/`)
```
plantvillage/color/<Class___name>/*.jpg     # 38 class folders
plantdoc/train/<PlantDoc class>/*.jpg
plantdoc/test/<PlantDoc class>/*.jpg
field/images/<PlantVillage class>/*.jpg      # your own photos
field/masks/<PlantVillage class>/<same stem>.png   # lesion masks (optional)
```

## Getting the datasets
- **PlantVillage**: use the `color` variant from `spMohanty/PlantVillage-Dataset`
  on GitHub (`raw/color`), or a Kaggle mirror of the same. Check the class
  folder count is 38.
- **PlantDoc**: the classification folders from `pratikkayal/PlantDoc-Dataset`
  on GitHub (`train/` and `test/`). **Both** go to the field test set; neither
  is ever trained on.
- Before the first run, check that the PlantDoc folder names match the keys in
  `configs/data/plantdoc_to_plantvillage.yaml`. The build fails loudly if one
  doesn't match.

For Colab, zip each dataset once (`plantvillage.zip`, `plantdoc.zip`,
`field.zip`, each containing the folder at the top level), put the zips in
`MyDrive/cropxai/raw/`, and the bootstrap notebook will unzip them to local disk.

## Own field photos
- Folder name = the PlantVillage class it shows (exact spelling, e.g.
  `Tomato___Early_blight`). Get labels confirmed by someone who knows plants.
- Lesion masks: export from CVAT / Label Studio as binary PNGs (lesion = 255),
  same file stem as the photo.
- Add new photos at any time and re-run `scripts/prepare_data.py`. Field photos
  only ever land in `field_test`, so adding them never changes the
  PlantVillage train/val/test split. (Exception: a PlantVillage image that
  near-duplicates a new field photo is removed from the pool, so re-check
  `manifest_v1_summary.json`.)

## What prepare_data does
1. Scans the three sources and maps every label to a PlantVillage class name.
2. Hashes every image (dHash). Unreadable files are dropped and counted.
3. Removes exact duplicates within the training pool and within the held-out pool.
4. Removes PlantVillage images within dHash distance ≤ 4 of any held-out image.
5. Splits PlantVillage 80/10/10, stratified by class, with fixed seed 0.
6. Runs the field-leak guard (also re-run every time a loader is built).
