# Contributing to crop-disease-xai

Thanks for your interest. Bug reports, fixes, documentation and new experiments are all welcome.
This is a research codebase, so a few rules exist to keep the results trustworthy; please read them
before opening a pull request.

## Ways to help

- **Report a bug** or a wrong number: open an issue with steps to reproduce (for the web app: the
  browser, device and, if you can share it, the photo).
- **Improve the web app** (`web/`): accessibility, performance on low-end phones, translations.
- **Review the remedies** in `knowledge/remedies.yaml`. They are general guidance that has not yet
  been checked by a plant pathologist; corrections with a source are especially valuable.
- **Add field photos or lesion masks** of supported crops (with permission to publish them), so the
  localisation metrics can be computed.
- **Run new experiments**: please open an issue first and state the GPU cost.

## Development setup

```bash
git clone https://github.com/ranjanvrma/crop-disease-xai.git
cd crop-disease-xai
python -m venv .venv
.venv/bin/python -m pip install -e ".[train,compress,dev,web]"   # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q
git config core.hooksPath .githooks                               # enables the commit checks

cd web
npm ci
npm run dev            # http://localhost:5173
npm test               # JavaScript vs. Python parity (run web/test/make_fixtures.py first) and header checks
npm run build && npm run test:e2e   # browser tests at phone, tablet and desktop sizes
```

The Python unit tests use small synthetic images, so they need no datasets. The full pipeline,
datasets and Colab workflow are described in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Research rules

These are not style preferences; a pull request that breaks one will not be merged.

1. **Never fabricate or hand-edit results.** Every number comes from `results/metrics/**.json`,
   written by the scripts; tables are regenerated with `python scripts/make_tables.py`.
2. **Never use the field test set (PlantDoc, own photos) for training, tuning or model selection.**
   `src/cropxai/data/guard.py` enforces this; do not work around it.
3. **Report three seeds** for any new trained result, or mark it `(n=1)`.
4. **Config-driven:** hyperparameters live in `configs/`, not in scripts.
5. **Do not hand-edit the Colab notebook:** edit `notebooks/build_colab_notebook.py` and regenerate.
6. **Cite real sources only.** Do not add a citation you have not read.

## Pull requests

- Keep each pull request focused on one change, and describe what it changes and how you tested it.
- Make sure `pytest` passes, and for web changes `npm test` and the browser tests.
- New behaviour needs a test; a bug fix should come with a test that fails without it.
- Match the style of the surrounding code (Python: type hints and docstrings as in `src/cropxai/`;
  JavaScript: plain ES modules, no frameworks).
- CI runs the tests and a repository check on every pull request.

## Licence

By contributing you agree that your contribution is released under the [MIT Licence](LICENSE).
Datasets keep their own licences; do not commit dataset images other than the small samples in
`web/public/samples/`.
