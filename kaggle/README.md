# Training on a Kaggle GPU

Kaggle gives about 30 GPU-hours per week for free (T4 x2 or P100), notebooks run up to 9–12 h and keep running when your browser is closed if you use *Save & Run All (Commit)*.

1. **Bundles** (on your computer, from the repo root): `kaggle/make_bundles.sh` creates in `kaggle_upload/`
   * `multitaskocta_code.zip` — the committed code (commit first; it uses `git archive HEAD`)
   * `octa500_b5_subset.zip` — the raw OCTA-500 files the builder needs for B5 / large vessels (about 60 MB zipped)
2. **Datasets:** kaggle.com → Datasets → New Dataset → upload each zip (Kaggle unzips it) → keep them private.
3. **Notebook:** Code → New Notebook → File → Import notebook → `kaggle/MultitaskOCTA_kaggle.ipynb`.
   Settings: *Accelerator* GPU T4 x2 (or P100), *Internet* On (pip + pretrained weights), then *Add Input* → the two datasets.
4. **Edit the parameter cell**: `FOLD`, `EPOCHS`, `STARTPOINT`, `EXTRA` (e.g. `["--scheduler", "onecycle"]`), `RUN_NAME`. Run one fold per notebook version, or loop over folds in a copy of the cell.
5. **Save Version → Save & Run All (Commit)**. When it finishes, download `<RUN_NAME>.zip` from the *Output* tab, unzip it into `runs/` and use `aggregate_folds.py`, `plot_runs.py` and TensorBoard as for local runs.

Notes
* The notebook sets `CUDA_VISIBLE_DEVICES=0`: with two GPUs `DataParallel` would split the batch of 4 and change the BatchNorm statistics.
* Smoke test: set `LIMIT = 30`, `EPOCHS = 2` (about 2 minutes).
* Output limit is about 20 GB; one run is ≈ 0.5 GB (best checkpoint + figures).
* Larger batches are possible on a GPU but change the method (`--batch_size 4` is the thesis value).
