# Training on a Kaggle GPU

Kaggle gives about 30 GPU-hours per week for free (T4 x2 or P100), notebooks run up to 9–12 h and keep running when your browser is closed if you use *Save & Run All (Commit)*.

1. **Bundles** (on your computer, from the repo root): `kaggle/make_bundles.sh` creates in `kaggle_upload/`
   * `multitaskocta_code.zip` — the committed code (commit first; it uses `git archive HEAD`)
   * `octa500_b5_subset.zip` — the raw OCTA-500 files the builder needs for B5 / large vessels (about 60 MB zipped)
2. **Datasets:** kaggle.com → Datasets → New Dataset → upload each zip (Kaggle unzips it) → keep them private.
3. **Notebook:** Code → New Notebook → File → Import notebook → `kaggle/MultitaskOCTA_kaggle.ipynb`.
   Settings: *Accelerator* GPU T4 x2 (or P100), *Internet* On (pip + pretrained weights), then *Add Input* → the two datasets.
4. **Edit the parameter cell**: `FOLDS`, `EPOCHS`, `STARTPOINT`, `SCRIPT`, `FOV_MODE`, `EXTRA` (e.g. `["--scheduler", "onecycle"]`), `RUN_NAME`. Several folds can be trained in one session by listing them in `FOLDS`.
5. **Save Version → Save & Run All (Commit)**. When it finishes, download `<RUN_NAME>.zip` from the *Output* tab, unzip it into `runs/` and use `aggregate_folds.py`, `plot_runs.py` and TensorBoard as for local runs.

Notes
* The notebook sets `CUDA_VISIBLE_DEVICES=0`: with two GPUs `DataParallel` would split the batch of 4 and change the BatchNorm statistics.
* Smoke test: set `LIMIT = 30`, `EPOCHS = 2`, `FOLDS = [0]` (about 2 minutes).
* Output limit is about 20 GB; one run is ≈ 0.5 GB (best checkpoint + figures).
* Larger batches are possible on a GPU but change the method (`--batch_size 4` is the thesis value).

## Full runs (checklist)

Before every full run
1. `git status` clean, then `kaggle/make_bundles.sh` — it packs **the last commit** (`git archive HEAD`). In Kaggle open the code dataset → *New Version* → upload `kaggle_upload/multitaskocta_code.zip` (the data dataset does not change). Wait until the notebook shows the new version number under *Input*.
2. Notebook settings: *Accelerator* GPU T4 x2 (or P100), *Internet* On, *Persistence* none; both datasets attached.
3. Edit **only the first cell** (parameters), then *Save Version → Save & Run All (Commit)*; the run continues without your browser. Output: `/kaggle/working/<RUN_NAME>.zip` (download from the version's *Output* tab).

Parameter sets (first cell)
| goal | FOLDS | EPOCHS / STARTPOINT | EXTRA | RUN_NAME |
|---|---|---|---|---|
| timing run | `[0]` | 40 / 8 | `["--full_eval"]` | `time_fold0` |
| joint model, 5-fold CV (reported protocol) | `[0,1,2,3,4]` | 40 / 8 | `["--full_eval"]` | `ILM_OPL_large_cv40` |
| joint model, thesis length | `[0,1,2,3,4]` | 100 / 20 | `["--full_eval"]` | `ILM_OPL_large_100` |
| regularised joint model | `[0,1,2,3,4]` | 40 / 8 | `["--full_eval","--strong_aug","--label_smoothing","0.1","--cls_dropout","0.3","--weight_decay","0.01"]` | `ILM_OPL_large_cv40reg` |
| ablation R4: no segmentation loss | `[0,1,2,3,4]` | 40 / 8 | regularised list + `["--seg_loss_weights","0","0","0"]` | `ILM_OPL_large_cv40reg_noseg` |
| ablation R3: shuffled masks | `[0,1,2,3,4]` | 40 / 8 | regularised list + `["--shuffle_masks"]` | `ILM_OPL_large_cv40reg_shufmask` |
| ablation R2: stop-gradient | `[2,3,4]` (folds 0–1 exist locally) | 40 / 8 | regularised list + `["--detach_clf_features"]` | `ILM_OPL_large_cv40reg_detach_f234` |
| classification-only baseline | `[0,1,2,3,4]` | 40 / 8 | `["--full_eval"]`, and `SCRIPT = "classification_baseline.py"` | `ILM_OPL_large_cls40` |
| common field of view | `[0,1,2,3,4]` | 40 / 8 | `["--full_eval"]` with `FOV_MODE = "center3mm"` | `ILM_OPL_large_fov3_cv40` |

Repeats with different seeds (to measure how much a single run varies)
* Add `SEED` to the first cell (default 2021; the cross-validation folds do not change with the seed) and give every run a name with the seed, e.g. `ILM_OPL_large_cv40reg_s1`, `..._s2`, `..._s3`. One 5-fold run of 40 epochs takes about 3 hours, so one seed per notebook version (Kaggle allows two GPU sessions at the same time).
* Runs made with the code of 2026-10-21 or later also log the class probabilities of the test scans at every evaluated epoch (`test_probs` and `test_names` in `results_fold*.json`). With them all variants can be compared under ONE epoch rule, pooled over the folds, without checkpoints: `python analysis/oof_rule.py --rule val_bal --ref C_plain --group A_joint runs/<A>_s4 --group B_shuffled runs/<B>_s3 --group C_plain runs/<C>_s3`. `--cls_ckpt balanced_acc` additionally saves `best_cls_fold<k>.pt` (a checkpoint chosen by validation balanced accuracy); it is not needed for that analysis and adds about 1.6 GB to the output.
* After downloading the zips into `runs/`: `python aggregate_seeds.py --group joint_reg runs/<A>_s1 runs/<A>_s2 runs/<A>_s3 --group cls_reg runs/<B>_s1 runs/<B>_s2 runs/<B>_s3` — mean ± std over seeds and paired differences per seed.

Test of the FAZ / background option (`--faz_bg_exclusion`)
* Why: in the seed-2 and seed-3 joint runs the background channel also outputs ~1.0 inside the FAZ, so the 3-class argmax loses FAZ pixels (`analysis/faz_channel_check.txt`). The option adds `weight * mean(background probability inside the ground-truth FAZ)` to the segmentation loss; default 0 = off, nothing else changes.
* Runs (two versions, one per seed, fold 0 only, about 35 minutes each): in the single-run notebook set `FOLDS = [0]`, `EPOCHS, STARTPOINT = 40, 8`, `SCRIPT = "faz_vessel_seg_cls.py"`, `EXTRA = ["--full_eval","--strong_aug","--label_smoothing","0.1","--cls_dropout","0.3","--weight_decay","0.01","--faz_bg_exclusion","1.0"]`, `SEED = 2` (then `3`), `RUN_NAME = "ILM_OPL_large_cv40reg_fazbg_f0_s2"` (then `..._s3`). One weight (1.0, the size of the Dice terms), no tuning.
* Controls (same seeds, same platform, no option, fold 0 test, best-validation epoch): seed 2: FAZ Dice argmax 0.914, threshold 0.947, vessel Dice 0.790; seed 3: argmax 0.889, threshold 0.952, vessel Dice 0.791.
* Success criteria, fixed before the runs: (1) mean background probability inside the ground-truth FAZ below 0.5 (controls ≈ 1.0; `python analysis/faz_channel_check.py runs/<run> --data ...`); (2) the gap between thresholded and argmax FAZ Dice shrinks to ≤ 0.01 in both seeds (controls 0.033 and 0.063); (3) vessel Dice within 0.01 of the control. Report the result whichever way it goes; classification is only reported, not claimed.

Planning
* Kaggle sessions are limited (about 9–12 h) and the weekly GPU quota is 30 h. Run the *timing run* first and read the minutes per fold from the output; if 5 folds do not fit into one session, split the folds over several versions with different `RUN_NAME` and put the unzipped folders together in one directory afterwards (`aggregate_folds.py` reads every `results_fold*.json` in a directory).
* Results are comparable with the Mac runs in accuracy, not bit for bit (different hardware).
* After downloading: unzip into `runs/`, then `python aggregate_folds.py runs/<RUN_NAME>`.
