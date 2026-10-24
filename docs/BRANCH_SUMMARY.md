# Branch `fix/thesis-consistency` — summary for review

Makes the repository runnable on current hardware (Apple Silicon / CPU / CUDA), fixes bugs that made the training objective differ
from the method described in the thesis, adds a leakage-free evaluation protocol and the experiments around it. The branch is pushed to `origin`
and is the repository's default branch; `master` (the thesis-time code) is untouched and nothing was merged into it. `docs/METHOD.md` is the reference for how the code
implements the thesis, `docs/SUMMARY_FOR_SUPERVISORS.md` for what the experiments show, `docs/MODEL_CARD.md` for the released model.

## Bug fixes (change what is trained)
* **clDice term was always zero and then overwritten** (`octa/losses.py`): the vendored soft-clDice sliced `[:, 1:]` on a single channel, and the
  multi-task loss then replaced the segmentation loss by a plain 3-channel Dice. Now `Dice(FAZ) + Dice(background) + soft-clDice(vessel)`.
* Metrics were computed on `round(logits)`; class weights were computed but unused; the classifier was unfrozen one epoch too early
  (`epoch >= startpoint` vs `>`); the "best" checkpoint kept the epoch with the highest validation loss; `.cuda()` was hard-coded.
* Signed-distance ground truth lost its inside half (PNG clipping).

## Reproducibility and environment
* `requirements-lock.txt` (Apple Silicon / CPU), `requirements-kaggle.txt`, device selection `cuda > mps > cpu`, `Preprocessing/build_processed_octa500.py`
  (builds the whole training set from raw OCTA-500; the MATLAB script produced a different layout and was not runnable), git commit + arguments stored with every run.
* Kaggle notebook and bundle script (`kaggle/`).

## Evaluation protocol
* Class-stratified 5-fold cross-validation (`--fold`), validation-based epoch selection, all scans evaluated (`--full_eval`), per-image class-wise Dice / Jaccard / clDice,
  topology metrics, balanced accuracy, per-class recall, per-field-of-view metrics, thesis-style 3-channel Dice, `aggregate_folds.py`, pooled out-of-fold bootstrap intervals.
* Classification-only baseline under the same protocol (`classification_baseline.py`); the old `classification.py` (ResNet50, unweighted loss, random split) is in `legacy/`.

## Extensions (all off by default, defaults reproduce the thesis method)
`--scheduler`, `--strong_aug` (mask-consistent), `--label_smoothing`, `--cls_dropout`, `--weight_decay`, `--vessel_bce`, `--exclusion`, `--aux_channels faz`,
`--cldice_alpha`, `--vessel_weight`, `--stratify_fov`, `--bg_dice 0`, builder options `--projection`, `--vessel`, `--fov_mode`.

## Experiments and findings (single-page version: `docs/SUMMARY_FOR_SUPERVISORS.md`; scripts and reports: `analysis/`)
Segmentation is accurate and stable (FAZ Dice ≈ 0.93 with a 0.5 threshold; 0.88–0.93 with the 3-class argmax depending on the run; vessel Dice ≈ 0.79, clDice ≈ 0.86). Each of the three main recipes (joint model, joint model on shuffled masks, plain classifier,
all regularised) was trained three times: classification reaches ≈ 0.75–0.77 balanced accuracy; joint model and plain classifier are equal within the run-to-run spread (about ±0.03) and the shuffled-mask control is lower on average but not significantly so, so the segmentation tasks
bring **no measurable classification benefit**; the auxiliary losses act as a regulariser. The choice of the reported epoch matters (the pipeline score uses segmentation metrics that are meaningless for the
shuffled-mask variant), so classification comparisons are also given under one classification-only rule. The field of view is not what the networks use, though both are sensitive to the image frame;
vessel disconnection is not a disease cue; dense capillary labels make soft-clDice collapse; clDice share 0.8 breaks training; dropping fragments < 30 px repairs connectivity metrics for free.
Thesis numbers are not treated as targets.

## Inference
`octa/predict.py`: image in, class probabilities and FAZ / vessel masks out (average of the five fold models); `docs/MODEL_CARD.md` describes the model, its data and its limits.

## Review checklist
* `python -m unittest discover -s tests` (33 tests; `SKIP_SLOW_TESTS=1` for the fast ones) — passes; CI (`.github/workflows/tests.yml`) runs lint and the fast tests.
* `python run_experiment.py configs/joint_cv40.yaml --dry-run` prints the commands.
* Files of interest: `octa/losses.py`, `octa/joint_training.py`, `octa/metrics.py`, `octa/data/augment.py`, `octa/predict.py`, `Preprocessing/build_processed_octa500.py`, `docs/METHOD.md`.
* Large artefacts (`runs*/`, `Datasets/`, `kaggle_upload/`, virtual environments) are git-ignored.
