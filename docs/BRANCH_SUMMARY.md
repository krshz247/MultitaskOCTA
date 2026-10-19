# Branch `fix/thesis-consistency` — summary for review

Makes the repository runnable on current hardware (Apple Silicon / CPU / CUDA), fixes bugs that made the training objective differ
from the method described in the thesis, adds a leakage-free evaluation protocol and the experiments around it. **Not pushed / no PR
opened yet.** `git log master..HEAD` lists 36 commits; `docs/METHOD.md` is the reference for how the code implements the thesis.

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

## Experiments and findings (full detail in the author's local experiment log, which is not part of the repository)
Segmentation is accurate and stable; the segmentation branches bring no measurable classification benefit once the baseline is regularised; the field of view is not
what the networks use; vessel disconnection is not a disease cue; dense capillary labels make soft-clDice collapse; clDice share 0.8 breaks training; dropping fragments < 30 px
repairs connectivity metrics for free. Thesis numbers are not treated as targets.

## Review checklist
* `python -m unittest discover -s tests` (13 tests) — passes.
* `python run_experiment.py configs/joint_cv40.yaml --dry-run` prints the commands.
* Files of interest: `octa/losses.py`, `octa/joint_training.py`, `octa/metrics.py`, `octa/data/augment.py`, `Preprocessing/build_processed_octa500.py`, `docs/METHOD.md`.
* Large artefacts (`runs*/`, `Datasets/`, `kaggle_upload/`, virtual environments) are git-ignored.
