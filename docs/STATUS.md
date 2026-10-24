# MultitaskOCTA — status

Last updated: 2026-10-24 — wrap-up and consistency check: test count 36 confirmed by a run, p-values and the prediction-equivalence check stored as scripts with outputs, FAZ caveat and data-terms limitation stated.
Branch: fix/thesis-consistency · Commit: 8d9fdac (the commit this status describes; the commit that adds this file sits on top of it)

## Current state
- Works:
  - Training and evaluation from raw OCTA-500 to per-fold results: `Preprocessing/build_processed_octa500.py`, `faz_vessel_seg_cls.py`, `classification_baseline.py`, `run_experiment.py` + `configs/*.yaml` (README "Usage").
  - 36 tests pass, including the golden regression runs at bit-identical tolerance `GOLDEN_TOL=1e-9` (`tests/`; run on commit `8d9fdac`, 2026-10-24: `Ran 36 tests ... OK`; the fast subset `SKIP_SLOW_TESTS=1` runs 36 with 9 skipped); CI runs lint and the fast tests (`.github/workflows/tests.yml`, green on `87d7fe4`).
  - Inference: `octa/predict.py`, `tests/test_predict.py`, `docs/MODEL_CARD.md` (commit `c924ba8`); one fold model's predictions equal the stored out-of-fold probabilities (max difference 5.4e-07 over all 73 fold-0 test scans; `analysis/predict_equivalence_check.py` → `analysis/predict_equivalence_check.txt`, needs the local data and weights). Weights are release assets of v1.0: https://github.com/krshz247/MultitaskOCTA/releases/tag/v1.0.
  - Runs made with the code of commit `40afc82` or later log test-scan probabilities at every evaluated epoch (`test_probs`, `test_names` in `results_fold*.json`); `analysis/oof_rule.py` pools them under one epoch rule (`tests/test_oof_rule.py`).
- Broken or untested:
  - The golden regression job is manual-only on GitHub (`workflow_dispatch`): the tiny trainings diverge on GitHub's CPUs (commit `11bae19`, `.github/workflows/tests.yml`).
  - `legacy/` scripts are not part of the pipeline; several do not import (`legacy/README.md`).
  - Runs with logged test probabilities (needed for pooled tests under one epoch rule) exist only for the shuffled-mask and classification-only seed-3 runs; the joint recipe has none (`analysis/oof_rule.py`). This limits the pooled comparison of all three recipes under one epoch rule: only shuffled masks vs classification-only (one run each, `analysis/oof_rule.py`; intervals include 0) is possible; the other comparisons use fold-mean epoch rules (`analysis/epoch_selection_results.txt`). No further run is planned.
  - The FAZ channel and the background channel are not mutually exclusive in the loss; in two of four joint runs this lowers the argmax FAZ Dice (`analysis/faz_channel_check.txt`). The option `--faz_bg_exclusion 1.0` (`octa/losses.py`, off by default) removes that error but creates argmax false positives outside the FAZ; on fold 0, seeds 2 and 3 the argmax FAZ Dice does not improve (`analysis/faz_bg_exclusion_test.txt`). Not fixed; the thresholded FAZ Dice (0.93 in every run) is reported.

## Environment
- Python 3.12 (README "Setup"); `requirements-lock.txt` pins the versions tested on Apple Silicon and CPU (e.g. `torch==2.14.1`, `timm==0.9.2`, `segmentation-models-pytorch==0.3.3`, `albumentations==2.0.8`, `kornia==0.8.3`); `requirements-kaggle.txt` (Kaggle image) pins only `segmentation-models-pytorch==0.3.3` and leaves `timm`, `albumentations`, `kornia` unpinned; `requirements.txt` pins `segmentation-models-pytorch==0.3.3` and `torchmetrics==1.6.3`.
- Device: `cuda` > `mps` > `cpu`, `--device` overrides (`octa/joint_training.py`, `octa/classification_training.py`, `octa/predict.py`); the hard-coded `.cuda()` calls were removed (commit `8a93ad0`; no `.cuda()` remains under `octa/`). Training runs were done on Apple Silicon (MPS) and on Kaggle GPUs (CUDA) (`results_fold*.json` record the device; `docs/RESULTS.md`).
- Check of the note "albumentations, segmentation_models_pytorch and kornia are unpinned; device selection may need fixes on MPS" (from the Vision AI chat, 2026-10-23): partly outdated. They are pinned in `requirements-lock.txt` (see above) and only partly in `requirements.txt` / `requirements-kaggle.txt`; MPS is supported and was used for the Mac runs.

## Results (only from committed logs or result files)
All on OCTA-500, B5 projection, 364 labelled scans (251 Normal / 64 DR / 49 AMD), class-stratified 5-fold cross-validation, 40 epochs per fold.

| Experiment | Metric | Value | Source |
|---|---|---|---|
| Joint model, regularised (4 runs) | FAZ Dice (3-class argmax) / vessel Dice / vessel clDice | 0.908 ± 0.026 / 0.788 ± 0.006 / 0.859 ± 0.006 | `docs/RESULTS.md` |
| Same, four runs | FAZ Dice with a 0.5 threshold on the FAZ channel | 0.937 / 0.925 / 0.933 / 0.936 | `analysis/faz_channel_check.txt` |
| Joint model, per run | pooled out-of-fold balanced accuracy | 0.747 / 0.758 / 0.789 / 0.771 (mean 0.766; ensemble of 4: 0.792) | `analysis/ensemble_runs_results.txt` |
| Joint model, shuffled masks, per run | pooled out-of-fold balanced accuracy | 0.762 / 0.659 / 0.748 / 0.692 (mean 0.715; ensemble 0.782) | `analysis/ensemble_runs_results.txt` |
| Classification-only ResNeSt50, regularised, per run | pooled out-of-fold balanced accuracy | 0.747 / 0.762 / 0.771 / 0.726 (mean 0.751; ensemble 0.801) | `analysis/ensemble_runs_results.txt` |
| Epoch rule, joint (4 runs) | mean over folds of balanced accuracy: pipeline score / best validation balanced accuracy / last epoch | 0.765 / 0.751 / 0.738 | `analysis/epoch_selection_results.txt` |
| Epoch rule, shuffled masks (4 runs) | same three rules | 0.717 / 0.737 / 0.736 | `analysis/epoch_selection_results.txt` |
| Epoch rule, classification-only (4 runs) | same three rules | 0.750 / 0.754 / 0.736 | `analysis/epoch_selection_results.txt` |
| AUROC (one-vs-rest, macro) | ensemble of four runs: joint / shuffled masks / classification-only; ECE of the ensembles | 0.930 / 0.910 / 0.932 (95 % intervals overlap); ECE 0.154 / 0.199 / 0.149 | `analysis/classification_report.txt` |
| Statistical power | SE of a paired balanced-accuracy difference (364 scans); smallest detectable difference, 80 % power | 0.038; 0.106 | `analysis/power_seeds_results.txt` |
| Image-frame ablation | balanced accuracy, outer 30 % replaced by grey: joint / classification-only (unchanged: 0.747 / 0.747) | 0.396 / 0.638 | `analysis/gradcam_results.txt` |
| Biomarkers from predicted masks | balanced accuracy of mask-derived features, regularised joint model | 0.546–0.611 depending on classifier and feature set (logistic regression on predicted-mask features 0.594, with the field of view added 0.611) | `analysis/biomarker_predicted_cv40reg.txt` |

Reading (see `docs/SUMMARY_FOR_SUPERVISORS.md`): the joint model and the plain regularised classifier agree within the run-to-run spread (0.766 vs 0.751, Welch p = 0.32 on the run values; `analysis/run_comparison_tests.txt`, four runs each); the shuffled-mask control is lower on average (0.715 vs 0.766, p = 0.12, not significant; under two classification-only epoch rules all three recipes lie within 0.017: 0.751 / 0.737 / 0.754 and 0.738 / 0.736 / 0.736), so the effect of correct masks can neither be shown nor excluded. No measurable benefit of the segmentation tasks over a plain regularised classifier. The stop-gradient and no-segmentation-loss ablations are single runs (`docs/RESULTS.md`).

Reference, thesis PDF: Table 4.1 (segmentation, stated to be obtained by 5-fold cross-validation): joint RV+FAZ Jaccard 95.3, Dice 98.1, clDice 91.3; Table 4.2 (classification; the text shows train / validation / test confusion matrices and does not state cross-validation or the split ratio): accuracy 95.2 (RV+FAZ) vs 94.1 (FAZ-only) vs 89.3 (no segmentation). Re-runs are logged above, not mixed into this line. Accuracy and balanced accuracy are never to be compared across thesis and re-run: see `docs/CLAIMS.md`. The thesis' Dice averaging is not specified in its text (hypothesis: background-dominated micro-average); the thesis itself concludes that segmenting superficial vessels did not substantially improve classification (PDF p. 41).

## Skills demonstrated
| Skill | What was done | Evidence (file or commit) |
|---|---|---|
| Debugging a deep-learning training objective | Found that the clDice term was computed and then overwritten, plus metrics on rounded logits, unused class weights, classifier unfreezing off by one | `octa/losses.py`, commit `8a93ad0`; `docs/BRANCH_SUMMARY.md` |
| Multi-task segmentation + classification model (PyTorch) | Shared encoder, three decoders, classifier fed by decoder features | `octa/models.py`, `octa/joint_training.py`, `docs/METHOD.md` |
| Data pipeline | Python port of the MATLAB preprocessing; boundary heat maps and signed distance maps | `Preprocessing/build_processed_octa500.py`, commit `e080206` |
| Evaluation design and statistics | Class-stratified CV, pooled out-of-fold bootstrap, McNemar, paired bootstrap, power analysis | `analysis/oof_compare.py` (`5523a50`), `analysis/power_seeds.py` (`9062cf1`), `analysis/oof_rule.py` |
| Reproducible experiments | YAML configs, seeds, aggregation over seeds, git commit stored with every run | `run_experiment.py` (`1f12cce`), `aggregate_seeds.py` (`5fcf651`) |
| Testing and CI | Golden regression tests, unit tests, GitHub Actions | `tests/test_golden.py` (`cee3699`), `.github/workflows/tests.yml` (`70e3b9b`) |
| Cloud GPU training | Kaggle notebooks, bundle script, multi-run presets | `kaggle/MultitaskOCTA_kaggle_multirun.ipynb` (`cb32b83`), `kaggle/make_bundles.sh` (`f11c5d9`) |
| Error analysis and interpretation | Class activation maps, image-frame ablation, epoch-selection rules, FAZ/background channel analysis | `analysis/gradcam_check.py` (`9062cf1`), `analysis/epoch_selection.py` (`61a1aa4`), `analysis/faz_channel_check.py` (`87d7fe4`) |
| Packaging and model documentation | Inference module, model card, GitHub release with weights | `octa/predict.py`, `docs/MODEL_CARD.md` (`c924ba8`), release v1.0 |
| Technical writing | README for readers, method map, one-page summary, claims table with safe and unsafe wording | `README.md`, `docs/METHOD.md`, `docs/SUMMARY_FOR_SUPERVISORS.md`, `docs/CLAIMS.md` |

## Open issues
- Thesis-text checks done 2026-10-23 (`docs/CLAIMS.md`): the README previously said "the thesis said yes"; the thesis reported a gain from segmentation (89.3 → 94.1 / 95.2 %) but itself found the vessel part not substantial. The FAZ-only BSDA-Net arm was not re-run here, so no claim is made about FAZ-only segmentation.
- Every main recipe has four runs (one Mac run, Kaggle seeds 1 to 3); the README and `docs/RESULTS.md` tables cover all four. The stop-gradient and no-segmentation-loss ablations are single runs.
- Only 49 AMD scans; the AMD figures have wide uncertainty (`docs/MODEL_CARD.md`).
- Mac and Kaggle runs are mixed within a recipe (hardware differs; `results_fold*.json` record the device).
- A common classification-only epoch rule exists for runs made with the logging code only; earlier runs can be compared at fold-mean level (`analysis/epoch_selection.py`).
- FAZ/background exclusivity in the loss (`analysis/faz_channel_check.txt`). Affected by the FAZ/background flaw: the FAZ Dice on the 3-class argmax map (`faz_dice`: README 0.908 ± 0.026, the pipeline's best-epoch score, and the FAZ mask that `octa/predict.py` returns); not affected: the thresholded FAZ Dice (`faz_dice_thr`, 0.933 ± 0.005), the vessel metrics and the classification; the default setting (`--faz_bg_exclusion 0`) still has the flaw, because the option that targets it is not a fix (`analysis/faz_bg_exclusion_test.txt`). The released weights come from the seed-1 run, in which the background probability inside the FAZ was 0.09 on fold 0 (`analysis/faz_channel_check.txt`).

## Next steps
- Optional, not planned: a different FAZ/background remedy (smaller weight, boundary-confined penalty, softmax over the channels); the test of the current option is documented in `analysis/faz_bg_exclusion_test.txt`.
- Optional: one more joint run with the current code (it logs test probabilities), so that all three recipes can be compared pooled under one epoch rule (`analysis/oof_rule.py`).
- Hand `docs/CLAIMS.md` to the Main Professional project for CV and LinkedIn text; no local CV drafts.
- Then continue with the Vision AI roadmap (Stage 1).
