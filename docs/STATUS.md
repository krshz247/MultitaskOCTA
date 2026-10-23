# MultitaskOCTA — status

Last updated: 2026-10-23 — status document and session wrap-up rule added (`CLAUDE.md`); link test line below; state after the v1.0 release, README rework, per-epoch probability logging and root clean-up.

link test 2026-10-23

## Current state
- Works:
  - Training and evaluation from raw OCTA-500 to per-fold results: `Preprocessing/build_processed_octa500.py`, `faz_vessel_seg_cls.py`, `classification_baseline.py`, `run_experiment.py` + `configs/*.yaml` (README "Usage").
  - 33 tests pass, including golden regression runs at bit-identical tolerance `GOLDEN_TOL=1e-9` (`tests/`, run on commit `c9cf727`); CI runs lint and the fast tests (`.github/workflows/tests.yml`, green on `87d7fe4`).
  - Inference: `octa/predict.py`, `tests/test_predict.py`, `docs/MODEL_CARD.md` (commit `c924ba8`); one-fold predictions equal the stored out-of-fold probabilities to 5e-7 (checked 2026-10-20, not stored as a file). Weights are release assets of v1.0: https://github.com/krshz247/MultitaskOCTA/releases/tag/v1.0.
  - Runs made with the code of commit `40afc82` or later log test-scan probabilities at every evaluated epoch (`test_probs`, `test_names` in `results_fold*.json`); `analysis/oof_rule.py` pools them under one epoch rule (`tests/test_oof_rule.py`).
- Broken or untested:
  - The golden regression job is manual-only on GitHub (`workflow_dispatch`): the tiny trainings diverge on GitHub's CPUs (commit `11bae19`, `.github/workflows/tests.yml`).
  - `legacy/` scripts are not part of the pipeline; several do not import (`legacy/README.md`).
  - Seed-3 runs of the shuffled-mask and the classification-only recipes do not exist yet (`analysis/ensemble_runs_results.txt` lists 4 runs of the joint recipe and 3 of each of the others).
  - The FAZ channel and the background channel are not mutually exclusive in the loss; in two of four joint runs this lowers the argmax FAZ Dice (`analysis/faz_channel_check.txt`). Not fixed.

## Environment
- Python 3.12 (README "Setup"); `requirements-lock.txt` pins the versions tested on Apple Silicon and CPU (e.g. `torch==2.14.1`, `timm==0.9.2`, `segmentation-models-pytorch==0.3.3`, `albumentations==2.0.8`, `kornia==0.8.3`); `requirements-kaggle.txt` (Kaggle image) pins only `segmentation-models-pytorch==0.3.3` and leaves `timm`, `albumentations`, `kornia` unpinned; `requirements.txt` pins `segmentation-models-pytorch==0.3.3` and `torchmetrics==1.6.3`.
- Device: `cuda` > `mps` > `cpu`, `--device` overrides (`octa/joint_training.py`, `octa/classification_training.py`, `octa/predict.py`); the hard-coded `.cuda()` calls were removed (commit `8a93ad0`; no `.cuda()` remains under `octa/`). Training runs were done on Apple Silicon (MPS) and on Kaggle GPUs (CUDA) (`results_fold*.json` record the device; `docs/RESULTS.md`).
- Check of the note "albumentations, segmentation_models_pytorch and kornia are unpinned; device selection may need fixes on MPS" (from the Vision AI chat, 2026-10-23): partly outdated. They are pinned in `requirements-lock.txt` (see above) and only partly in `requirements.txt` / `requirements-kaggle.txt`; MPS is supported and was used for the Mac runs.

## Results (only from committed logs or result files)
All on OCTA-500, B5 projection, 364 labelled scans (251 Normal / 64 DR / 49 AMD), class-stratified 5-fold cross-validation, 40 epochs per fold.

| Experiment | Metric | Value | Source |
|---|---|---|---|
| Joint model, regularised (3 runs) | FAZ Dice (3-class argmax) / vessel Dice / vessel clDice | 0.916 ± 0.025 / 0.790 ± 0.004 / 0.862 ± 0.004 | `docs/RESULTS.md` |
| Same, four runs | FAZ Dice with a 0.5 threshold on the FAZ channel | 0.937 / 0.925 / 0.933 / 0.936 | `analysis/faz_channel_check.txt` |
| Joint model, per run | pooled out-of-fold balanced accuracy | 0.747 / 0.758 / 0.789 / 0.771 (mean 0.766; ensemble of 4: 0.792) | `analysis/ensemble_runs_results.txt` |
| Joint model, shuffled masks, per run | pooled out-of-fold balanced accuracy | 0.762 / 0.659 / 0.748 (mean 0.723; ensemble 0.781) | `analysis/ensemble_runs_results.txt` |
| Classification-only ResNeSt50, regularised, per run | pooled out-of-fold balanced accuracy | 0.747 / 0.762 / 0.771 (mean 0.760; ensemble 0.805) | `analysis/ensemble_runs_results.txt` |
| Epoch rule, joint (3 runs) | mean over folds of balanced accuracy: pipeline score / best validation balanced accuracy / last epoch | 0.763 / 0.753 / 0.734 | `analysis/epoch_selection_results.txt` |
| Epoch rule, shuffled masks (3 runs) | same three rules | 0.726 / 0.749 / 0.741 | `analysis/epoch_selection_results.txt` |
| Epoch rule, classification-only (3 runs) | same three rules | 0.759 / 0.771 / 0.746 | `analysis/epoch_selection_results.txt` |
| Statistical power | SE of a paired balanced-accuracy difference (364 scans); smallest detectable difference, 80 % power | 0.038; 0.106 | `analysis/power_seeds_results.txt` |
| Image-frame ablation | balanced accuracy, outer 30 % replaced by grey: joint / classification-only (unchanged: 0.747 / 0.747) | 0.396 / 0.638 | `analysis/gradcam_results.txt` |
| Biomarkers from predicted masks | balanced accuracy of mask-derived features, regularised joint model | 0.546–0.611 depending on classifier and feature set (logistic regression on predicted-mask features 0.594, with the field of view added 0.611) | `analysis/biomarker_predicted_cv40reg.txt` |

Reading (see `docs/SUMMARY_FOR_SUPERVISORS.md`): the three recipes lie within the run-to-run spread; no measurable classification benefit of the segmentation tasks on this data. The stop-gradient and no-segmentation-loss ablations are single runs (`docs/RESULTS.md`).

Reference, thesis PDF Tables 4.1 and 4.2 (5-fold CV, OCTA-500): joint RV+FAZ Jaccard 95.3, Dice 98.1, clDice 91.3; classification accuracy 95.2 vs 94.1 (FAZ-only) vs 89.3 (no segmentation) (also in the collapsed block of `docs/RESULTS.md`). Re-runs are logged above, not mixed into this line; the thesis Dice most likely is the background-dominated 3-channel micro-average (`docs/METHOD.md`).

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
| Technical writing | README for readers, method map, one-page summary | `README.md`, `docs/METHOD.md`, `docs/SUMMARY_FOR_SUPERVISORS.md` |

## Open issues
- B and C seed 3 pending (Kaggle quota); README and `docs/RESULTS.md` show three runs per recipe, the joint recipe has a fourth run that is not yet in the README table.
- Only 49 AMD scans; the AMD figures have wide uncertainty (`docs/MODEL_CARD.md`).
- Mac and Kaggle runs are mixed within a recipe (hardware differs; `results_fold*.json` record the device).
- A common classification-only epoch rule exists for runs made with the logging code only; earlier runs can be compared at fold-mean level (`analysis/epoch_selection.py`).
- FAZ/background exclusivity in the loss (`analysis/faz_channel_check.txt`).
- The Main Professional project's GitHub sync may read only `master`; the work is on `fix/thesis-consistency` (the default branch), `master` is intentionally untouched. The link test below decides whether the sync follows the default branch.

## Next steps
- Run B and C seed 3 on Kaggle (and optionally the joint recipe with another seed), import them, update README and `docs/RESULTS.md` to four runs per recipe, regenerate `Figs/findings.png` (`analysis/make_figures.py`), run `analysis/oof_rule.py`.
- Link test: ask the Main Professional chat to read `docs/STATUS.md` from the synced repo and report the line "link test 2026-10-23".
- Refresh the CV and LinkedIn text from this repository (local drafts, not in git).
- Then continue with the Vision AI roadmap (Stage 1).
