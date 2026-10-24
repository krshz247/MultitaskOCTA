# analysis/

Scripts that answer one question each from stored runs (checkpoints and predictions in `runs/`); none of them trains a model. Run them with the project environment
(`.venv-smoke/bin/python analysis/<script>.py ...`). Out-of-fold = every scan is predicted by the model of the fold in which it is a test scan.
Result files are written next to the scripts (`*.txt` = printed report, committed; `*.json` = machine-readable copy, git-ignored); figures go to `analysis/figs/` (git-ignored).

| script | question | needs | output |
|---|---|---|---|
| `biomarker_baseline.py` | How much class information is in FAZ / vessel measures taken from the *ground-truth* masks? | processed data | `biomarker_baseline_results.json` |
| `biomarker_predicted.py` | Same with the *predicted* masks of a joint model (19 features: FAZ geometry, vessel density, skeleton, width, branch points, fractal dimension, annuli, quadrants), and does adding them to the CNN probabilities help? `--out` sets the result file | a joint run (`--joint`) | `biomarker_predicted_results.json` (thesis-setting run), `biomarker_predicted_cv40reg.txt` (regularised run) |
| `gradcam_check.py` | Where does the classifier look (class activation maps of the last stage), and what happens if the outer part of the scan is replaced by grey? | a joint run and a classification-only run | `gradcam_results.txt`, `figs/gradcam_amd_examples.png` |
| `fov_shortcut_test.py` | Does the classifier depend on the field of view (3 mm vs 6 mm)? Scale-swap test by cropping 6 mm scans | joint and classification-only runs | `fov_shortcut_results.json` |
| `perturbation_tests.py` | Do random erasing and small vessel gaps change the decision? | joint run | `perturbation_results.json` |
| `connectivity_vs_class.py` | Is vessel fragmentation of the predicted maps related to the disease class? | joint run | `connectivity_results.json` |
| `oof_compare.py` | Pooled out-of-fold comparison of the joint and the classification-only model (bootstrap interval, McNemar, flip TTA); stores `oof_probs.npz` | two runs | `oof_probs.npz` |
| `oof_variants.py` | The same for any number of variants against a reference | runs | `oof_variants_reg_results.txt` (R1–R4 and plain regularised) |
| `epoch_selection.py` | How much does the rule that picks the reported epoch matter (pipeline score, validation F1 / balanced accuracy, last epoch, fixed epoch, mean of the last evaluations, oracle)? Uses the per-epoch records in `results_fold*.json` | runs | `epoch_selection_results.txt` |
| `oof_rule.py` | Pooled out-of-fold comparison of recipes under ONE epoch rule (validation balanced accuracy, validation F1, last epoch, pipeline score) from the probabilities that runs of 2026-10-21 or later log in `results_fold*.json`; ensembles and paired bootstrap against a reference group | runs with logged probabilities | printed |
| `ensemble_runs.py` | Do repeats of a recipe make the same mistakes (agreement between runs), and what does averaging 2–3 runs give? Caches out-of-fold probabilities as `<run>/oof_probs.npy` | runs | `ensemble_runs_results.txt` |
| `make_figures.py` | Draws the README figures (balanced accuracy per run and recipe, image-frame ablation, predicted masks) from stored results; uses no scan pixels | cached `oof_probs.npy` of the runs, `gradcam_results.json`, optionally exported weights | `Figs/findings.png`, `Figs/predictions.png` |
| `faz_channel_check.py` | Why does the FAZ Dice of the 3-class argmax differ between runs of one recipe? Which channel claims the FAZ pixels (report in `faz_channel_check.txt`) | joint runs with `best_fold0.pt` | printed |
| `faz_argmax_errors.py` | Precision and recall of the argmax and thresholded FAZ map on the fold-0 test scans; used to read the test of `--faz_bg_exclusion` (report `faz_bg_exclusion_test.txt`) | joint runs with `best_fold0.pt` | printed |
| `classification_report.py` | One-vs-rest AUROC (with bootstrap interval for the ensembles), pooled confusion matrices, calibration (ECE, Brier) from the cached out-of-fold probabilities; figure `Figs/confusion.png` | cached `oof_probs.npy` of the runs | `classification_report.txt` |
| `power_seeds.py` | How large must a difference be to be detectable, and what do seed repeats buy? | `oof_probs.npz` | `power_seeds_results.txt` |
| `postprocess_vessels.py` | Does removing small vessel fragments change the metrics? | joint run | printed |

Reading rules used throughout: one run is one sample of the training noise (single folds differ by 5–9 balanced-accuracy points between repeats of the same recipe), so differences
between variants are judged on the pooled out-of-fold predictions and, once available, on seed repeats (`aggregate_seeds.py`); a pooled paired difference needs about ±0.06 balanced accuracy
(364 scans, 49 AMD) to be called different (see `power_seeds_results.txt`).
