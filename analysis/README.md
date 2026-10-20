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
| `power_seeds.py` | How large must a difference be to be detectable, and what do seed repeats buy? | `oof_probs.npz` | `power_seeds_results.txt` |
| `postprocess_vessels.py` | Does removing small vessel fragments change the metrics? | joint run | printed |

Reading rules used throughout: one run is one sample of the training noise (single folds differ by 5–9 balanced-accuracy points between repeats of the same recipe), so differences
between variants are judged on the pooled out-of-fold predictions and, once available, on seed repeats (`aggregate_seeds.py`); a pooled paired difference needs about ±0.06 balanced accuracy
(364 scans, 49 AMD) to be called different (see `power_seeds_results.txt`).
