# Does segmentation information help OCTA disease classification? — status of the re-evaluation

Data: OCTA-500, B5 (ILM–OPL) en-face projection, 364 scans (251 Normal, 64 DR, 49 AMD), 3 mm and 6 mm fields of view. Model: shared-encoder segmentation network
(FAZ, vessel, background) with boundary-heatmap and signed-distance branches; a ResNeSt50 classifier receives the decoder features of all three branches (the thesis method).
Protocol: class-stratified 5-fold cross-validation, epoch chosen on a validation fold, all numbers on the test folds; pooled out-of-fold predictions with bootstrap intervals.
The thesis numbers are not used as targets; the method is re-tested on its own protocol. Details: `docs/METHOD.md`, `README.md`.

## Question
Does the joint model classify better *because* it has learned FAZ/vessel segmentation, as the thesis argues?

## What was found (one or two training runs per variant; seed repeats are running on Kaggle)
| | balanced accuracy (pooled out-of-fold, 95 % interval) |
|---|---|
| joint model, regularised | 0.747 [0.688, 0.809] (Mac run) · 0.758 [0.700, 0.812] (Kaggle, seed 1) |
| same, but trained on shuffled masks (no valid segmentation information) | 0.762 [0.703, 0.818] (Mac run) · **0.659 [0.598, 0.720]** (Kaggle, seed 1) |
| classification-only ResNeSt50, same regularisation | 0.747 [0.685, 0.801] |
| joint, classifier gradient stopped at the decoder features | 0.658 [0.597, 0.722] |
| joint, segmentation loss switched off | 0.684 [0.624, 0.741] |

1. **Segmentation works** (FAZ Dice 0.93, vessel Dice 0.79, clDice 0.86, std ≈ 1 point across folds).
2. **Classification is ≈ 0.75 balanced accuracy; whether the segmentation content matters is not settled.** A plain regularised classifier is not different from the joint model (paired difference −0.000 [−0.065, +0.059]), so segmentation brings no measurable gain over regularisation alone. The joint model reproduces on a second run (0.747 and 0.758). The shuffled-mask control does not: 0.762 in the first run, 0.659 in the second (different seed and machine) — a 10-point swing for the same recipe, so "shuffled masks = correct masks" cannot be claimed until the seed repeats are in. Removing the segmentation loss or the end-to-end coupling gave 0.68 and 0.66 in single runs, which is within the range the shuffled-mask recipe has now shown.
3. **Segmentation-derived measures are weak on their own**: FAZ and vessel biomarkers from perfect masks reach ≈ 0.6 balanced accuracy, from predicted masks the same, and adding them to the CNN probabilities does not improve them
   (checked for the thesis-setting model and the regularised model).
4. **Field of view**: 6 mm scans hold 88 % of the AMD cases. Training on a common 3 mm field of view gives results within noise of the native-field-of-view runs, so the model does not need the scale cue. Attention maps of the joint model concentrate on the fovea/FAZ (12–26× the area share), those of the classification-only network are more diffuse and more often off-centre; DR scans receive more peripheral attention.
   Replacing the outer 30 % of each scan by grey changes the decisions strongly (joint: balanced accuracy 0.75 → 0.40, AMD recall 0.71 → 0.06; classification-only: 0.75 → 0.64, with errors in the opposite direction), so both models are sensitive to the image frame. Because the two models react in different directions and the manipulation is out of distribution, this shows lack of robustness, not a demonstrated shortcut.

## Limits
* 49 AMD scans; a paired difference between two models must be about ±0.06 balanced accuracy to be distinguishable (scan sampling alone; `analysis/power_seeds_results.txt`).
* Single runs differ by 5–9 balanced-accuracy points per fold between repeats; pooled estimates are steadier. Three seeds per variant can only resolve differences of roughly 0.05–0.09; the R1/R3/plain comparison (differences ≈ 0–0.015) will stay unresolved, the degradations of R2/R4 (0.06–0.09) are at the edge.
* Superficial B5 projection only; AMD mainly changes the outer retina, which this input does not show. Features and stacking in the biomarker analysis are exploratory.

## Open
Seed repeats (3 seeds × joint regularised / shuffled masks / plain regularised) on Kaggle; an input that contains the deeper layers (B6, 3D) would be the natural next test of whether segmentation of the *relevant* structure can help AMD.
