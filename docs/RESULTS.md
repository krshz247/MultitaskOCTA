# Full results

See the README for the summary. Everything below is 5-fold cross-validation on 364 OCTA-500 scans; scripts and reports behind it are in `analysis/`.

## Tables

Own results of this implementation, 5-fold cross-validation on the 364 labelled OCTA-500 scans (Normal / DR / AMD), B5 en-face projection,
ground truth = GT_FAZ + GT_LargeVessel, 40 epochs per fold, test metrics at the best-validation epoch (`configs/*.yaml`). The thesis numbers are **not** used as targets
(see the note below the tables). A single training run varies by about ±0.03 in balanced accuracy (pooled) and 5–9 points per fold, so the three main recipes were each
trained **three times** (one run on the Mac, two Kaggle runs with seeds 1 and 2); the tables give mean ± sd over the runs of the fold-averaged metric.
Rows marked "1 run" are single runs and cannot be separated from that noise.

**Segmentation** (per image, class-wise):

| model | runs | FAZ Dice | vessel Dice | vessel clDice | Dice incl. background* |
|---|---|---|---|---|---|
| joint RV + FAZ model, regularised (`joint_regularised`) | 3 | 0.916 ± 0.025 | 0.790 ± 0.004 | 0.862 ± 0.004 | 0.959 ± 0.005 |
| joint RV + FAZ model, thesis settings (`joint_cv40`) | 1 | 0.926 ± 0.009 | 0.787 ± 0.011 | 0.857 ± 0.011 | 0.962 ± 0.003 |
| common field of view (`joint_common_fov`) | 1 | 0.933 ± 0.018 | 0.738 ± 0.012 | 0.826 ± 0.012 | 0.956 ± 0.003 |

\* micro-average over the three channels including the background — the kind of number the thesis' Dice table most likely reports; it is dominated by the background and not comparable with the class-wise values.
(FAZ Dice here is measured on the 3-class argmax map. In the seed-2 run (0.888) and the seed-3 run (0.884) the background channel also outputs probability ≈ 1 inside the FAZ, so the argmax often assigns FAZ pixels to the background; with a 0.5 threshold on the FAZ channel itself all four runs reach 0.925–0.937. The segmentation is equally good in every run, the class assignment is not; see `analysis/faz_channel_check.txt`.)

**Classification** (Normal / DR / AMD):

| model | runs | accuracy | balanced accuracy | macro-F1 | AMD recall | balanced acc., epoch chosen on validation balanced acc. |
|---|---|---|---|---|---|---|
| joint model, regularised | 3 | 0.788 ± 0.014 | 0.763 ± 0.021 | 0.715 ± 0.016 | 0.73 ± 0.05 | 0.753 ± 0.022 |
| joint model, regularised, **shuffled masks** (segmentation supervision meaningless) | 3 | 0.771 ± 0.045 | 0.726 ± 0.052 | 0.690 ± 0.058 | 0.67 ± 0.17 | 0.749 ± 0.017 |
| classification-only ResNeSt50, regularised | 3 | 0.819 ± 0.030 | 0.759 ± 0.017 | 0.742 ± 0.032 | 0.63 ± 0.04 | 0.771 ± 0.004 |
| average of the three runs' probabilities (pooled out-of-fold, balanced acc.): joint / shuffled / classification-only | | | 0.784 / 0.781 / 0.805 | | | |
| joint model, thesis settings | 1 | 0.819 ± 0.036 | 0.763 ± 0.035 | 0.740 ± 0.038 | 0.63 | |
| classification-only ResNeSt50, no regularisation | 1 | 0.799 ± 0.062 | 0.687 ± 0.068 | 0.689 ± 0.074 | 0.49 | |
| joint, classifier gradient stopped at the decoder features | 1 | 0.722 ± 0.090 | 0.656 ± 0.079 | 0.623 ± 0.096 | 0.53 | 0.667 |
| joint, segmentation loss switched off | 1 | 0.692 ± 0.103 | 0.682 ± 0.112 | 0.629 ± 0.108 | 0.69 | 0.698 |
| joint, common field of view | 1 | 0.813 ± 0.028 | 0.743 ± 0.085 | 0.708 ± 0.069 | 0.66 | |
| classification-only, common field of view | 1 | 0.805 ± 0.026 | 0.712 ± 0.027 | 0.688 ± 0.043 | 0.49 | |

The last column re-selects the reported epoch with a classification-only rule (best validation balanced accuracy) for every row. The default rule of the pipeline also uses the
segmentation metrics on the validation fold, which for the shuffled-mask and no-segmentation variants are meaningless and make their epoch choice partly random with respect to classification.

**What the experiments say** (details, uncertainty and caveats in `docs/METHOD.md` and `docs/SUMMARY_FOR_SUPERVISORS.md`):
- Segmentation of the FAZ and the large vessels is accurate and stable (vessel Dice 0.79 ± 0.004 over runs; clDice 0.86 reflects vessel connectivity).
- Classification reaches about 0.75–0.77 balanced accuracy with large fold-to-fold spread (only 49 AMD scans). **The segmentation tasks bring no measurable benefit:** the joint model, the joint model trained
  on shuffled masks (no valid segmentation information) and a plain regularised classifier are equal within the run-to-run spread (paired difference joint − plain ≈ 0.00, 95 % interval ±0.06).
  The auxiliary losses act as a regulariser; the stop-gradient and no-segmentation-loss variants are lower, but are single runs and were not repeated.
- A single run is not enough to decide any of this: the shuffled-mask recipe gave 0.762, 0.659 and 0.748 pooled balanced accuracy, and two runs of the same recipe agree on only 76–82 % of the scans.
- Field of view: 6 mm scans hold 88 % of the AMD cases. Training on a common 3 mm field of view gives results within the noise, so the model does not need the scale cue; both model families do change their decisions
  strongly when the outer part of the scan is replaced by grey, i.e. they are sensitive to the image frame (`analysis/gradcam_results.txt`). Vessel disconnection is not a disease cue the classifier relies on.
- Dense capillary labels (`GT_Capillary`) make the soft-clDice loss collapse to an all-vessel prediction; the thesis' large-vessel ground truth does not.
- **Why this fork differs from the thesis code:** in the thesis-time code the clDice term was computed and then overwritten by a plain Dice loss, so the topology loss never reached the optimiser; this branch fixes it, together
  with other bugs listed in `docs/BRANCH_SUMMARY.md`.

<details><summary>Numbers reported in the thesis (not reproduced; metrics are defined differently, see docs/METHOD.md)</summary>

| Method | Jaccard | Dice | clDice |
|---|---|---|---|
| DeepLabV3+ | 88.4 ± 2.2 | 92.5 ± 1.6 | 77.6 ± 2.7 |
| UNet | 91.3 ± 1.5 | 97.1 ± 2.5 | 92.3 ± 1.9 |
| UNet++ | 92.2 ± 1.6 | 97.4 ± 2.1 | 93.4 ± 2.5 |
| BSDA-Net (RV + FAZ) | 95.3 ± 1.7 | 98.1 ± 1.1 | 91.3 ± 2.4 |

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNeSt50 (no segmentation) | 89.3 | 87.1 | 88.4 | 87.5 |
| BSDA-Net (FAZ-only segmentation) | 94.1 | 95.4 | 96.2 | 95.3 |
| BSDA-Net (RV + FAZ) | 95.2 | 94.7 | 95.6 | 95.4 |

The thesis' classification accuracy (≈ 95 %) is far above what a leakage-free, class-stratified protocol gives here (≈ 0.79–0.82); one possible reason is that augmented copies of the same scan were split across train and test in the original pipeline (not verified).
</details>
