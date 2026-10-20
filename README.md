# MultitaskOCTA

**Joint retinal vessel + FAZ segmentation for OCTA-based eye disease classification.**
Master's thesis project, Technische Universität München (Department of Physics — Biomedical Physics), submitted October 2023.

This repository extends [**BSDA-Net**](https://www.researchgate.net/publication/354793161_BSDA-Net_A_Boundary_Shape_and_Distance_Aware_Joint_Learning_Framework_for_Segmenting_and_Classifying_OCTA_Images) (Lin et al., MICCAI 2021) — originally built to jointly segment the Foveal Avascular Zone (FAZ) and classify disease from OCTA (Optical Coherence Tomography Angiography) images — with **retinal vessel segmentation** as a second semantic task, and asks a focused research question: *does teaching the network to see the vasculature, not just the FAZ, actually improve disease classification?*

## Overview

Optical Coherence Tomography Angiography (OCTA) provides non-invasive, capillary-level images of the retinal vasculature. The shape of the FAZ and the morphology of the surrounding vessels are both known to shift with diabetic retinopathy (DR) and age-related macular degeneration (AMD), but most prior work segments only one of the two and rarely measures whether the added segmentation signal actually helps a downstream diagnosis.

This project:
- Integrates the **OCTA-500** dataset's vessel annotations into the BSDA-Net pipeline alongside its existing FAZ annotations, turning it into a joint RV + FAZ segmentation and 3-way (Healthy / DR / AMD) classification model.
- Adds a **clDice** loss term ([Shit et al., CVPR 2021](https://github.com/jocpae/clDice)) to the segmentation objective to preserve vessel topology/connectivity, which plain Dice loss tends to break on thin capillaries.
- Runs a controlled ablation comparing classification performance with **no segmentation** (plain ResNeSt50), **FAZ-only segmentation** (original BSDA-Net), and **joint RV+FAZ segmentation** (this work), to isolate the effect of vascular semantic information on diagnostic accuracy.

<p align="center">
  <img src="Figs/model_architecture.png" alt="Model architecture: shared encoder with segmentation, boundary heatmap, and signed distance map decoder branches feeding a disease classifier" width="700">
  <br>
  <em>Shared encoder with a segmentation decoder (vessel + FAZ) and two auxiliary decoders (boundary heatmap, signed distance map) feeding a joint disease classifier.</em>
</p>

## Results

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
(One of the three regularised runs learned the FAZ late and ended at FAZ Dice 0.888; the other two reach 0.93.)

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

<p align="center">
  <img src="Figs/segmentation_results.png" alt="Segmentation comparison across DeepLabV3+, UNet, UNet++, and BSDA-Net against ground truth (thesis figure)" width="600">
</p>

## Repository structure

```
faz_vessel_seg_cls.py          entry point: train / evaluate the joint RV + FAZ segmentation + classification model (this work), one CV fold per call
octa/                          the package behind it: joint_training.py, classification_training.py, config.py (all options), common.py (split, schedules),
                               losses.py, models.py, metrics.py, data/ (dataset, loaders, mask-consistent augmentation)
classification_baseline.py     entry point: classification-only ResNeSt50 baseline, same protocol (data, folds, loss weights, metrics)
aggregate_folds.py             mean ± std over folds at each fold's best-validation epoch
plot_runs.py, eval_checkpoint.py, rename_tb_tags.py   curves, re-evaluation of a checkpoint, TensorBoard tag conversion
run_experiment.py, configs/    run an experiment from a YAML file (all folds, builds the data on demand); run_cv.sh, run_sweep.sh: shell equivalents
legacy/                        scripts, helpers and models of the original BSDA-Net code base (not part of the pipeline; see legacy/README.md)
Preprocessing/build_processed_octa500.py   builds the training data from the raw OCTA-500 release
octa/predict.py                inference: image in, class probabilities + FAZ / vessel masks out (docs/MODEL_CARD.md)
analysis/                      one script per question (biomarkers, attention maps, epoch rules, ensembles, power); see analysis/README.md
tests/                         unit tests (python -m unittest discover -s tests)
kaggle/                        notebook + bundle script to train on a Kaggle GPU
docs/METHOD.md                 thesis → code map, metric definitions, known deviations  (read this first)
docs/SUMMARY_FOR_SUPERVISORS.md, docs/MODEL_CARD.md   findings in one page; what the released model is and is not
clDice/                        official clDice implementation (MIT)
```

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt      # versions verified on Apple Silicon (MPS) and CPU
# Kaggle / Colab: requirements-kaggle.txt (keeps the platform's CUDA torch), see kaggle/README.md
```

The code selects CUDA, then MPS, then CPU (`--device` overrides). **Data:** download OCTA-500 (see [Li et al., 2024](https://ieee-dataport.org/open-access/octa-500); not redistributed here) and build the training set:

```bash
python Preprocessing/build_processed_octa500.py --octa500 <OCTA500> --out Datasets/processed_OCTA500/ILM_OPL_large \
       --projection ILM_OPL --vessel large     # B5 input, GT_FAZ + GT_LargeVessel (thesis setup); 364 scans (Normal / DR / AMD)
```

## Usage

```bash
python run_experiment.py configs/joint_cv40.yaml          # 5 folds of the joint model; builds the dataset if missing, saves config.yaml + git commit
python run_experiment.py configs/classification_baseline.yaml --folds 0 1 --dry-run   # see configs/README.md for all experiments
```

The same experiment by hand:

```bash
D=Datasets/processed_OCTA500/ILM_OPL_large
# one fold of the joint model with the thesis hyper-parameters (batch 4, LR 1e-4 / 2e-5, startpoint 20, loss weights 3:1:1)
python faz_vessel_seg_cls.py --img_path $D/image_surface --gt_path $D/gt_full/mask --save_path runs/thesis \
       --train_type Vessel_FAZ --distance_type dist_signed11 --num_epochs 100 --fold 0 --eval_every 3 --full_eval
./run_cv.sh ILM_OPL_large 100 20 _thesis --full_eval          # all five folds in turn
python aggregate_folds.py runs/ILM_OPL_large_thesis           # mean ± std of the test metrics at the best validation epoch
SCRIPT=classification_baseline.py ./run_cv.sh ILM_OPL_large 100 20 _cls --full_eval   # classification-only arm
```

Outputs per run: `results_fold<k>.json` (arguments, git commit, per-image metrics every 3 epochs), `best_fold<k>.pt`, `preds_fold<k>/` (prediction figures), TensorBoard logs (`tensorboard --logdir runs`).
Optional extensions, **off by default** (see `docs/METHOD.md`): `--scheduler onecycle|cosine`, `--vessel_bce`, `--exclusion`, `--aux_channels faz`, `--cldice_alpha`, `--vessel_weight`, `--stratify_fov`, `--bg_dice 0`.

**Reading the numbers.** Segmentation metrics are reported class-wise (FAZ, vessel) per image on the argmax class map, plus `dice_3ch_micro`, the background-dominated micro-average that the original training loop printed as "Dice_Coeff" and that the thesis' segmentation table most likely reports. Do not compare class-wise values with that table. Known confound: 88 % of AMD scans are 6 mm and 64 % of Normal scans 3 mm; report per-FOV and balanced metrics (`balanced_acc`, `recall_*`, `acc_3mm`, `acc_6mm`).

## Tests

```bash
python -m unittest discover -s tests            # 29 tests: fast unit tests + golden regression (the golden trainings take several minutes on a CPU)
SKIP_SLOW_TESTS=1 python -m unittest discover -s tests   # unit tests only (seconds)
python tests/verify_checkpoints.py              # re-evaluates stored checkpoints (needs the local runs/ and Datasets/)
```
`tests/test_golden.py` trains tiny models for 2 epochs on a synthetic data set and compares every metric with `tests/golden/*.json`; on one machine the result is bit-for-bit reproducible, so any change of behaviour is detected (`GOLDEN_TOL=1e-9`). A GitHub Actions workflow (`.github/workflows/tests.yml`) runs pyflakes and the unit tests.

## Prediction with trained weights

```bash
python -m octa.predict predict --weights weights/joint_reg_s1 --image scan.bmp --out predictions/     # class probabilities, FAZ / vessel overlay
```
Weights: see `docs/MODEL_CARD.md` (research use only; not a medical device).

## Future work (not done here)
Swap the encoder for ConvNeXt or a ViT and check vessel connectivity (clDice); self-supervised pretraining on the unused OCTA-500 scans; deeper projections or the 3D volumes for AMD; repeats of the stop-gradient and no-segmentation-loss ablations; one classification-only checkpoint rule for all variants.

## Full thesis

**"OCTA-based Foveal Avascular Zone (FAZ) and Retinal Vasculature Segmentation and Disease Identification"** — Kourosh Zargari, TUM, October 2023.
Supervisors: Prof. Dr. Julia Herzen, Prof. Dr. Nassir Navab · Advisor: Dr. Shahrooz Faghihroohi.

The full write-up covers the OCT/OCTA imaging background, related work on RV/FAZ segmentation, the complete methodology (loss functions, network design, evaluation metrics), and an in-depth discussion of the results above, including dataset and class-imbalance limitations and directions for future work (unsupervised microvessel segmentation, larger annotated OCTA datasets).

## Citation

If you build on this code, please cite the original BSDA-Net paper and, if you use the vessel segmentation loss, clDice:

```bibtex
@inproceedings{lin2021bsda,
  title={BSDA-Net: A Boundary Shape and Distance Aware Joint Learning Framework for Segmenting and Classifying OCTA Images},
  author={Lin, Li and Wang, Zhonghua and Wu, Jiewei and Huang, Yijin and Lyu, Junyan and Cheng, Pujin and Wu, Jiang and Tang, Xiaoying},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021}
}

@inproceedings{cldice2021,
  title={clDice -- a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},
  author={Shit, Suprosanna and Paetzold, Johannes C and Sekuboyina, Anjany and Ezhov, Ivan and Unger, Alexander and Zhylka, Andrey and Pluim, Josien PW and Bauer, Ulrich and Menze, Bjoern H},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16560--16569},
  year={2021}
}
```

## Acknowledgements

This work builds directly on [llmir/MultitaskOCTA](https://github.com/llmir/MultitaskOCTA) (BSDA-Net, MICCAI 2021) and the [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library. Thanks to Prof. Julia Herzen, Prof. Nassir Navab, Dr. Shahrooz Faghihroohi, and Shervin Dehghani for supervision and guidance throughout the thesis.

## Licence

No licence is granted: this is research code, shared publicly with full attribution for academic and portfolio purposes. The base BSDA-Net code from its original authors has no explicit licence either; the vendored `clDice/` folder is MIT-licensed.
