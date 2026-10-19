# Method: how the code implements the thesis

Thesis: *OCTA-based FAZ and Retinal Vasculature Segmentation and Disease Identification* (K. Zargari, TUM, 2023).
This file maps every element of the thesis method to the code, says where they differ and why, and defines every metric.
Status: **match** = implemented as the thesis states · **approx** = implemented, thesis leaves details open · **deviation** = differs on purpose · **unknown** = cannot be verified.

## 1. Pipeline

```
raw OCTA-500 ──Preprocessing/build_processed_octa500.py──▶ Datasets/processed_OCTA500/<projection>_<vessel>/
      │                                                     image_surface/  gt_{full,vessel,faz}/mask/
      │                                                     gt_{vessel,faz}/{contour_img,dist_signed11_img}/
      ▼
faz_vessel_seg_cls.py --fold k   (train: octa/data + octa/losses.py + octa/models.py; evaluate: octa/metrics.py)
      ▼
<save_path>/results_fold<k>.json · best_fold<k>.pt · preds_fold<k>/ · tb_fold<k>/ · train.log
      ▼
aggregate_folds.py (mean ± std over folds) · plot_runs.py (curves) · eval_checkpoint.py (re-evaluate a checkpoint)
```

Every experiment is described by a YAML file in `configs/` and started with `python run_experiment.py configs/<file>.yaml` (all folds, builds the data on demand, stores `config.yaml` and the git commit with the results).

Reference command (thesis setup, one fold, without the launcher):
```bash
python Preprocessing/build_processed_octa500.py --octa500 ../OCTA500 --out Datasets/processed_OCTA500/ILM_OPL_large --projection ILM_OPL --vessel large
D=Datasets/processed_OCTA500/ILM_OPL_large
python faz_vessel_seg_cls.py --img_path $D/image_surface --gt_path $D/gt_full/mask --save_path runs/thesis --train_type Vessel_FAZ \
       --distance_type dist_signed11 --fold 0 --eval_every 3        # defaults = thesis hyper-parameters
```
Every run stores its arguments and the git commit in `results_fold<k>.json`.

## 2. Data

| Item | Value |
|---|---|
| Source | OCTA-500 (Li et al., 2024): 200 subjects 3 mm × 3 mm (304 × 304 px), 300 subjects 6 mm × 6 mm (400 × 400 px); one eye per subject |
| Input | OCTA en-face **B5 = ILM-OPL** maximum projection (superficial) for both FOVs, resized to 384 × 384 (bilinear); labels resized nearest-neighbour |
| Labels used | `GT_FAZ` and `GT_LargeVessel` (user's original ground truth). `GT_Capillary` already contains the large vessels (checked on all 500 scans) and is an optional dense variant (`--vessel large_capillary`) |
| Classes | Normal / DR / AMD from `Text labels`; CNV, CSC and other diseases (136 scans) are excluded → **364 scans** |

Class × field-of-view counts (important confound: the classifier could use the field of view as a shortcut; tested, see README):

| | 3 mm | 6 mm | share at 6 mm |
|---|---|---|---|
| Normal | 160 | 91 | 36 % |
| DR | 29 | 35 | 55 % |
| AMD | 6 | 43 | 88 % |

Split: stratified (by class) 5-fold cross-validation, `kfold_split` in `octa/common.py` (seed 2021). Fold *k* is the test set, fold *k+1* the validation set, the other three train (≈ 218 / 73 / 73 scans). Model selection uses the validation set only.

## 3. Thesis → code map

| Thesis | Code | Status |
|---|---|---|
| §3.1 input B5, 384 × 384 | `octa/data/dataset.py:load_image` (÷ per-image max, then ÷ dataset mean/std), `--input_size` | match (normalisation details not in thesis) |
| §3.3 shared encoder + 3 decoders S, B, D; ResNeSt50 | `octa/models.py:MyMultibranchModel` (encoder ``timm-resnest50d``, decoders `MyUnetDecoder`); `--encoder` default | match |
| S has 3 channels | segmentation head: [FAZ, vessel, background]; B and D have 2 channels [FAZ, vessel] | match |
| B features concatenated into S | `cat_conv(cat[decoder1_out, decoder2_out])` before the S head; D only feeds its own loss and the classifier | match |
| Heatmap GT, Eq. 3.9–3.11 (Gaussian, Heatsum, < 0.001 → 0, normalised) | `Preprocessing/build_processed_octa500.py:boundary_heatmap`: σ = 1.6, amplitude 0.06 (as `pre_dis.m`), noisy-or sum, rescale to [0,1], then < 0.001 → 0 | approx (σ and the order of threshold / normalisation are not given in the thesis) |
| SDM GT, Eq. 3.8 (negative inside, positive outside, normalised) | `signed_distance`: `pre_dis.m` construction, stored as (x+1)/2 → 0.5 on the boundary, < 0.5 inside, > 0.5 outside | approx (affine map of the thesis definition) |
| Eq. 3.12–3.13 MSE for B and D | `Vessel_FAZ_multiLoss` (`criterion2/3`, `nn.MSELoss`) | match |
| Eq. 3.14 `L_sh = Dice(FAZ) + softclDice(vessel)` (note: the code of the thesis submission computed this sum and then overwrote it with a plain 3-channel Dice, so the clDice term never reached the optimiser; fixed here) | `octa/losses.py:VesselClDiceLoss` (0.5·Dice + 0.5·clDice, 3 skeleton iterations) + `Dice(FAZ)` | match; clDice parameters are not in the thesis |
| background channel | `+ Dice(background)` in `seg_loss` (`--bg_dice`, default 1) | deviation (the original code computed this sum; the thesis equation omits it; the 3-channel argmax needs it) |
| Eq. 3.15 `L_seg = λ1 L_sh + λ2 L_bd + λ3 L_sd` | `loss_weights = [3, 1, 1]` | match |
| Eq. 3.16 joint loss, classifier frozen for epoch ≤ τ | `if epoch > startpoint` (unfreeze and add CE); `--startpoint 20` | match |
| classifier: ResNeSt50, ImageNet, features of all decoder blocks concatenated, 1×1 convs | `octa/models.py:MyResNestEncoderMulti`, `CotrainingModelMulti.clf_model` | match |
| weighted cross-entropy (Discussion) | `CrossEntropyLoss(weight=[0.49, 1.88, 2.35])` ≈ inverse class frequency | match (values not in thesis) |
| Adam, LR 1e-4 (seg) / 2e-5 (cls), batch 4, 100 epochs, τ = 20 | `octa/config.py` defaults (`TrainConfig`, listed in docs/OPTIONS.md) | match |
| no LR schedule | `--scheduler none` (default); `onecycle` / `cosine` optional | match |
| augmentation: random 90° rotations, h/v flips | `octa/data/dataset.py:random_rot_flip`, train split only | match |
| 5-fold cross-validation, mean ± std | `--fold k`, `--num_folds 5`; `aggregate_folds.py` | match (needs 5 runs) |
| Dice / Jaccard / clDice, Eq. 3.17–3.19 | see §4 | **definition differs**, see §4 |

Optional additions, all **off by default**: `--vessel_bce`, `--exclusion`, `--aux_channels faz`, `--cldice_alpha`, `--vessel_weight`, `--scheduler`, `--stratify_fov` (folds stratified by class × field of view), `--full_eval` (evaluate every val/test scan instead of dropping the last incomplete batch), `--bg_dice 0` (no background term).

**Classification-only baseline.** The original `classification.py` is *not* the thesis baseline: it trains a torchvision ResNet50 (not ResNeSt50) with an unweighted cross-entropy on a random 70/20 split without test set. `classification_baseline.py` replaces it: same ResNeSt50 backbone and ImageNet weights as the joint model's classifier, same folds, augmentation, class weights, batch size / LR, selection rule (best validation macro-F1) and metrics, so the two arms can be compared fold by fold.

## 4. Metrics (what the numbers mean)

Per image on the 384 × 384 maps, averaged over images (`octa/metrics.py:evaluate_loader`). The class map is the **argmax over the 3 sigmoid channels**.

| Name | Definition |
|---|---|
| `faz_dice`, `vessel_dice` (+ `_jaccard`) | class-wise overlap of the argmax map with the ground truth |
| `vessel_cldice` | clDice of the vessel map: 2·Tprec·Tsens/(Tprec+Tsens) with skeletons from `skimage.skeletonize` |
| `vessel_topo_prec`, `vessel_topo_sens` | Tprec / Tsens separately; `vessel_components`, `vessel_fragments` (< 30 px) count connectivity breaks |
| `dice_3ch_micro` | **micro-averaged Dice over all three channels including the background** of the rounded sigmoid outputs; this is the "Dice_Coeff" printed by the original training loop |
| `acc`, `f1_macro`, `kappa`, `recall_*`, `balanced_acc` | classification on the image level (macro = unweighted over Normal/DR/AMD) |
| `acc_3mm`, `acc_6mm`, `balanced_acc_*mm` | the same per field of view (the FOV is confounded with the class) |
| `jaccard_3ch_micro` | Jaccard counterpart of `dice_3ch_micro` |
| `heat_*`, `sdm_*` `_mse`, `_r2` | auxiliary-branch regression error; R² is relative to predicting each image's mean |

**Why the thesis Dice (98.1 %) and ours differ.** The model of the 100-epoch reference run scores FAZ Dice 0.94 and vessel Dice 0.79 class-wise, but `dice_3ch_micro` = 0.965 — the background channel dominates the micro-average. The thesis table most likely reports that background-dominated quantity (same family of value, 98.1 %). Report both and label them; do not compare class-wise numbers with the thesis table.

## 5. Known limitations of the current implementation

* Without `--full_eval` the validation/test loaders use `drop_last=True`, so 72 of 73 scans are evaluated per split (finished early runs); all newer cross-validation runs use `--full_eval`.
* Evaluation at 384 × 384, not at the native resolution of each scan.
* One fold per invocation; five invocations are needed for the thesis protocol.
* The thesis does not state σ, heatmap amplitude, clDice parameters, class-weight values or seeds; the values above are recorded here and in each run's `results_fold<k>.json`.
* The original preprocessed data and training logs are not available, so exact numerical reproduction is not possible; the goal is method agreement.
