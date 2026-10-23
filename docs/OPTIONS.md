| option | default | description |
|---|---|---|
| `--train_path` | `None` | path to training img jpg files |
| `--val_path` | `None` | path to validation img jpg files |
| `--test_path` | `None` | path to test img jpg files |
| `--img_path` | `None` | path to img bmp files |
| `--gt_path` | `None` | path to ground truth bmp files |
| `--train_percentage` | `0.7` | percentage of the files in the dataset used for training |
| `--val_percentage` | `0.2` | percentage of the files in the dataset used for validation |
| `--test_percentage` | `0.1` | percentage of the files in the dataset used for test |
| `--num_folds` | `5` | Stratified k-fold cross-validation (thesis: 5). 0 or 1 = legacy percentage split. |
| `--fold` | `0` | Fold used as the test set (validation = next fold) when --num_folds > 1. |
| `--stratify_fov` | `False` | stratify the folds by class x field of view (default: class only, as in the thesis protocol). |
| `--full_eval` | `False` | evaluate every validation/test scan (drop_last=False); default keeps the original drop_last=True. |
| `--distance_type` | `dist_mask` | select distance transform type - dist_mask,dist_contour,dist_signed |
| `--input_size` | `[384, 384]` | input image and mask size |
| `--num_workers` | `0` | DataLoader worker processes (0 = main process). |
| `--clahe` | `False` | do clahe. |
| `--train_type` | `cotraining` | Select training type, including single classification, segmentation, cotraining and multitask.  |
| `--seed` | `2021` | random seed of the training (weight initialisation of the new layers, batch order, augmentation); the cross-validation split is fixed. |
| `--num_epochs` | `100` | number of epochs |
| `--batch_size` | `4` | train batch size |
| `--val_batch_size` | `4` | validation batch size |
| `--startpoint` | `20` | start cotraining point. |
| `--LR_seg` | `0.0001` | learning rate. |
| `--LR_clf` | `2e-05` | learning rate. |
| `--scheduler` | `none` | per-iteration LR schedule (thesis: none = constant LR). |
| `--warmup_frac` | `0.1` | fraction of all steps used for LR warm-up. |
| `--weight_decay` | `0.0` | classification-only baseline: > 0 switches Adam to AdamW with this weight decay. |
| `--use_scheduler` | `False` | unused, see --scheduler. |
| `--eval_every` | `0` | every N epochs write per-image val/test metrics to results_fold<k>.json in save_path (0 = off). |
| `--encoder` | `timm-resnest50d` | encoder (thesis: ResNeSt50 = timm-resnest50d). |
| `--pretrain` | `imagenet` | choose pretrain. |
| `--attention` | `None` | decoder_attention_type. |
| `--usenorm` | `True` | encoder use bn |
| `--classnum` | `3` | clf class number. |
| `--aux` | `False` | choose to do classification |
| `--model_type` | `None` | select model type: unet,dcan,dmtn,psinet,convmcd |
| `--object_type` | `None` | Dataset. |
| `--loss_type` | `None` | loss type. |
| `--use_pretrained` | `False` | Load pretrained checkpoint. |
| `--pretrained_model_path` | `None` | If use_pretrained is true, provide checkpoint. |
| `--seg_loss_weights` | `[3.0, 1.0, 1.0]` | lambda_1 (segmentation), lambda_2 (boundary heatmap), lambda_3 (SDM); thesis 3 1 1. 0 0 0 trains the segmentation network without any segmentation supervision. |
| `--bg_dice` | `1.0` | weight of the background-channel Dice term in the segmentation loss (the 3-channel head is [FAZ, vessel, background]). |
| `--cldice_alpha` | `0.5` | share of soft clDice in the vessel term (1 - alpha is Dice); thesis/original: 0.5. |
| `--vessel_weight` | `1.0` | weight of the vessel (Dice + clDice) term in the segmentation loss. |
| `--vessel_bce` | `0.0` | weight of an extra BCE on the vessel channel (0 = thesis loss). |
| `--exclusion` | `0.0` | weight of the FAZ/vessel overlap penalty (0 = off). |
| `--faz_bg_exclusion` | `0.0` | weight of the mean background probability inside the ground-truth FAZ (0 = off); keeps the background channel from claiming FAZ pixels in the 3-class argmax. |
| `--aux_channels` | `both` | channels used by the heatmap/SDM auxiliary losses (thesis: both). |
| `--label_smoothing` | `0.0` | classification-only baseline: label smoothing of the cross-entropy. |
| `--strong_aug` | `False` | classification-only baseline: random affine + intensity + noise + erasing on top of the rotations/flips. |
| `--cls_dropout` | `0.0` | classification-only baseline: dropout before the linear layer. |
| `--detach_clf_features` | `False` | ablation: stop-gradient on the decoder features fed to the classifier. |
| `--shuffle_masks` | `False` | control: give every training image the masks of a different training scan. |
| `--augmentation` | `True` | using data augmentation. |
| `--cls_ckpt` | `none` | also save best_cls_fold<k>.pt, the epoch with the best validation balanced accuracy / macro-F1 (a rule without segmentation terms, the same for every variant); none = only the pipeline checkpoint best_fold<k>.pt. |
| `--save_path` | `None` | Model save path. |
| `--wandb_path` | `None` | wandb log path. |
| `--log_mode` | `True` | logging in wandb |
| `--device` | `auto` | auto (cuda > mps > cpu), cuda, mps or cpu. |
| `--cuda_no` | `0` | cuda number |
| `--local_rank` | `0` | node rank for distributed training |
