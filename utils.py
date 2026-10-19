import argparse
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist, mean_and_std, DatasetImageFazMaskContourDist, DatasetImageClassification


def generate_dataset(train_img_names, val_img_names, test_img_names, input_size, batch_size, val_batch_size, distance_type, do_clahe, train_type, train_gt_names=None, val_gt_names=None, test_gt_names=None, num_workers=0, full_eval=False):

    train_mean, train_std = mean_and_std(train_img_names)

    if train_type == "Vessel_FAZ":

        train_dataset = DatasetImageMaskContourDist(train_img_names, train_gt_names, input_size, distance_type, train_mean, train_std, do_clahe, augment=True)
        valid_dataset = DatasetImageMaskContourDist(val_img_names, val_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageMaskContourDist(test_img_names, test_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)

    elif train_type == "FAZ":

        train_dataset = DatasetImageFazMaskContourDist(train_img_names, train_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        valid_dataset = DatasetImageFazMaskContourDist(val_img_names, val_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageFazMaskContourDist(test_img_names, test_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)

    elif train_type == "CLS":
        train_dataset = DatasetImageClassification(train_img_names, input_size, train_mean, train_std, do_clahe)
        valid_dataset = DatasetImageClassification(val_img_names, input_size, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageClassification(test_img_names, input_size, train_mean, train_std, do_clahe)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, drop_last=not full_eval, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, drop_last=not full_eval, num_workers=num_workers, persistent_workers=num_workers > 0)

    return train_loader, valid_loader, test_loader


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to training img jpg files")
    parser.add_argument("--val_path", type=str, help="path to validation img jpg files")
    parser.add_argument("--test_path", type=str, help="path to test img jpg files")
    
    parser.add_argument("--img_path", type=str, help="path to img bmp files")
    parser.add_argument("--gt_path", type=str, help="path to ground truth bmp files")

    # parser.add_argument("--img_depth", default="surface",type=str, help="depth of the OCTA projection map [full, surface, deep]")
    # parser.add_argument("--gt_type", default="Vessel + FAZ",type=str, help="Vessel, FAZ, Vessel + FAZ")

    parser.add_argument("--train_percentage", default= 0.7, type=float, help="percentage of the files in the dataset used for training")
    parser.add_argument("--val_percentage", default= 0.2, type=float, help="percentage of the files in the dataset used for validation")
    parser.add_argument("--test_percentage", default= 0.1, type=float, help="percentage of the files in the dataset used for test")

    parser.add_argument("--train_type", type=str, default="cotraining", help="Select training type, including single classification, segmentation, cotraining and multitask. ")
    parser.add_argument("--model_type", type=str, help="select model type: unet,dcan,dmtn,psinet,convmcd")
    parser.add_argument("--log_mode", type=bool, default=True, help="logging in wandb")
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument("--distance_type", type=str, default="dist_mask", help="select distance transform type - dist_mask,dist_contour,dist_signed")
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--input_size", type=list, default=[384, 384], help="input image and mask size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="validation batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="auto (cuda > mps > cpu), cuda, mps or cpu.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument("--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="If use_pretrained is true, provide checkpoint.")
    parser.add_argument("--save_path", type=str, help="Model save path.")
    parser.add_argument("--wandb_path", type=str, help="wandb log path.")
    parser.add_argument("--encoder", type=str, default="timm-resnest50d", help="encoder (thesis: ResNeSt50 = timm-resnest50d).")
    parser.add_argument("--pretrain", type=str, default="imagenet", help="choose pretrain.")
    parser.add_argument("--loss_type", type=str, default=None, help="loss type.")
    parser.add_argument("--local_rank", default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--LR_seg", default=1e-4, type=float, help='learning rate.')
    parser.add_argument("--LR_clf", default=2e-5, type=float, help='learning rate.')
    parser.add_argument("--use_scheduler", type=bool, default=False, help="unused, see --scheduler.")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "onecycle", "cosine"], help="per-iteration LR schedule (thesis: none = constant LR).")
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="fraction of all steps used for LR warm-up.")
    parser.add_argument("--aux", type=bool, default=False, help="choose to do classification")
    parser.add_argument("--attention", type=str, default=None, help="decoder_attention_type.")
    parser.add_argument("--usenorm", type=bool, default=True, help="encoder use bn")
    parser.add_argument("--startpoint", type=int, default=20, help="start cotraining point.")
    parser.add_argument("--clahe", type=bool, default=False, help="do clahe.")
    parser.add_argument("--classnum", type=int, default=3, help="clf class number.")
    parser.add_argument("--fold", type=int, default=0, help="Fold used as the test set (validation = next fold) when --num_folds > 1.")
    parser.add_argument("--eval_every", type=int, default=0, help="every N epochs write per-image val/test metrics to results_fold<k>.json in save_path (0 = off).")
    parser.add_argument("--vessel_bce", type=float, default=0.0, help="weight of an extra BCE on the vessel channel (0 = thesis loss).")
    parser.add_argument("--bg_dice", type=float, default=1.0, help="weight of the background-channel Dice term in the segmentation loss (the 3-channel head is [FAZ, vessel, background]).")
    parser.add_argument("--cldice_alpha", type=float, default=0.5, help="share of soft clDice in the vessel term (1 - alpha is Dice); thesis/original: 0.5.")
    parser.add_argument("--vessel_weight", type=float, default=1.0, help="weight of the vessel (Dice + clDice) term in the segmentation loss.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (0 = main process).")
    parser.add_argument("--exclusion", type=float, default=0.0, help="weight of the FAZ/vessel overlap penalty (0 = off).")
    parser.add_argument("--aux_channels", type=str, default="both", choices=["both", "faz"], help="channels used by the heatmap/SDM auxiliary losses (thesis: both).")
    parser.add_argument("--stratify_fov", action="store_true", help="stratify the folds by class x field of view (default: class only, as in the thesis protocol).")
    parser.add_argument("--full_eval", action="store_true", help="evaluate every validation/test scan (drop_last=False); default keeps the original drop_last=True.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="classification-only baseline: label smoothing of the cross-entropy.")
    parser.add_argument("--cls_dropout", type=float, default=0.0, help="classification-only baseline: dropout before the linear layer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="classification-only baseline: > 0 switches Adam to AdamW with this weight decay.")
    parser.add_argument("--strong_aug", action="store_true", help="classification-only baseline: random affine + intensity + noise + erasing on top of the rotations/flips.")
    parser.add_argument("--seg_loss_weights", type=float, nargs=3, default=[3.0, 1.0, 1.0], help="lambda_1 (segmentation), lambda_2 (boundary heatmap), lambda_3 (SDM); thesis 3 1 1. 0 0 0 trains the segmentation network without any segmentation supervision.")
    parser.add_argument("--detach_clf_features", action="store_true", help="ablation: stop-gradient on the decoder features fed to the classifier.")
    parser.add_argument("--shuffle_masks", action="store_true", help="control: give every training image the masks of a different training scan.")
    parser.add_argument("--num_folds", type=int, default=5, help="Stratified k-fold cross-validation (thesis: 5). 0 or 1 = legacy percentage split.")
    parser.add_argument("--augmentation", type=str, default=True, help="using data augmentation.")
    return parser
