"""Typed configuration of the training scripts (single source of truth for every command-line option).

TrainConfig is a dataclass; build_parser() derives the argparse parser from its fields (flag = field name, default = field default,
type / choices / nargs / action / help from the field metadata), parse_config() returns a TrainConfig. The options, defaults and help
texts are pinned by tests/test_config.py. `python -m octa.config --markdown` prints the option table (docs/OPTIONS.md).
"""
import argparse
from dataclasses import MISSING, dataclass, field, fields
from typing import List, Optional

TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list": list}


@dataclass
class TrainConfig:
    # ---- Data (paths and split)
    train_path: Optional[str] = field(default=None, metadata=dict(help='path to training img jpg files', type='str'))
    val_path: Optional[str] = field(default=None, metadata=dict(help='path to validation img jpg files', type='str'))
    test_path: Optional[str] = field(default=None, metadata=dict(help='path to test img jpg files', type='str'))
    img_path: Optional[str] = field(default=None, metadata=dict(help='path to img bmp files', type='str'))
    gt_path: Optional[str] = field(default=None, metadata=dict(help='path to ground truth bmp files', type='str'))
    train_percentage: float = field(default=0.7, metadata=dict(help='percentage of the files in the dataset used for training', type='float'))
    val_percentage: float = field(default=0.2, metadata=dict(help='percentage of the files in the dataset used for validation', type='float'))
    test_percentage: float = field(default=0.1, metadata=dict(help='percentage of the files in the dataset used for test', type='float'))
    num_folds: int = field(default=5, metadata=dict(help='Stratified k-fold cross-validation (thesis: 5). 0 or 1 = legacy percentage split.', type='int'))
    fold: int = field(default=0, metadata=dict(help='Fold used as the test set (validation = next fold) when --num_folds > 1.', type='int'))
    stratify_fov: bool = field(default=False, metadata=dict(help='stratify the folds by class x field of view (default: class only, as in the thesis protocol).', action='store_true'))
    full_eval: bool = field(default=False, metadata=dict(help='evaluate every validation/test scan (drop_last=False); default keeps the original drop_last=True.', action='store_true'))
    distance_type: str = field(default='dist_mask', metadata=dict(help='select distance transform type - dist_mask,dist_contour,dist_signed', type='str'))
    input_size: list = field(default_factory=lambda: [384, 384], metadata=dict(help='input image and mask size', type='list'))
    num_workers: int = field(default=0, metadata=dict(help='DataLoader worker processes (0 = main process).', type='int'))
    clahe: bool = field(default=False, metadata=dict(help='do clahe.', type='bool'))

    # ---- Training schedule and optimisation
    train_type: str = field(default='cotraining', metadata=dict(help='Select training type, including single classification, segmentation, cotraining and multitask. ', type='str'))
    seed: int = field(default=2021, metadata=dict(help='random seed of the training (weight initialisation of the new layers, batch order, augmentation); the cross-validation split is fixed.', type='int'))
    num_epochs: int = field(default=100, metadata=dict(help='number of epochs', type='int'))
    batch_size: int = field(default=4, metadata=dict(help='train batch size', type='int'))
    val_batch_size: int = field(default=4, metadata=dict(help='validation batch size', type='int'))
    startpoint: int = field(default=20, metadata=dict(help='start cotraining point.', type='int'))
    LR_seg: float = field(default=0.0001, metadata=dict(help='learning rate.', type='float'))
    LR_clf: float = field(default=2e-05, metadata=dict(help='learning rate.', type='float'))
    scheduler: str = field(default='none', metadata=dict(help='per-iteration LR schedule (thesis: none = constant LR).', choices=['none', 'onecycle', 'cosine'], type='str'))
    warmup_frac: float = field(default=0.1, metadata=dict(help='fraction of all steps used for LR warm-up.', type='float'))
    weight_decay: float = field(default=0.0, metadata=dict(help='classification-only baseline: > 0 switches Adam to AdamW with this weight decay.', type='float'))
    use_scheduler: bool = field(default=False, metadata=dict(help='unused, see --scheduler.', type='bool'))
    eval_every: int = field(default=0, metadata=dict(help='every N epochs write per-image val/test metrics to results_fold<k>.json in save_path (0 = off).', type='int'))

    # ---- Model
    encoder: str = field(default='timm-resnest50d', metadata=dict(help='encoder (thesis: ResNeSt50 = timm-resnest50d).', type='str'))
    pretrain: str = field(default='imagenet', metadata=dict(help='choose pretrain.', type='str'))
    attention: Optional[str] = field(default=None, metadata=dict(help='decoder_attention_type.', type='str'))
    usenorm: bool = field(default=True, metadata=dict(help='encoder use bn', type='bool'))
    classnum: int = field(default=3, metadata=dict(help='clf class number.', type='int'))
    aux: bool = field(default=False, metadata=dict(help='choose to do classification', type='bool'))
    model_type: Optional[str] = field(default=None, metadata=dict(help='select model type: unet,dcan,dmtn,psinet,convmcd', type='str'))
    object_type: Optional[str] = field(default=None, metadata=dict(help='Dataset.', type='str'))
    loss_type: Optional[str] = field(default=None, metadata=dict(help='loss type.', type='str'))
    use_pretrained: bool = field(default=False, metadata=dict(help='Load pretrained checkpoint.', type='bool'))
    pretrained_model_path: Optional[str] = field(default=None, metadata=dict(help='If use_pretrained is true, provide checkpoint.', type='str'))

    # ---- Loss (thesis defaults; extensions are off by default)
    seg_loss_weights: List[float] = field(default_factory=lambda: [3.0, 1.0, 1.0], metadata=dict(help='lambda_1 (segmentation), lambda_2 (boundary heatmap), lambda_3 (SDM); thesis 3 1 1. 0 0 0 trains the segmentation network without any segmentation supervision.', nargs=3, type='float'))
    bg_dice: float = field(default=1.0, metadata=dict(help='weight of the background-channel Dice term in the segmentation loss (the 3-channel head is [FAZ, vessel, background]).', type='float'))
    cldice_alpha: float = field(default=0.5, metadata=dict(help='share of soft clDice in the vessel term (1 - alpha is Dice); thesis/original: 0.5.', type='float'))
    vessel_weight: float = field(default=1.0, metadata=dict(help='weight of the vessel (Dice + clDice) term in the segmentation loss.', type='float'))
    vessel_bce: float = field(default=0.0, metadata=dict(help='weight of an extra BCE on the vessel channel (0 = thesis loss).', type='float'))
    exclusion: float = field(default=0.0, metadata=dict(help='weight of the FAZ/vessel overlap penalty (0 = off).', type='float'))
    aux_channels: str = field(default='both', metadata=dict(help='channels used by the heatmap/SDM auxiliary losses (thesis: both).', choices=['both', 'faz'], type='str'))
    label_smoothing: float = field(default=0.0, metadata=dict(help='classification-only baseline: label smoothing of the cross-entropy.', type='float'))

    # ---- Augmentation, regularisation and ablation switches
    strong_aug: bool = field(default=False, metadata=dict(help='classification-only baseline: random affine + intensity + noise + erasing on top of the rotations/flips.', action='store_true'))
    cls_dropout: float = field(default=0.0, metadata=dict(help='classification-only baseline: dropout before the linear layer.', type='float'))
    detach_clf_features: bool = field(default=False, metadata=dict(help='ablation: stop-gradient on the decoder features fed to the classifier.', action='store_true'))
    shuffle_masks: bool = field(default=False, metadata=dict(help='control: give every training image the masks of a different training scan.', action='store_true'))
    augmentation: str = field(default=True, metadata=dict(help='using data augmentation.', type='str'))

    # ---- Output, hardware and logging
    cls_ckpt: str = field(default='none', metadata=dict(help='also save best_cls_fold<k>.pt, the epoch with the best validation balanced accuracy / macro-F1 (a rule without segmentation terms, the same for every variant); none = only the pipeline checkpoint best_fold<k>.pt.', choices=['none', 'balanced_acc', 'f1_macro'], type='str'))
    save_path: Optional[str] = field(default=None, metadata=dict(help='Model save path.', type='str'))
    wandb_path: Optional[str] = field(default=None, metadata=dict(help='wandb log path.', type='str'))
    log_mode: bool = field(default=True, metadata=dict(help='logging in wandb', type='bool'))
    device: str = field(default='auto', metadata=dict(help='auto (cuda > mps > cpu), cuda, mps or cpu.', type='str'))
    cuda_no: int = field(default=0, metadata=dict(help='cuda number', type='int'))
    local_rank: int = field(default=0, metadata=dict(help='node rank for distributed training', type='int'))


def build_parser():
    """argparse parser with exactly the options of TrainConfig."""
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    for f in fields(TrainConfig):
        md = dict(f.metadata)
        kwargs = {}
        if md.get("action"):
            kwargs["action"] = md["action"]
        else:
            kwargs["type"] = TYPES[md["type"]]
            kwargs["default"] = f.default_factory() if f.default is MISSING else f.default
        for key in ("nargs", "choices", "help"):
            if key in md:
                kwargs[key] = md[key]
        parser.add_argument("--" + f.name, **kwargs)
    return parser


create_train_arg_parser = build_parser  # name used by the legacy scripts


def parse_config(argv=None) -> TrainConfig:
    """Parse the command line (or `argv`) into a TrainConfig."""
    return TrainConfig(**vars(build_parser().parse_args(argv)))


def option_table() -> str:
    rows = ["| option | default | description |", "|---|---|---|"]
    for f in fields(TrainConfig):
        default = f.default_factory() if f.default is MISSING else f.default
        rows.append(f"| `--{f.name}` | `{default}` | {f.metadata.get('help', '')} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    print(option_table() if "--markdown" in sys.argv else build_parser().format_help())
