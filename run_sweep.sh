#!/usr/bin/env bash
# Pilot comparison of input projection x vessel ground truth on one CV fold.
# Usage: ./run_sweep.sh <fold> <epochs> <startpoint> <config> [<config> ...]
#   config = <ILM_OPL|FULL|OPL_BM>_<large_capillary|large>, built with Preprocessing/build_processed_octa500.py
set -euo pipefail
FOLD=$1; EPOCHS=$2; START=$3; shift 3
for CFG in "$@"; do
  D=Datasets/processed_OCTA500/$CFG
  OUT=runs/$CFG; mkdir -p "$OUT/summary"
  WANDB_MODE=disabled PYTHONWARNINGS=ignore .venv-smoke/bin/python faz_vessel_seg_cls.py \
    --img_path "$D/image_surface" --gt_path "$D/gt_full/mask" --save_path "$OUT" \
    --train_type Vessel_FAZ --distance_type dist_signed11 \
    --num_epochs "$EPOCHS" --startpoint "$START" --fold "$FOLD" --eval_every 3 > "$OUT/train.log" 2>&1
done
