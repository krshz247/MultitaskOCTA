#!/usr/bin/env bash
# Cross-validation of one configuration: every fold in turn, one training at a time.
# Usage: ./run_cv.sh <config> <epochs> <startpoint> <run_tag> [extra args for faz_vessel_seg_cls.py ...]
#   e.g. ./run_cv.sh ILM_OPL_large 40 8 _cv40 --full_eval
# Results: runs/<config><run_tag>/results_fold{0..4}.json, train_fold{k}.log;  aggregate with aggregate_folds.py
set -euo pipefail
PYTHON=${PYTHON:-.venv-smoke/bin/python}; DATA_DIR=${DATA_DIR:-Datasets/processed_OCTA500}; RUNS_DIR=${RUNS_DIR:-runs}; SCRIPT=${SCRIPT:-faz_vessel_seg_cls.py}   # SCRIPT=classification_baseline.py for the classification-only arm
CFG=$1; EPOCHS=$2; START=$3; TAG=$4; shift 4
D=$DATA_DIR/$CFG; OUT=$RUNS_DIR/$CFG$TAG; mkdir -p "$OUT/summary"
for FOLD in 0 1 2 3 4; do
  WANDB_MODE=disabled PYTHONWARNINGS=ignore $PYTHON $SCRIPT \
    --img_path "$D/image_surface" --gt_path "$D/gt_full/mask" --save_path "$OUT" \
    --train_type Vessel_FAZ --distance_type dist_signed11 \
    --num_epochs "$EPOCHS" --startpoint "$START" --fold "$FOLD" --eval_every 3 "$@" > "$OUT/train_fold$FOLD.log" 2>&1 || echo "fold $FOLD failed" >> "$OUT/FAILED"
done
