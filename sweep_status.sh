#!/usr/bin/env bash
# Progress of the pilot sweep: epochs done per config (of the total passed as $1, default 30) and latest time.
TOTAL=${1:-30}
for d in runs/*/; do
  n=$(grep -c '^Epoch' "$d/train.log" 2>/dev/null || echo 0)
  echo "$(basename "$d"): epoch $n/$TOTAL  (log updated $(date -r "$d/train.log" +%H:%M:%S))"
done
pgrep -f faz_vessel_seg_cls >/dev/null && echo "training process: running" || echo "training process: not running"
