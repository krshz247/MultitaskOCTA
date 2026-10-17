#!/usr/bin/env bash
# Build the two files to upload to Kaggle (Datasets -> New Dataset):
#   kaggle_upload/multitaskocta_code.zip     the committed code (git archive HEAD)
#   kaggle_upload/octa500_b5_subset.zip      only what the builder needs for the B5 / large-vessel setup (~260 MB)
# Usage: kaggle/make_bundles.sh [out_dir] [path/to/OCTA500]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-kaggle_upload}; RAW=$(cd "${2:-../OCTA500}" && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
git archive --format=zip --prefix=MultitaskOCTA/ -o "$OUT/multitaskocta_code.zip" HEAD
rm -f "$OUT/octa500_b5_subset.zip"
(cd "$RAW" && zip -qr "$OUT/octa500_b5_subset.zip" \
  "OCTA_3mm/Projection Maps/OCTA(ILM_OPL)" "OCTA_6mm/Projection Maps/OCTA(ILM_OPL)" \
  Label/GT_FAZ Label/GT_LargeVessel Label/GT_Capillary Label/TextLabels)
ls -lh "$OUT"
