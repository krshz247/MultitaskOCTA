# Legacy scripts

Training / model scripts inherited from the original BSDA-Net code base. They are **not part of the current pipeline**, are not
tested with the pinned environment and may need `PYTHONPATH=..` to run (they import modules from the repository root).
The current entry points are `faz_vessel_seg_cls.py` (joint model) and `classification_baseline.py` (classification-only arm),
started through `run_experiment.py`. `classification.py` was the original classification baseline (torchvision ResNet50, unweighted
loss, random split without test set), replaced by `classification_baseline.py`.

Layout after the clean-up of 2026-10-19: besides the early training scripts this folder now also holds the old
`faz_seg_cls.py` (FAZ-only BSDA-Net), `train_seg_clf.py`, `evaluate.py`, `models.py` (UNet variants), `augmentation.py`, `inference/`
and `legacy_utils.py` (helper functions that used to live in `utils.py`), `smp_model_legacy.py` and four tiny compatibility shims (`utils.py`, `losses.py`, `dataset.py`, `smp_model.py`) that forward to the `octa` package. Run any of them from the repository root with
`PYTHONPATH=.:legacy python legacy/<script>.py`. Their import status is unchanged by the move: `faz_seg_cls`, `models`, `augmentation`,
`legacy_utils` and `train_smp_y` import; `train_seg_clf`, `evaluate`, `classification` (missing `My_multiLoss`), `train_cls` (`resnest`),
`train_smp` (`apex`), `basic_jointlearning` (`smp_model_basic`) and `criterion` (`surface_distance`) were already broken before the move.
