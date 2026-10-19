# Legacy scripts

Training / model scripts inherited from the original BSDA-Net code base. They are **not part of the current pipeline**, are not
tested with the pinned environment and may need `PYTHONPATH=..` to run (they import modules from the repository root).
The current entry points are `faz_vessel_seg_cls.py` (joint model) and `classification_baseline.py` (classification-only arm),
started through `run_experiment.py`. `classification.py` was the original classification baseline (torchvision ResNet50, unweighted
loss, random split without test set), replaced by `classification_baseline.py`.
