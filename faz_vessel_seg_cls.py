"""Joint RV + FAZ segmentation and classification, one cross-validation fold per call.

    python faz_vessel_seg_cls.py --img_path <D>/image_surface --gt_path <D>/gt_full/mask --save_path runs/x --fold 0 ...

All options: docs/OPTIONS.md (or --help). The implementation is octa/joint_training.py; experiments are normally started with run_experiment.py.
"""
from octa.joint_training import main

if __name__ == "__main__":
    main()
