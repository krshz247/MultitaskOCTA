"""Classification-only ResNeSt50 baseline under the same protocol as the joint model (see octa/classification_training.py).

    python classification_baseline.py --img_path <D>/image_surface --gt_path <D>/gt_full/mask --save_path runs/x --fold 0 ...
"""
from octa.classification_training import main

if __name__ == "__main__":
    main()
