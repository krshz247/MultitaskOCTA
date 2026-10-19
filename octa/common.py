"""Helpers shared by the training scripts: seeding, run provenance, LR schedules and the cross-validation split."""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def git_commit():
    """Short hash of the code that produced a run (+ '-dirty' if there are uncommitted changes)."""
    import subprocess
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, stderr=subprocess.DEVNULL).decode().strip()
        return h + ("-dirty" if dirty else "")
    except Exception:  # no git (e.g. a Kaggle bundle): kaggle/make_bundles.sh stores the commit in a GIT_COMMIT file
        try:
            return "bundle-" + open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GIT_COMMIT")).read().strip()
        except OSError:
            return "unknown"


def make_scheduler(optimizer, name, total_steps, peak_lrs, warmup_frac=0.1):
    """Per-iteration schedule whose peak learning rates are the (thesis) LRs given on the command line.
    onecycle: warm-up from peak/25 to the peak, then cosine decay to peak/1e4.
    cosine:   linear warm-up from peak/100, then cosine decay to 0."""
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=peak_lrs, total_steps=total_steps, pct_start=warmup_frac,
                                                   anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)
    warm = max(1, int(total_steps * warmup_frac))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warm))], milestones=[warm])


def kfold_split(img_names, num_folds, fold, seed=2021, stratify_fov=False):
    """Stratified (by the N/D/M prefix of the file name) k-fold split: fold `fold` is the test set,
    fold `fold + 1` the validation set and the remaining folds the training set.
    stratify_fov=True stratifies by class x field of view (file names are <N|D|M>_<3|6>_<id>)."""
    assert 0 <= fold < num_folds
    rng = random.Random(seed)
    folds = [[] for _ in range(num_folds)]
    by_class = {}
    for name in img_names:
        key = os.path.basename(name)[:3] if stratify_fov else os.path.basename(name)[0]
        by_class.setdefault(key, []).append(name)
    offset = 0
    for cls in sorted(by_class):
        names = sorted(by_class[cls])
        rng.shuffle(names)
        for i, name in enumerate(names):
            folds[(i + offset) % num_folds].append(name)
        offset += len(names)  # keeps fold sizes balanced across classes
    test = folds[fold]
    val = folds[(fold + 1) % num_folds]
    train = [n for i, f in enumerate(folds) if i not in (fold, (fold + 1) % num_folds) for n in f]
    return train, val, test
