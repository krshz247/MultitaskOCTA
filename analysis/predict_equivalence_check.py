"""Check that octa/predict.py reproduces the stored out-of-fold probabilities: load ONE exported fold model (fold 0) and predict the fold-0 test scans;
compare with <run>/oof_probs.npy (written by analysis/ensemble_runs.py, same checkpoints). No training. Needs the local data and the exported weights.
Usage: python analysis/predict_equivalence_check.py --weights weights/joint_reg_s1 --run runs/ILM_OPL_large_cv40reg_s1 --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, shutil, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.common import kfold_split
from octa.predict import CLASSES, load_ensemble, predict

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); idx = {n: i for i, n in enumerate(names)}
oof = np.load(os.path.join(a.run, "oof_probs.npy"))
_, _, te = kfold_split(names, 5, 0)
with tempfile.TemporaryDirectory() as d:
    for f in ("fold0.pt", "stats.json"):
        shutil.copy(os.path.join(a.weights, f), d)
    ens = load_ensemble(d, "cpu")
    sel = te[:a.n] if a.n else te
    diff = [np.abs(np.array([predict(ens, n)["probabilities"][c] for c in CLASSES]) - oof[idx[n]]).max() for n in sel]
print(f"{len(diff)} fold-0 test scans: max |probability difference| between predict() and the stored out-of-fold probabilities = {max(diff):.2e}")
