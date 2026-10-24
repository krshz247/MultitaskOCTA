"""Errors of the FAZ class map on the fold-0 test scans: precision and recall of the 3-class argmax FAZ and of the thresholded FAZ channel (> 0.5),
and how many argmax false positives have a FAZ probability below 0.5 (i.e. win only because the other channels are lower).
Used to read the test of --faz_bg_exclusion (analysis/faz_bg_exclusion_test.txt). Needs runs/<run>/best_fold0.pt.
Usage: python analysis/faz_argmax_errors.py RUN [RUN ...] --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, io, contextlib, os, sys
import numpy as np, cv2, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.common import kfold_split
from octa.data.dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--data", default="Datasets/processed_OCTA500/ILM_OPL_large")
a = ap.parse_args()
S = 384
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
tr, _, te = kfold_split(names, 5, 0)
with contextlib.redirect_stdout(io.StringIO()):
    mean, std = mean_and_std(tr)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dice = lambda p, g: 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
for run in a.runs:
    m = CotrainingModelMulti("timm-resnest50d", None, True, None, 3)
    m.load_state_dict(torch.load(os.path.join(run, "best_fold0.pt"), map_location="cpu"))
    m.to(dev).eval()
    c = dict(tp=0, fp=0, fn=0, tpt=0, fpt=0, fnt=0, fp_low=0)
    dA, dT = [], []
    with torch.no_grad():
        for n in te:
            g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max()
            x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
            p = torch.sigmoid(m.seg_forward(x)[0])[0].cpu().numpy()  # channels: FAZ, vessel, background
            gt = cv2.resize(cv2.imread(n.replace("image_surface", "gt_faz/mask"), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
            am, th = p.argmax(0) == 0, p[0] > 0.5
            dA.append(dice(am, gt)); dT.append(dice(th, gt))
            c["tp"] += (am & gt).sum(); c["fp"] += (am & ~gt).sum(); c["fn"] += (~am & gt).sum()
            c["tpt"] += (th & gt).sum(); c["fpt"] += (th & ~gt).sum(); c["fnt"] += (~th & gt).sum()
            c["fp_low"] += (am & ~gt & (p[0] < 0.5)).sum()
    print(f"{os.path.basename(run):28s} argmax: precision {c['tp'] / (c['tp'] + c['fp']):.3f} recall {c['tp'] / (c['tp'] + c['fn']):.3f} (per-scan Dice {np.mean(dA):.3f})"
          f" | threshold: precision {c['tpt'] / (c['tpt'] + c['fpt']):.3f} recall {c['tpt'] / (c['tpt'] + c['fnt']):.3f} (Dice {np.mean(dT):.3f})"
          f" | argmax false positive pixels {c['fp']}, of them with FAZ probability < 0.5: {c['fp_low'] / max(c['fp'], 1):.2f}")
