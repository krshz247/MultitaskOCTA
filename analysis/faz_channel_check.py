"""Which channel claims the FAZ pixels? For fold-0 test scans, the share of ground-truth FAZ pixels that the 3-class argmax assigns to FAZ / vessel / background,
and the mean probability of each channel inside the FAZ. Explains why the FAZ Dice of the argmax map differs between runs although the FAZ channel itself
is equally good (see analysis/faz_channel_check.txt). Needs runs/<run>/best_fold0.pt.
Usage: python analysis/faz_channel_check.py RUN [RUN ...] --data Datasets/processed_OCTA500/ILM_OPL_large
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
for run in a.runs:
    m = CotrainingModelMulti("timm-resnest50d", None, True, None, 3)
    m.load_state_dict(torch.load(os.path.join(run, "best_fold0.pt"), map_location="cpu"))
    m.to(dev).eval()
    cnt, prob = np.zeros(3), []
    with torch.no_grad():
        for n in te:
            g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max()
            x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
            p = torch.sigmoid(m.seg_forward(x)[0])[0].cpu().numpy()  # channels: FAZ, vessel, background
            faz = cv2.resize(cv2.imread(n.replace("image_surface", "gt_faz/mask"), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
            cnt += np.bincount(p.argmax(0)[faz], minlength=3)
            prob.append([p[c][faz].mean() for c in range(3)])
    cnt /= cnt.sum(); pm = np.mean(prob, axis=0)
    print(f"{os.path.basename(run)}: argmax inside the FAZ -> FAZ {cnt[0]:.2f} vessel {cnt[1]:.2f} background {cnt[2]:.2f} | mean probability FAZ {pm[0]:.2f} vessel {pm[1]:.2f} background {pm[2]:.2f}")
