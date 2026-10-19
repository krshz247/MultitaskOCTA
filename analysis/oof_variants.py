"""Pooled out-of-fold comparison of several variants (joint models and classification-only models) with bootstrap intervals,
paired bootstrap differences to a reference and McNemar tests. Every scan is predicted by the model of the fold in which it is a test scan.
Usage: python analysis/oof_variants.py --ref NAME --variant NAME DIR [--variant NAME DIR ...] --data Datasets/processed_OCTA500/ILM_OPL_large
(a DIR with results of classification_baseline.py is recognised automatically)"""
import argparse, glob, io, contextlib, json, os, sys, warnings
import numpy as np, cv2, torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.classification_training import ResNeStClassifier
from octa.common import kfold_split
from octa.data.dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); idx = {n: i for i, n in enumerate(names)}
CLS = {"N": 0, "D": 1, "M": 2}; y = np.array([CLS[os.path.basename(n)[0]] for n in names]); S = 384
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
P = {n: np.zeros((len(names), 3)) for n, _ in a.variant}
for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    xs = {}
    for n in te:
        g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max(); xs[n] = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
    for name, d in a.variant:
        is_cls = json.load(open(os.path.join(d, f"results_fold{k}.json"))).get("model") == "classification_baseline"
        if is_cls:
            m = ResNeStClassifier(pretrain=None); m.load_state_dict(torch.load(os.path.join(d, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
        else:
            m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(d, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
        with torch.no_grad():
            for n, x in xs.items():
                if is_cls: p = m(x)
                else: so = m.module.seg_forward(x); p = m.module.clf_forward(x, so[3], so[4], so[5])
                P[name][idx[n]] = torch.softmax(p, 1)[0].cpu().numpy()
    print("fold", k, "done", flush=True)
rng = np.random.default_rng(0); boots = rng.integers(0, len(y), (2000, len(y)))
metrics = lambda yt, yp: [accuracy_score(yt, yp), balanced_accuracy_score(yt, yp), f1_score(yt, yp, average="macro")]
pred = {n: P[n].argmax(1) for n in P}
print(f"\n{'variant':24s} {'accuracy':>22s} {'balanced acc':>22s} {'macro-F1':>22s}")
for n in pred:
    pt = metrics(y, pred[n]); bs = np.array([metrics(y[b], pred[n][b]) for b in boots]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"{n:24s} " + " ".join(f"{pt[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]" for i in range(3)))
print(f"\npaired differences to {a.ref} (pooled out-of-fold, 364 scans, 95 % bootstrap interval; McNemar on the correct/incorrect decisions)")
for n in pred:
    if n == a.ref: continue
    d = np.array([np.subtract(metrics(y[b], pred[n][b]), metrics(y[b], pred[a.ref][b])) for b in boots]); mean = np.subtract(metrics(y, pred[n]), metrics(y, pred[a.ref])); lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    on, orr = int(((pred[n] == y) & (pred[a.ref] != y)).sum()), int(((pred[n] != y) & (pred[a.ref] == y)).sum()); p = binomtest(on, on + orr, 0.5).pvalue if on + orr else 1.0
    print(f"{n:24s} acc {mean[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]  bal-acc {mean[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]  F1 {mean[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]  McNemar p={p:.3f} (only {n} right {on}, only {a.ref} right {orr})")
