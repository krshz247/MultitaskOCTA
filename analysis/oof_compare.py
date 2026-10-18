"""Pooled out-of-fold comparison of the joint model and the classification-only baseline, with uncertainty.
Every scan is predicted by the model of the fold in which it is a test scan (364 predictions per model).
Reports accuracy / balanced accuracy / macro-F1 with 95 % bootstrap intervals, a paired bootstrap of the difference,
McNemar's test, and the effect of flip test-time augmentation (TTA).
Usage: python analysis/oof_compare.py --joint runs/ILM_OPL_large_cv40 --cls runs/ILM_OPL_large_cls40 --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, sys, io, contextlib, warnings
import numpy as np, cv2, torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faz_vessel_seg_cls import CotrainingModelMulti, kfold_split
from classification_baseline import ResNeStClassifier
from dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--cls", required=True); ap.add_argument("--data", required=True)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLS = {"N": 0, "D": 1, "M": 2}
y = np.array([CLS[os.path.basename(n)[0]] for n in names]); fov = np.array([os.path.basename(n).split("_")[1] for n in names])
idx = {n: i for i, n in enumerate(names)}
TTA = [lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2)]
P = {m: {t: np.zeros((len(names), 3)) for t in ("plain", "tta")} for m in ("joint", "cls")}

def prep(n, mean, std):
    g = cv2.resize(cv2.imread(n, 0), (384, 384)); g = g / g.max()
    return torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)

for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    j = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); j.module.load_state_dict(torch.load(os.path.join(a.joint, f"best_fold{k}.pt"), map_location="cpu")); j.to(dev).eval()
    c = ResNeStClassifier(pretrain=None); c.load_state_dict(torch.load(os.path.join(a.cls, f"best_fold{k}.pt"), map_location="cpu")); c.to(dev).eval()
    def joint_p(x): so = j.module.seg_forward(x); return torch.softmax(j.module.clf_forward(x, so[3], so[4], so[5]), 1)[0].cpu().numpy()
    def cls_p(x): return torch.softmax(c(x), 1)[0].cpu().numpy()
    with torch.no_grad():
        for n in te:
            x = prep(n, mean, std)
            for m, f in (("joint", joint_p), ("cls", cls_p)):
                ps = [f(t(x)) for t in TTA]; P[m]["plain"][idx[n]] = ps[0]; P[m]["tta"][idx[n]] = np.mean(ps, 0)
    print("fold", k, "done", flush=True)
np.savez("analysis/oof_probs.npz", y=y, fov=fov, **{f"{m}_{t}": P[m][t] for m in P for t in P[m]})

rng = np.random.default_rng(0); B = 2000
def metrics(yt, yp): return [accuracy_score(yt, yp), balanced_accuracy_score(yt, yp), f1_score(yt, yp, average="macro")]
boots = rng.integers(0, len(y), (B, len(y)))
def ci(pred):
    pt = metrics(y, pred); bs = np.array([metrics(y[b], pred[b]) for b in boots]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi
print(f"\n{'model':22s} {'accuracy':>22s} {'balanced acc':>22s} {'macro-F1':>22s}")
preds = {}
for m in P:
    for t in ("plain", "tta"):
        p = P[m][t].argmax(1); preds[(m, t)] = p; pt, lo, hi = ci(p)
        print(f"{m + ' ' + t:22s} " + " ".join(f"{pt[i]:.3f} [{lo[i]:.3f},{hi[i]:.3f}]" for i in range(3)))
print("\npaired difference joint - cls-only (pooled out-of-fold, 364 scans):")
for t in ("plain", "tta"):
    pj, pc = preds[("joint", t)], preds[("cls", t)]
    d = np.array([np.subtract(metrics(y[b], pj[b]), metrics(y[b], pc[b])) for b in boots]); mean = np.subtract(metrics(y, pj), metrics(y, pc))
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    only_j = int(((pj == y) & (pc != y)).sum()); only_c = int(((pj != y) & (pc == y)).sum())
    pval = binomtest(only_j, only_j + only_c, 0.5).pvalue if only_j + only_c else 1.0
    print(f"  {t:5s}: acc {mean[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]  bal-acc {mean[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]  F1 {mean[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]  McNemar p={pval:.3f} (joint-only correct {only_j}, cls-only correct {only_c})")
print("\nper field of view (plain):")
for m in P:
    p = preds[(m, "plain")]
    for f in ("3", "6"):
        s = fov == f; print(f"  {m:5s} {f} mm: n={s.sum()} acc {accuracy_score(y[s], p[s]):.3f} balanced acc {balanced_accuracy_score(y[s], p[s]):.3f}")
