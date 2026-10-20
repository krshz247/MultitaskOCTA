"""Do repeats of the same recipe, and different recipes, make the same mistakes? Does averaging several runs help?
For every run the out-of-fold class probabilities are computed once (each scan by the model of the fold in which it is a test scan, 364 scans)
and cached in <run>/oof_probs.npy. Then, per group of runs (repeats of one recipe):
  * balanced accuracy of every single run and of the average of the probabilities of 2 runs (all pairs) and of all runs of the group,
  * agreement of the hard predictions between two runs of the same group (seed noise) and between runs of different groups (recipe difference):
    if joint and plain models disagree no more than two seeds of the same recipe do, they behave as one model.
Usage: python analysis/ensemble_runs.py --data Datasets/processed_OCTA500/ILM_OPL_large --group A_joint RUN RUN ... --group C_plain RUN RUN
"""
import argparse, glob, io, contextlib, itertools, json, os, sys, warnings
import numpy as np, cv2, torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.classification_training import ResNeStClassifier
from octa.common import kfold_split
from octa.data.dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); idx = {n: i for i, n in enumerate(names)}
CLS = {"N": 0, "D": 1, "M": 2}; y = np.array([CLS[os.path.basename(n)[0]] for n in names]); S = 384
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def oof_probs(run):
    cache = os.path.join(run, "oof_probs.npy")
    if os.path.exists(cache): return np.load(cache)
    is_cls = json.load(open(os.path.join(run, "results_fold0.json"))).get("model") == "classification_baseline"
    P = np.zeros((len(names), 3))
    for k in range(5):
        tr, va, te = kfold_split(names, 5, k)
        with contextlib.redirect_stdout(io.StringIO()):
            mean, std = mean_and_std(tr)
        if is_cls:
            m = ResNeStClassifier(pretrain=None); m.load_state_dict(torch.load(os.path.join(run, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
        else:
            m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(run, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
        with torch.no_grad():
            for n in te:
                g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max(); x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
                if is_cls: p = m(x)
                else: so = m.module.seg_forward(x); p = m.module.clf_forward(x, so[3], so[4], so[5])
                P[idx[n]] = torch.softmax(p, 1)[0].cpu().numpy()
    np.save(cache, P); return P

bal = lambda p: balanced_accuracy_score(y, p.argmax(1))
G = {g[0]: {r: oof_probs(r) for r in g[1:]} for g in a.group}
for name, runs in G.items():
    print(f"\n=== {name}")
    for r, p in runs.items(): print(f"  single run {os.path.basename(r):36s} balanced acc {bal(p):.3f}  acc {accuracy_score(y, p.argmax(1)):.3f}")
    ps = list(runs.values())
    pairs = [bal(np.mean(c, 0)) for c in itertools.combinations(ps, 2)]
    singles = [bal(p) for p in ps]
    print(f"  mean single {np.mean(singles):.3f}" + (f" | ensemble of 2 (mean over {len(pairs)} pairs) {np.mean(pairs):.3f}" if pairs else "") + (f" | ensemble of all {len(ps)}: {bal(np.mean(ps, 0)):.3f}" if len(ps) > 2 else ""))
print("\nagreement of hard predictions between two runs (fraction of the 364 scans with the same predicted class; mean over pairs)")
agree = lambda p, q: float((p.argmax(1) == q.argmax(1)).mean())
for (n1, r1), (n2, r2) in itertools.combinations_with_replacement(G.items(), 2):
    v = [agree(p, q) for (ra, p), (rb, q) in itertools.product(r1.items(), r2.items()) if (n1 != n2 or ra < rb)]
    if v: print(f"  {n1:14s} vs {n2:14s} {np.mean(v):.3f}  ({len(v)} pair(s))")
