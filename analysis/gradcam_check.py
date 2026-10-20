"""TODO T2 (item 6): where does the classifier look? Class activation maps (CAM) of the last ResNeSt stage for the joint model and the
classification-only model, both out-of-fold (every scan is evaluated by the model of the fold in which it is a test scan).
Both classifiers end in global average pooling + one linear layer, so the CAM of class k is exactly sum_c W[k,c] * A_c (no gradients needed).

Reported, per true class and field of view (CAM of the predicted class, relu, normalised to sum 1):
  * share of the CAM mass in the outer band (outside the central 70 % square) and its enrichment = mass share / area share
  * enrichment on the FAZ, on the ground-truth vessels, and in the central 50 % square (= the 3 mm view of a 6 mm scan)
Border ablation: the outer band of every test scan is replaced by the median grey value and the scan is classified again
(a model that uses the borders / the scale of the field of view loses accuracy, a model that uses the centre does not).
Usage: python analysis/gradcam_check.py --data Datasets/processed_OCTA500/ILM_OPL_large --joint runs/ILM_OPL_large_cv40reg --cls runs/ILM_OPL_large_cls40reg
"""
import argparse, glob, io, contextlib, json, os, sys, warnings
import numpy as np, cv2, torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.classification_training import ResNeStClassifier
from octa.common import kfold_split
from octa.data.dataset import mean_and_std

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--joint", required=True); ap.add_argument("--cls", required=True)
ap.add_argument("--out", default="analysis/gradcam_results.json"); ap.add_argument("--fig_dir", default="analysis/figs")
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); idx = {n: i for i, n in enumerate(names)}
CLS = {"N": 0, "D": 1, "M": 2}; CN = ["Normal", "DR", "AMD"]; S = 384
y = np.array([CLS[os.path.basename(n)[0]] for n in names]); fov = np.array([0 if os.path.basename(n).split("_")[1] == "3" else 1 for n in names])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
yy, xx = np.mgrid[:S, :S]
def square(frac):  # central square covering frac of the side
    lo, hi = int(round(S * (1 - frac) / 2)), int(round(S * (1 + frac) / 2)); m = np.zeros((S, S), bool); m[lo:hi, lo:hi] = True; return m
INNER70, INNER50 = square(0.7), square(0.5)
BORDER = ~INNER70
def gt(n, key): return cv2.resize(cv2.imread(n.replace("image_surface", key), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
def load_joint(d, k):
    m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(d, f"best_fold{k}.pt"), map_location="cpu"))
    return m.to(dev).eval()
def load_cls(d, k):
    m = ResNeStClassifier(pretrain=None); m.load_state_dict(torch.load(os.path.join(d, f"best_fold{k}.pt"), map_location="cpu")); return m.to(dev).eval()

class Runner:
    """forward(x) -> (logits, cam (S,S) of the predicted class)"""
    def __init__(self, m, joint):
        self.m, self.joint = m, joint; self.feat = None
        core = m.module.clf_model if joint else m.encoder
        self.fc = core.my_fc if joint else m.fc
        core.layer4.register_forward_hook(lambda mod, i, o: setattr(self, "feat", o.detach()))
    def __call__(self, x):
        with torch.no_grad():
            if self.joint: so = self.m.module.seg_forward(x); logits = self.m.module.clf_forward(x, so[3], so[4], so[5])
            else: logits = self.m(x)
            k = int(logits.argmax(1)); w = self.fc.weight[k]
            cam = F.relu(torch.einsum("c,chw->hw", w, self.feat[0]))[None, None]
            cam = F.interpolate(cam, size=(S, S), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        return logits[0].cpu().numpy(), cam / max(cam.sum(), 1e-12)

stats = {nm: {} for nm in ("joint", "cls")}; pred = {nm: {c: np.zeros(len(names), int) for c in ("none", "band15", "band30")} for nm in ("joint", "cls")}
examples = {nm: [] for nm in ("joint", "cls")}
rows = {nm: [] for nm in ("joint", "cls")}
for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    runners = {"joint": Runner(load_joint(a.joint, k), True), "cls": Runner(load_cls(a.cls, k), False)}
    for n in te:
        g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max(); xs = ((g - mean) / std).astype(np.float32); x = torch.from_numpy(xs[None, None]).to(dev)
        faz, ves = gt(n, "gt_faz/mask"), gt(n, "gt_vessel/mask"); i = idx[n]
        for nm, r in runners.items():
            logits, cam = r(x); pred[nm]["none"][i] = int(logits.argmax())
            band_share = float(cam[BORDER].sum()); row = {"i": i, "band_share": band_share, "band_enrich": band_share / BORDER.mean(),
                   "faz_enrich": float(cam[faz].sum() / max(faz.mean(), 1e-9)) if faz.any() else np.nan, "ves_enrich": float(cam[ves].sum() / max(ves.mean(), 1e-9)),
                   "centre50_share": float(cam[INNER50].sum()), "centre50_enrich": float(cam[INNER50].sum() / INNER50.mean())}
            rows[nm].append(row)
            if y[i] == 2 and len(examples[nm]) < 8 and (len(examples[nm]) == 0 or i % 3 == 0): examples[nm].append((n, g.copy(), cam, int(logits.argmax())))
            for tag, frac in (("band15", 0.7), ("band30", 0.4)):  # keep the central square, fill the rest with the median grey value
                keep = square(frac); xa = xs.copy(); xa[~keep] = np.median(xs); pred[nm][tag][i] = int(r(torch.from_numpy(xa[None, None]).to(dev))[0].argmax())
    print("fold", k, "done", flush=True)

out = {}
print(f"\n{'model':6s} {'true class':10s} {'FOV':>4s} {'n':>4s} {'outer-band share':>17s} {'band enrich':>12s} {'FAZ enrich':>11s} {'vessel enrich':>14s} {'centre50 enrich':>16s}")
for nm in ("joint", "cls"):
    R = {r["i"]: r for r in rows[nm]}
    for c in range(3):
        for f, fn in ((0, "3mm"), (1, "6mm"), (None, "all")):
            sel = [i for i in range(len(names)) if y[i] == c and (f is None or fov[i] == f)]
            if not sel: continue
            m = lambda key: float(np.nanmean([R[i][key] for i in sel]))
            out[f"{nm}/{CN[c]}/{fn}"] = {"n": len(sel), **{key: m(key) for key in ("band_share", "band_enrich", "faz_enrich", "ves_enrich", "centre50_enrich")}}
            print(f"{nm:6s} {CN[c]:10s} {fn:>4s} {len(sel):4d} {m('band_share'):17.3f} {m('band_enrich'):12.2f} {m('faz_enrich'):11.2f} {m('ves_enrich'):14.2f} {m('centre50_enrich'):16.2f}")
print("\n(outer band = outside the central 70 % square, 51 % of the area; enrichment 1.0 = attention proportional to area)")
print(f"\nborder ablation (out-of-fold, 364 scans): balanced accuracy / recall Normal, DR, AMD")
for nm in ("joint", "cls"):
    for tag, lab in (("none", "unchanged"), ("band15", "outer 30 % replaced"), ("band30", "outer 60 % replaced (centre 40 % kept)")):
        p = pred[nm][tag]; rec = recall_score(y, p, average=None, labels=[0, 1, 2]); r3 = recall_score(y[fov == 1], p[fov == 1], average=None, labels=[0, 1, 2])
        out[f"ablation/{nm}/{tag}"] = {"balanced_acc": float(balanced_accuracy_score(y, p)), "recall": [float(v) for v in rec], "recall_6mm": [float(v) for v in r3]}
        print(f"{nm:6s} {lab:40s} {balanced_accuracy_score(y, p):.3f}   {rec[0]:.2f} {rec[1]:.2f} {rec[2]:.2f}   (6 mm scans only: {r3[0]:.2f} {r3[1]:.2f} {r3[2]:.2f})")
json.dump(out, open(a.out, "w"), indent=1)

os.makedirs(a.fig_dir, exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 6, figsize=(15, 10))
for r, nm in enumerate(("joint", "cls")):
    for j, (n, g, cam, p) in enumerate(examples[nm][:6]):
        for q, (im, ttl) in enumerate(((g, "image"), (None, "CAM"))):
            axx = ax[2 * r + q, j]; axx.axis("off")
            axx.imshow(g, cmap="gray")
            if q == 1: axx.imshow(cam / cam.max(), cmap="jet", alpha=0.45)
            axx.set_title(f"{nm} {os.path.basename(n)[:-4]} → {CN[p]}" if q == 1 else os.path.basename(n)[:-4], fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(a.fig_dir, "gradcam_amd_examples.png"), dpi=110); print("figure:", os.path.join(a.fig_dir, "gradcam_amd_examples.png"))
