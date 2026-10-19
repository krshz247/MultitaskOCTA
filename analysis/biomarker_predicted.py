"""TODO T1: do segmentation-derived measures carry class information?
Out-of-fold predicted masks (each scan is segmented by the model of the fold in which it is a test scan) and the ground-truth
masks are turned into OCTA biomarkers (FAZ geometry, vessel density, skeleton length, width, branching, fractal dimension,
parafoveal and quadrant densities); simple classifiers are trained with the same 5 class-stratified folds. The out-of-fold
class probabilities of the joint CNN are added as a stacking feature for an exploratory comparison.
Usage: python analysis/biomarker_predicted.py --joint runs/ILM_OPL_large_cv40 --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, sys, json, io, contextlib, warnings
import numpy as np, cv2, torch
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.common import kfold_split
from octa.data.dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--data", required=True)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
CLS = {"N": 0, "D": 1, "M": 2}; MM = {"3": 3.0, "6": 6.0}
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
S = 384

def box_count_dim(sk):
    sizes = [2, 4, 8, 16, 32]; counts = []
    for s in sizes:
        h = sk[: S // s * s, : S // s * s].reshape(S // s, s, S // s, s).any(axis=(1, 3)); counts.append(max(h.sum(), 1))
    return -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

def features(faz, ves, fov_mm):
    mm = fov_mm / S; a_mm2 = (mm ** 2)
    f = []
    if faz.any():
        pr = max(regionprops(label(faz)), key=lambda r: r.area); area = pr.area * a_mm2; per = pr.perimeter * mm
        f += [area, per, 4 * np.pi * area / max(per ** 2, 1e-9), pr.eccentricity, pr.solidity]; cy, cx = pr.centroid
    else:
        f += [0, 0, 0, 0, 0]; cy, cx = S / 2, S / 2
    sk = skeletonize(ves); nb = ndi.convolve(sk.astype(int), np.ones((3, 3)), mode="constant") * sk
    branch = ((nb >= 4) & sk).sum(); ends = ((nb == 2) & sk).sum()
    dist = ndi.distance_transform_edt(ves); valid = ~faz
    f += [ves[valid].mean(), sk.sum() * mm / (valid.sum() * a_mm2), 2 * dist[sk].mean() * mm if sk.any() else 0, ndi.label(ves, structure=np.ones((3, 3)))[1] / (S * S * a_mm2),
          branch / (S * S * a_mm2), ends / (S * S * a_mm2), box_count_dim(sk)]
    yy, xx = np.mgrid[:S, :S]; r = np.hypot(yy - cy, xx - cx) * mm
    for lo, hi in ((0.0, 0.6), (0.6, 1.2), (1.2, 2.0)):  # parafoveal annuli (mm around the FAZ centroid)
        ring = (r >= lo) & (r < hi) & valid; f.append(ves[ring].mean() if ring.any() else 0)
    for qy in (0, 1):
        for qx in (0, 1):
            q = np.zeros((S, S), bool); q[qy * S // 2:(qy + 1) * S // 2, qx * S // 2:(qx + 1) * S // 2] = True; q &= valid; f.append(ves[q].mean())
    return f
FEATS = ["faz_area", "faz_perimeter", "faz_circ", "faz_ecc", "faz_solidity", "ves_density", "skel_len", "width", "components", "branch_pts", "end_pts", "fractal_dim",
         "dens_0-0.6mm", "dens_0.6-1.2mm", "dens_1.2-2mm", "dens_q1", "dens_q2", "dens_q3", "dens_q4"]

def load_gt(n):
    r = lambda k: cv2.resize(cv2.imread(n.replace("image_surface", k), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
    return r("gt_faz/mask"), r("gt_vessel/mask")

Fgt, Fpred, Y, FOV, P_cnn = {}, {}, {}, {}, {}
for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(a.joint, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
    with torch.no_grad():
        for n in te:
            b = os.path.basename(n); g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max()
            x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
            so = m.module.seg_forward(x); cls = m.module.clf_forward(x, so[3], so[4], so[5])
            pm = torch.sigmoid(so[0]).cpu().numpy()[0].argmax(0)
            Fpred[n] = features(pm == 0, pm == 1, MM[b.split("_")[1]]); Fgt[n] = features(*load_gt(n), MM[b.split("_")[1]])
            Y[n] = CLS[b[0]]; FOV[n] = 0 if b.split("_")[1] == "3" else 1; P_cnn[n] = torch.softmax(cls, 1)[0].cpu().numpy()
    print("fold", k, "done", flush=True)
Xg = np.array([Fgt[n] for n in names]); Xp = np.array([Fpred[n] for n in names]); y = np.array([Y[n] for n in names]); fv = np.array([FOV[n] for n in names]); pc = np.array([P_cnn[n] for n in names])
cnn_pred = pc.argmax(1)
models = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
          "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, class_weight="balanced", random_state=0)}
sets = {"GT masks": Xg, "predicted masks": Xp, "GT masks + FOV": np.c_[Xg, fv], "predicted masks + FOV": np.c_[Xp, fv],
        "CNN probs only (stack)": pc, "CNN probs + predicted-mask biomarkers": np.c_[pc, Xp]}
print(f"\n{len(names)} scans. Joint CNN, out-of-fold, as reference: acc {accuracy_score(y, cnn_pred):.3f}, balanced acc {balanced_accuracy_score(y, cnn_pred):.3f}, macro-F1 {f1_score(y, cnn_pred, average='macro'):.3f}")
print(f"{'features [model]':48s} {'acc':>12s} {'balanced acc':>14s} {'macro-F1':>12s}")
rows = {}
for sname, X in sets.items():
    for mname, mk in models.items():
        acc, bal, f1 = [], [], []
        for k in range(5):
            _, _, te = kfold_split(names, 5, k); ti = np.array([names.index(n) for n in te]); tri = np.setdiff1d(np.arange(len(names)), ti)
            p = mk().fit(X[tri], y[tri]).predict(X[ti])
            acc.append(accuracy_score(y[ti], p)); bal.append(balanced_accuracy_score(y[ti], p)); f1.append(f1_score(y[ti], p, average="macro"))
        rows[f"{sname} [{mname}]"] = [float(np.mean(acc)), float(np.std(acc, ddof=1)), float(np.mean(bal)), float(np.std(bal, ddof=1)), float(np.mean(f1)), float(np.std(f1, ddof=1))]
        r = rows[f"{sname} [{mname}]"]; print(f"{sname + ' [' + mname + ']':48s} {r[0]:.3f}±{r[1]:.3f} {r[2]:.3f}±{r[3]:.3f}  {r[4]:.3f}±{r[5]:.3f}")
# agreement between predicted and GT biomarkers
corr = [np.corrcoef(Xg[:, j], Xp[:, j])[0, 1] for j in range(Xg.shape[1])]
print("\ncorrelation GT vs predicted-mask feature:", {FEATS[j]: round(float(corr[j]), 2) for j in range(len(FEATS))})
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)).fit(Xp, y)
imp = np.abs(lr[-1].coef_).mean(0); order = np.argsort(-imp)[:6]
print("largest |logistic coefficient| (predicted-mask features):", [(FEATS[j], round(float(imp[j]), 2)) for j in order])
json.dump(rows, open("analysis/biomarker_predicted_results.json", "w"), indent=1)
