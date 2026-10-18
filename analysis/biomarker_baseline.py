"""Does the label leak through the field of view, and how far do simple mask biomarkers go?

Features are computed from the ground-truth FAZ / large-vessel masks (an upper bound for what a perfect
segmentation could provide) and classified with logistic regression / gradient boosting under the same
class-stratified 5-fold split as the networks (fold k = test, the rest train).
Usage: python analysis/biomarker_baseline.py [data_dir]
"""
import glob, os, sys, json
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faz_vessel_seg_cls import kfold_split

D = sys.argv[1] if len(sys.argv) > 1 else "Datasets/processed_OCTA500/ILM_OPL_large"
names = sorted(glob.glob(os.path.join(D, "image_surface", "*.bmp")))
CLS = {"N": 0, "D": 1, "M": 2}
MM_PER_PX = {"3": 3 / 304, "6": 6 / 400}
rows, y, fov = {}, {}, {}
for n in names:
    b = os.path.basename(n); c, f = b[0], b.split("_")[1]
    faz = cv2.imread(n.replace("image_surface", "gt_faz/mask"), 0) > 127
    ves = cv2.imread(n.replace("image_surface", "gt_vessel/mask"), 0) > 127
    mm = MM_PER_PX[f]
    props = max(regionprops(label(faz)), key=lambda r: r.area)
    area_mm2 = faz.sum() * mm * mm
    per_mm = props.perimeter * mm
    sk = skeletonize(ves)
    dist = ndi.distance_transform_edt(ves)
    n_cc = ndi.label(ves, structure=np.ones((3, 3)))[1]
    h, w = faz.shape
    rows[n] = [area_mm2, per_mm, 4 * np.pi * area_mm2 / max(per_mm ** 2, 1e-9), props.eccentricity, props.solidity,
               ves.sum() / (h * w), sk.sum() * mm / (h * w * mm * mm), 2 * dist[sk].mean() * mm if sk.any() else 0, n_cc / (h * w * mm * mm)]
    y[n] = CLS[c]; fov[n] = 0 if f == "3" else 1
FEATS = ["faz_area_mm2", "faz_perimeter_mm", "faz_circularity", "faz_eccentricity", "faz_solidity", "vessel_density", "skeleton_len_per_mm2", "vessel_width_mm", "components_per_mm2"]
X = np.array([rows[n] for n in names]); Y = np.array([y[n] for n in names]); F = np.array([fov[n] for n in names])
print(f"{len(names)} scans; class counts {np.bincount(Y).tolist()}; 6 mm share per class {[round(F[Y==k].mean(),2) for k in range(3)]}")
models = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
          "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, class_weight="balanced", random_state=0)}
sets = {"majority (always Normal)": None, "FOV only": np.c_[F], "biomarkers (no FOV)": X, "biomarkers + FOV": np.c_[X, F]}
res = {}
for sname, feats in sets.items():
    for mname, mk in models.items():
        if feats is None and mname == "gbm":
            continue
        accs, bal, f1s, perfov = [], [], [], {0: [], 1: []}
        for k in range(5):
            _, _, te = kfold_split(names, 5, k)
            te_idx = np.array([names.index(n) for n in te]); tr_idx = np.setdiff1d(np.arange(len(names)), te_idx)
            if feats is None:
                pred = np.zeros(len(te_idx), int)
            else:
                m = mk().fit(feats[tr_idx], Y[tr_idx]); pred = m.predict(feats[te_idx])
            accs.append(accuracy_score(Y[te_idx], pred)); bal.append(balanced_accuracy_score(Y[te_idx], pred)); f1s.append(f1_score(Y[te_idx], pred, average="macro"))
            for f in (0, 1):
                s = F[te_idx] == f; perfov[f].append(accuracy_score(Y[te_idx][s], pred[s]))
        key = f"{sname} [{mname}]" if feats is not None else sname
        res[key] = dict(acc=np.mean(accs), acc_sd=np.std(accs, ddof=1), bal=np.mean(bal), bal_sd=np.std(bal, ddof=1), f1=np.mean(f1s), f1_sd=np.std(f1s, ddof=1), acc3=np.mean(perfov[0]), acc6=np.mean(perfov[1]))
print(f"{'features [model]':36s} {'acc':>13s} {'balanced acc':>14s} {'macro-F1':>13s} {'acc 3mm':>8s} {'acc 6mm':>8s}")
for k, r in res.items():
    print(f"{k:36s} {r['acc']:.3f}±{r['acc_sd']:.3f} {r['bal']:.3f}±{r['bal_sd']:.3f} {r['f1']:.3f}±{r['f1_sd']:.3f} {r['acc3']:8.3f} {r['acc6']:8.3f}")
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in res.items()}, open("analysis/biomarker_baseline_results.json", "w"), indent=1)
