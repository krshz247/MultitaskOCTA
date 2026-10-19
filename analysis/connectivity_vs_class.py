"""Is vessel (dis)connectivity of the segmentation related to the disease class?
Out-of-fold predicted vessel masks -> per scan: predicted components / small fragments, ground-truth components,
clDice / topology precision / sensitivity against the ground truth. Reports the class means, how well these measures alone
predict the class (balanced accuracy, chance 0.33), and whether the CNN's mistakes relate to segmentation quality.
Usage: python analysis/connectivity_vs_class.py --joint runs/ILM_OPL_large_cv40 --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, sys, io, contextlib, json, warnings
import numpy as np, cv2, torch
from scipy import ndimage as ndi
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faz_vessel_seg_cls import CotrainingModelMulti, kfold_split
from eval_metrics import _cl_dice, _components, _dice_jaccard
from dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--data", required=True); a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); idx = {n: i for i, n in enumerate(names)}
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); S = 384; CLS = {"N": 0, "D": 1, "M": 2}
M = np.zeros((len(names), 7)); correct = np.zeros(len(names)); y = np.array([CLS[os.path.basename(n)[0]] for n in names]); fov = np.array([os.path.basename(n).split("_")[1] == "6" for n in names])
COLS = ["pred_components", "pred_fragments", "gt_components", "clDice", "topo_precision", "topo_sensitivity", "vessel_dice"]
for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(a.joint, f"best_fold{k}.pt"), map_location="cpu")); m.to(dev).eval()
    with torch.no_grad():
        for n in te:
            g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max(); x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(dev)
            so = m.module.seg_forward(x); cls = m.module.clf_forward(x, so[3], so[4], so[5]).argmax(1).item()
            P = torch.sigmoid(so[0]).cpu().numpy()[0].argmax(0) == 1
            G = cv2.resize(cv2.imread(n.replace("image_surface", "gt_vessel/mask"), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
            cl, tp, ts = _cl_dice(P, G); np_, nf = _components(P)
            M[idx[n]] = [np_, nf, _components(G)[0], cl, tp, ts, _dice_jaccard(P, G)[0]]; correct[idx[n]] = float(cls == CLS[os.path.basename(n)[0]])
    print("fold", k, "done", flush=True)
nm = ["Normal", "DR", "AMD"]
print(f"\n{'measure':18s} " + " ".join(f"{c:>9s}" for c in nm) + "   Kruskal-Wallis p (class)   p within 6 mm only")
for j, c in enumerate(COLS):
    ps = kruskal(*[M[y == k, j] for k in range(3)]).pvalue; p6 = kruskal(*[M[(y == k) & fov, j] for k in range(3)]).pvalue
    print(f"{c:18s} " + " ".join(f"{M[y == k, j].mean():9.3f}" for k in range(3)) + f"   {ps:10.4f}              {p6:10.4f}")
X = M[:, [0, 1, 3, 4, 5]]; bal = []
for k in range(5):
    _, _, te = kfold_split(names, 5, k); ti = np.array([idx[n] for n in te]); tri = np.setdiff1d(np.arange(len(names)), ti)
    bal.append(balanced_accuracy_score(y[ti], make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tri], y[tri]).predict(X[ti])))
print(f"\nconnectivity/quality features alone -> class: balanced accuracy {np.mean(bal):.3f} ± {np.std(bal, ddof=1):.3f} (chance 0.333)")
print("Spearman correlation between segmentation clDice and the CNN being correct: %.3f (p=%.3f)" % spearmanr(M[:, 3], correct))
json.dump({"class_means": {c: [float(M[y == k, j].mean()) for k in range(3)] for j, c in enumerate(COLS)}, "balanced_acc_connectivity_only": float(np.mean(bal))}, open("analysis/connectivity_results.json", "w"), indent=1)
