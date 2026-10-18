"""Scale-swap test for the field-of-view shortcut (TODO T2).

6 mm scans of the test fold are centre-cropped (to 80 % and to 50 % of the side; 50 % = the 3 mm field of view of the same
retina) and resized to 384 x 384 like every other scan, so they look like 3 mm scans. Each out-of-fold model (fold k is
evaluated on its own test scans) predicts the original and the cropped scan. If the networks relied on the image scale /
field of view to detect AMD and DR, the predictions of AMD and DR scans would move towards 'Normal' (3 mm scans are mostly
Normal) when only the scale changes.
Usage: python analysis/fov_shortcut_test.py --joint runs/ILM_OPL_large_cv40 --cls runs/ILM_OPL_large_cls40 --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, sys, json, io, contextlib
import numpy as np, cv2, torch
from torch import nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faz_vessel_seg_cls import CotrainingModelMulti, kfold_split
from classification_baseline import ResNeStClassifier
from dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--cls", required=True); ap.add_argument("--data", required=True)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FRACS = [1.0, 0.8, 0.5]
res = {m: {f: {c: [] for c in (0, 1, 2)} for f in FRACS} for m in ("joint", "cls")}   # model -> crop -> true class -> list of prob vectors
probs_n = {m: {f: [] for f in FRACS} for m in res}
CLS = {"N": 0, "D": 1, "M": 2}

def prep(path, frac, mean, std):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE); h, w = g.shape
    if frac < 1:
        ch, cw = int(h * frac), int(w * frac); t, l = (h - ch) // 2, (w - cw) // 2; g = g[t:t + ch, l:l + cw]
    g = cv2.resize(g, (384, 384)); g = g / g.max()
    return torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32))

for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    joint = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); joint.module.load_state_dict(torch.load(os.path.join(a.joint, f"best_fold{k}.pt"), map_location="cpu")); joint.to(dev).eval()
    clsm = ResNeStClassifier(pretrain=None); clsm.load_state_dict(torch.load(os.path.join(a.cls, f"best_fold{k}.pt"), map_location="cpu")); clsm.to(dev).eval()
    with torch.no_grad():
        for n in te:
            b = os.path.basename(n)
            if b.split("_")[1] != "6":
                continue
            c = CLS[b[0]]
            for f in FRACS:
                x = prep(n, f, mean, std).to(dev)
                so = joint.module.seg_forward(x); pj = torch.softmax(joint.module.clf_forward(x, so[3], so[4], so[5]), 1)[0].cpu().numpy()
                pc = torch.softmax(clsm(x), 1)[0].cpu().numpy()
                res["joint"][f][c].append(pj); res["cls"][f][c].append(pc)

names_c = ["Normal", "DR", "AMD"]
out = {}
for m in res:
    print(f"\n=== {m} model, 6 mm test scans (out-of-fold). Crop 1.0 = original; 0.5 = 3 mm field of view")
    print(f"{'true class':10s} {'n':>3s} {'crop':>5s} | predicted share Normal / DR / AMD | mean P(Normal) | accuracy")
    for c in (0, 1, 2):
        for f in FRACS:
            P = np.array(res[m][f][c])
            if len(P) == 0: continue
            pred = P.argmax(1); share = [(pred == j).mean() for j in range(3)]
            print(f"{names_c[c]:10s} {len(P):3d} {f:5.1f} | {share[0]:.2f} / {share[1]:.2f} / {share[2]:.2f}          | {P[:, 0].mean():.3f}          | {(pred == c).mean():.2f}")
            out[f"{m}|{names_c[c]}|{f}"] = dict(n=len(P), share=[float(s) for s in share], mean_p_normal=float(P[:, 0].mean()), acc=float((pred == c).mean()))
json.dump(out, open("analysis/fov_shortcut_results.json", "w"), indent=1)
