"""Do random erasing and vessel breaks change what the classifiers decide? (questions about augmentation and connectivity)

For every test scan (out-of-fold models) the standardised 384 x 384 input is perturbed:
  * erase_random : one random square patch covering 2-8 % of the image, filled with the image mean (as in the 'erasing' augmentation)
  * gaps         : 15 small discs (radius 5 px) centred on ground-truth vessel pixels -> local vessel breaks
  * control      : 15 discs of the same size on non-vessel pixels outside the FAZ
We report how often the predicted class changes, and the mean drop of the probability of the true class. If the classifier
decided on vessel connectivity, 'gaps' would change decisions much more than 'control'. For the joint model we also report what
the segmentation does inside the erased patch (does it hallucinate or remove vessels?).
Usage: python analysis/perturbation_tests.py --joint runs/ILM_OPL_large_cv40 --cls runs/ILM_OPL_large_cls40 --clsreg runs/ILM_OPL_large_cls40reg --data Datasets/processed_OCTA500/ILM_OPL_large
"""
import argparse, glob, os, sys, io, contextlib, json, warnings
import numpy as np, cv2, torch
from torch import nn
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faz_vessel_seg_cls import CotrainingModelMulti, kfold_split
from classification_baseline import ResNeStClassifier
from dataset import mean_and_std

ap = argparse.ArgumentParser(); ap.add_argument("--joint", required=True); ap.add_argument("--cls", required=True); ap.add_argument("--clsreg", required=True); ap.add_argument("--data", required=True); ap.add_argument("--draws", type=int, default=3)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); S = 384
CLS = {"N": 0, "D": 1, "M": 2}; rng = np.random.default_rng(0)
yy, xx = np.mgrid[:S, :S]

def prep(n, mean, std):
    g = cv2.resize(cv2.imread(n, 0), (S, S)); g = g / g.max(); return ((g - mean) / std).astype(np.float32)

def erase_patch(img):
    side = int(np.sqrt(rng.uniform(0.02, 0.08)) * S); t, l = rng.integers(0, S - side, 2); out = img.copy(); out[t:t + side, l:l + side] = 0.0
    m = np.zeros((S, S), bool); m[t:t + side, l:l + side] = True; return out, m

def discs(img, centres, r=5):
    out = img.copy(); m = np.zeros((S, S), bool)
    for cy, cx in centres: d = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r; out[d] = 0.0; m |= d
    return out, m

rows = {m: {p: [] for p in ("erase_random", "gaps", "control")} for m in ("joint", "cls", "cls+reg")}   # (true class, flipped, drop in P(true))
seg = {"erase_random": []}
for k in range(5):
    tr, va, te = kfold_split(names, 5, k)
    with contextlib.redirect_stdout(io.StringIO()):
        mean, std = mean_and_std(tr)
    j = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); j.module.load_state_dict(torch.load(os.path.join(a.joint, f"best_fold{k}.pt"), map_location="cpu")); j.to(dev).eval()
    cs = {}
    for nm, d in (("cls", a.cls), ("cls+reg", a.clsreg)):
        c = ResNeStClassifier(pretrain=None); c.load_state_dict(torch.load(os.path.join(d, f"best_fold{k}.pt"), map_location="cpu")); cs[nm] = c.to(dev).eval()
    def predict(img):
        x = torch.from_numpy(img[None, None]).to(dev)
        with torch.no_grad():
            so = j.module.seg_forward(x); pj = torch.softmax(j.module.clf_forward(x, so[3], so[4], so[5]), 1)[0].cpu().numpy()
            return {"joint": pj, "cls": torch.softmax(cs["cls"](x), 1)[0].cpu().numpy(), "cls+reg": torch.softmax(cs["cls+reg"](x), 1)[0].cpu().numpy()}, so[0]
    for n in te:
        y = CLS[os.path.basename(n)[0]]; img = prep(n, mean, std)
        ves = cv2.resize(cv2.imread(n.replace("image_surface", "gt_vessel/mask"), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
        faz = cv2.resize(cv2.imread(n.replace("image_surface", "gt_faz/mask"), 0), (S, S), interpolation=cv2.INTER_NEAREST) > 127
        base, seg0 = predict(img); pred0 = torch.sigmoid(seg0).cpu().numpy()[0].argmax(0) == 1
        vy, vx = np.nonzero(ves); by, bx = np.nonzero(~ves & ~cv2.dilate(faz.astype(np.uint8), np.ones((15, 15))).astype(bool))
        for _ in range(a.draws):
            perts = {"erase_random": erase_patch(img)}
            iv = rng.choice(len(vy), 15, replace=False); ib = rng.choice(len(by), 15, replace=False)
            perts["gaps"] = discs(img, zip(vy[iv], vx[iv])); perts["control"] = discs(img, zip(by[ib], bx[ib]))
            for pn, (pimg, pm) in perts.items():
                pr, segp = predict(pimg)
                for m in rows: rows[m][pn].append((y, int(pr[m].argmax() != base[m].argmax()), float(base[m][y] - pr[m][y])))
                if pn == "erase_random":
                    pv = torch.sigmoid(segp).cpu().numpy()[0].argmax(0) == 1
                    seg["erase_random"].append((float(pv[pm].mean()), float(pred0[pm].mean()), float(ves[pm].mean()), float((pv != pred0)[~pm].mean())))
    print("fold", k, "done", flush=True)

nm = ["Normal", "DR", "AMD"]
print(f"\n{'model':8s} {'perturbation':13s} {'decision flips':>15s}   flips by true class (N / DR / AMD)   mean drop of P(true class)")
out = {}
for m in rows:
    for pn, v in rows[m].items():
        v = np.array(v); fl = v[:, 1].mean(); byc = [v[v[:, 0] == c][:, 1].mean() for c in range(3)]
        print(f"{m:8s} {pn:13s} {100 * fl:14.1f}%   {100 * byc[0]:5.1f}% / {100 * byc[1]:5.1f}% / {100 * byc[2]:5.1f}%        {v[:, 2].mean():+.4f}")
        out[f"{m}|{pn}"] = dict(flip=float(fl), flip_by_class=[float(b) for b in byc], drop_p_true=float(v[:, 2].mean()), n=len(v))
s = np.array(seg["erase_random"])
print(f"\njoint model segmentation under random erasing: vessel fraction inside the erased patch: predicted {s[:, 0].mean():.3f} (before erasing {s[:, 1].mean():.3f}, ground truth {s[:, 2].mean():.3f}); fraction of pixels outside the patch whose vessel label changes: {s[:, 3].mean():.4f}")
out["seg_erase"] = dict(pred_in_patch=float(s[:, 0].mean()), pred_before=float(s[:, 1].mean()), gt=float(s[:, 2].mean()), changed_outside=float(s[:, 3].mean()))
json.dump(out, open("analysis/perturbation_results.json", "w"), indent=1)
