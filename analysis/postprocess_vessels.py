"""Effect of simple vessel post-processing (remove small fragments / close tiny gaps) on a saved checkpoint.
Usage: python analysis/postprocess_vessels.py --run runs/<config> --data Datasets/processed_OCTA500/<config> [--fold 0]"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy import ndimage as ndi
from torch import nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from octa.joint_training import CotrainingModelMulti
from octa.common import kfold_split
from octa.metrics import _cl_dice, _components, _dice_jaccard
from octa.data.loaders import generate_dataset

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--fold", type=int, default=0)
a = ap.parse_args()
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp"))); tr, va, te = kfold_split(names, 5, a.fold)
gt = lambda L: [n.replace("image_surface", "gt_full/mask") for n in L]
_, _, testL = generate_dataset(tr, va, te, [384, 384], 4, 4, "dist_signed11", False, "Vessel_FAZ", gt(tr), gt(va), gt(te), full_eval=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = nn.DataParallel(CotrainingModelMulti("timm-resnest50d", None, True, None, 3)); m.module.load_state_dict(torch.load(os.path.join(a.run, f"best_fold{a.fold}.pt"), map_location="cpu")); m.to(dev).eval()

def drop_small(b, k):
    lab, n = ndi.label(b, structure=np.ones((3, 3))); sizes = np.bincount(lab.ravel()); keep = sizes >= k; keep[0] = False; return keep[lab]
def close(b): return ndi.binary_closing(b, structure=np.ones((3, 3)))
variants = {"raw": lambda b: b, "drop <30 px": lambda b: drop_small(b, 30), "drop <60 px": lambda b: drop_small(b, 60), "close 3x3": close, "close + drop <30": lambda b: drop_small(close(b), 30)}
acc = {k: [] for k in variants}
with torch.no_grad():
    for x, mask, *_ in testL:
        pr = torch.sigmoid(m.module.seg_forward(x.float().to(dev))[0]).cpu().numpy().argmax(1)
        for b in range(len(x)):
            G = mask[b, 1].numpy() > .5
            for k, f in variants.items():
                P = f(pr[b] == 1); cl, tp, ts = _cl_dice(P, G)
                acc[k].append([_dice_jaccard(P, G)[0], cl, tp, ts, _components(P)[0]])
print(f"{'variant':20s} {'dice':>6s} {'clDice':>7s} {'topoP':>6s} {'topoS':>6s} {'components (GT ~%d)' % 19}")
for k, v in acc.items():
    v = np.mean(v, 0); print(f"{k:20s} {v[0]:6.3f} {v[1]:7.3f} {v[2]:6.3f} {v[3]:6.3f} {v[4]:8.1f}")
