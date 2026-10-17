"""Re-evaluate a saved best_fold<k>.pt on the validation and test split of its fold (per-image metrics).

Usage: python eval_checkpoint.py --run runs/<config> --data Datasets/processed_OCTA500/<config> [--fold 0]
"""
import argparse, glob, json, os, sys
import torch
from torch import nn
from faz_vessel_seg_cls import CotrainingModelMulti, kfold_split
from eval_metrics import evaluate_loader
from utils import generate_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--encoder", default="timm-resnest50d")
a = ap.parse_args()

res = json.load(open(os.path.join(a.run, "results_fold{}.json".format(a.fold))))["args"]
names = sorted(glob.glob(os.path.join(a.data, "image_surface", "*.bmp")))
tr, va, te = kfold_split(names, res.get("num_folds", 5), a.fold)
gt = lambda L: [n.replace("image_surface", "gt_full/mask") for n in L]
_, devL, testL = generate_dataset(tr, va, te, [384, 384], res.get("batch_size", 4), res.get("val_batch_size", 4), "dist_signed11", False, "Vessel_FAZ", gt(tr), gt(va), gt(te))
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = nn.DataParallel(CotrainingModelMulti(a.encoder, None, True, None, 3))
model.module.load_state_dict(torch.load(os.path.join(a.run, "best_fold{}.pt".format(a.fold)), map_location="cpu"))
model.to(device)
out = {"val": evaluate_loader(model, devL, device), "test": evaluate_loader(model, testL, device)}
print(json.dumps({s: {k: round(v, 4) for k, v in m.items()} for s, m in out.items()}, indent=1))
