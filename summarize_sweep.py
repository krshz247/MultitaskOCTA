"""Print the test metrics of each runs/<config>/results_fold<k>.json at the epoch with the best validation score."""
import glob, json, os, sys
fold = sys.argv[1] if len(sys.argv) > 1 else "0"
rows = []
for f in sorted(glob.glob(f"runs/*/results_fold{fold}.json")):
    h = json.load(open(f))["history"]
    joint = [e for e in h if e["epoch"] > json.load(open(f))["args"]["startpoint"]] or h
    score = lambda e: e["val"]["faz_dice"] + e["val"]["vessel_dice"] + e["val"]["vessel_cldice"] + e["val"]["f1_macro"]
    best = max(joint, key=score)
    t = best["test"]
    rows.append((os.path.basename(os.path.dirname(f)), best["epoch"], h[-1]["epoch"], t))
cols = ["faz_dice", "vessel_dice", "vessel_cldice", "acc", "f1_macro", "kappa", "heat_faz_r2", "heat_vessel_r2", "sdm_faz_r2", "sdm_vessel_r2"]
print(f"{'config':26s} ep/last " + " ".join(f"{c:>13s}" for c in cols))
for name, ep, last, t in rows:
    print(f"{name:26s} {ep:>3d}/{last:<4d} " + " ".join(f"{t.get(c, float("nan")):13.3f}" for c in cols))
