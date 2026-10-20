"""How much does the choice of the reported epoch matter? Uses only the per-epoch records stored in results_fold<k>.json
(val and test metrics every 3 epochs), so nothing is trained or evaluated again.

Rules compared (all use epochs after the classifier started, i.e. > startpoint, for the joint model):
  current      validation score of the pipeline (joint: FAZ Dice + vessel Dice + clDice + macro-F1; classification-only: macro-F1)
  val_f1       highest validation macro-F1
  val_bal      highest validation balanced accuracy
  last         the final epoch
  fixed30      epoch 30 (no selection at all)
  last3        mean of the test metrics of the last three evaluations (epochs 36, 39, 40) = a cheap "average over epochs"
  oracle       epoch with the best *test* balanced accuracy (upper bound; shows how much selection could matter at most)
Reported: mean over the 5 folds of each test metric, per run; then mean and sd over the runs of a group (a group = repeats of one recipe).
Also: Spearman correlation between validation and test macro-F1 across the epochs of a fold (how informative is the validation fold?).
Usage: python analysis/epoch_selection.py --group A_joint RUN RUN ... --group C_plain RUN ...
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aggregate_folds import best_epoch

METRICS = ["balanced_acc", "f1_macro", "acc", "recall_amd", "faz_dice", "vessel_dice"]
ap = argparse.ArgumentParser(); ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True)
a = ap.parse_args()

def pick(hist, sp, rule):
    h = [e for e in hist if e["epoch"] > sp] or hist
    if rule == "current": return [best_epoch(hist, sp)]
    if rule == "val_f1": return [max(h, key=lambda e: e["val"]["f1_macro"])]
    if rule == "val_bal": return [max(h, key=lambda e: e["val"]["balanced_acc"])]
    if rule == "last": return [hist[-1]]
    if rule == "fixed30": return [e for e in hist if e["epoch"] == 30] or [hist[-1]]
    if rule == "last3": return hist[-3:]
    if rule == "oracle": return [max(h, key=lambda e: e["test"]["balanced_acc"])]
RULES = ["current", "val_f1", "val_bal", "last", "fixed30", "last3", "oracle"]

def run_table(run):
    folds = []
    for f in sorted(glob.glob(os.path.join(run, "results_fold*.json"))):
        d = json.load(open(f)); sp = d["args"].get("startpoint", 0) if "faz_dice" in d["history"][0]["val"] else 0
        folds.append((d["history"], sp))
    out = {}
    for r in RULES:
        out[r] = {m: float(np.mean([np.mean([e["test"][m] for e in pick(h, sp, r) if m in e["test"]] or [np.nan]) for h, sp in folds])) for m in METRICS}
    rho = [spearmanr([e["val"]["f1_macro"] for e in h if e["epoch"] > sp], [e["test"]["f1_macro"] for e in h if e["epoch"] > sp])[0] for h, sp in folds]
    ep = [pick(h, sp, "current")[0]["epoch"] for h, sp in folds]
    return out, float(np.nanmean(rho)), ep

for g in a.group:
    name, runs = g[0], g[1:]
    tabs = [run_table(r) for r in runs]
    print(f"\n=== {name}: {len(runs)} run(s) — test metrics, mean over folds per rule (group mean ± sd over runs)")
    print(f"{'rule':9s} " + " ".join(f"{m:>16s}" for m in METRICS))
    for r in RULES:
        cells = []
        for m in METRICS:
            v = [t[0][r][m] for t in tabs]
            cells.append(f"{np.mean(v):.3f}" + (f"±{np.std(v, ddof=1):.3f}" if len(v) > 1 else "     ") + "     ")
        print(f"{r:9s} " + " ".join(c.rjust(16) for c in cells))
    for run, t in zip(runs, tabs):
        print(f"  {os.path.basename(run):34s} balanced acc by rule: " + "  ".join(f"{r} {t[0][r]['balanced_acc']:.3f}" for r in RULES) + f"   | val-test Spearman (macro-F1 over epochs) {t[1]:.2f}, chosen epochs {t[2]}")
