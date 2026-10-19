"""Mean +- std over cross-validation folds of the test metrics at each fold's best-validation epoch.

Usage: python aggregate_folds.py runs/<config> [more run dirs ...]
(one results_fold<k>.json per fold, written by faz_vessel_seg_cls.py --fold k with the same --save_path)
"""
import glob, json, os, statistics as st, sys

KEYS = ["faz_dice", "vessel_dice", "vessel_cldice", "dice_3ch_micro", "acc", "balanced_acc", "f1_macro", "kappa",
        "recall_normal", "recall_dr", "recall_amd", "acc_3mm", "acc_6mm", "balanced_acc_6mm", "vessel_topo_sens", "vessel_components"]


def best_epoch(history, startpoint):
    # joint model: FAZ Dice + vessel Dice + vessel clDice + macro-F1; classification-only baseline: macro-F1
    score = lambda e: sum(e["val"][k] for k in ("faz_dice", "vessel_dice", "vessel_cldice", "f1_macro") if k in e["val"])
    return max([e for e in history if e["epoch"] > startpoint] or history, key=score)


def fold_results(run):
    """[(results file, best-validation-epoch record)] of one run directory."""
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "results_fold*.json"))):
        d = json.load(open(f))
        rows.append((f, best_epoch(d["history"], d["args"].get("startpoint", 0))))
    return rows


def main(runs):
    for run in runs:
        rows = fold_results(run)
        print(f"== {run}: {len(rows)} fold(s)")
        for f, e in rows:
            print(f"   {os.path.basename(f)}: best-validation epoch {e['epoch']}")
        for k in KEYS:
            vals = [e["test"][k] for _, e in rows if k in e["test"]]
            if vals:
                print(f"   {k:18s} {st.mean(vals):.3f}" + (f" +- {st.stdev(vals):.3f}" if len(vals) > 1 else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
