"""Progress of cross-validation runs: for every run folder (or the given ones) the last epoch of each fold, and FAILED markers.
Usage: python run_status.py [runs/<name> ...]      (default: runs modified in the last 3 days)"""
import glob, json, os, sys, time

dirs = sys.argv[1:] or sorted((d for d in glob.glob("runs/*") if os.path.isdir(d) and time.time() - os.path.getmtime(d) < 3 * 86400), key=os.path.getmtime)
for d in dirs:
    folds = []
    for k in range(5):
        f = os.path.join(d, f"results_fold{k}.json")
        if os.path.exists(f):
            h = json.load(open(f))
            total = h["args"].get("num_epochs", "?")
            folds.append(f"fold{k}: {h['history'][-1]['epoch']}/{total}")
    state = "FAILED folds: " + open(os.path.join(d, "FAILED")).read().replace("\n", ", ") if os.path.exists(os.path.join(d, "FAILED")) else ""
    print(f"{os.path.basename(d):42s} {'  '.join(folds) or '(no results yet)'}  {state}")
