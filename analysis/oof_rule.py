"""Pooled out-of-fold comparison under ONE epoch-selection rule for every variant, from the probabilities logged in results_fold<k>.json
(`test_probs` / `test_names`, written by runs made with the code of 2026-10-21 or later; no checkpoints are needed).
For every fold the epoch is chosen with the rule from the validation metrics, the test probabilities of that epoch are pooled over the five folds
(364 scans), and balanced accuracy is computed once. Rules: pipeline (the training script's own score), val_bal (best validation balanced accuracy),
val_f1 (best validation macro-F1), last (final epoch).
Per group (= repeats of one recipe): balanced accuracy of every run, mean, and the ensemble (probabilities of the runs averaged). With --ref the
ensemble of every other group is compared with the reference group's by a paired bootstrap over scans (95 % interval).
Usage: python analysis/oof_rule.py --rule val_bal --ref C_plain --group A_joint RUN RUN --group B_shuffled RUN --group C_plain RUN RUN
"""
import argparse, glob, json, os, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aggregate_folds import best_epoch

CLASS = {"N": 0, "D": 1, "M": 2}
RULES = ("pipeline", "val_bal", "val_f1", "last")


def choose(history, startpoint, rule):
    h = [e for e in history if e["epoch"] > startpoint] or history
    if rule == "pipeline":
        return best_epoch(history, startpoint)
    if rule == "val_bal":
        return max(h, key=lambda e: e["val"]["balanced_acc"])
    if rule == "val_f1":
        return max(h, key=lambda e: e["val"]["f1_macro"])
    if rule == "last":
        return history[-1]
    raise ValueError(rule)


def pooled(run, rule):
    """(names, probabilities) of all test scans of a run, each from the epoch the rule picks in its fold; None if the run has no logged probabilities."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(run, "results_fold*.json"))):
        d = json.load(open(f))
        if "test_names" not in d or "test_probs" not in d["history"][-1]:
            return None
        e = choose(d["history"], d["args"].get("startpoint", 0) if "faz_dice" in d["history"][0]["val"] else 0, rule)
        for n, p in zip(d["test_names"], e["test_probs"]):
            rows[n] = p
    names = sorted(rows)
    return names, np.array([rows[n] for n in names])


def labels(names):
    return np.array([CLASS[n[0]] for n in names])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="val_bal", choices=RULES)
    ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True)
    ap.add_argument("--ref"); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    ens, names_ref = {}, None
    print(f"rule: {a.rule}")
    for g in a.group:
        name, runs = g[0], g[1:]
        got = [(r, pooled(r, a.rule)) for r in runs]
        ok = [(r, p) for r, p in got if p is not None]
        for r, p in got:
            if p is None:
                print(f"  {os.path.basename(r)}: no logged probabilities (run made before the logging was added); skipped")
        if not ok:
            continue
        names = ok[0][1][0]
        assert all(p[0] == names for _, p in ok), "runs cover different scans"
        names_ref = names_ref or names
        y = labels(names)
        singles = [balanced_accuracy_score(y, p[1].argmax(1)) for _, p in ok]
        ens[name] = np.mean([p[1] for _, p in ok], axis=0)
        print(f"{name}: " + " ".join(f"{s:.3f}" for s in singles) + f"  mean {np.mean(singles):.3f}  ensemble of {len(ok)} {balanced_accuracy_score(y, ens[name].argmax(1)):.3f}")
    if a.ref and a.ref in ens:
        y = labels(names_ref); rng = np.random.default_rng(0); boots = rng.integers(0, len(y), (a.boot, len(y)))
        bal = lambda p, b: balanced_accuracy_score(y[b], p[b].argmax(1))
        for name, p in ens.items():
            if name == a.ref:
                continue
            d = np.array([bal(p, b) - bal(ens[a.ref], b) for b in boots]); pt = balanced_accuracy_score(y, p.argmax(1)) - balanced_accuracy_score(y, ens[a.ref].argmax(1))
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"ensemble {name} - {a.ref}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
