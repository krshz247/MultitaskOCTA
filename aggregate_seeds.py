"""Repeated cross-validation: mean +- std over training seeds, and paired differences between variants.

Every run directory holds one complete cross-validation (results_fold0..4.json) trained with one seed (--seed); the folds are identical
for all seeds. For each directory the test metrics at the best-validation epoch are averaged over the folds; the numbers shown are the
mean and the standard deviation of these cross-validation means over the seeds (the spread that a single run hides). Variant B is compared
with variant A seed by seed (same seed index), which removes part of the shared noise.

  python aggregate_seeds.py --group joint_reg runs/jr_s1 runs/jr_s2 runs/jr_s3 --group cls_reg runs/cr_s1 runs/cr_s2 runs/cr_s3
"""
import argparse, statistics as st
from aggregate_folds import fold_results

METRICS = ["acc", "balanced_acc", "f1_macro", "kappa", "recall_amd", "recall_dr", "faz_dice", "vessel_dice", "vessel_cldice"]


def cv_means(run):
    rows = fold_results(run)
    return {k: st.mean(e["test"][k] for _, e in rows) for k in METRICS if all(k in e["test"] for _, e in rows)}, len(rows)


def sd(v):
    return st.stdev(v) if len(v) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True, help="NAME followed by one run directory per seed")
    a = ap.parse_args()
    groups = {}
    for g in a.group:
        name, runs = g[0], g[1:]
        res = [cv_means(r) for r in runs]
        groups[name] = [r for r, _ in res]
        print(f"{name}: {len(runs)} seed(s), folds per run {[n for _, n in res]}")
    names = list(groups)
    print(f"\n{'variant':16s}" + "".join(f"{m:>16s}" for m in METRICS))
    for n in names:
        row = ""
        for m in METRICS:
            v = [r[m] for r in groups[n] if m in r]
            row += f"{st.mean(v):10.3f}±{sd(v):.3f}" if v else f"{'-':>16s}"
        print(f"{n:16s}{row}")
    for n in names[1:]:
        print(f"\n{n} minus {names[0]} (per seed, then mean ± std over seeds; wins = seeds with a positive difference)")
        for m in METRICS:
            d = [b[m] - c[m] for b, c in zip(groups[n], groups[names[0]]) if m in b and m in c]
            if d:
                print(f"   {m:14s} {st.mean(d):+.3f} ± {sd(d):.3f}   wins {sum(x > 0 for x in d)}/{len(d)}")


if __name__ == "__main__":
    main()
