"""How large must a difference in balanced accuracy be to be detected, and what do seed repeats buy?
Two separate sources of uncertainty:
  (1) scan sampling: 364 scans (49 AMD) are one sample of patients; the 95 % interval of a *paired* difference between two models
      is estimated by a bootstrap over scans (out-of-fold predictions in analysis/oof_probs.npz). Repeating the training with other seeds does NOT shrink it.
  (2) training noise: the same recipe with another seed gives other weights; only repeats can average it. For S seeds per variant and a paired
      t-test over the per-seed differences (sd sigma_d = sqrt(2) * sigma_seed if the seeds of the two variants are independent), the smallest
      difference detected with 80 % power at alpha = 0.05 (two-sided) is (t_{0.975,S-1} + t_{0.8,S-1}) * sigma_d / sqrt(S).
sigma_seed is not known yet: use the table for a range of values and replace it by the measured value once the Kaggle seed runs are aggregated
(python aggregate_seeds.py ... prints the sd of the per-seed pooled balanced accuracy).
Usage: python analysis/power_seeds.py [--seed_sd 0.015]
"""
import argparse, os
import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

ap = argparse.ArgumentParser(); ap.add_argument("--seed_sd", type=float, default=None, help="measured sd of the pooled out-of-fold balanced accuracy over seeds")
a = ap.parse_args()
d = np.load(os.path.join(os.path.dirname(__file__), "oof_probs.npz")); y = d["y"]
pj, pc = d["joint_plain"].argmax(1), d["cls_plain"].argmax(1)
rng = np.random.default_rng(0); boots = rng.integers(0, len(y), (4000, len(y)))
bal = lambda p, b: balanced_accuracy_score(y[b], p[b])
diff = np.array([bal(pj, b) - bal(pc, b) for b in boots]); single = np.array([bal(pj, b) for b in boots])
print(f"(1) scan sampling, n = {len(y)}: SE of one model's balanced accuracy {single.std():.3f}; SE of the paired difference joint - classification-only {diff.std():.3f}")
print(f"    smallest difference detectable with 80 % power from scan sampling alone: {(1.96 + 0.84) * diff.std():.3f}")
print("\n(2) smallest detectable difference (balanced accuracy, 80 % power) from S seeds per variant, by training-noise sd per run (pooled out-of-fold)")
sds = [a.seed_sd] if a.seed_sd else [0.01, 0.015, 0.02, 0.03]
print(f"{'S seeds':>8s} " + " ".join(f"sd={s:.3f}".rjust(9) for s in sds))
for S in (1, 2, 3, 5, 8, 10):
    if S == 1: print(f"{S:8d} " + " ".join("n/a".rjust(9) for _ in sds)); continue
    f = (stats.t.ppf(0.975, S - 1) + stats.t.ppf(0.8, S - 1)) / np.sqrt(S)
    print(f"{S:8d} " + " ".join(f"{f * np.sqrt(2) * s:9.3f}" for s in sds))
print("\nobserved so far (pooled out-of-fold, one run each): joint regularised 0.747, shuffled masks 0.762, classification-only 0.747 -> spread ~0.01;"
      "\nsingle folds differ by 5-9 points between two runs of the same recipe, which is why the pooled estimate is the one to use.")
