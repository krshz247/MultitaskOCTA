"""Welch tests on the per-run pooled out-of-fold balanced accuracies of the recipes (the p-values quoted in docs/CLAIMS.md row 5 and docs/STATUS.md).
Reads the cached <run>/oof_probs.npy (analysis/ensemble_runs.py), computes balanced accuracy of every run on the 364 pooled scans, then Welch's t-test
between recipes (runs are the units; n = 4 per recipe). Output: analysis/run_comparison_tests.txt.
Usage: python analysis/run_comparison_tests.py
"""
import os
import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
y = np.load(os.path.join(os.path.dirname(__file__), "oof_probs.npz"))["y"]
P = os.path.join(ROOT, "runs", "ILM_OPL_large_")
RECIPES = {"A_joint": ["cv40reg", "cv40reg_s1", "cv40reg_s2", "cv40reg_s3"],
           "B_shuffled": ["cv40reg_shufmask", "cv40reg_shufmask_s1", "cv40reg_shufmask_s2", "cv40reg_shufmask_s3"],
           "C_plain": ["cls40reg", "cls40reg_s1", "cls40reg_s2", "cls40reg_s3"]}
vals = {k: [balanced_accuracy_score(y, np.load(P + r + "/oof_probs.npy").argmax(1)) for r in rs] for k, rs in RECIPES.items()}
for k, v in vals.items():
    print(f"{k:11s} per-run pooled balanced accuracy " + " ".join(f"{x:.3f}" for x in v) + f"  mean {np.mean(v):.3f}  sd {np.std(v, ddof=1):.3f}")
for a, b in (("A_joint", "C_plain"), ("A_joint", "B_shuffled"), ("C_plain", "B_shuffled")):
    t = stats.ttest_ind(vals[a], vals[b], equal_var=False)
    print(f"Welch {a} vs {b}: mean difference {np.mean(vals[a]) - np.mean(vals[b]):+.3f}, t = {t.statistic:.2f}, p = {t.pvalue:.2f}")
