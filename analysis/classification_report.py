"""AUROC, confusion matrices and calibration of the pooled out-of-fold class probabilities (364 scans) from the cached <run>/oof_probs.npy
(written by analysis/ensemble_runs.py), per run and for the ensemble (probabilities of the runs averaged) of every recipe. No model is run.
  AUROC: one-vs-rest per class (Normal, DR, AMD) and the macro mean; 95 % bootstrap interval over scans for the ensemble.
  Calibration: expected calibration error (ECE, 10 equal-width bins of the top-class probability) and the multi-class Brier score.
  Confusion matrix of the ensemble (rows = true class).
Usage: python analysis/classification_report.py --group A_joint RUN RUN ... --group C_plain RUN ...     (figures: --fig Figs/confusion.png)
"""
import argparse, os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score

CLASSES = ("Normal", "DR", "AMD")
ap = argparse.ArgumentParser()
ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "RUN"), required=True)
ap.add_argument("--fig"); ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
y = np.load(os.path.join(os.path.dirname(__file__), "oof_probs.npz"))["y"]


def aucs(p, yy):
    return [roc_auc_score(yy == c, p[:, c]) for c in range(3)]


def ece(p, yy, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == yy[m]).mean() - conf[m].mean())
    return e


brier = lambda p, yy: float(np.mean(np.sum((p - np.eye(3)[yy]) ** 2, axis=1)))
rng = np.random.default_rng(0); boots = rng.integers(0, len(y), (a.boot, len(y)))
ens = {}
print(f"{'':36s} {'AUROC Normal':>12s} {'DR':>6s} {'AMD':>6s} {'macro':>7s} {'ECE':>6s} {'Brier':>6s}")
for g in a.group:
    name, runs = g[0], g[1:]
    P = [np.load(os.path.join(r, "oof_probs.npy")) for r in runs]
    for r, p in zip(runs, P):
        u = aucs(p, y)
        print(f"  {os.path.basename(r):34s} {u[0]:12.3f} {u[1]:6.3f} {u[2]:6.3f} {np.mean(u):7.3f} {ece(p, y):6.3f} {brier(p, y):6.3f}")
    pe = np.mean(P, axis=0); ens[name] = pe; u = aucs(pe, y)
    lo, hi = np.percentile([np.mean(aucs(pe[b], y[b])) for b in boots], [2.5, 97.5])
    print(f"{name + ' ensemble (' + str(len(P)) + ' runs)':36s} {u[0]:12.3f} {u[1]:6.3f} {u[2]:6.3f} {np.mean(u):7.3f} {ece(pe, y):6.3f} {brier(pe, y):6.3f}   macro AUROC 95 % [{lo:.3f}, {hi:.3f}]")
print("\nconfusion matrices of the ensembles (rows = true class Normal / DR / AMD, columns = predicted)")
for name, pe in ens.items():
    cm = confusion_matrix(y, pe.argmax(1), labels=[0, 1, 2])
    print(f"{name}: balanced accuracy {balanced_accuracy_score(y, pe.argmax(1)):.3f}\n" + "\n".join("   " + "  ".join(f"{v:4d}" for v in row) for row in cm))
if a.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(ens), figsize=(3.6 * len(ens), 3.4))
    for ax, (name, pe) in zip(np.atleast_1d(axs), ens.items()):
        cm = confusion_matrix(y, pe.argmax(1), labels=[0, 1, 2]); r = cm / cm.sum(1, keepdims=True)
        ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{r[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8, color="white" if r[i, j] > 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8); ax.set_title(name, fontsize=9)
    plt.tight_layout(); plt.savefig(a.fig, dpi=150, facecolor="white"); print("figure:", a.fig)
