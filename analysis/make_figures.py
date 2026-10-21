"""Figures for the README, drawn from stored results. No scan pixels are used (the OCTA-500 terms say nothing about publishing images).
  Figs/findings.png      (a) pooled out-of-fold balanced accuracy of every run, per recipe; (b) effect of replacing the outer part of the scan by grey
  Figs/predictions.png   predicted FAZ / vessel masks of three held-out scans on a plain background (needs --weights, a folder with fold<k>.pt + stats.json)
Needs the cached out-of-fold probabilities <run>/oof_probs.npy (written by analysis/ensemble_runs.py) and analysis/gradcam_results.json.
Usage: python analysis/make_figures.py [--weights weights/fold0_only --fold 0 --data Datasets/processed_OCTA500/ILM_OPL_large]
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--weights"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--data", default="Datasets/processed_OCTA500/ILM_OPL_large")
a = ap.parse_args()
os.chdir(ROOT)
os.makedirs("Figs", exist_ok=True)

GROUPS = [("Joint model", "#1b7f79", ["cv40reg", "cv40reg_s1", "cv40reg_s2", "cv40reg_s3"]),
          ("Joint model,\nshuffled masks", "#c8661a", ["cv40reg_shufmask", "cv40reg_shufmask_s1", "cv40reg_shufmask_s2"]),
          ("Classifier only", "#6b4fa0", ["cls40reg", "cls40reg_s1", "cls40reg_s2"])]
SINGLE = [("Stop-gradient", "cv40reg_detach"), ("No segmentation\nloss", "cv40reg_noseg")]
y = np.load("analysis/oof_probs.npz")["y"]
bal = lambda r: balanced_accuracy_score(y, np.load(f"runs/ILM_OPL_large_{r}/oof_probs.npy").argmax(1))
ens = lambda rs: balanced_accuracy_score(y, np.mean([np.load(f"runs/ILM_OPL_large_{r}/oof_probs.npy") for r in rs], 0).argmax(1))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#222", "ytick.color": "#222"})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
xs, labels = [], []
rng = np.random.default_rng(0)
for i, (name, col, runs) in enumerate(GROUPS):
    v = [bal(r) for r in runs if os.path.exists(f"runs/ILM_OPL_large_{r}/oof_probs.npy")]
    ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=46, color=col, zorder=3, label="single run" if i == 0 else None)
    ax.hlines(np.mean(v), i - 0.22, i + 0.22, color=col, lw=2.2, zorder=2, label="mean of the runs" if i == 0 else None)
    e = ens([r for r in runs if os.path.exists(f"runs/ILM_OPL_large_{r}/oof_probs.npy")])
    ax.scatter([i + 0.3], [e], marker="D", s=40, facecolor="white", edgecolor=col, lw=1.6, zorder=3, label="ensemble: class probabilities of\nall runs averaged, then scored" if i == 0 else None)
    labels.append(f"{name}\n({len(v)} runs)")
for j, (name, r) in enumerate(SINGLE):
    i = len(GROUPS) + j
    ax.scatter([i], [bal(r)], s=46, facecolor="white", edgecolor="#777", lw=1.6, zorder=3, label="single run, not repeated" if j == 0 else None)
    labels.append(f"{name}\n(1 run)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Balanced accuracy (pooled out-of-fold, 364 scans)"); ax.set_ylim(0.62, 0.90)
ax.axhspan(0.747 - 0.06, 0.747 + 0.06, color="#bbb", alpha=0.25, lw=0, zorder=0)
ax.text(1.5, 0.697, "grey band: ±0.06, the 95 % interval of one run", ha="left", fontsize=8, color="#555")
ax.set_title("(a) Same recipe, different seed: the spread is as large as the differences", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")

g = json.load(open("analysis/gradcam_results.json"))
conds = [("none", "unchanged"), ("band15", "outer 30 %\nreplaced"), ("band30", "outer 60 %\nreplaced")]
w = 0.36
for k, (key, col, name) in enumerate((("joint", "#1b7f79", "Joint model"), ("cls", "#6b4fa0", "Classifier only"))):
    vals = [g[f"ablation/{key}/{c}"]["balanced_acc"] for c, _ in conds]
    bx.bar(np.arange(3) + (k - 0.5) * w, vals, w, color=col, label=name)
bx.set_xticks(range(3)); bx.set_xticklabels([l for _, l in conds], fontsize=8.5)
bx.set_ylabel("Balanced accuracy"); bx.set_ylim(0, 0.9); bx.axhline(1 / 3, color="#777", lw=1, ls=":")
bx.text(-0.45, 1 / 3 + 0.015, "chance", ha="left", fontsize=8, color="#555")
bx.set_title("(b) Both models react strongly to the image frame", fontsize=10, loc="left"); bx.legend(frameon=False, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("Figs/findings.png", dpi=150, facecolor="white")
print("Figs/findings.png")

if a.weights:
    import glob as _g
    from octa.common import kfold_split
    from octa.predict import load_ensemble, predict
    names = sorted(_g.glob(os.path.join(a.data, "image_surface", "*.bmp")))
    _, _, te = kfold_split(names, 5, a.fold)
    ens_ = load_ensemble(a.weights, "auto")
    picks = [next(n for n in te if os.path.basename(n)[0] == c) for c in "NDM"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for axx, n in zip(axs, picks):
        r = predict(ens_, n)
        img = np.zeros(r["faz_mask"].shape + (3,), np.float32)
        img[r["vessel_mask"] > 0] = (0.11, 0.50, 0.47)
        img[r["faz_mask"] > 0] = (0.85, 0.30, 0.20)
        axx.imshow(img); axx.axis("off")
        true = {"N": "Normal", "D": "DR", "M": "AMD"}[os.path.basename(n)[0]]
        axx.set_title(f"true {true}, predicted {r['label']}\nP = {max(r['probabilities'].values()):.2f}", fontsize=9)
    fig.suptitle("Predicted FAZ (red) and large vessels (green); first Normal, DR and AMD scan of one test fold, masks only, one fold model", fontsize=9.5)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig("Figs/predictions.png", dpi=150, facecolor="white")
    print("Figs/predictions.png")
