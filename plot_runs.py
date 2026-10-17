"""Training curves from runs/<config>/train.log and results_fold<k>.json.

Usage: python plot_runs.py [fold] [out_dir]   -> runs/<config>/curves.png and <out_dir>/all_curves.png
"""
import glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fold = sys.argv[1] if len(sys.argv) > 1 else "0"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "runs"
NUM = r"([-+]?[0-9.]+(?:e[-+]?\d+)?|nan)"
TR = re.compile(rf"TrainSeg Loss:{NUM}, TrMutiLoss:{NUM}, Dice_Coeff: {NUM}, Jaccard: {NUM}, TrainClf Loss:{NUM}, Acc: {NUM}, Kappa:{NUM}")
VA = re.compile(rf"ValSeg Loss:{NUM}, VaMutiLoss:{NUM}, Dice_Coeff: {NUM}, Jaccard: {NUM}, ValClf Loss:{NUM}, Acc: {NUM}, Kappa:{NUM}")


def parse(log):
    tr, va = [], []
    for line in open(log, errors="ignore").read().replace("\r", "\n").splitlines():
        m = TR.search(line)
        if m:
            tr.append([float(x) for x in m.groups()])
        m = VA.search(line)
        if m:
            va.append([float(x) for x in m.groups()])
    return tr, va


def draw(axs, name, tr, va, hist, startpoint):
    ep = range(1, len(tr) + 1)
    panels = [("Segmentation loss (Dice FAZ + clDice vessel)", 0), ("Total multi-task loss", 1), ("Classification loss", 4), ("Classification accuracy", 5)]
    for ax, (title, i) in zip(axs[:4], panels):
        ax.plot(ep, [r[i] for r in tr], label="train")
        ax.plot(range(1, len(va) + 1), [r[i] for r in va], label="val")
        ax.axvline(startpoint, color="gray", ls=":", lw=0.8)
        ax.set_title(title, fontsize=8)
    axs[0].legend(fontsize=7)
    e = [h["epoch"] for h in hist]
    for key in ("faz_dice", "vessel_dice", "vessel_cldice"):
        axs[4].plot(e, [h["val"][key] for h in hist], label="val " + key)
        axs[4].plot(e, [h["test"][key] for h in hist], ls="--", color=axs[4].lines[-1].get_color())
    axs[4].set_title("per-image val (solid) / test (dashed)", fontsize=8)
    axs[4].legend(fontsize=6)
    axs[5].plot(e, [h["val"]["f1_macro"] for h in hist], label="val F1 macro")
    axs[5].plot(e, [h["test"]["f1_macro"] for h in hist], ls="--", label="test F1 macro")
    for k in ("heat_faz_r2", "sdm_faz_r2"):
        if k in hist[0]["val"]:
            axs[5].plot(e, [max(h["val"][k], -1) for h in hist], label="val " + k + " (clipped at -1)")
    axs[5].set_title("classification F1 / aux-branch R2", fontsize=8)
    axs[5].legend(fontsize=6)
    axs[0].set_ylabel(name, fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7)


runs = sorted(glob.glob(f"runs/*/results_fold{fold}.json"))
fig_all, axs_all = plt.subplots(len(runs), 6, figsize=(24, 3 * len(runs)), squeeze=False)
for r, jf in enumerate(runs):
    d = os.path.dirname(jf)
    name = os.path.basename(d)
    tr, va = parse(os.path.join(d, "train.log"))
    res = json.load(open(jf))
    fig, axs = plt.subplots(1, 6, figsize=(24, 3))
    for a_set in (axs, axs_all[r]):
        draw(a_set, name, tr, va, res["history"], res["args"].get("startpoint", 0))
    fig.tight_layout(); fig.savefig(os.path.join(d, "curves.png"), dpi=70); plt.close(fig)
fig_all.tight_layout(); fig_all.savefig(os.path.join(out_dir, "all_curves.png"), dpi=55)
print("wrote", len(runs), "run figures +", os.path.join(out_dir, "all_curves.png"))
