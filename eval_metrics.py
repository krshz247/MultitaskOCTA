"""Per-image segmentation (Dice / Jaccard / clDice) and classification metrics over a whole loader.

The metrics logged inside seg_clf_iteration are running averages of batch statistics (vessel Dice and clDice come
from the last batch only), so the comparison runs use this instead.
"""
import numpy as np
import torch
from skimage.morphology import skeletonize
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

EPS = 1e-7


def _dice_jaccard(p, t):
    inter = float((p & t).sum())
    p_sum, t_sum = float(p.sum()), float(t.sum())
    return (2 * inter + EPS) / (p_sum + t_sum + EPS), (inter + EPS) / (p_sum + t_sum - inter + EPS)


def _cl_dice(p, t):
    sp, st = skeletonize(p), skeletonize(t)
    tprec = (float((sp & t).sum()) + EPS) / (float(sp.sum()) + EPS)
    tsens = (float((st & p).sum()) + EPS) / (float(st.sum()) + EPS)
    return 2 * tprec * tsens / (tprec + tsens)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Channels of the segmentation output / target are [FAZ, vessel, background]."""
    model.eval()
    seg = {k: [] for k in ("faz_dice", "faz_jaccard", "vessel_dice", "vessel_jaccard", "vessel_cldice")}
    # auxiliary branches: sums of squared error, and of squared deviation from the target's own per-image mean
    # (the error of the best constant prediction), per branch and channel
    aux_sse, aux_const, aux_n = np.zeros((2, 2)), np.zeros((2, 2)), 0
    y_true, y_pred = [], []
    for inputs, targets1, targets2, targets3, targets4 in loader:
        inputs = inputs.to(torch.float32).to(device)
        seg_outputs = model.module.seg_forward(inputs)
        clf_outputs = model.module.clf_forward(inputs, seg_outputs[3], seg_outputs[4], seg_outputs[5])
        for branch, (out, tgt) in enumerate(((seg_outputs[1], targets2), (seg_outputs[2], targets3))):
            out, tgt = out.float().cpu(), tgt.float()
            for ch in range(2):
                aux_sse[branch, ch] += float(((out[:, ch] - tgt[:, ch]) ** 2).sum())
                dev = tgt[:, ch] - tgt[:, ch].mean(dim=(1, 2), keepdim=True)
                aux_const[branch, ch] += float((dev ** 2).sum())
        aux_n += inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
        preds = (torch.sigmoid(seg_outputs[0]) > 0.5).cpu().numpy()
        gts = (targets1 > 0.5).numpy()
        for b in range(preds.shape[0]):
            d, j = _dice_jaccard(preds[b, 0], gts[b, 0])
            seg["faz_dice"].append(d)
            seg["faz_jaccard"].append(j)
            d, j = _dice_jaccard(preds[b, 1], gts[b, 1])
            seg["vessel_dice"].append(d)
            seg["vessel_jaccard"].append(j)
            seg["vessel_cldice"].append(_cl_dice(preds[b, 1], gts[b, 1]))
        y_true.extend(torch.argmax(targets4, dim=2).squeeze(1).tolist())
        y_pred.extend(torch.argmax(clf_outputs, dim=1).cpu().tolist())
    out = {k: float(np.mean(v)) for k, v in seg.items()}
    out.update(
        n=len(y_true),
        acc=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        precision_macro=precision_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2]),
    )
    for branch, name in enumerate(("heat", "sdm")):
        for ch, region in enumerate(("faz", "vessel")):
            out[f"{name}_{region}_mse"] = float(aux_sse[branch, ch] / aux_n)
            # 1 = perfect, 0 = no better than predicting each image's mean, <0 = worse
            out[f"{name}_{region}_r2"] = float(1.0 - aux_sse[branch, ch] / max(aux_const[branch, ch], EPS))
    return out
