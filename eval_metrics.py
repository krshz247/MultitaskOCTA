"""Per-image segmentation (Dice / Jaccard / clDice) and classification metrics over a whole loader.

The metrics logged inside seg_clf_iteration are running averages of batch statistics (vessel Dice and clDice come
from the last batch only), so the comparison runs use this instead.
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

EPS = 1e-7


def _dice_jaccard(p, t):
    inter = float((p & t).sum())
    p_sum, t_sum = float(p.sum()), float(t.sum())
    return (2 * inter + EPS) / (p_sum + t_sum + EPS), (inter + EPS) / (p_sum + t_sum - inter + EPS)


def _cl_dice(p, t):
    """Returns (clDice, topology precision, topology sensitivity)."""
    sp, st = skeletonize(p), skeletonize(t)
    tprec = (float((sp & t).sum()) + EPS) / (float(sp.sum()) + EPS)
    tsens = (float((st & p).sum()) + EPS) / (float(st.sum()) + EPS)
    return 2 * tprec * tsens / (tprec + tsens), tprec, tsens


def _components(b):
    """(number of 8-connected components, number of fragments smaller than 30 px)."""
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    return n, int((sizes < 30).sum())


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Channels of the segmentation output / target are [FAZ, vessel, background]."""
    model.eval()
    seg = {k: [] for k in ("faz_dice", "faz_jaccard", "vessel_dice", "vessel_jaccard", "vessel_cldice", "faz_dice_thr", "vessel_dice_thr",
                                     "vessel_topo_prec", "vessel_topo_sens", "vessel_components", "vessel_fragments", "vessel_components_gt")}
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
        probs = torch.sigmoid(seg_outputs[0]).cpu().numpy()
        preds_thr = probs > 0.5                      # independent per-channel threshold (channels may overlap)
        cls_map = probs.argmax(axis=1)               # 3-class map: 0 = FAZ, 1 = vessel, 2 = background
        preds = np.stack([cls_map == 0, cls_map == 1, cls_map == 2], axis=1)
        gts = (targets1 > 0.5).numpy()
        for b in range(preds.shape[0]):
            seg["faz_dice_thr"].append(_dice_jaccard(preds_thr[b, 0], gts[b, 0])[0])
            seg["vessel_dice_thr"].append(_dice_jaccard(preds_thr[b, 1], gts[b, 1])[0])
            d, j = _dice_jaccard(preds[b, 0], gts[b, 0])
            seg["faz_dice"].append(d)
            seg["faz_jaccard"].append(j)
            d, j = _dice_jaccard(preds[b, 1], gts[b, 1])
            seg["vessel_dice"].append(d)
            seg["vessel_jaccard"].append(j)
            cl, tprec, tsens = _cl_dice(preds[b, 1], gts[b, 1])
            seg["vessel_cldice"].append(cl)
            seg["vessel_topo_prec"].append(tprec)
            seg["vessel_topo_sens"].append(tsens)
            n_p, frag_p = _components(preds[b, 1])
            seg["vessel_components"].append(n_p)
            seg["vessel_fragments"].append(frag_p)
            seg["vessel_components_gt"].append(_components(gts[b, 1])[0])
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


@torch.no_grad()
def save_prediction_figure(model, dataset, indices, device, path, writer=None, tag="predictions", step=0):
    """One row per scan: input | ground truth (R=FAZ, G=vessel) | prediction | vessel probability |
    predicted boundary heatmaps (R=FAZ, G=vessel) | predicted signed-distance maps (R=FAZ, G=vessel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    fig, axs = plt.subplots(len(indices), 6, figsize=(18, 3 * len(indices)), squeeze=False)
    titles = ["input", "GT (R=FAZ G=vessel)", "prediction (argmax)", "vessel prob.", "pred. heatmap", "pred. SDM"]
    for r, i in enumerate(indices):
        image, mask, _, _, cls = dataset[i]
        out = model.module.seg_forward(image.unsqueeze(0).float().to(device))
        prob = torch.sigmoid(out[0])[0].cpu().numpy()
        rgb = lambda a: np.stack([a[0], a[1], np.zeros_like(a[0])], axis=-1).clip(0, 1)
        cls_map = prob.argmax(axis=0)
        panels = [image[0].numpy(), rgb(np.asarray(mask)), rgb(np.stack([cls_map == 0, cls_map == 1]).astype(float)), prob[1],
                  rgb(out[1][0].cpu().numpy()), rgb(out[2][0].cpu().numpy())]
        for c, p in enumerate(panels):
            axs[r, c].imshow(p, cmap="gray", vmin=0, vmax=1 if c else None)
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(titles[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    if writer is not None:
        writer.add_figure(tag, fig, step)
    plt.close(fig)
