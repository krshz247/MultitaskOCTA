"""Classification-only baseline ("ResNeSt50 without segmentation" in the thesis), trained with the same protocol as the
joint model: same data, same stratified folds, same augmentation, weighted cross-entropy, Adam with LR_clf, batch size,
validation-based model selection and the same image-level metrics. Writes results_fold<k>.json in the same layout.

Usage: python classification_baseline.py --img_path <D>/image_surface --gt_path <D>/gt_full/mask --save_path runs/cls_baseline \
                                         --fold k --num_epochs 40 [--full_eval]
"""
import glob, json, os, random, sys
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import segmentation_models_pytorch as smp
from eval_metrics import classification_metrics
from faz_vessel_seg_cls import kfold_split, git_commit, set_seed
from utils import create_train_arg_parser, generate_dataset


class ResNeStClassifier(nn.Module):
    """Same backbone and ImageNet weights as the classifier of the joint model, plus global pooling and a linear layer."""

    def __init__(self, encoder="timm-resnest50d", pretrain="imagenet", num_classes=3):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder, in_channels=1, depth=5, weights=pretrain)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.encoder(x)[-1]).flatten(1))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y, p, loss, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss()
    for inputs, _, _, _, cls in loader:
        out = model(inputs.float().to(device))
        t = cls.argmax(2).squeeze(1)
        loss += crit(out.cpu(), t).item() * len(t); n += len(t)
        y += t.tolist(); p += out.argmax(1).cpu().tolist()
    return y, p, loss / max(n, 1)


def main():
    args = create_train_arg_parser().parse_args()
    set_seed(2021)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    names = sorted(glob.glob(os.path.join(args.img_path, "*.bmp")))
    gt = lambda L: [n.replace("image_surface", args.gt_path.split("/")[-2] + "/mask") for n in L]
    tr, va, te = kfold_split(names, args.num_folds, args.fold, stratify_fov=args.stratify_fov)
    trainL, devL, testL = generate_dataset(tr, va, te, args.input_size, args.batch_size, args.val_batch_size, "dist_signed11", args.clahe,
                                           "Vessel_FAZ", gt(tr), gt(va), gt(te), num_workers=args.num_workers, full_eval=args.full_eval)
    model = ResNeStClassifier(args.encoder, args.pretrain if args.pretrain in ("imagenet", "ssl", "swsl") else None).to(device)
    opt = Adam(model.parameters(), lr=args.LR_clf)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.49, 1.88, 2.35]).to(device))
    os.makedirs(args.save_path, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.num_epochs + 1):
        model.train(); tl, tn, tc = 0.0, 0, 0
        for inputs, _, _, _, cls in trainL:
            t = cls.argmax(2).squeeze(1).to(device)
            out = model(inputs.float().to(device)); loss = crit(out, t)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(t); tn += len(t); tc += (out.argmax(1) == t).sum().item()
        rec = {"epoch": epoch, "train_loss": tl / tn, "train_acc": tc / tn}
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.num_epochs):
            yv, pv, lv = predict(model, devL, device); yt, pt, lt = predict(model, testL, device)
            rec["val"] = dict(classification_metrics(yv, pv, devL.dataset.img_names), cls_loss=lv)
            rec["test"] = dict(classification_metrics(yt, pt, testL.dataset.img_names), cls_loss=lt)
            history.append(rec)
            if rec["val"]["f1_macro"] > best:
                best = rec["val"]["f1_macro"]; torch.save(model.state_dict(), os.path.join(args.save_path, "best_fold{}.pt".format(args.fold)))
            with open(os.path.join(args.save_path, "results_fold{}.json".format(args.fold)), "w") as fh:
                json.dump({"args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}, "git_commit": git_commit(),
                           "model": "classification_baseline", "history": history}, fh, indent=1)
        print("epoch {} train loss {:.3f} acc {:.3f}".format(epoch, rec["train_loss"], rec["train_acc"]) +
              (" | val acc {:.3f} f1 {:.3f} | test acc {:.3f} f1 {:.3f}".format(rec["val"]["acc"], rec["val"]["f1_macro"], rec["test"]["acc"], rec["test"]["f1_macro"]) if "val" in rec else ""), flush=True)


if __name__ == "__main__":
    main()
