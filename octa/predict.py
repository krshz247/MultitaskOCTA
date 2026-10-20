"""Inference with trained joint models: one OCTA en-face image in, class probabilities and FAZ / vessel masks out.

The model of one cross-validation fold is trained on four fifths of the scans; the five fold models are averaged (class probabilities and
segmentation probabilities) for the prediction. The weights are not part of the repository (see docs/MODEL_CARD.md).

    python -m octa.predict export --run runs/ILM_OPL_large_cv40reg_s1 --data Datasets/processed_OCTA500/ILM_OPL_large --out weights/joint_reg
    python -m octa.predict predict --weights weights/joint_reg --image scan.bmp --out predictions/

    from octa.predict import load_ensemble, predict
    ens = load_ensemble("weights/joint_reg")
    result = predict(ens, "scan.bmp")      # {'label', 'probabilities', 'faz_mask', 'vessel_mask', ...}

Input: the B5 (ILM-OPL) OCTA projection, 3 mm or 6 mm field of view, any image size, grey-scale. It is resized to 384 x 384 like in training.
Research code: not a medical device and not validated for clinical use.
"""
import argparse, contextlib, glob, io, json, os, shutil
import cv2
import numpy as np
import torch

from .common import kfold_split
from .data.dataset import mean_and_std
from .joint_training import CotrainingModelMulti

CLASSES = ("Normal", "DR", "AMD")  # label order of the training code: N, D, M
SIZE = 384
ENCODER = "timm-resnest50d"


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    return CotrainingModelMulti(ENCODER, None, True, None, len(CLASSES))


def export_weights(run_dir, data_dir, out_dir):
    """Copy the best checkpoints of a cross-validation run and store the normalisation statistics of each fold's training set.

    Each fold model was trained with the mean and standard deviation of its own training images, so they have to travel with the weights.
    """
    names = sorted(glob.glob(os.path.join(data_dir, "image_surface", "*.bmp")))
    if not names:
        raise FileNotFoundError(f"no images in {data_dir}/image_surface")
    os.makedirs(out_dir, exist_ok=True)
    folds = {}
    for k in range(5):
        src = os.path.join(run_dir, f"best_fold{k}.pt")
        if not os.path.exists(src):
            continue
        train, _, _ = kfold_split(names, 5, k)
        with contextlib.redirect_stdout(io.StringIO()):
            mean, std = mean_and_std(train)
        shutil.copyfile(src, os.path.join(out_dir, f"fold{k}.pt"))
        folds[str(k)] = {"mean": float(mean), "std": float(std)}
    info = json.load(open(os.path.join(run_dir, "results_fold0.json")))
    meta = {"classes": list(CLASSES), "input_size": SIZE, "encoder": ENCODER, "folds": folds,
            "source_run": os.path.basename(os.path.normpath(run_dir)), "git_commit": info.get("git_commit"),
            "seed": info["args"].get("seed")}
    json.dump(meta, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    return meta


class Ensemble:
    def __init__(self, models, stats, device):
        self.models, self.stats, self.device = models, stats, device


def load_ensemble(weights_dir, device="auto"):
    """Load every fold model found in `weights_dir` (fold<k>.pt + stats.json written by export_weights)."""
    dev = pick_device(device)
    meta = json.load(open(os.path.join(weights_dir, "stats.json")))
    models, stats = [], []
    for k, s in sorted(meta["folds"].items()):
        path = os.path.join(weights_dir, f"fold{k}.pt")
        if not os.path.exists(path):
            continue
        model = build_model()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        models.append(model.to(dev).eval())
        stats.append((s["mean"], s["std"]))
    if not models:
        raise FileNotFoundError(f"no fold<k>.pt found in {weights_dir}")
    return Ensemble(models, stats, dev)


def read_image(image):
    """Grey-scale float array in [0, 1] of size SIZE x SIZE (resized, divided by its maximum, like load_image in training)."""
    if isinstance(image, (str, os.PathLike)):
        arr = cv2.imread(os.fspath(image), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"cannot read image {image}")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, (SIZE, SIZE))
    top = arr.max()
    return arr / top if top > 0 else arr.astype(np.float64)


def predict(ens, image):
    """Class probabilities (mean over the fold models), predicted class and the FAZ / vessel masks (384 x 384, uint8 0/1)."""
    g = read_image(image)
    probs, seg = [], []
    with torch.no_grad():
        for model, (mean, std) in zip(ens.models, ens.stats):
            x = torch.from_numpy(((g - mean) / std)[None, None].astype(np.float32)).to(ens.device)
            out = model.seg_forward(x)
            logits = model.clf_forward(x, out[3], out[4], out[5])
            probs.append(torch.softmax(logits, 1)[0].cpu().numpy())
            seg.append(torch.sigmoid(out[0])[0].cpu().numpy())  # channels: FAZ, vessel, background
    p = np.mean(probs, axis=0)
    cls = np.argmax(np.mean(seg, axis=0), axis=0)
    faz, vessel = (cls == 0).astype(np.uint8), (cls == 1).astype(np.uint8)
    return {"label": CLASSES[int(p.argmax())], "probabilities": {c: float(v) for c, v in zip(CLASSES, p)},
            "faz_mask": faz, "vessel_mask": vessel, "faz_area_fraction": float(faz.mean()), "vessel_fraction": float(vessel.mean()),
            "n_models": len(ens.models)}


def overlay(image, result):
    g = (read_image(image) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for mask, colour in ((result["faz_mask"], (0, 0, 255)), (result["vessel_mask"], (0, 200, 0))):
        rgb[mask > 0] = 0.55 * rgb[mask > 0] + 0.45 * np.array(colour, dtype=np.float32)
    return rgb.astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="copy the checkpoints of a CV run and store the normalisation statistics")
    e.add_argument("--run", required=True); e.add_argument("--data", required=True); e.add_argument("--out", required=True)
    p = sub.add_parser("predict", help="classify and segment images")
    p.add_argument("--weights", required=True); p.add_argument("--image", nargs="+", required=True); p.add_argument("--out", default="predictions")
    p.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export_weights(a.run, a.data, a.out), indent=1))
        return
    ens = load_ensemble(a.weights, a.device)
    os.makedirs(a.out, exist_ok=True)
    for path in a.image:
        try:
            r = predict(ens, path)
        except FileNotFoundError as err:
            print(f"skipped: {err}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, stem + "_overlay.png"), overlay(path, r))
        summary = {k: v for k, v in r.items() if not k.endswith("_mask")}
        json.dump(summary, open(os.path.join(a.out, stem + ".json"), "w"), indent=1)
        print(stem, r["label"], {c: round(v, 3) for c, v in r["probabilities"].items()}, f"FAZ {r['faz_area_fraction']:.3f} of the image")


if __name__ == "__main__":
    main()
