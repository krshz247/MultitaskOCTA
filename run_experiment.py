"""Run one experiment (all folds, or a chosen few) from a YAML configuration.

    python run_experiment.py configs/joint_cv40.yaml                 # all five folds, one after the other
    python run_experiment.py configs/joint_cv40.yaml --folds 0 1     # chosen folds
    python run_experiment.py configs/joint_cv40.yaml --dry-run       # print the commands only
    python run_experiment.py configs/joint_cv40.yaml --set args.num_epochs=2 data.limit=30 paths.runs_dir=/tmp/runs   # override values

A configuration has four sections (see configs/README.md):
  name, base (optional parent file whose values are overridden), script (joint | classification),
  paths (data_root, raw_octa500, runs_dir), data (builder options), args (every key becomes --key value of the training script).
The resolved configuration is saved as <run dir>/config.yaml together with the git commit, the dataset is built on demand.
"""
import argparse, copy, difflib, os, subprocess, sys
from dataclasses import fields
import yaml
from octa.config import TrainConfig

SCRIPTS = {"joint": "faz_vessel_seg_cls.py", "classification": "classification_baseline.py"}


def load(path):
    cfg = yaml.safe_load(open(path)) or {}
    if "base" in cfg:
        parent = load(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = merge(parent, cfg)
    return cfg


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_override(cfg, item):
    key, value = item.split("=", 1)
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def validate_args(args):
    """Options of the 'args' section that the training script does not have, with close matches: [(name, [suggestions])]."""
    names = [f.name for f in fields(TrainConfig)]
    return [(k, difflib.get_close_matches(k, names, n=2)) for k in args if k not in names]


def cli(args):
    out = []
    for k, v in args.items():
        if v is True:
            out.append(f"--{k}")
        elif v is not False and v is not None:
            out += [f"--{k}"] + (list(map(str, v)) if isinstance(v, list) else [str(v)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--folds", type=int, nargs="*"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", help="repeat the whole cross-validation with these training seeds; run folders get the suffix _s<seed> (aggregate with aggregate_seeds.py)")
    ap.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    a = ap.parse_args()
    cfg = load(a.config)
    for item in a.set:
        set_override(cfg, item)
    paths, data, args = cfg["paths"], cfg.get("data", {}), dict(cfg.get("args", {}))
    unknown = validate_args(args)
    if unknown:
        sys.exit("unknown option(s) in the 'args' section: " + "; ".join(f"{k}" + (f" (did you mean {', '.join(m)}?)" if m else "") for k, m in unknown))
    folds = a.folds if a.folds is not None and len(a.folds) else list(range(args.get("num_folds", 5)))
    data_name = "{}_{}".format(data.get("projection", "ILM_OPL"), data.get("vessel", "large")) + {"native": "", "center3mm": "_fov3"}[data.get("fov_mode", "native")]
    data_dir = os.path.join(paths["data_root"], data_name)
    py = sys.executable
    build = [py, "Preprocessing/build_processed_octa500.py", "--octa500", paths["raw_octa500"], "--out", data_dir, "--projection", data.get("projection", "ILM_OPL"),
             "--vessel", data.get("vessel", "large"), "--fov_mode", data.get("fov_mode", "native")] + (["--limit", str(data["limit"])] if data.get("limit") else [])
    if not os.path.isdir(os.path.join(data_dir, "image_surface")) or data.get("rebuild"):
        print("dataset:", " ".join(build))
        if not a.dry_run:
            subprocess.run(build, check=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONWARNINGS="ignore")
    for seed in (a.seeds if a.seeds else [None]):
        run_dir = os.path.join(paths["runs_dir"], cfg["name"] + (f"_s{seed}" if seed is not None else ""))
        run_args = dict(args, **({"seed": seed} if seed is not None else {}))
        base = [py, SCRIPTS[cfg.get("script", "joint")], "--img_path", os.path.join(data_dir, "image_surface"), "--gt_path", os.path.join(data_dir, "gt_full", "mask"),
                "--save_path", run_dir, "--train_type", "Vessel_FAZ", "--distance_type", "dist_signed11"] + cli(run_args)
        if not a.dry_run:
            os.makedirs(os.path.join(run_dir, "summary"), exist_ok=True)
            yaml.safe_dump({**cfg, "args": run_args, "git_commit": git}, open(os.path.join(run_dir, "config.yaml"), "w"), sort_keys=False)
        for fold in folds:
            cmd = base + ["--fold", str(fold)]
            print(f"seed {seed} fold {fold}:" if seed is not None else f"fold {fold}:", " ".join(cmd))
            if a.dry_run:
                continue
            with open(os.path.join(run_dir, f"train_fold{fold}.log"), "w") as log:
                if subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode:
                    print(f"fold {fold} FAILED, see {log.name}", file=sys.stderr)
        if not a.dry_run:
            print("results:", run_dir, "-> python aggregate_folds.py", run_dir)


if __name__ == "__main__":
    main()
