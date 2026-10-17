"""Rewrite TensorBoard logs of old runs to the current tag scheme.

  val/<m>, test/<m>        ->  eval/val/<m>, eval/test/<m>        (per-image metrics, every --eval_every epochs)
  <split>_epoch/<m>        ->  epoch_avg/<split>/<m>               (legacy per-epoch batch averages)

Old event directories are moved to <backup>/<run>/ (outside the TensorBoard logdir). Needs tensorboard (use .venv-tb).
Usage: .venv-tb/bin/python rename_tb_tags.py [runs_dir] [backup_dir]
"""
import glob, io, os, re, shutil, sys
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter  # needs the training env's tensorboardX; falls back below

runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
backup = sys.argv[2] if len(sys.argv) > 2 else "runs_tb_backup"


def new_tag(t):
    m = re.match(r"^(train|val|test)_epoch/(.+)$", t)
    if m:
        return "epoch_avg/{}/{}".format(m.group(1), m.group(2))
    if re.match(r"^(val|test)/", t):
        return "eval/" + t
    return t


for d in sorted(glob.glob(os.path.join(runs, "*", "tb_fold*"))):
    ea = EventAccumulator(d, size_guidance={"scalars": 0, "images": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if any(t.startswith(("eval/", "epoch_avg/")) for t in tags):
        print("already converted:", d)
        continue
    run = os.path.basename(os.path.dirname(d))
    os.makedirs(os.path.join(backup, run), exist_ok=True)
    dst_backup = os.path.join(backup, run, os.path.basename(d))
    shutil.move(d, dst_backup)
    w = SummaryWriter(d)
    for t in tags:
        for ev in ea.Scalars(t):
            w.add_scalar(new_tag(t), ev.value, ev.step, walltime=ev.wall_time)
    for t in ea.Tags()["images"]:
        for ev in ea.Images(t):
            arr = np.array(Image.open(io.BytesIO(ev.encoded_image_string)).convert("RGB"))
            w.add_image(t, arr, ev.step, walltime=ev.wall_time, dataformats="HWC")
    w.close()
    print("converted", d, "({} scalar tags)".format(len(tags)), "-> backup", dst_backup)
