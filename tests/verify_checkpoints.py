"""Manual check (needs the local runs/ and Datasets/ folders): re-evaluate saved best checkpoints of finished runs with eval_checkpoint.py
and compare with the metrics stored in their results_fold<k>.json at the best-validation epoch.  Run after every refactoring step:
    python tests/verify_checkpoints.py            (about 4-5 minutes on the Apple GPU)
"""
import json, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CASES = [("ILM_OPL_large_cv40", 0, "ILM_OPL_large"), ("ILM_OPL_large_cv40", 3, "ILM_OPL_large"),
         ("ILM_OPL_large_cv40reg", 2, "ILM_OPL_large"), ("ILM_OPL_large_fov3_cv40", 1, "ILM_OPL_large_fov3")]
KEYS = ["faz_dice", "vessel_dice", "vessel_cldice", "acc", "f1_macro", "balanced_acc"]
TOL = 2e-3
bad = 0
for run, fold, data in CASES:
    d = json.load(open(os.path.join(ROOT, "runs", run, f"results_fold{fold}.json")))
    score = lambda e: sum(e["val"][k] for k in ("faz_dice", "vessel_dice", "vessel_cldice", "f1_macro"))
    best = max([e for e in d["history"] if e["epoch"] > d["args"]["startpoint"]], key=score)
    out = subprocess.run([sys.executable, "eval_checkpoint.py", "--run", f"runs/{run}", "--data", f"Datasets/processed_OCTA500/{data}", "--fold", str(fold), "--full_eval"],
                         cwd=ROOT, env=dict(os.environ, WANDB_MODE="disabled", PYTHONWARNINGS="ignore"), capture_output=True, text=True).stdout
    got = json.loads(out[out.index('{'):out.rindex('}') + 1])["test"]
    diffs = {k: round(got[k] - best["test"][k], 5) for k in KEYS}
    ok = all(abs(v) < TOL for v in diffs.values()); bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {run} fold {fold} (epoch {best['epoch']}): max |diff| {max(abs(v) for v in diffs.values()):.5f}")
sys.exit(1 if bad else 0)
