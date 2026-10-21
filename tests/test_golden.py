"""Golden-file regression tests: a tiny deterministic CPU training run must reproduce the stored metrics.

A synthetic data set (24 'scans' with a disc-shaped FAZ and random vessel lines, built with the real ground-truth-map code of
Preprocessing/build_processed_octa500.py) is trained for 2 epochs on the CPU without pretrained weights, for the joint model and
the classification-only baseline. On one machine such a run is bit-for-bit reproducible; the tolerance (1e-4) only absorbs
differences between machines. Use these tests to prove that a refactoring did not change behaviour.
  python -m unittest tests.test_golden            (about 3-4 minutes)      SKIP_SLOW_TESTS=1 skips them
  UPDATE_GOLDEN=1 python -m unittest tests.test_golden    rewrites tests/golden/*.json (only after an intended behaviour change)
"""
import json, os, shutil, subprocess, sys, tempfile, unittest
import numpy as np, cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
sys.path.insert(0, os.path.join(ROOT, "Preprocessing"))
import build_processed_octa500 as B

KEYS_EXCLUDE = {"epoch"}
TOL = float(os.environ.get("GOLDEN_TOL", "1e-4"))  # GOLDEN_TOL=1e-9 demands bit-identical results (same machine)


def make_dataset(out, n_per=(12, 6, 6), size=384, seed=0):
    rng = np.random.default_rng(seed)
    for d in ("image_surface", "gt_full/mask", "gt_vessel/mask", "gt_faz/mask", "gt_vessel/contour_img", "gt_faz/contour_img", "gt_vessel/dist_signed11_img", "gt_faz/dist_signed11_img"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    i = 0
    for cls, n in zip("NDM", n_per):
        for _ in range(n):
            name = f"{cls}_{3 if i % 2 else 6}_{10000 + i}"
            faz = np.zeros((size, size), np.uint8); cv2.circle(faz, (size // 2 + int(rng.integers(-8, 8)), size // 2 + int(rng.integers(-8, 8))), int(rng.integers(20, 34)), 255, -1)
            ves = np.zeros((size, size), np.uint8)
            for _ in range(14):
                p = rng.integers(0, size, 4); cv2.line(ves, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 255, int(rng.integers(2, 5)))
            ves[faz > 0] = 0
            img = np.clip(0.15 * 255 * rng.random((size, size)) + 0.8 * ves + 0.05 * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out, "image_surface", name + ".bmp"), img)
            cv2.imwrite(os.path.join(out, "gt_full/mask", name + ".bmp"), np.maximum(faz, ves))
            for key, m in (("faz", faz), ("vessel", ves)):
                cv2.imwrite(os.path.join(out, f"gt_{key}/mask", name + ".bmp"), m)
                perim, heat = B.boundary_heatmap(m)
                cv2.imwrite(os.path.join(out, f"gt_{key}/contour_img", name + ".png"), B.to_u8(heat))
                cv2.imwrite(os.path.join(out, f"gt_{key}/dist_signed11_img", name + ".png"), B.to_u8(B.signed_distance(m, perim)))
            i += 1


def run(script, data, save, extra=(), block=()):
    os.makedirs(os.path.join(save, "summary"), exist_ok=True)
    launcher = [sys.executable, script]
    if block:  # run the script with some modules made un-importable (to test optional dependencies)
        launcher = [sys.executable, "-c", f"import sys, runpy; [sys.modules.__setitem__(m, None) for m in {list(block)!r}]; sys.argv = [{script!r}] + sys.argv[1:]; runpy.run_path({script!r}, run_name='__main__')"]
    cmd = launcher + ["--img_path", os.path.join(data, "image_surface"), "--gt_path", os.path.join(data, "gt_full", "mask"), "--save_path", save,
           "--train_type", "Vessel_FAZ", "--distance_type", "dist_signed11", "--device", "cpu", "--pretrain", "none", "--num_epochs", "2", "--startpoint", "1",
           "--eval_every", "1", "--full_eval", "--num_folds", "3", "--fold", "0", "--batch_size", "4", "--val_batch_size", "4"] + list(extra)
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, WANDB_MODE="disabled", PYTHONWARNINGS="ignore"), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.load(open(os.path.join(save, "results_fold0.json")))["history"]


def flatten(history):
    return {f"ep{h['epoch']}/{split}/{k}": float(v) for h in history for split in ("val", "test") for k, v in h[split].items() if isinstance(v, (int, float)) and k not in KEYS_EXCLUDE}


@unittest.skipIf(os.environ.get("SKIP_SLOW_TESTS"), "slow")
class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cls.data = os.path.join(cls.tmp, "data"); make_dataset(cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check(self, name, script, extra=(), block=(), golden=None):
        got = flatten(run(script, self.data, os.path.join(self.tmp, name), extra, block)); path = os.path.join(GOLDEN, (golden or name) + ".json")
        if os.environ.get("UPDATE_GOLDEN"):
            os.makedirs(GOLDEN, exist_ok=True); json.dump(got, open(path, "w"), indent=0, sort_keys=True); self.skipTest("golden file rewritten")
        want = json.load(open(path))
        self.assertEqual(sorted(got), sorted(want), "different set of logged metrics")
        bad = {k: (got[k], want[k]) for k in want if not np.isclose(got[k], want[k], rtol=TOL, atol=TOL)}
        self.assertFalse(bad, f"{len(bad)} metrics changed, e.g. {dict(list(bad.items())[:5])}")

    def test_joint_model(self):
        self.check("joint", "faz_vessel_seg_cls.py")

    def test_joint_model_regularised(self):
        self.check("joint_reg", "faz_vessel_seg_cls.py", ["--strong_aug", "--label_smoothing", "0.1", "--cls_dropout", "0.3", "--weight_decay", "0.01", "--scheduler", "onecycle"])

    def test_joint_model_ablation_switches(self):  # stop-gradient on the decoder features, shuffled masks, explicit loss weights
        self.check("joint_ablation", "faz_vessel_seg_cls.py", ["--detach_clf_features", "--shuffle_masks", "--seg_loss_weights", "2", "1", "1"])

    def test_joint_model_without_wandb(self):  # wandb is an optional dependency: same numbers when it is not installed
        self.check("joint_nowandb", "faz_vessel_seg_cls.py", block=["wandb"], golden="joint")

    def test_joint_model_other_seed(self):  # a different training seed is reproducible and gives different numbers
        self.check("joint_seed7", "faz_vessel_seg_cls.py", ["--seed", "7"])
        a, b = json.load(open(os.path.join(GOLDEN, "joint_seed7.json"))), json.load(open(os.path.join(GOLDEN, "joint.json")))
        self.assertNotEqual(a, b)

    def test_classification_baseline(self):
        self.check("cls", "classification_baseline.py")

    def _probabilities_logged(self, name, script):
        save = os.path.join(self.tmp, name)
        run(script, self.data, save, ["--cls_ckpt", "balanced_acc"])
        d = json.load(open(os.path.join(save, "results_fold0.json")))
        n = len(d["test_names"])
        self.assertGreater(n, 0)
        for h in d["history"]:  # one row of class probabilities per test scan and evaluated epoch
            p = np.array(h["test_probs"])
            self.assertEqual(p.shape, (n, 3))
            self.assertTrue(np.allclose(p.sum(1), 1.0, atol=1e-3))
        self.assertTrue(os.path.exists(os.path.join(save, "best_cls_fold0.pt")))
        self.assertTrue(os.path.exists(os.path.join(save, "best_fold0.pt")))

    def test_joint_logs_probabilities_and_second_checkpoint(self):
        self._probabilities_logged("joint_probs", "faz_vessel_seg_cls.py")

    def test_baseline_logs_probabilities_and_second_checkpoint(self):
        self._probabilities_logged("cls_probs", "classification_baseline.py")


if __name__ == "__main__":
    unittest.main()
