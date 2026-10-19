"""Fast sanity tests for the pieces the thesis protocol depends on. Run: python -m unittest discover -s tests -v"""
import os, sys, unittest, collections, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Preprocessing"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")
from faz_vessel_seg_cls import kfold_split, make_scheduler
from dataset import random_rot_flip
from losses import VesselClDiceLoss, Vessel_FAZ_multiLoss
from eval_metrics import _cl_dice, _components, _dice_jaccard
import build_processed_octa500 as B

NAMES = [f"/x/image_surface/{c}_{f}_{i:05d}.bmp" for i, (c, f) in enumerate([("N", "3")] * 40 + [("N", "6")] * 20 + [("D", "3")] * 8 + [("D", "6")] * 8 + [("M", "3")] * 3 + [("M", "6")] * 12)]


class KFold(unittest.TestCase):
    def test_partition_and_stratification(self):
        tests = []
        for k in range(5):
            tr, va, te = kfold_split(NAMES, 5, k)
            self.assertFalse(set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            self.assertEqual(len(tr) + len(va) + len(te), len(NAMES))
            tests += te
            c = collections.Counter(os.path.basename(n)[0] for n in te)
            self.assertGreaterEqual(c["M"], 2)  # 15 AMD scans over 5 folds -> 3 each
        self.assertEqual(sorted(tests), sorted(NAMES))  # every scan is a test scan exactly once

    def test_fov_stratification(self):
        for k in range(5):
            te = kfold_split(NAMES, 5, k, stratify_fov=True)[2]
            c = collections.Counter(os.path.basename(n)[:3] for n in te)
            self.assertTrue(all(abs(c[g] - n / 5) <= 1 for g, n in (("N_3", 40), ("N_6", 20), ("M_6", 12))))

    def test_deterministic(self):
        self.assertEqual(kfold_split(NAMES, 5, 2), kfold_split(NAMES, 5, 2))


class Augmentation(unittest.TestCase):
    def test_image_and_maps_stay_aligned(self):
        img = np.random.rand(16, 16); maps = np.stack([img * 2, img + 1])
        for _ in range(50):
            a, m = random_rot_flip(img, maps)
            np.testing.assert_allclose(m[0], a * 2); np.testing.assert_allclose(m[1], a + 1)


class Losses(unittest.TestCase):
    def setUp(self):
        t = torch.zeros(1, 3, 48, 48); t[:, 0, 20:26, 20:26] = 1; t[:, 1, :, 40:] = 1; t[:, 2] = 1 - t[:, 0] - t[:, 1]
        self.t = t

    def test_perfect_prediction_has_near_zero_loss(self):
        z = torch.zeros(1, 2, 48, 48)
        _, seg, *_ = Vessel_FAZ_multiLoss([3, 1, 1])(self.t * 20 - 10, z, z, self.t, z, z)
        self.assertLess(seg.item(), 0.01)

    def test_cldice_term_is_nonzero(self):  # regression test: the vendored loss sliced [:,1:] and returned 0
        t = torch.zeros(1, 1, 48, 48); t[..., 10:12, :] = 1
        pred = torch.full_like(t, -10.0)  # predicts nothing -> clDice loss must be large
        self.assertGreater(VesselClDiceLoss()(pred, t).item(), 0.9)

    def test_scheduler_peak_and_end(self):
        p = torch.nn.Parameter(torch.zeros(1)); opt = torch.optim.Adam([{"params": [p], "lr": 1e-4}])
        s = make_scheduler(opt, "onecycle", 100, [1e-4], 0.1); lrs = []
        for _ in range(100): opt.step(); s.step(); lrs.append(opt.param_groups[0]["lr"])
        self.assertAlmostEqual(max(lrs), 1e-4, delta=1e-6); self.assertLess(lrs[-1], 1e-7)


class Metrics(unittest.TestCase):
    def test_cldice_and_components(self):
        g = np.zeros((64, 64), bool); g[10:12, :] = True; g[:, 30:32] = True
        self.assertAlmostEqual(_cl_dice(g, g)[0], 1.0, places=3)
        b = g.copy(); b[10:12, 28:34] = False
        self.assertLess(_cl_dice(b, g)[0], 1.0); self.assertGreater(_components(b)[0], _components(g)[0])

    def test_dice_jaccard(self):
        a = np.zeros((10, 10), bool); a[:5] = True; b = np.zeros((10, 10), bool); b[:, :5] = True
        d, j = _dice_jaccard(a, b); self.assertAlmostEqual(d, 0.5, places=3); self.assertAlmostEqual(j, 1 / 3, places=3)


class GroundTruthMaps(unittest.TestCase):
    def setUp(self):
        m = np.zeros((64, 64), np.uint8); m[20:40, 20:40] = 255; self.m = m

    def test_heatmap_is_a_ring(self):
        perim, heat = B.boundary_heatmap(self.m)
        self.assertAlmostEqual(heat.max(), 1.0, places=6); self.assertEqual(heat[30, 30], 0.0); self.assertGreater(heat[20, 30], 0.5)

    def test_sdm_sign_convention(self):
        perim, _ = B.boundary_heatmap(self.m); s = B.signed_distance(self.m, perim)
        self.assertLess(s[30, 30], 0.5); self.assertGreater(s[2, 2], 0.5)  # inside < 0.5 < outside


class Configs(unittest.TestCase):
    def test_every_config_loads_and_inherits(self):
        import glob, run_experiment as R
        for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.yaml")):
            cfg = R.load(f)
            self.assertIn(cfg["script"], R.SCRIPTS); self.assertIn("paths", cfg); self.assertGreaterEqual(cfg["args"]["num_epochs"], 1)
        reg = R.load(os.path.join(os.path.dirname(__file__), "..", "configs", "joint_regularised.yaml"))
        self.assertTrue(reg["args"]["strong_aug"]); self.assertEqual(reg["args"]["batch_size"], 4)  # inherited from joint_cv40

    def test_cli_and_override(self):
        import run_experiment as R
        self.assertEqual(R.cli({"a": 1, "flag": True, "off": False, "none": None, "xs": [1, 2]}), ["--a", "1", "--flag", "--xs", "1", "2"])
        cfg = {"args": {"num_epochs": 40}}; R.set_override(cfg, "args.num_epochs=2"); R.set_override(cfg, "data.limit=30")
        self.assertEqual(cfg, {"args": {"num_epochs": 2}, "data": {"limit": 30}})


if __name__ == "__main__":
    unittest.main()
