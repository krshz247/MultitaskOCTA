"""Inference module: a tiny random-weight ensemble loads, predicts, is deterministic and returns consistent shapes."""
import json, os, sys, tempfile, unittest
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")
from octa import predict as P


class Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        for k in (0, 1):
            torch.save(P.build_model().state_dict(), os.path.join(cls.dir.name, f"fold{k}.pt"))
        json.dump({"folds": {"0": {"mean": 0.3, "std": 0.2}, "1": {"mean": 0.35, "std": 0.25}}}, open(os.path.join(cls.dir.name, "stats.json"), "w"))
        cls.ens = P.load_ensemble(cls.dir.name, device="cpu")
        rng = np.random.default_rng(0)
        cls.image = (rng.random((400, 400)) * 255).astype(np.uint8)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_output(self):
        r = P.predict(self.ens, self.image)
        self.assertEqual(r["n_models"], 2)
        self.assertIn(r["label"], P.CLASSES)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=5)
        self.assertEqual(r["faz_mask"].shape, (P.SIZE, P.SIZE))
        self.assertEqual(r["vessel_mask"].shape, (P.SIZE, P.SIZE))
        self.assertFalse((r["faz_mask"] & r["vessel_mask"]).any())  # one class per pixel (argmax)

    def test_deterministic(self):
        a = P.predict(self.ens, self.image)
        b = P.predict(self.ens, self.image)
        self.assertEqual(a["probabilities"], b["probabilities"])
        self.assertTrue((a["faz_mask"] == b["faz_mask"]).all())

    def test_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            P.predict(self.ens, "/does/not/exist.bmp")


if __name__ == "__main__":
    unittest.main()
