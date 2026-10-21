"""analysis/oof_rule.py: epoch choice per fold from validation metrics, pooling of the logged test probabilities."""
import json, os, sys, tempfile, unittest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import oof_rule as O


def fake_run(root, names_by_fold, good_epoch):
    for k, names in names_by_fold.items():
        hist = []
        for ep in (1, 2):
            p = [[0.8, 0.1, 0.1] if ep == good_epoch else [0.1, 0.8, 0.1]] * len(names)  # class 0 predicted at the good epoch
            hist.append({"epoch": ep, "val": {"balanced_acc": 0.9 if ep == good_epoch else 0.1, "f1_macro": 0.9 if ep == good_epoch else 0.1}, "test": {}, "test_probs": p})
        json.dump({"args": {}, "test_names": names, "history": hist}, open(os.path.join(root, f"results_fold{k}.json"), "w"))


class OofRule(unittest.TestCase):
    def test_rule_picks_the_validation_best_epoch_and_pools(self):
        with tempfile.TemporaryDirectory() as d:
            fake_run(d, {0: ["N_3_1.bmp", "N_6_2.bmp"], 1: ["D_3_3.bmp", "M_6_4.bmp"]}, good_epoch=1)
            names, p = O.pooled(d, "val_bal")
            self.assertEqual(names, ["D_3_3.bmp", "M_6_4.bmp", "N_3_1.bmp", "N_6_2.bmp"])
            self.assertEqual(p.argmax(1).tolist(), [0, 0, 0, 0])  # epoch 1 chosen everywhere
            self.assertEqual(O.pooled(d, "last")[1].argmax(1).tolist(), [1, 1, 1, 1])
            self.assertEqual(O.labels(names).tolist(), [1, 2, 0, 0])

    def test_run_without_probabilities_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"args": {}, "history": [{"epoch": 1, "val": {}, "test": {}}]}, open(os.path.join(d, "results_fold0.json"), "w"))
            self.assertIsNone(O.pooled(d, "val_bal"))


if __name__ == "__main__":
    unittest.main()
