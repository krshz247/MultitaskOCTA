"""The network definitions must keep their parameter names and shapes: stored checkpoints (state dicts) have to keep loading.
For one encoder of every supported family the segmentation model and the classifier are built without pretrained weights, run on a
64 x 64 input (evaluation mode) and the hash of all (parameter name, shape) pairs is compared with tests/golden/model_structure.json.
  UPDATE_GOLDEN=1 python -m unittest tests.test_model_structure   rewrites the file (only after an intended change of the architecture)"""
import hashlib, json, os, sys, unittest, warnings
import torch
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import octa.models as M

GOLDEN = os.path.join(ROOT, "tests", "golden", "model_structure.json")
FAMILIES = {"resnet": "resnet50", "resnest": "timm-resnest50d", "res2net": "timm-res2net50_26w_4s", "sknet": "timm-skresnet18", "vgg": "vgg11"}


def describe(name):
    torch.manual_seed(0)
    seg = M.MyMultibranchModel(encoder_name=name, encoder_depth=5, encoder_weights=None, decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
                               decoder_attention_type=None, in_channels=1, out_channel=2, classes=3, activation=None, aux_params=None).eval()
    clf = M.my_get_encoder(name, in_channels=1, depth=5, weights=None, decoder_channels=(256, 128, 64, 32, 16), num_classes=3).eval()
    with torch.no_grad():
        so = seg(torch.randn(1, 1, 64, 64)); co = clf(torch.randn(1, 1, 64, 64), so[3], so[4], so[5])
    pairs = [("seg." + k, list(v.shape)) for k, v in seg.state_dict().items()] + [("clf." + k, list(v.shape)) for k, v in clf.state_dict().items()]
    return {"hash": hashlib.md5(json.dumps(pairs).encode()).hexdigest(), "n_tensors": len(pairs), "seg_out": list(so[0].shape), "cls_out": list(co.shape)}


class ModelStructure(unittest.TestCase):
    def test_parameter_names_and_shapes_are_stable(self):
        got = {fam: describe(name) for fam, name in FAMILIES.items()}
        if os.environ.get("UPDATE_GOLDEN"):
            json.dump(got, open(GOLDEN, "w"), indent=1, sort_keys=True); self.skipTest("rewritten")
        self.assertEqual(got, json.load(open(GOLDEN)))


if __name__ == "__main__":
    unittest.main()
