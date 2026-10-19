"""The command line of the training scripts must not change: every option with its default, type, choices, nargs and help text is
compared with tests/golden/cli_parser.json, and the default configuration object must equal the parser's defaults.
  UPDATE_GOLDEN=1 python -m unittest tests.test_config    (only after an intended change of the options)"""
import json, os, sys, unittest, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from utils import create_train_arg_parser

GOLDEN = os.path.join(ROOT, "tests", "golden", "cli_parser.json")


def snapshot(parser):
    out = {}
    for a in parser._actions:
        if a.dest == "help":
            continue
        out[a.dest] = {"flags": a.option_strings, "default": a.default, "type": getattr(a.type, "__name__", None) if a.type else None, "nargs": a.nargs,
                       "choices": a.choices, "const": a.const, "action": type(a).__name__, "help": a.help}
    return out


class CommandLine(unittest.TestCase):
    def test_parser_is_unchanged(self):
        got = json.loads(json.dumps(snapshot(create_train_arg_parser())))
        if os.environ.get("UPDATE_GOLDEN"):
            json.dump(got, open(GOLDEN, "w"), indent=1); self.skipTest("rewritten")
        want = json.load(open(GOLDEN))
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            self.assertEqual(got[k], want[k], k)

    def test_defaults_namespace(self):
        ns = vars(create_train_arg_parser().parse_args([]))
        want = {k: v["default"] for k, v in json.load(open(GOLDEN)).items()}
        self.assertEqual(ns, want)


if __name__ == "__main__":
    unittest.main()


class TypedConfig(unittest.TestCase):
    def test_parse_config_matches_parser(self):
        import train_config as T
        self.assertEqual(vars(T.parse_config([])), vars(create_train_arg_parser().parse_args([])))
        c = T.parse_config(["--fold", "3", "--strong_aug", "--seg_loss_weights", "0", "0", "0", "--scheduler", "onecycle"])
        self.assertEqual((c.fold, c.strong_aug, c.seg_loss_weights, c.scheduler), (3, True, [0.0, 0.0, 0.0], "onecycle"))
        self.assertIsInstance(c, T.TrainConfig)

    def test_defaults_are_independent_copies(self):  # mutable defaults must not be shared between configurations
        import train_config as T
        a, b = T.parse_config([]), T.parse_config([])
        a.seg_loss_weights[0] = 99.0
        self.assertEqual(b.seg_loss_weights, [3.0, 1.0, 1.0])

    def test_every_yaml_option_exists(self):
        import glob, yaml, run_experiment as R
        for f in glob.glob(os.path.join(ROOT, "configs", "*.yaml")):
            self.assertEqual(R.validate_args(R.load(f).get("args", {})), [], f)
        self.assertEqual(R.validate_args({"num_epoch": 3}), [("num_epoch", ["num_epochs"])])
