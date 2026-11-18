"""Checks for train.py / predict.py: resume after a disconnect, the run-directory
guard, automatic dev scoring, the test-set guard, the optional shared short
convolution, full-precision recurrences under autocast, and summarize_runs.py.

Runs on CPU in a few seconds. Skipped entirely when PyTorch is not installed.
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if HAVE_TORCH:
    import torch.nn.functional as F
    import predict
    import summarize_runs
    import train
    from models.attention import ATTENTION, linear_attention_chunked
    from models.model import LanguageModel, ModelConfig
    from score import evaluate as score_evaluate
    from synthdata.generator import generate_dataset


def run_quietly(fn, argv):
    """Call fn(argv) and return what it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(argv)
    return buf.getvalue()


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TrainingRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data = cls.tmp / "data"
        generate_dataset(str(cls.data), n_train=200, n_dev=40, n_test=10, seed=7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def args(self, out, steps=40, *extra):
        return ["--attention", "linear", "--out", str(self.tmp / out), "--data-dir", str(self.data),
                "--d-model", "32", "--n-layers", "2", "--n-heads", "2", "--steps", str(steps),
                "--max-tokens", "2048", "--eval-every", "10", "--eval-examples", "20",
                "--log-every", "10", "--warmup-steps", "5", "--device", "cpu",
                "--max-context-length", "256", *extra]

    def test_resume_matches_an_uninterrupted_run(self):
        run_quietly(train.main, self.args("straight", 40, "--no-final-eval"))

        # Simulate a disconnect right after the step-20 checkpoint.
        real = train.lr_at

        def disconnect(step, args):
            if step == 20:
                raise KeyboardInterrupt
            return real(step, args)
        train.lr_at = disconnect
        try:
            with self.assertRaises(KeyboardInterrupt):
                run_quietly(train.main, self.args("resumed", 40, "--no-final-eval"))
        finally:
            train.lr_at = real
        printed = run_quietly(train.main, self.args("resumed", 40, "--no-final-eval"))

        self.assertIn("at step 20", printed)            # continued, didn't start over
        lines = (self.tmp / "resumed" / "metrics.jsonl").read_text().splitlines()
        steps = [json.loads(l)["step"] for l in lines if json.loads(l)["type"] == "train"]
        self.assertEqual(steps, [10, 20, 30, 40])        # every step trained exactly once
        a = torch.load(self.tmp / "straight" / "last.pt", weights_only=False)["model"]
        b = torch.load(self.tmp / "resumed" / "last.pt", weights_only=False)["model"]
        for name in a:
            torch.testing.assert_close(a[name], b[name], msg=name)

    def test_run_dir_refuses_different_settings(self):
        run_quietly(train.main, self.args("guarded", 10, "--no-final-eval"))
        with self.assertRaises(SystemExit) as cm:
            run_quietly(train.main, self.args("guarded", 20, "--no-final-eval"))
        self.assertIn("steps: 10 -> 20", str(cm.exception.code))

    def test_final_eval_scores_the_filtered_dev_split(self):
        run_quietly(train.main, self.args("scored", 10))
        out = self.tmp / "scored"
        n_dev = sum(1 for l in (self.data / "dev.jsonl").read_text().splitlines()
                    if json.loads(l)["context_length"] <= 256)
        preds = (out / "dev_pred.jsonl").read_text().splitlines()
        self.assertEqual(len(preds), n_dev)
        report = json.loads((out / "dev_scores.json").read_text())
        self.assertEqual(report["overall"]["n"], n_dev)
        self.assertEqual(report, score_evaluate(out / "dev_gold.jsonl", out / "dev_pred.jsonl"))

        # predict.py refuses the test split unless --final is given.
        base = ["--checkpoint", str(out / "best.pt"), "--data-dir", str(self.data), "--device", "cpu"]
        with self.assertRaises(SystemExit) as cm:
            run_quietly(predict.main, base + ["--split", "test", "--out", str(out / "test_pred.jsonl")])
        self.assertIn("--final", str(cm.exception.code))
        self.assertFalse((out / "test_pred.jsonl").exists())
        run_quietly(predict.main, base + ["--split", "test", "--final", "--out", str(out / "test_pred.jsonl")])
        self.assertEqual(len((out / "test_pred.jsonl").read_text().splitlines()), 10)

        # summarize_runs.py lists the finished run.
        rows = []
        with contextlib.redirect_stdout(io.StringIO()):
            rows = summarize_runs.main(["--runs-dir", str(self.tmp), "--out", str(self.tmp / "s.csv")])
        self.assertIn("scored", [r["run"] for r in rows])
        self.assertEqual(next(r for r in rows if r["run"] == "scored")["attention"], "linear")


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class ShortConvTest(unittest.TestCase):
    def cfg(self, attention, **kw):
        return ModelConfig(vocab_size=50, attention=attention, d_model=32, n_layers=2, n_heads=2,
                           linear_chunk=8, **kw)

    def test_off_by_default(self):
        m = LanguageModel(self.cfg("linear"))
        self.assertFalse(any(".conv." in n for n, _ in m.named_parameters() if ".attn." not in n))

    def test_causal_and_same_for_every_attention(self):
        extra = set()
        for name in ATTENTION:
            with self.subTest(name):
                torch.manual_seed(0)
                with_conv = LanguageModel(self.cfg(name, short_conv=4)).double().eval()
                without = LanguageModel(self.cfg(name))
                extra.add(with_conv.num_params() - without.num_params())
                ids = torch.randint(4, 50, (2, 20))
                ids2 = ids.clone()
                ids2[:, 12:] = torch.randint(4, 50, (2, 8))
                torch.testing.assert_close(with_conv(ids)[:, :12], with_conv(ids2)[:, :12])
        self.assertEqual(len(extra), 1)       # the shared conv adds the same parameters to every model


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class FullPrecisionTest(unittest.TestCase):
    """Under bf16 autocast, the reference recurrences must still compute in float32."""

    def test_linear_ignores_autocast(self):
        torch.manual_seed(0)
        q, k = (F.elu(torch.randn(2, 2, 300, 16)) + 1 for _ in range(2))
        v = torch.randn(2, 2, 300, 16)
        ref = linear_attention_chunked(q, k, v)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            got = linear_attention_chunked(q, k, v)
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
