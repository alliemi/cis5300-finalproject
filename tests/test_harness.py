"""Tests for the model harness. Skipped automatically when torch isn't installed.

    .venv/bin/python -m unittest discover -s tests -v

The contract tests run for every mixer registered in models.MIXERS, so a
newly added architecture is checked without writing new tests.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from models import MIXERS, build_model
    from models.linear_attention import feature_map, linear_attention_chunked, linear_attention_recurrent
    from training.config import load_config
    from training.data import BOS, EOS, SEP, BucketBatcher, collate, encode_record, encode_split, load_vocab
    from training.trainer import train
    from synthdata.generator import generate_dataset

needs_torch = unittest.skipIf(torch is None, "torch not installed")

TINY = ["model.d_model=32", "model.n_layers=2", "model.n_heads=2", "train.batch_size=8",
        "train.warmup_steps=5", "train.log_every=10", "train.eval_every=10", "train.eval_examples=40"]


def tiny_model(name):
    cfg = {"mixer": name, "mixer_cfg": {},
           "model": {"d_model": 32, "n_layers": 2, "n_heads": 2, "mlp_ratio": 2, "short_conv": 4}}
    return build_model(cfg, vocab_size=50).double()


@needs_torch
class TestMixerContract(unittest.TestCase):
    """Every registered mixer must satisfy these, inside the full model."""

    def setUp(self):
        torch.manual_seed(0)

    def runnable(self):
        return [n for n, cls in sorted(MIXERS.items())
                if not getattr(cls, "requires_cuda", False) or torch.cuda.is_available()]

    def test_shapes_and_odd_lengths(self):
        for name in self.runnable():
            model = tiny_model(name)
            for T in (1, 7, 64, 130):
                with self.subTest(mixer=name, T=T):
                    x = torch.randint(0, 50, (2, T))
                    self.assertEqual(model(x).shape, (2, T, 50))
                    self.assertEqual(model(x, torch.tensor([T - 1, 0])).shape, (2, 50))

    def test_causal(self):
        """Changing later tokens must not change earlier outputs."""
        for name in self.runnable():
            with self.subTest(mixer=name):
                model = tiny_model(name)
                x = torch.randint(0, 50, (2, 130))
                y = x.clone()
                y[:, 70:] = torch.randint(0, 50, (2, 60))
                torch.testing.assert_close(model(x)[:, :70], model(y)[:, :70])

    def test_finite_gradients(self):
        for name in self.runnable():
            with self.subTest(mixer=name):
                model = tiny_model(name)
                model(torch.randint(0, 50, (2, 100))).sum().backward()
                for pname, p in model.named_parameters():
                    self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(), pname)


@needs_torch
class TestLinearAttention(unittest.TestCase):
    def test_chunked_matches_recurrent(self):
        torch.manual_seed(0)
        for T in (1, 7, 64, 130):
            for C in (1, 16, 64):
                with self.subTest(T=T, chunk=C):
                    q, k, v = (torch.randn(2, 3, T, 8, dtype=torch.float64) for _ in range(3))
                    q, k = feature_map(q), feature_map(k)
                    torch.testing.assert_close(linear_attention_chunked(q, k, v, C),
                                               linear_attention_recurrent(q, k, v))

    def test_memory_only_adds(self):
        """Vanilla linear attention can't overwrite: after seeing the same key with
        two values, the output is their average, not the latest value."""
        k = torch.ones(1, 1, 2, 4, dtype=torch.float64)
        v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
        out = linear_attention_chunked(k, k, v, chunk_size=64)
        torch.testing.assert_close(out[0, 0, 1], torch.tensor([0.5, 0.5], dtype=torch.float64))


class _TinyData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.data = os.path.join(cls.tmp, "data")
        generate_dataset(cls.data, n_train=200, n_dev=40, n_test=10, seed=7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)


@needs_torch
class TestData(_TinyData):
    def test_encoding(self):
        vocab = load_vocab(self.data)
        tok2id = {t: i for i, t in enumerate(vocab)}
        with open(os.path.join(self.data, "dev.jsonl")) as f:
            rec = json.loads(f.readline())
        ids, a = encode_record(rec, tok2id)
        self.assertEqual(ids[0], tok2id[BOS])
        self.assertEqual(ids[a - 1], tok2id[SEP])
        self.assertEqual(vocab[ids[a]], rec["answer"])
        self.assertEqual(ids[-1], tok2id[EOS])
        # bos + context + sep + 9 query tokens + sep + answer + eos
        self.assertEqual(len(ids), rec["context_tokens"] + 14)

    def test_longest_input(self):
        split = encode_split(self.data, "train")
        longest = max(split.input_length(i) + 2 for i in range(len(split)))
        self.assertLessEqual(longest, 2062)

    def test_cache_and_batches(self):
        a = encode_split(self.data, "train")
        b = encode_split(self.data, "train")       # second call reads the cache
        self.assertTrue(torch.equal(a.tokens, b.tokens))
        batcher = BucketBatcher(a, range(len(a)), batch_size=8, seed=3)
        batches = batcher.epoch_batches(0)
        self.assertEqual(sorted(i for bt in batches for i in bt), list(range(len(a))))
        for bt in batches:
            self.assertEqual(len({int(a.context_length[i]) for i in bt}), 1)   # one bucket per batch
        self.assertEqual(batches, batcher.epoch_batches(0))
        self.assertNotEqual(batches, batcher.epoch_batches(1))
        batch = collate(a, batches[0], pad_id=0)
        for row, i in enumerate(batches[0]):
            self.assertEqual(int(batch["last"][row]), a.input_length(i) - 1)

    def test_select_filters(self):
        split = encode_split(self.data, "train")
        short = split.select(max_context_length=256, limit=20)
        self.assertLessEqual(len(short), 20)
        self.assertTrue(all(int(split.context_length[i]) <= 256 for i in short))


@needs_torch
class TestConfig(unittest.TestCase):
    def test_merge_and_overrides(self):
        cfg = load_config(os.path.join(ROOT, "configs", "linear.json"), ["train.lr=3e-4", "model.n_layers=2"])
        self.assertEqual(cfg["mixer"], "linear")
        self.assertEqual(cfg["train"]["lr"], 3e-4)
        self.assertEqual(cfg["model"]["n_layers"], 2)
        self.assertEqual(cfg["model"]["d_model"], 256)      # inherited from base.json

    def test_every_config_names_a_registered_mixer(self):
        for name in os.listdir(os.path.join(ROOT, "configs")):
            if name != "base.json" and name.endswith(".json"):
                with self.subTest(config=name):
                    self.assertIn(load_config(os.path.join(ROOT, "configs", name))["mixer"], MIXERS)


@needs_torch
class TestTraining(_TinyData):
    def run_train(self, run, steps, mixer="none", final_eval=False):
        cfg = load_config(os.path.join(ROOT, "configs", f"{mixer}.json"), TINY + [f"train.max_steps={steps}"])
        return train(cfg, os.path.join(self.tmp, run), data_dir=self.data, device="cpu",
                     max_context_length=256, final_eval=final_eval, log=lambda msg: None)

    def test_run_dir_rejects_a_different_config(self):
        self.run_train("guarded", 10)
        with self.assertRaisesRegex(ValueError, "different config"):
            self.run_train("guarded", 20)     # changed max_steps changes the LR schedule

    def test_resume_is_exact(self):
        cfg = load_config(os.path.join(ROOT, "configs", "linear.json"), TINY + ["train.max_steps=40"])
        straight = os.path.join(self.tmp, "exact-straight")
        resumed = os.path.join(self.tmp, "exact-resumed")
        train(cfg, straight, data_dir=self.data, device="cpu", max_context_length=256,
              final_eval=False, log=lambda msg: None)
        # Simulate a disconnect after step 20: train to 20 with the same config by
        # stopping early, then rerun the identical command.
        import training.trainer as T
        real = T.lr_at

        def interrupt(step, t):
            if step == 20:
                raise KeyboardInterrupt
            return real(step, t)
        T.lr_at = interrupt
        try:
            with self.assertRaises(KeyboardInterrupt):
                train(cfg, resumed, data_dir=self.data, device="cpu", max_context_length=256,
                      final_eval=False, log=lambda msg: None)
        finally:
            T.lr_at = real
        messages = []
        summary = train(cfg, resumed, data_dir=self.data, device="cpu", max_context_length=256,
                        final_eval=False, log=messages.append)
        self.assertIn("at step 20", messages[0])      # it resumed instead of starting over
        self.assertEqual(summary["step"], 40)
        with open(os.path.join(resumed, "log.jsonl")) as f:
            steps = [json.loads(line)["step"] for line in f if '"train"' in line]
        self.assertEqual(steps, [10, 20, 30, 40])      # each step trained exactly once
        a = torch.load(os.path.join(straight, "last.pt"), weights_only=False)["model"]
        b = torch.load(os.path.join(resumed, "last.pt"), weights_only=False)["model"]
        for name in a:
            torch.testing.assert_close(a[name], b[name], msg=name)

    def test_train_predict_score_round_trip(self):
        summary = self.run_train("roundtrip", 20, mixer="linear", final_eval=True)
        run = os.path.join(self.tmp, "roundtrip")
        for f in ("config.json", "log.jsonl", "best.pt", "last.pt", "dev_predictions.jsonl", "dev_scores.json"):
            self.assertTrue(os.path.exists(os.path.join(run, f)), f)
        with open(os.path.join(run, "dev_scores.json")) as f:
            report = json.load(f)
        self.assertEqual(report["overall"]["n"], 40)        # full dev, not just the eval subset
        self.assertEqual(summary["dev_accuracy"], report["overall"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
