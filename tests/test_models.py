"""Checks for the attention layers, backbone, data pipeline and decoding.

Runs on CPU. Skipped entirely when PyTorch is not installed.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # the data and scoring tests only need the standard library
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[1]

if HAVE_TORCH:
    import torch.nn.functional as F
    from models.attention import (ATTENTION, build_attention, linear_attention_chunked,
                                  linear_attention_recurrent)
    from models.data import (EXTRA_QUERY_LEN, IGNORE_INDEX, BucketBatchSampler, Vocab, collate,
                             encode_record, eval_batches, load_split)
    from models.inference import greedy_decode, load_checkpoint, save_checkpoint
    from models.model import LanguageModel, ModelConfig

    def tiny_cfg(attention, **kw):
        base = dict(vocab_size=50, attention=attention, d_model=32, n_layers=2, n_heads=2,
                    linear_chunk=8)
        base.update(kw)
        return ModelConfig(**base)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class AttentionLayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_all_layers(self):
        for name in ATTENTION:
            with self.subTest(name):
                layer = build_attention(tiny_cfg(name))
                x = torch.randn(3, 21, 32)
                self.assertEqual(layer(x).shape, x.shape)

    def test_causal_all_layers(self):
        """Changing tokens after position t must not change outputs at <= t."""
        T, t = 20, 11
        for name in ATTENTION:
            with self.subTest(name):
                layer = build_attention(tiny_cfg(name)).double()
                x = torch.randn(2, T, 32, dtype=torch.float64)
                x2 = x.clone()
                x2[:, t + 1:] = torch.randn(2, T - t - 1, 32, dtype=torch.float64)
                y, y2 = layer(x), layer(x2)
                torch.testing.assert_close(y[:, :t + 1], y2[:, :t + 1])
                self.assertFalse(torch.allclose(y[:, t + 1:], y2[:, t + 1:]))

    def test_gradients_flow(self):
        for name in ATTENTION:
            with self.subTest(name):
                layer = build_attention(tiny_cfg(name))
                layer(torch.randn(2, 9, 32)).pow(2).mean().backward()
                for pname, p in layer.named_parameters():
                    self.assertIsNotNone(p.grad, pname)
                    self.assertTrue(torch.isfinite(p.grad).all(), pname)

    def test_linear_chunked_matches_recurrent(self):
        for T, chunk in [(1, 4), (7, 4), (16, 4), (33, 8)]:
            with self.subTest(T=T, chunk=chunk):
                q, k = (F.elu(torch.randn(2, 3, T, 5, dtype=torch.float64)) + 1 for _ in range(2))
                v = torch.randn(2, 3, T, 6, dtype=torch.float64)
                ref = linear_attention_recurrent(q, k, v).double()
                got = linear_attention_chunked(q, k, v, chunk=chunk).double()
                torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class BackboneTest(unittest.TestCase):
    def test_only_mixer_differs(self):
        counts = {}
        for name in ATTENTION:
            m = LanguageModel(tiny_cfg(name))
            counts[name] = m.num_params() - m.num_mixer_params()
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_right_padding_does_not_change_real_positions(self):
        for name in ATTENTION:
            with self.subTest(name):
                torch.manual_seed(0)
                m = LanguageModel(tiny_cfg(name)).double().eval()
                ids = torch.randint(4, 50, (1, 13))
                padded = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)
                torch.testing.assert_close(m(ids), m(padded)[:, :13])

    def test_config_ignores_removed_settings(self):
        """Checkpoints saved before gated_deltanet/kda were removed still load."""
        old = dict(tiny_cfg("softmax").to_dict(), conv_size=4, expand_v=1.0, kernel="auto")
        self.assertEqual(ModelConfig.from_dict(old), tiny_cfg("softmax"))

    def test_checkpoint_roundtrip(self):
        vocab = Vocab(["<pad>", "<bos>", "<sep>", "<eos>"] + [f"w{i}" for i in range(46)])
        m = LanguageModel(tiny_cfg("linear"))
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(Path(d) / "c.pt", m, vocab, step=3)
            m2, v2, ck = load_checkpoint(Path(d) / "c.pt", torch.device("cpu"))
        ids = torch.randint(4, 50, (2, 9))
        torch.testing.assert_close(m.eval()(ids), m2(ids))
        self.assertEqual(v2.itos, vocab.itos)
        self.assertEqual(ck["step"], 3)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        from synthdata.tokenizer import build_vocab
        (d / "vocab.txt").write_text("\n".join(build_vocab()) + "\n")
        recs = [
            {"id": "x-1", "task": "recall", "context_length": 128,
             "context": "Project Trout status: oasis. The team lunch ran late on Friday.",
             "query": "What is the current status of Project Trout?", "answer": "oasis"},
            {"id": "x-2", "task": "update", "context_length": 256,
             "context": "Project Opossum is onion. Project Opossum is merged.",
             "query": "What is the current status of Project Opossum?", "answer": "merged"},
            {"id": "x-3", "task": "update", "context_length": 128,
             "context": "Project Shark status: jasmine. Project Shark is velvet.",
             "query": "What is the current status of Project Shark?", "answer": "velvet"},
            {"id": "x-4", "task": "update", "context_length": 128,
             "context": "Project Trout status: oasis. Project Heron is red. The team lunch ran late "
                        "on Friday. Project Falcon is pending. Project Trout is merged. "
                        "The status of Project Heron is blue.",
             "query": "What is the current status of Project Trout?", "answer": "merged"},
        ]
        (d / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        cls.dir, cls.recs = d, recs
        cls.vocab = Vocab.load(d / "vocab.txt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_vocab_file_matches_data_vocab(self):
        real = ROOT / "data" / "vocab.txt"
        if real.exists():
            self.assertEqual(Vocab.load(real).itos, self.vocab.itos)

    def test_encode_format(self):
        v = self.vocab
        seq, plen, _ = encode_record(self.recs[0], v)
        self.assertEqual(seq[0], v.bos_id)
        self.assertEqual(seq[plen - 1], v.sep_id)
        self.assertEqual([v.itos[i] for i in seq[plen:]], ["oasis", "<eos>"])
        self.assertEqual(sum(1 for i in seq if i == v.sep_id), 2)
        self.assertEqual(v.decode(seq[plen:]), "oasis")

    def test_encode_extra_pairs_are_other_projects_current_values(self):
        v = self.vocab
        _, _, extra = encode_record(self.recs[3], v)
        self.assertEqual([(v.itos[e], v.itos[val]) for e, val in extra],
                         [("Heron", "blue"), ("Falcon", "pending")])
        self.assertEqual(encode_record(self.recs[0], v)[2], [])
        bad = dict(self.recs[3], answer="oasis")      # stale value given as answer
        with self.assertRaises(ValueError):
            encode_record(bad, v)

    def test_load_split_and_cache(self):
        a = load_split(self.dir, "dev", self.vocab)
        b = load_split(self.dir, "dev", self.vocab)      # from cache
        self.assertEqual(a.ids, b.ids)
        self.assertTrue((a.tokens == b.tokens).all())
        self.assertTrue((a.extra == b.extra).all())
        short = load_split(self.dir, "dev", self.vocab, max_context_length=128)
        self.assertEqual(short.ids, ["x-1", "x-3", "x-4"])
        self.assertEqual(short.num_extra(2), 2)          # extra pairs survive subsetting
        self.assertEqual(len(load_split(self.dir, "dev", self.vocab, max_examples=2, seed=1)), 2)

    def test_collate_supervises_only_answer_and_eos(self):
        split = load_split(self.dir, "dev", self.vocab)
        x, y, official = collate(split, [0, 1, 3], self.vocab)
        for row, i in enumerate([0, 1, 3]):
            p, n = int(split.prompt_lens[i]), split.seq_len(i)
            labels = y[row][y[row] != IGNORE_INDEX].tolist()
            self.assertEqual(labels, split.seq(i)[p:].tolist())
            self.assertEqual(x[row, p - 1].item(), self.vocab.sep_id)   # predicts the answer
            self.assertTrue((x[row, n - 1:] == self.vocab.pad_id).all())
            self.assertEqual(official[row].nonzero().flatten().tolist(), [p - 1])

    def test_collate_multi_query(self):
        v = self.vocab
        split = load_split(self.dir, "dev", v)
        x, y, official = collate(split, [3], v, max_extra=-1, rng=random.Random(0))
        p = int(split.prompt_lens[3])
        # prompt is untouched, so evaluation (which decodes after the prompt) is unchanged
        self.assertEqual(x[0, :p].tolist(), split.seq(3)[:p].tolist())
        self.assertEqual(official[0].nonzero().flatten().tolist(), [p - 1])
        self.assertEqual(y[0, p - 1].item(), v.stoi["merged"])
        # labels: official answer, then each other project's current value, then <eos>
        labels = [v.itos[t] for t in y[0][y[0] != IGNORE_INDEX].tolist()]
        self.assertEqual(labels[0], "merged")
        self.assertEqual(sorted(labels[1:-1]), ["blue", "pending"])
        self.assertEqual(labels[-1], "<eos>")
        # each extra value is predicted right after "<sep> What is ... Project E ? <sep>"
        text = [v.itos[t] for t in x[0].tolist()]
        for pos in (y[0] != IGNORE_INDEX).nonzero().flatten().tolist()[1:-1]:
            self.assertEqual(text[pos], "<sep>")
            entity = text[pos - 2]
            q_len = EXTRA_QUERY_LEN - 3
            self.assertEqual(text[pos - q_len:pos], [v.itos[t] for t in v.query_ids(v.stoi[entity])])
            self.assertEqual(text[pos - q_len - 1], "<sep>")
            self.assertEqual({"Heron": "blue", "Falcon": "pending"}[entity], v.itos[y[0, pos].item()])
        self.assertEqual(len(split.seq(3)) + 2 * EXTRA_QUERY_LEN, x.shape[1] + 1)
        # a cap of 1 adds exactly one extra query
        x1, y1, _ = collate(split, [3], v, max_extra=1, rng=random.Random(0))
        self.assertEqual((y1 != IGNORE_INDEX).sum().item(), 3)
        self.assertEqual(split.train_len(3, 1), x1.shape[1] + 1)

    def test_samplers(self):
        split = load_split(self.dir, "dev", self.vocab)
        s = BucketBatchSampler(split, max_tokens=10_000, seed=0, curriculum_steps=10)
        s_mq = BucketBatchSampler(split, max_tokens=10_000, seed=0, max_extra=-1)
        self.assertLess(s_mq.batch_sizes[0], s.batch_sizes[0])   # room for the extra queries
        self.assertEqual(s.allowed(0), 1)
        self.assertEqual(s.allowed(10), 2)
        for step in range(5):     # curriculum: only the 128 bucket at the start
            self.assertTrue(all(split.buckets[i] == 128 for i in s.batch(step)))
        covered = sorted(i for b in eval_batches(split, 40) for i in b)
        self.assertEqual(covered, [0, 1, 2, 3])

    def test_greedy_decode_returns_tokens(self):
        split = load_split(self.dir, "dev", self.vocab)
        m = LanguageModel(tiny_cfg("softmax", vocab_size=len(self.vocab)))
        out = greedy_decode(m, split, [0, 1, 2], self.vocab, torch.device("cpu"), max_new_tokens=2)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(1 <= len(o) <= 2 for o in out))

    def test_model_can_overfit_tiny_batch(self):
        """End-to-end sanity: the answer-only loss drives the model to memorize."""
        torch.manual_seed(0)
        split = load_split(self.dir, "dev", self.vocab)
        m = LanguageModel(tiny_cfg("softmax", vocab_size=len(self.vocab)))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x, y, _ = collate(split, [0, 1, 2], self.vocab)
        for _ in range(150):
            sel = y != IGNORE_INDEX
            loss = F.cross_entropy(m.head(m.hidden(x)[sel]), y[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        out = greedy_decode(m.eval(), split, [0, 1, 2], self.vocab, torch.device("cpu"))
        self.assertEqual([self.vocab.decode(o) for o in out], ["oasis", "merged", "velvet"])


if __name__ == "__main__":
    unittest.main()
