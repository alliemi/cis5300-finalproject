"""Checks for the attention layers, backbone, data pipeline and decoding.

Runs on CPU. Tests that compare against the fla Triton kernels run only when
CUDA and flash-linear-attention are available (run them once on the GPU box).
Skipped entirely when PyTorch is not installed.
"""

import json
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
    from models.attention import (ATTENTION, build_attention, delta_rule_recurrent, _fla_kernel,
                                  linear_attention_chunked, linear_attention_recurrent)
    from models.data import (IGNORE_INDEX, BucketBatchSampler, Vocab, collate, encode_record,
                             eval_batches, load_split)
    from models.inference import greedy_decode, load_checkpoint, save_checkpoint
    from models.model import LanguageModel, ModelConfig

    def tiny_cfg(attention, **kw):
        base = dict(vocab_size=50, attention=attention, d_model=32, n_layers=2, n_heads=2,
                    kernel="torch", linear_chunk=8)
        base.update(kw)
        return ModelConfig(**base)

HAVE_FLA_GPU = HAVE_TORCH and torch.cuda.is_available() and _fla_kernel(False) is not None


def recurrent_ref_loop(q, k, v, g, beta, scale):
    """Independent, unbatched float64 gated delta rule, one (example, head) at a time:
    S <- Diag(a) S;  S <- S - b k (k^T S) + b k v^T;  o = S^T (scale q)."""
    B, T, H, K = q.shape
    out = torch.zeros(B, T, H, v.shape[-1], dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(K, v.shape[-1], dtype=torch.float64)
            for t in range(T):
                a = g[b, t, h].double().exp()
                S = (a[:, None] if a.dim() else a) * S
                kt = k[b, t, h].double()
                S = S - beta[b, t, h].double() * torch.outer(kt, kt @ S) \
                      + beta[b, t, h].double() * torch.outer(kt, v[b, t, h].double())
                out[b, t, h] = S.T @ (q[b, t, h].double() * scale)
    return out


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

    def test_delta_rule_matches_unbatched_loop(self):
        B, T, H, K, V = 2, 7, 2, 4, 3
        q, k = (F.normalize(torch.randn(B, T, H, K, dtype=torch.float64), dim=-1) for _ in range(2))
        v = torch.randn(B, T, H, V, dtype=torch.float64)
        beta = torch.rand(B, T, H, dtype=torch.float64)
        for g in (-torch.rand(B, T, H, dtype=torch.float64),          # GDN: scalar per head
                  -torch.rand(B, T, H, K, dtype=torch.float64)):      # KDA: per channel
            with self.subTest(g_dim=g.dim()):
                got = delta_rule_recurrent(q, k, v, g, beta, scale=0.5).double()
                torch.testing.assert_close(got, recurrent_ref_loop(q, k, v, g, beta, 0.5))

    def test_delta_rule_overwrites_value_under_same_key(self):
        """The property that should help on Update examples: with beta = 1 and
        no decay, writing key k twice replaces the first value, while plain
        linear attention returns a mix of both."""
        K = 4
        k = F.normalize(torch.randn(K, dtype=torch.float64), dim=0)
        keys = k.expand(1, 2, 1, K).clone()
        vals = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64).view(1, 2, 1, 2)
        o = delta_rule_recurrent(keys, keys, vals, torch.zeros(1, 2, 1, dtype=torch.float64),
                                 torch.ones(1, 2, 1, dtype=torch.float64), scale=1.0)
        torch.testing.assert_close(o[0, -1, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))
        lin = linear_attention_recurrent(keys.transpose(1, 2) + 1, keys.transpose(1, 2) + 1,
                                         vals.transpose(1, 2))
        torch.testing.assert_close(lin[0, 0, -1], torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_decay_forgets(self):
        """A strongly negative log-decay should erase earlier memory."""
        K = 4
        k = F.normalize(torch.randn(1, 2, 1, K, dtype=torch.float64), dim=-1)
        v = torch.randn(1, 2, 1, 3, dtype=torch.float64)
        g = torch.tensor([0.0, -50.0], dtype=torch.float64).view(1, 2, 1)
        beta = torch.tensor([1.0, 0.0], dtype=torch.float64).view(1, 2, 1)
        o = delta_rule_recurrent(k, k, v, g, beta, 1.0)
        self.assertLess(o[0, 1].abs().max().item(), 1e-12)

    @unittest.skipUnless(HAVE_FLA_GPU, "needs CUDA + flash-linear-attention")
    def test_fla_kernels_match_reference(self):
        from models.attention import delta_rule_fla
        torch.manual_seed(0)
        B, T, H, K = 2, 300, 4, 64
        dev = "cuda"
        q, k = (F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1) for _ in range(2))
        v = torch.randn(B, T, H, K, device=dev)
        beta = torch.rand(B, T, H, device=dev)
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                shape = (B, T, H, K) if per_channel else (B, T, H)
                g = F.logsigmoid(torch.randn(*shape, device=dev) + 3)
                ref = delta_rule_recurrent(q, k, v, g, beta, K ** -0.5)
                got = delta_rule_fla(q, k, v, g, beta, K ** -0.5, per_channel).float()
                torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-3)

    @unittest.skipUnless(HAVE_FLA_GPU, "needs CUDA + flash-linear-attention")
    def test_fla_layer_matches_torch_layer(self):
        for name in ("gated_deltanet", "kda"):
            with self.subTest(name):
                torch.manual_seed(0)
                layer = build_attention(tiny_cfg(name, d_model=128, n_heads=2)).cuda()
                x = torch.randn(2, 200, 128, device="cuda")
                layer.kernel = "torch"
                ref = layer(x)
                layer.kernel = "fla"
                torch.testing.assert_close(layer(x), ref, rtol=2e-2, atol=2e-3)


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

    def test_checkpoint_roundtrip(self):
        vocab = Vocab(["<pad>", "<bos>", "<sep>", "<eos>"] + [f"w{i}" for i in range(46)])
        m = LanguageModel(tiny_cfg("kda"))
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
        seq, plen = encode_record(self.recs[0], v)
        self.assertEqual(seq[0], v.bos_id)
        self.assertEqual(seq[plen - 1], v.sep_id)
        self.assertEqual([v.itos[i] for i in seq[plen:]], ["oasis", "<eos>"])
        self.assertEqual(sum(1 for i in seq if i == v.sep_id), 2)
        self.assertEqual(v.decode(seq[plen:]), "oasis")

    def test_load_split_and_cache(self):
        a = load_split(self.dir, "dev", self.vocab)
        b = load_split(self.dir, "dev", self.vocab)      # from cache
        self.assertEqual(a.ids, b.ids)
        self.assertTrue((a.tokens == b.tokens).all())
        short = load_split(self.dir, "dev", self.vocab, max_context_length=128)
        self.assertEqual(short.ids, ["x-1", "x-3"])
        self.assertEqual(len(load_split(self.dir, "dev", self.vocab, max_examples=2, seed=1)), 2)

    def test_collate_supervises_only_answer_and_eos(self):
        split = load_split(self.dir, "dev", self.vocab)
        x, y = collate(split, [0, 1], self.vocab.pad_id)
        for row in range(2):
            p, n = int(split.prompt_lens[row]), split.seq_len(row)
            labels = y[row][y[row] != IGNORE_INDEX].tolist()
            self.assertEqual(labels, split.seq(row)[p:].tolist())
            self.assertEqual(x[row, p - 1].item(), self.vocab.sep_id)   # predicts the answer
            self.assertTrue((x[row, n - 1:] == self.vocab.pad_id).all())

    def test_samplers(self):
        split = load_split(self.dir, "dev", self.vocab)
        s = BucketBatchSampler(split, max_tokens=10_000, seed=0, curriculum_steps=10)
        self.assertEqual(s.allowed(0), 1)
        self.assertEqual(s.allowed(10), 2)
        for step in range(5):     # curriculum: only the 128 bucket at the start
            self.assertTrue(all(split.buckets[i] == 128 for i in s.batch(step)))
        covered = sorted(i for b in eval_batches(split, 40) for i in b)
        self.assertEqual(covered, [0, 1, 2])

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
        x, y = collate(split, [0, 1, 2], self.vocab.pad_id)
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
