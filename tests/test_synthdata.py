"""Tests for the synthetic data generator.  Run:  python3 -m unittest discover tests -v"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from synthdata import wordlists as W                                  # noqa: E402
from synthdata.checks import check_record, iter_jsonl, load_vocab, validate_dir   # noqa: E402
from synthdata.errors import classify_prediction, describe_value, parse_facts      # noqa: E402
from synthdata.generator import (Rng, design_cells, generate_dataset,             # noqa: E402
                                 make_filler, position_range)
from synthdata.tokenizer import build_vocab, render, tokenize        # noqa: E402

SIZES = dict(n_train=600, n_dev=150, n_test=150)


def read(path):
    with open(path, "rb") as f:
        return f.read()


class TestWordlistsAndTokenizer(unittest.TestCase):
    def test_wordlists_disjoint_and_large(self):
        W.check_wordlists()   # raises on overlap between names/values/filler/templates
        self.assertGreaterEqual(len(W.VALUES), 256)

    def test_value_pool_covers_largest_example(self):
        # Every value in an example is distinct, so the pool must exceed the max fact count.
        from synthdata.generator import LENGTHS, max_statements
        self.assertLess(max_statements(max(LENGTHS)), len(W.VALUES))

    def test_render_tokenize_roundtrip(self):
        toks = ["Meanwhile", ",", "the", "team", "lunch", "ran", "late", ".", "Project", "Falcon",
                "status", ":", "shipped", ".", "What", "is", "it", "?"]
        self.assertEqual(tokenize(render(toks)), toks)
        self.assertEqual(render(toks[8:14]), "Project Falcon status: shipped.")

    def test_vocab_closed(self):
        vocab = set(build_vocab())
        rng = Rng("vocab")
        for _ in range(2000):
            for sent in make_filler(rng, 37):
                self.assertTrue(set(sent) <= vocab, sent)

    def test_filler_exact_length(self):
        rng = Rng("fill")
        for n in range(0, 300):
            got = sum(len(s) for s in make_filler(rng, n))
            self.assertEqual(got, n - 1 if n == 1 else n)

    def test_every_design_cell_has_valid_position_range(self):
        cells = list(design_cells())
        self.assertAlmostEqual(sum(c[-1] for c in cells), 1.0)
        for task, L, K, U, b, _ in cells:
            lo, hi = position_range(L, U, b)
            self.assertLess(lo, hi)


class TestErrorAnalysis(unittest.TestCase):
    CTX = ("Project Falcon status: blocked. The team lunch ran late. Project Heron is testing. "
           "The status of Project Falcon is shipped. Project Heron status: paused.")
    REC = {"answer": "shipped", "stale_values": ["blocked"], "distractor_values": ["testing", "paused"]}

    def test_parse_facts(self):
        facts = [(f.entity, f.value) for f in parse_facts(self.CTX)]
        self.assertEqual(facts, [("Falcon", "blocked"), ("Heron", "testing"),
                                 ("Falcon", "shipped"), ("Heron", "paused")])

    def test_classify(self):
        self.assertEqual(classify_prediction("shipped", self.REC), "correct")
        self.assertEqual(classify_prediction(" shipped\n", self.REC), "correct")
        self.assertEqual(classify_prediction("shipped.", self.REC), "other")    # as in score.py
        self.assertEqual(classify_prediction("Shipped", self.REC), "other")
        self.assertEqual(classify_prediction("blocked", self.REC), "stale")
        self.assertEqual(classify_prediction("testing", self.REC), "distractor")
        self.assertEqual(classify_prediction("paused", self.REC), "distractor")
        self.assertEqual(classify_prediction("velvet", self.REC), "other")

    def test_classify_agrees_with_score_py(self):
        from collections import Counter
        from score import evaluate
        gold = os.path.join(ROOT, "examples", "gold.jsonl")
        preds_path = os.path.join(ROOT, "examples", "predictions.jsonl")
        preds = {r["id"]: r["prediction"] for r in iter_jsonl(preds_path)}
        mine = Counter(classify_prediction(preds[r["id"]], r) for r in iter_jsonl(gold))
        theirs = evaluate(gold, preds_path)["overall"]
        self.assertEqual(mine["correct"], theirs["correct"])
        self.assertEqual({k: mine[k] for k in ("stale", "distractor", "other")}, theirs["error_counts"])

    def test_describe_value(self):
        self.assertEqual(describe_value("testing", self.CTX),
                         {"entity": "Heron", "statement_index": 0, "is_current": False, "num_statements": 2})
        self.assertIsNone(describe_value("velvet", self.CTX))


class TestGeneratedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.a = os.path.join(cls.tmp, "a")
        generate_dataset(cls.a, **SIZES, seed=5300)
        cls.report = validate_dir(cls.a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_all_checks_pass(self):
        for split, sr in self.report["splits"].items():
            self.assertEqual(sr["records_with_errors"], 0, (split, sr["errors"]))
            self.assertEqual(sr["duplicates_within_split"] + sr["duplicates_across_splits"], 0)
        self.assertTrue(self.report["ok"])

    def test_split_counts(self):
        for split, n in (("train", 600), ("dev", 150), ("test", 150)):
            self.assertEqual(self.report["splits"][split]["count"], n)

    def test_task_rules(self):
        for rec in iter_jsonl(os.path.join(self.a, "train.jsonl")):
            self.assertNotIn(rec["answer"], rec["stale_values"] + rec["distractor_values"])
            if rec["task"] == "recall":
                self.assertEqual(rec["stale_values"], [])
                self.assertEqual(rec["num_updates"], 0)
                self.assertEqual([f.entity for f in parse_facts(rec["context"])].count(rec["target_entity"]), 1)
            else:
                self.assertEqual(len(rec["stale_values"]), rec["num_updates"])

    def test_validator_catches_corruption(self):
        vocab = load_vocab(self.a)
        rec = next(iter_jsonl(os.path.join(self.a, "dev.jsonl")))
        self.assertEqual(check_record(rec, "dev", 1, vocab), [])
        bad_values = [rec["distractor_values"][0], rec["stale_values"][0] if rec["stale_values"] else "velvet"]
        for field, val in (("answer", bad_values[0]), ("answer", bad_values[1]),
                           ("context_tokens", rec["context_tokens"] + 1), ("num_entities", 99),
                           ("needle_position", 0.0 if rec["needle_position"] > 0.5 else 0.99),
                           ("id", "dev-000002"), ("task", "update" if rec["task"] == "recall" else "recall")):
            broken = dict(rec, **{field: val})
            self.assertNotEqual(check_record(broken, "dev", 1, vocab), [], field)
        # A value word hidden in filler text must be caught.
        broken = dict(rec, context=rec["context"].replace("Meanwhile,", "Meanwhile velvet,", 1))
        if broken["context"] != rec["context"]:
            self.assertNotEqual(check_record(broken, "dev", 1, vocab), [])

    def test_same_seed_is_byte_identical(self):
        b = os.path.join(self.tmp, "b")
        generate_dataset(b, **SIZES, seed=5300)
        for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "vocab.txt", "generation_info.json"):
            self.assertEqual(read(os.path.join(self.a, f)), read(os.path.join(b, f)), f)

    def test_different_seed_differs(self):
        c = os.path.join(self.tmp, "c")
        generate_dataset(c, **SIZES, seed=1)
        self.assertNotEqual(read(os.path.join(self.a, "test.jsonl")), read(os.path.join(c, "test.jsonl")))

    def test_train_size_does_not_change_dev_or_test(self):
        d = os.path.join(self.tmp, "d")
        generate_dataset(d, n_train=50, n_dev=SIZES["n_dev"], n_test=SIZES["n_test"], seed=5300)
        for f in ("dev.jsonl", "test.jsonl"):
            self.assertEqual(read(os.path.join(self.a, f)), read(os.path.join(d, f)), f)

    def test_records_are_single_line_json(self):
        with open(os.path.join(self.a, "test.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.assertIsInstance(json.loads(line), dict)


class TestCommandLine(unittest.TestCase):
    def test_pipeline_and_archive_reproducible(self):
        tmp = tempfile.mkdtemp()
        try:
            hashes = []
            for run in ("r1", "r2"):
                out = os.path.join(tmp, run)
                os.makedirs(out)
                data = os.path.join(out, "data")
                for cmd in (["generate_data.py", "--out", data, "--n-train", "200", "--n-dev", "40", "--n-test", "40"],
                            ["validate_data.py", data],
                            ["make_archive.py", data, "--out-dir", out, "--force-sample", "--sample-train", "50"]):
                    subprocess.run([sys.executable, os.path.join(ROOT, cmd[0])] + cmd[1:], check=True,
                                   cwd=ROOT, stdout=subprocess.DEVNULL)
                hashes.append((read(os.path.join(out, "data.tar.gz")), read(os.path.join(out, "data_sample.tar.gz"))))
            self.assertEqual(hashes[0], hashes[1])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
