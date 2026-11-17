"""Checks for scoring arithmetic, ID alignment, and invalid evaluation inputs."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from score import evaluate


ROOT = Path(__file__).resolve().parents[1]


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gold = [json.loads(line) for line in
                     (ROOT / "examples/gold.jsonl").read_text().splitlines()]
        self.predictions = [json.loads(line) for line in
                            (ROOT / "examples/predictions.jsonl").read_text().splitlines()]

    def run_score(self, gold=None, predictions=None):
        gold_path = Path(self.temp.name) / "gold.jsonl"
        predictions_path = Path(self.temp.name) / "predictions.jsonl"
        for path, rows in ((gold_path, self.gold if gold is None else gold),
                           (predictions_path, self.predictions if predictions is None else predictions)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return evaluate(gold_path, predictions_path)

    def test_known_scores_and_group_denominators(self):
        report = self.run_score()
        overall = report["overall"]
        self.assertEqual(overall["accuracy"], 0.25)
        self.assertEqual(overall["error_counts"], {"stale": 1, "distractor": 1, "other": 1})
        self.assertEqual(overall["error_rates"]["stale"], 0.25)
        self.assertAlmostEqual(overall["error_fractions"]["stale"], 1 / 3)
        recall, update = report["breakdowns"]["task"]
        self.assertEqual((recall["accuracy"], update["accuracy"]), (0.5, 0.0))
        self.assertEqual(update["error_rates"]["stale"], 0.5)
        for groups in report["breakdowns"].values():
            self.assertEqual(sum(row["n"] for row in groups), 4)
            self.assertEqual(sum(row["correct"] for row in groups), 1)

    def test_predictions_are_aligned_by_id(self):
        self.assertEqual(self.run_score(), self.run_score(predictions=self.predictions[::-1]))

    def test_overall_accuracy_weights_examples_not_task_groups(self):
        gold = [copy.deepcopy(self.gold[0]) for _ in range(3)] + [self.gold[1]]
        for index, row in enumerate(gold):
            row["id"] = f"unequal-{index}"
        predictions = [{"id": row["id"], "prediction": "shipped"} for row in gold]
        report = self.run_score(gold=gold, predictions=predictions)
        self.assertEqual(report["overall"]["accuracy"], 0.75)
        recall, update = report["breakdowns"]["task"]
        self.assertEqual((recall["n"], update["n"]), (3, 1))
        self.assertEqual((recall["accuracy"], update["accuracy"]), (1.0, 0.0))

    def test_perfect_and_all_wrong(self):
        perfect = [{"id": row["id"], "prediction": " \t" + row["answer"] + "\n"}
                   for row in self.gold]
        result = self.run_score(predictions=perfect)["overall"]
        self.assertEqual(result["accuracy"], 1.0)
        self.assertTrue(all(value is None for value in result["error_fractions"].values()))
        wrong = [{"id": row["id"], "prediction": ""} for row in self.gold]
        result = self.run_score(predictions=wrong)["overall"]
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["error_fractions"]["other"], 1.0)

    def test_case_punctuation_and_explanations_are_not_normalized(self):
        for prediction in ("Shipped", "shipped.", "The answer is shipped"):
            with self.subTest(prediction=prediction):
                result = self.run_score(gold=self.gold[:1], predictions=[
                    {"id": "example-1", "prediction": prediction}])
                self.assertEqual(result["overall"]["error_counts"]["other"], 1)

    def test_missing_extra_and_duplicate_ids(self):
        cases = [
            (self.predictions[:-1], "missing prediction"),
            (self.predictions + [{"id": "unknown", "prediction": "x"}], "not found in gold"),
            (self.predictions + self.predictions[:1], "duplicate id"),
        ]
        for predictions, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.run_score(predictions=predictions)
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.run_score(gold=self.gold + self.gold[:1])

    def test_invalid_gold_and_predictions(self):
        for field, value in (("answer", None), ("stale_values", "onion"),
                             ("num_entities", True), ("needle_position_bin", 5),
                             ("distractor_values", ["shipped"])):
            gold = copy.deepcopy(self.gold)
            gold[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_score(gold=gold)
        for prediction in (None, 1, ["shipped"]):
            with self.subTest(prediction=prediction), self.assertRaisesRegex(ValueError, "must be a string"):
                self.run_score(predictions=[{"id": "example-1", "prediction": prediction}])
        with self.assertRaisesRegex(ValueError, "no examples"):
            self.run_score(gold=[], predictions=[])
        with self.assertRaisesRegex(ValueError, "prediction must be a string"):
            self.run_score(predictions=[{"id": "example-1", "answer": "shipped"}])

    def test_invalid_json_has_line_number(self):
        bad_path = Path(self.temp.name) / "bad.jsonl"
        bad_path.write_text('{"id": "ok", "prediction": "x"}\nnot json\n')
        with self.assertRaisesRegex(ValueError, r"bad.jsonl:2: invalid JSON"):
            evaluate(ROOT / "examples/gold.jsonl", bad_path)

    def test_stale_and_distractor_values_are_normalized(self):
        predictions = copy.deepcopy(self.predictions)
        for row in predictions:
            row["prediction"] = " \t" + row["prediction"] + "\n"
        gold = copy.deepcopy(self.gold)
        for row in gold:
            for field in ("stale_values", "distractor_values"):
                row[field] = [" " + value + " " for value in row[field]]
        self.assertEqual(self.run_score(), self.run_score(gold=gold, predictions=predictions))

    def test_overlapping_error_categories_are_rejected(self):
        gold = copy.deepcopy(self.gold)
        gold[1]["distractor_values"].append("onion")
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            self.run_score(gold=gold)

    def test_cli_writes_report_and_handles_errors(self):
        output = Path(self.temp.name) / "scores.json"
        command = [sys.executable, str(ROOT / "score.py"),
                   "--gold", str(ROOT / "examples/gold.jsonl"),
                   "--predictions", str(ROOT / "examples/predictions.jsonl")]
        completed = subprocess.run(command + ["--output", str(output)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Accuracy: 0.250000 (1/4)", completed.stdout)
        self.assertEqual(json.loads(output.read_text()), self.run_score())

        invalid = Path(self.temp.name) / "invalid.jsonl"
        invalid.write_text('{"id":"example-1","prediction":"shipped"}\n')
        failed = subprocess.run(command[:4] + ["--predictions", str(invalid)],
                                capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("missing prediction", failed.stderr)
        self.assertEqual(failed.stdout, "")
        self.assertNotIn("Traceback", failed.stderr)

        # Use temporary inputs so a regression cannot damage the example fixture.
        gold_path = Path(self.temp.name) / "gold.jsonl"
        predictions_path = Path(self.temp.name) / "predictions.jsonl"
        for protected in (gold_path, predictions_path):
            before = protected.read_bytes()
            failed = subprocess.run(
                [sys.executable, str(ROOT / "score.py"), "--gold", str(gold_path),
                 "--predictions", str(predictions_path), "--output", str(protected)],
                capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("must not overwrite", failed.stderr)
            self.assertEqual(protected.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
