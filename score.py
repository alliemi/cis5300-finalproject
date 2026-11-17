#!/usr/bin/env python3
"""Score prData status predictions using exact match and retrieval diagnostics."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


OUTCOMES = ("correct", "stale", "distractor", "other")
GROUPS = {
    "task": ("task",),
    "context_length": ("context_length",),
    "num_entities": ("num_entities",),
    "num_updates": ("num_updates",),
    "needle_position_bin": ("needle_position_bin",),
    "task_context_length": ("task", "context_length"),
    "task_num_updates": ("task", "num_updates"),
    "task_context_length_num_entities": ("task", "context_length", "num_entities"),
}


def read_jsonl(path):
    """Yield records with unique IDs, reporting malformed input with line numbers."""
    seen = set()
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            where = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{where}: expected a JSON object")
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{where}: id must be a nonempty string")
            if record_id in seen:
                raise ValueError(f"{where}: duplicate id {record_id!r}")
            seen.add(record_id)
            yield record


def normalize(value):
    """Ignore surrounding whitespace only; retain case and punctuation."""
    return value.strip()


def validate_gold(row):
    """Validate the fields used for scoring; context text is never needed."""
    prefix = f"gold {row['id']!r}"
    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
        raise ValueError(f"{prefix}: answer must be a nonempty string")
    if row.get("task") not in ("recall", "update"):
        raise ValueError(f"{prefix}: task must be recall or update")
    for field in ("context_length", "num_entities", "num_updates", "needle_position_bin"):
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{prefix}: {field} must be a nonnegative integer")
    if row["context_length"] == 0 or row["num_entities"] == 0:
        raise ValueError(f"{prefix}: context_length and num_entities must be positive")
    if row["needle_position_bin"] > 4:
        raise ValueError(f"{prefix}: needle_position_bin must be between 0 and 4")
    values = {}
    for field in ("stale_values", "distractor_values"):
        items = row.get(field)
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"{prefix}: {field} must be a list of nonempty strings")
        values[field] = {normalize(item) for item in items}
    answer = normalize(row["answer"])
    if (answer in values["stale_values"] or answer in values["distractor_values"]
            or values["stale_values"] & values["distractor_values"]):
        raise ValueError(f"{prefix}: answer, stale values and distractors must be disjoint")
    if row["task"] == "recall" and (row["num_updates"] != 0 or row["stale_values"]):
        raise ValueError(f"{prefix}: recall must have no updates or stale values")
    if row["task"] == "update" and row["num_updates"] == 0:
        raise ValueError(f"{prefix}: update must have at least one update")


def summarize(counts):
    """Rates use all examples in this group; error fractions use only errors."""
    total = sum(counts.values())
    errors = total - counts["correct"]
    return {
        "n": total,
        "correct": counts["correct"],
        "accuracy": counts["correct"] / total,
        "error_counts": {key: counts[key] for key in OUTCOMES[1:]},
        "error_rates": {key: counts[key] / total for key in OUTCOMES[1:]},
        "error_fractions": {
            key: counts[key] / errors if errors else None for key in OUTCOMES[1:]
        },
    }


def evaluate(gold_path, predictions_path):
    """Align predictions by ID and stream gold records to avoid loading contexts."""
    predictions = {}
    for row in read_jsonl(predictions_path):
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"prediction {row['id']!r}: prediction must be a string")
        predictions[row["id"]] = normalize(row["prediction"])

    overall = Counter()
    grouped = {name: defaultdict(Counter) for name in GROUPS}
    for row in read_jsonl(gold_path):
        validate_gold(row)
        if row["id"] not in predictions:
            raise ValueError(f"missing prediction for gold id {row['id']!r}")
        prediction = predictions.pop(row["id"])
        if prediction == normalize(row["answer"]):
            outcome = "correct"
        elif prediction in {normalize(value) for value in row["stale_values"]}:
            outcome = "stale"
        elif prediction in {normalize(value) for value in row["distractor_values"]}:
            outcome = "distractor"
        else:
            outcome = "other"
        overall[outcome] += 1
        for name, fields in GROUPS.items():
            key = tuple(row[field] for field in fields)
            grouped[name][key][outcome] += 1

    if not overall:
        raise ValueError("gold file contains no examples")
    if predictions:
        raise ValueError(f"{len(predictions)} prediction id(s) not found in gold, "
                         f"including {next(iter(predictions))!r}")
    return {
        "metric": "exact_match_accuracy",
        "normalization": "strip surrounding whitespace; case-sensitive",
        "overall": summarize(overall),
        "breakdowns": {
            name: [
                {**dict(zip(GROUPS[name], key)), **summarize(counts)}
                for key, counts in sorted(buckets.items())
            ]
            for name, buckets in grouped.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="Gold split JSONL")
    parser.add_argument("--predictions", required=True, type=Path,
                        help='JSONL records with "id" and "prediction" strings')
    parser.add_argument("--output", type=Path, help="Save full JSON report to this file")
    args = parser.parse_args()
    try:
        report = evaluate(args.gold, args.predictions)
        if args.output:
            if args.output.resolve() in (args.gold.resolve(), args.predictions.resolve()):
                raise ValueError("output must not overwrite an input file")
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    overall = report["overall"]
    print(f"Accuracy: {overall['accuracy']:.6f} ({overall['correct']}/{overall['n']})")
    for row in report["breakdowns"]["task"]:
        print(f"{row['task']} accuracy: {row['accuracy']:.6f} ({row['correct']}/{row['n']})")
    for category, count in overall["error_counts"].items():
        print(f"{category} errors: {count} (rate={overall['error_rates'][category]:.6f})")
    if args.output:
        print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
