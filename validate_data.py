#!/usr/bin/env python3
"""Validate a generated data directory and write validation_report.json into it.

    python3 validate_data.py data

Exits with status 1 if any check fails.
"""

import argparse
import json
import os
import sys

from synthdata.checks import validate_dir


def _print_split(split, sr):
    print(f"\n== {split}: {sr['count']} examples ==")
    print(f"  records with errors: {sr['records_with_errors']}   duplicates within split: "
          f"{sr['duplicates_within_split']}   across splits: {sr['duplicates_across_splits']}")
    for e in sr["errors"]:
        print("   !", e)
    print(f"  balance ok: {sr['balance_ok']}")
    for field, rows in sr["balance"].items():
        cells = "  ".join(f"{v}:{r['observed']:.3f}/{r['expected']:.3f}{'' if r['ok'] else '(!)'}"
                          for v, r in rows.items())
        print(f"    {field:20s} observed/expected  {cells}")
    toks = "  ".join(f"{L}:[{s['min']},{s['max']}]" for L, s in sr["context_tokens_by_length"].items())
    print(f"  context tokens [min,max] per length bucket: {toks}")
    if "baselines_accuracy" in sr:
        print("  shortcut baselines (accuracy):")
        for task, row in sr["baselines_accuracy"].items():
            cells = "  ".join(f"{k}={v}" for k, v in row.items())
            print(f"    {task:6s} {cells}")
        print("  query-free 'latest value of most-mentioned project' by num_entities:")
        for task, rows in sr["baselines_accuracy_by_num_entities"].items():
            cells = "  ".join(f"K={K}:{r['latest_value_of_most_mentioned_project']}" for K, r in rows.items())
            print(f"    {task:6s} {cells}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", nargs="?", default="data")
    args = ap.parse_args()

    report = validate_dir(args.data_dir)
    for split, sr in report["splits"].items():
        _print_split(split, sr)
    out = os.path.join(args.data_dir, "validation_report.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nALL CHECKS PASSED" if report["ok"] else "\nCHECKS FAILED")
    print(f"report written to {out}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
