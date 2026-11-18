#!/usr/bin/env python3
"""Compare finished runs: one row per run from <run>/config.json and <run>/dev_scores.json.

    python summarize_runs.py                 # all runs under runs/
    python summarize_runs.py --runs-dir runs --out runs/summary.csv

Standard library only.
"""

import argparse
import csv
import glob
import json
import os

LENGTHS = (128, 256, 512, 1024, 2048)
ENTITIES = (2, 4, 8, 16, 32, 64)


def summarize(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        info = json.load(f)
    with open(os.path.join(run_dir, "dev_scores.json"), encoding="utf-8") as f:
        report = json.load(f)
    by = report["breakdowns"]
    overall = report["overall"]
    row = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "mixer": info["config"]["mixer"],
        "seed": info["config"]["train"]["seed"],
        "params": info.get("params"),
        "max_steps": info["config"]["train"]["max_steps"],
        "train_max_context_length": info["data"]["max_context_length"],
        "accuracy": overall["accuracy"],
    }
    for g in by["task"]:
        row[f"{g['task']}_accuracy"] = g["accuracy"]
    for kind in ("stale", "distractor", "other"):
        row[f"{kind}_rate"] = overall["error_rates"][kind]
    acc = {("L", g["context_length"]): g["accuracy"] for g in by["context_length"]}
    acc.update({("K", g["num_entities"]): g["accuracy"] for g in by["num_entities"]})
    for L in LENGTHS:
        row[f"acc_len{L}"] = acc.get(("L", L))
    for K in ENTITIES:
        row[f"acc_projects{K}"] = acc.get(("K", K))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", help="CSV path (default: <runs-dir>/summary.csv)")
    args = ap.parse_args()

    run_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.runs_dir, "*", "dev_scores.json")))
    if not run_dirs:
        raise SystemExit(f"no finished runs (with dev_scores.json) under {args.runs_dir}/")
    rows = [summarize(d) for d in run_dirs]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = args.out or os.path.join(args.runs_dir, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))
    shown = ["run", "mixer", "seed", "accuracy", "recall_accuracy", "update_accuracy",
             "stale_rate", "distractor_rate", "other_rate"] + [f"acc_len{L}" for L in LENGTHS]
    widths = {k: max(len(k), *(len(fmt(r.get(k))) for r in rows)) for k in shown}
    print("  ".join(k.ljust(widths[k]) for k in shown))
    for r in rows:
        print("  ".join(fmt(r.get(k)).ljust(widths[k]) for k in shown))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
