#!/usr/bin/env python3
"""Simple baseline: a small language model with linear attention
(Katharopoulos et al., 2020), trained from scratch on the Recall / Update data.

This script trains the model with train.py using `--attention linear` and the
shared training recipe from the README. It then uses best.pt to write
predictions for one split in the format score.py reads:
{"id": ..., "prediction": ...}.

Train, then predict on test (GPU; the test split needs --final):
    python simple-baseline.py --out runs/linear-cur12k --data-dir data_big \
        --split test --final
    python3 score.py --gold data/test.jsonl --predictions runs/linear-cur12k/test_pred.jsonl

Predict only, from an existing checkpoint:
    python simple-baseline.py --out runs/linear-cur12k --skip-train --split test --final

Any other flag is passed on to train.py and overrides the recipe below, e.g.
`--steps 300 --max-context-length 128` for a quick CPU check. Rerunning the
same command resumes training from <out>/last.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ATTENTION = "linear"

# Shared recipe (README, "GPU runs"). The strong baseline uses the same flags
# with --attention softmax.
RECIPE = ["--multi-query", "16", "--curriculum-steps", "12000",
          "--steps", "20000", "--max-tokens", "65536"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run directory, e.g. runs/linear-cur12k")
    p.add_argument("--split", default="dev", choices=["dev", "test"], help="split to predict")
    p.add_argument("--final", action="store_true", help="required for --split test")
    p.add_argument("--eval-data-dir", default="data",
                   help="data directory holding the dev/test split to predict")
    p.add_argument("--predictions", default=None,
                   help="output JSONL (default: <out>/<split>_pred.jsonl)")
    p.add_argument("--skip-train", action="store_true", help="only predict from <out>/best.pt")
    args, train_args = p.parse_known_args(argv)
    if "--attention" in train_args:
        p.error("--attention is fixed to 'linear' for the simple baseline")
    if args.split == "test" and not args.final:
        p.error("--split test needs --final: the test set is for the final evaluation only")

    # Imported here so `--help` works without PyTorch installed.
    import predict
    import train

    out = Path(args.out)
    if not args.skip_train:
        train.main(["--attention", ATTENTION, "--out", str(out), *RECIPE, *train_args])

    checkpoint = out / "best.pt"
    if not checkpoint.exists():
        sys.exit(f"no checkpoint at {checkpoint}; train first or check --out")
    predictions = args.predictions or str(out / f"{args.split}_pred.jsonl")
    predict.main(["--checkpoint", str(checkpoint), "--split", args.split,
                  "--data-dir", args.eval_data_dir, "--out", predictions]
                 + (["--final"] if args.final else []))
    print(f"score with: python3 score.py --gold {args.eval_data_dir}/{args.split}.jsonl "
          f"--predictions {predictions}")


if __name__ == "__main__":
    main()
