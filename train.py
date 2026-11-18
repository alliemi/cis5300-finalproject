#!/usr/bin/env python3
"""Train one model on the synthetic retrieval data.

    python train.py --config configs/linear.json --run runs/linear-s0
    python train.py --config configs/linear.json --run runs/timing --max-steps 200 --no-final-eval
    python train.py --config configs/linear.json --run runs/small --max-context-length 128 --set train.max_steps=2000

Rerunning the same command resumes from the run's last checkpoint. When
training ends, best.pt is used to predict the full dev split and score.py
writes dev_scores.json into the run directory.
"""

import argparse

from training.config import load_config
from training.trainer import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<model>.json")
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/linear-s0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--seed", type=int, help="shortcut for --set train.seed=N")
    ap.add_argument("--max-steps", type=int, help="shortcut for --set train.max_steps=N")
    ap.add_argument("--max-context-length", type=int,
                    help="train and evaluate only on contexts up to this many tokens (laptop runs)")
    ap.add_argument("--train-subset", type=int, help="use only the first N training examples")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config value, e.g. --set train.lr=3e-4")
    ap.add_argument("--no-final-eval", action="store_true", help="skip full-dev prediction and scoring")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.seed is not None:
        overrides.append(f"train.seed={args.seed}")
    if args.max_steps is not None:
        overrides.append(f"train.max_steps={args.max_steps}")
    config = load_config(args.config, overrides)
    summary = train(config, args.run, data_dir=args.data_dir, device=args.device,
                    max_context_length=args.max_context_length, train_subset=args.train_subset,
                    final_eval=not args.no_final_eval)
    print("done:", summary)


if __name__ == "__main__":
    main()
