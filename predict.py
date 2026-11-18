#!/usr/bin/env python3
"""Write predictions for a split with a trained model, then score them.

    python predict.py --run runs/linear-s0 --split dev
    python predict.py --run runs/linear-s0 --split test --final     # final evaluation only

Output: <run>/<split>_predictions.jsonl ({"id", "prediction"} per line, the
format score.py reads) and <run>/<split>_scores.json.
"""

import argparse
import json
import os
import sys

import torch

from models import build_model, check_device
from training.data import PAD, encode_split, load_vocab
from training.evaluate import predict_ids, write_predictions
from training.trainer import amp_dtype_for, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--checkpoint", help="checkpoint file (default: <run>/best.pt)")
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        sys.exit("refusing to predict on test without --final: the test set is for the final evaluation only")

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint or os.path.join(args.run, "best.pt"), map_location=device, weights_only=False)
    check_device(ckpt["config"], device)
    vocab = load_vocab(args.data_dir)
    model = build_model(ckpt["config"], len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    split = encode_split(args.data_dir, args.split)
    preds = predict_ids(model, split, range(len(split)), args.batch_size, device,
                        vocab.index(PAD), amp_dtype_for(device))
    out = os.path.join(args.run, f"{args.split}_predictions.jsonl")
    write_predictions(out, split, preds, vocab)
    print(f"wrote {len(preds)} predictions to {out}")

    if not args.no_score:
        from score import evaluate
        report = evaluate(os.path.join(args.data_dir, f"{args.split}.jsonl"), out)
        with open(os.path.join(args.run, f"{args.split}_scores.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        o = report["overall"]
        print(f"accuracy {o['accuracy']:.4f} ({o['correct']}/{o['n']}); errors {o['error_counts']}")


if __name__ == "__main__":
    main()
