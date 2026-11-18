#!/usr/bin/env python3
"""Write predictions in the format score.py reads: {"id": ..., "prediction": ...} per line.

Full split:
    python predict.py --checkpoint runs/gdn/best.pt --split dev --out runs/gdn/dev_pred.jsonl
    python score.py --gold data/dev.jsonl --predictions runs/gdn/dev_pred.jsonl --output runs/gdn/dev_scores.json

Subset (score.py requires a prediction for every gold id, so also write the matching gold):
    python predict.py --checkpoint runs/smoke/best.pt --split dev --max-examples 200 \
        --max-context-length 128 --out runs/smoke/dev_pred.jsonl --write-gold runs/smoke/dev_gold.jsonl

The test split is for the final evaluation only, so it also needs --final:
    python predict.py --checkpoint runs/gdn/best.pt --split test --final --out runs/gdn/test_pred.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from models.data import load_split
from models.inference import (load_checkpoint, pick_device, predict_split, write_gold_subset,
                              write_predictions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-context-length", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=12345,
                   help="seed for --max-examples; 12345 matches train.py's dev subset, "
                        "train.py's --seed matches its train subset")
    p.add_argument("--write-gold", default=None, help="write the gold records of the predicted subset here")
    p.add_argument("--max-tokens", type=int, default=65536, help="tokens per batch")
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--kernel", default=None, choices=["auto", "torch", "fla"],
                   help="override the GDN/KDA kernel stored in the checkpoint")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--final", action="store_true", help="required for --split test")
    args = p.parse_args(argv)
    if args.split == "test" and not args.final:
        sys.exit("refusing to predict on test without --final: the test set is for the final "
                 "evaluation only (use dev for model selection)")

    device = pick_device(args.device)
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device, kernel=args.kernel)
    print(f"loaded {args.checkpoint}: {model.cfg.attention}, step {ckpt.get('step')}, device {device}")
    split = load_split(args.data_dir, args.split, vocab, max_examples=args.max_examples,
                       max_context_length=args.max_context_length, seed=args.subset_seed)
    t = time.time()
    preds = predict_split(model, split, vocab, device, args.max_tokens, args.max_new_tokens,
                          amp=not args.no_amp, progress=True)

    out = Path(args.out)
    write_predictions(out, split, preds)
    print(f"wrote {len(preds)} predictions to {out} ({time.time() - t:.1f}s)")

    if args.write_gold:
        write_gold_subset(args.data_dir, args.split, split.ids, args.write_gold)
        print(f"wrote matching gold subset to {args.write_gold}")


if __name__ == "__main__":
    main()
