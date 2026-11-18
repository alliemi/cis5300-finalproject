#!/usr/bin/env python3
"""Write predictions in the format score.py reads: {"id": ..., "prediction": ...} per line.

Full split:
    python predict.py --checkpoint runs/gdn/best.pt --split dev --out runs/gdn/dev_pred.jsonl
    python score.py --gold data/dev.jsonl --predictions runs/gdn/dev_pred.jsonl --output runs/gdn/dev_scores.json

Subset (score.py requires a prediction for every gold id, so also write the matching gold):
    python predict.py --checkpoint runs/smoke/best.pt --split dev --max-examples 200 \
        --max-context-length 128 --out runs/smoke/dev_pred.jsonl --write-gold runs/smoke/dev_gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from models.data import load_split
from models.inference import load_checkpoint, pick_device, predict_split


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
    args = p.parse_args(argv)

    device = pick_device(args.device)
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device, kernel=args.kernel)
    print(f"loaded {args.checkpoint}: {model.cfg.attention}, step {ckpt.get('step')}, device {device}")
    split = load_split(args.data_dir, args.split, vocab, max_examples=args.max_examples,
                       max_context_length=args.max_context_length, seed=args.subset_seed)
    t = time.time()
    preds = predict_split(model, split, vocab, device, args.max_tokens, args.max_new_tokens,
                          amp=not args.no_amp, progress=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex_id in split.ids:
            f.write(json.dumps({"id": ex_id, "prediction": preds[ex_id]}) + "\n")
    print(f"wrote {len(preds)} predictions to {out} ({time.time() - t:.1f}s)")

    if args.write_gold:
        keep = set(split.ids)
        gold = Path(args.write_gold)
        gold.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(args.data_dir) / f"{args.split}.jsonl", encoding="utf-8") as src, \
                open(gold, "w", encoding="utf-8") as dst:
            for line in src:
                if json.loads(line)["id"] in keep:
                    dst.write(line)
        print(f"wrote matching gold subset to {gold}")


if __name__ == "__main__":
    main()
