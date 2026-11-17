#!/usr/bin/env python3
"""Generate the synthetic Recall/Update dataset.

    python3 generate_data.py --n-train 80000 --n-dev 10000 --n-test 10000 --out data

Same seed + same sizes -> byte-identical files. dev/test depend only on the
seed and their own sizes, so changing --n-train never changes them.
"""

import argparse
import os
import time

from synthdata.generator import DEFAULT_SEED, generate_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-train", type=int, default=80000)
    ap.add_argument("--n-dev", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()

    start = time.time()
    info = generate_dataset(args.out, args.n_train, args.n_dev, args.n_test, seed=args.seed)
    elapsed = time.time() - start
    for split in ("train", "dev", "test"):
        path = os.path.join(args.out, f"{split}.jsonl")
        print(f"{split:5s} {info['splits'][split]['count']:7d} examples  "
              f"{os.path.getsize(path) / 1e6:8.1f} MB  "
              f"(duplicates rejected: {info['splits'][split]['duplicates_rejected']})")
    print(f"vocab {info['vocab_size']} tokens; generated in {elapsed:.1f}s with seed {args.seed}")


if __name__ == "__main__":
    main()
