#!/usr/bin/env python3
"""Write SHA256SUMS and package the data as reproducible .tar.gz archives.

    python3 make_archive.py data                 # data.tar.gz (+ sample if large)
    python3 make_archive.py data --sample-train 2000 --force-sample

The full archive is always written. If it is larger than --sample-threshold-mb
(or --force-sample is given), data_sample.tar.gz is also written: the first
--sample-train training examples plus the complete dev and test sets.
Archives are byte-reproducible (sorted entries, fixed timestamps/owners).
"""

import argparse
import gzip
import hashlib
import io
import os
import tarfile

ARCHIVE_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl", "vocab.txt",
                 "generation_info.json", "validation_report.json", "SHA256SUMS")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tar_gz(out_path: str, entries):
    """entries: list of (archive_name, bytes) written in sorted order."""
    with open(out_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name, data in sorted(entries):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))


def sums_text(named_bytes):
    return "".join(f"{sha256_bytes(b)}  {n}\n" for n, b in sorted(named_bytes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--sample-train", type=int, default=2000)
    ap.add_argument("--sample-threshold-mb", type=float, default=50.0)
    ap.add_argument("--force-sample", action="store_true")
    args = ap.parse_args()

    d = args.data_dir
    missing = [f for f in ARCHIVE_FILES if f != "SHA256SUMS" and not os.path.exists(os.path.join(d, f))]
    if missing:
        raise SystemExit(f"missing {missing}; run generate_data.py and validate_data.py first")

    # Full archive. SHA256SUMS covers every other file.
    files = {f: open(os.path.join(d, f), "rb").read() for f in ARCHIVE_FILES if f != "SHA256SUMS"}
    sums = sums_text(files.items())
    with open(os.path.join(d, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.write(sums)
    files["SHA256SUMS"] = sums.encode()
    full = os.path.join(args.out_dir, "data.tar.gz")
    write_tar_gz(full, [(f"data/{n}", b) for n, b in files.items()])
    full_mb = os.path.getsize(full) / 1e6
    print(f"{full}: {full_mb:.1f} MB  sha256 {sha256_file(full)[:16]}...")

    if full_mb > args.sample_threshold_mb or args.force_sample:
        lines = files["train.jsonl"].splitlines(keepends=True)
        sample = {
            "train_sample.jsonl": b"".join(lines[:args.sample_train]),
            "dev.jsonl": files["dev.jsonl"], "test.jsonl": files["test.jsonl"],
            "vocab.txt": files["vocab.txt"], "generation_info.json": files["generation_info.json"],
            "validation_report.json": files["validation_report.json"],
        }
        sample["SHA256SUMS"] = sums_text(sample.items()).encode()
        out = os.path.join(args.out_dir, "data_sample.tar.gz")
        write_tar_gz(out, [(f"data_sample/{n}", b) for n, b in sample.items()])
        print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB  "
              f"(first {min(args.sample_train, len(lines))} train examples + full dev + full test)")


if __name__ == "__main__":
    main()
