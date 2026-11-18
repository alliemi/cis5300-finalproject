"""Tokenization, caching and batching for the Recall / Update data.

Every example becomes one token sequence

    <bos> context <sep> query <sep> answer <eos>

The prompt is everything up to and including the second <sep>. Training
supervises only the tokens after the prompt (the answer and <eos>).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from synthdata.tokenizer import SPECIAL_TOKENS, render, tokenize

PAD, BOS, SEP, EOS = SPECIAL_TOKENS
IGNORE_INDEX = -100
TASK_IDS = {"recall": 0, "update": 1}
CACHE_VERSION = 1


class Vocab:
    def __init__(self, tokens: Sequence[str]):
        self.itos = list(tokens)
        if self.itos[:4] != list(SPECIAL_TOKENS):
            raise ValueError(f"vocab must start with {SPECIAL_TOKENS}, got {self.itos[:4]}")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("vocab has duplicate tokens")
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.sep_id, self.eos_id = (self.stoi[t] for t in SPECIAL_TOKENS)

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls([t for t in Path(path).read_text(encoding="utf-8").splitlines() if t])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        try:
            return [self.stoi[t] for t in tokens]
        except KeyError as exc:
            raise ValueError(f"token {exc.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: Sequence[int]) -> str:
        """Text of the generated tokens: stop at <eos>, drop special tokens."""
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i < len(SPECIAL_TOKENS):
                continue
            out.append(self.itos[i])
        return render(out)

    def fingerprint(self) -> str:
        return hashlib.sha1("\n".join(self.itos).encode()).hexdigest()[:10]


def encode_record(rec: Dict, vocab: Vocab):
    """Return (full token ids, prompt length)."""
    prompt = ([vocab.bos_id] + vocab.encode(tokenize(rec["context"])) + [vocab.sep_id]
              + vocab.encode(tokenize(rec["query"])) + [vocab.sep_id])
    target = vocab.encode(tokenize(rec["answer"])) + [vocab.eos_id]
    return prompt + target, len(prompt)


@dataclass
class EncodedSplit:
    """A split stored as one flat int16 array plus offsets (cheap in memory)."""
    tokens: np.ndarray       # all sequences concatenated
    offsets: np.ndarray      # int64, len n+1
    prompt_lens: np.ndarray  # int32
    buckets: np.ndarray      # int32, dataset `context_length` bucket
    tasks: np.ndarray        # int8, 0 = recall, 1 = update
    ids: List[str]

    def __len__(self) -> int:
        return len(self.ids)

    def seq(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def seq_len(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def subset(self, indices: Sequence[int]) -> "EncodedSplit":
        indices = list(indices)
        seqs = [self.seq(i) for i in indices]
        lens = np.array([len(s) for s in seqs], dtype=np.int64)
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        return EncodedSplit(
            tokens=np.concatenate(seqs) if seqs else np.zeros(0, np.int16),
            offsets=offsets,
            prompt_lens=self.prompt_lens[indices],
            buckets=self.buckets[indices],
            tasks=self.tasks[indices],
            ids=[self.ids[i] for i in indices],
        )


def _encode_file(path: Path, vocab: Vocab) -> EncodedSplit:
    seqs, prompt_lens, buckets, tasks, ids = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seq, plen = encode_record(rec, vocab)
            seqs.append(np.asarray(seq, dtype=np.int16))
            prompt_lens.append(plen)
            buckets.append(rec["context_length"])
            tasks.append(TASK_IDS[rec["task"]])
            ids.append(rec["id"])
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return EncodedSplit(np.concatenate(seqs), offsets, np.array(prompt_lens, np.int32),
                        np.array(buckets, np.int32), np.array(tasks, np.int8), ids)


def load_split(data_dir, split: str, vocab: Vocab, *, max_examples: Optional[int] = None,
               max_context_length: Optional[int] = None, seed: int = 0,
               use_cache: bool = True) -> EncodedSplit:
    """Load (and cache) a tokenized split, optionally filtered and subsampled.

    `max_context_length` keeps only buckets <= that value. `max_examples` then
    takes a seeded random subset (the same subset for the same seed).
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.jsonl"
    cache = data_dir / "cache" / f"{split}-v{CACHE_VERSION}-{vocab.fingerprint()}.npz"
    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        z = np.load(cache, allow_pickle=False)
        enc = EncodedSplit(z["tokens"], z["offsets"], z["prompt_lens"], z["buckets"],
                           z["tasks"], z["ids"].tolist())
    else:
        enc = _encode_file(path, vocab)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez(tmp, tokens=enc.tokens, offsets=enc.offsets, prompt_lens=enc.prompt_lens,
                     buckets=enc.buckets, tasks=enc.tasks, ids=np.array(enc.ids))
            tmp.replace(cache)

    keep = list(range(len(enc)))
    if max_context_length is not None:
        keep = [i for i in keep if enc.buckets[i] <= max_context_length]
    if max_examples is not None and max_examples < len(keep):
        keep = sorted(random.Random(seed).sample(keep, max_examples))
    if len(keep) != len(enc):
        enc = enc.subset(keep)
    if len(enc) == 0:
        raise ValueError(f"no examples left in {split} after filtering")
    return enc


def collate(split: EncodedSplit, indices: Sequence[int], pad_id: int):
    """Teacher-forcing batch: inputs = seq[:-1], labels = seq[1:] on target tokens only."""
    T = max(split.seq_len(i) for i in indices) - 1
    x = torch.full((len(indices), T), pad_id, dtype=torch.long)
    y = torch.full((len(indices), T), IGNORE_INDEX, dtype=torch.long)
    for row, i in enumerate(indices):
        s = torch.from_numpy(split.seq(i).astype(np.int64))
        n = len(s) - 1
        x[row, :n] = s[:-1]
        p = int(split.prompt_lens[i])
        y[row, p - 1:n] = s[p:]          # position p-1 (the last <sep>) predicts the answer
    return x, y


class BucketBatchSampler:
    """Infinite, step-based sampler. Each batch comes from one length bucket.

    Batch size is `max_tokens // bucket_seq_len`, so every step processes about
    the same number of tokens. Buckets are drawn in proportion to their number
    of batches, so in expectation every example is seen equally often.

    `curriculum_steps > 0` unlocks buckets from short to long: at step s the
    first 1 + floor((s / curriculum_steps) * (n_buckets - 1)) buckets are allowed.
    """

    def __init__(self, split: EncodedSplit, max_tokens: int, seed: int = 0,
                 curriculum_steps: int = 0):
        self.rng = random.Random(seed)
        self.curriculum_steps = curriculum_steps
        groups: Dict[int, List[int]] = {}
        for i, b in enumerate(split.buckets.tolist()):
            groups.setdefault(b, []).append(i)
        self.buckets = sorted(groups)
        self.groups = [groups[b] for b in self.buckets]
        self.batch_sizes = [max(1, max_tokens // max(split.seq_len(i) for i in g)) for g in self.groups]
        self.weights = [len(g) / bs for g, bs in zip(self.groups, self.batch_sizes)]
        self._perm = [[] for _ in self.groups]

    def state_dict(self) -> Dict:
        """Everything needed to continue the exact same batch sequence after a resume."""
        return {"rng": self.rng.getstate(), "perm": [list(p) for p in self._perm]}

    def load_state_dict(self, state: Dict) -> None:
        self.rng.setstate(state["rng"])
        self._perm = [list(p) for p in state["perm"]]

    def allowed(self, step: int) -> int:
        if self.curriculum_steps <= 0:
            return len(self.buckets)
        frac = min(1.0, step / self.curriculum_steps)
        return 1 + int(frac * (len(self.buckets) - 1))

    def batch(self, step: int) -> List[int]:
        n = self.allowed(step)
        b = self.rng.choices(range(n), weights=self.weights[:n])[0]
        out = []
        while len(out) < self.batch_sizes[b]:
            if not self._perm[b]:
                self._perm[b] = self.groups[b][:]
                self.rng.shuffle(self._perm[b])
            out.append(self._perm[b].pop())
        return out


def eval_batches(split: EncodedSplit, max_tokens: int) -> Iterator[List[int]]:
    """Deterministic batches covering every example once, grouped by length."""
    order = sorted(range(len(split)), key=lambda i: split.seq_len(i))
    batch: List[int] = []
    for i in order:
        if batch and (len(batch) + 1) * split.seq_len(i) > max_tokens:
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch
