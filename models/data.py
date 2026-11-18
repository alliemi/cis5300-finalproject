"""Tokenization, caching and batching for the Recall / Update data.

Every example becomes one token sequence

    <bos> context <sep> query <sep> answer <eos>

The prompt is everything up to and including the second <sep>. Training
supervises only the tokens after the prompt (the answer and <eos>).

Multi-query training (`max_extra > 0` in `collate`) appends questions about
other projects in the same context, so one sequence supervises many answers:

    <bos> context <sep> query <sep> answer  (<sep> query_e <sep> value_e)*  <eos>

The official query always comes first, so evaluation (which decodes only
after the official prompt) is unchanged. The extra answers come from the
independent fact parser in synthdata/errors.py.
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

from synthdata.errors import parse_facts, parse_query, value_history
from synthdata.tokenizer import SPECIAL_TOKENS, render, tokenize
from synthdata.wordlists import QUERY_TEMPLATE

PAD, BOS, SEP, EOS = SPECIAL_TOKENS
IGNORE_INDEX = -100
TASK_IDS = {"recall": 0, "update": 1}
CACHE_VERSION = 2
EXTRA_QUERY_LEN = len(QUERY_TEMPLATE) + 3   # <sep> query <sep> value


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

    def query_ids(self, entity_id: int) -> List[int]:
        """Token ids of the query template for the given project-name token."""
        if not hasattr(self, "_query_parts"):
            self._query_parts = [None if w == "{E}" else self.stoi[w] for w in QUERY_TEMPLATE]
        return [entity_id if t is None else t for t in self._query_parts]

    def fingerprint(self) -> str:
        return hashlib.sha1("\n".join(self.itos).encode()).hexdigest()[:10]


def encode_record(rec: Dict, vocab: Vocab):
    """Return (full token ids, prompt length, extra (entity id, value id) pairs).

    The extra pairs give every *other* project in the context with its current
    (last stated) value, in order of first mention; they feed multi-query training.
    """
    prompt = ([vocab.bos_id] + vocab.encode(tokenize(rec["context"])) + [vocab.sep_id]
              + vocab.encode(tokenize(rec["query"])) + [vocab.sep_id])
    target = vocab.encode(tokenize(rec["answer"])) + [vocab.eos_id]
    hist = value_history(parse_facts(rec["context"]))
    entity = parse_query(rec["query"])
    if entity not in hist or hist[entity][-1] != rec["answer"]:
        raise ValueError(f"{rec.get('id')}: parsed current value of {entity} does not match answer")
    extra = [(vocab.stoi[e], vocab.stoi[vals[-1]]) for e, vals in hist.items() if e != entity]
    return prompt + target, len(prompt), extra


@dataclass
class EncodedSplit:
    """A split stored as one flat int16 array plus offsets (cheap in memory)."""
    tokens: np.ndarray       # all sequences concatenated
    offsets: np.ndarray      # int64, len n+1
    prompt_lens: np.ndarray  # int32
    buckets: np.ndarray      # int32, dataset `context_length` bucket
    tasks: np.ndarray        # int8, 0 = recall, 1 = update
    ids: List[str]
    extra: np.ndarray        # int16 (M, 2): (entity id, current value id) of other projects
    extra_offsets: np.ndarray  # int64, len n+1

    def __len__(self) -> int:
        return len(self.ids)

    def seq(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def seq_len(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def extra_pairs(self, i: int) -> np.ndarray:
        return self.extra[self.extra_offsets[i]:self.extra_offsets[i + 1]]

    def num_extra(self, i: int) -> int:
        return int(self.extra_offsets[i + 1] - self.extra_offsets[i])

    def train_len(self, i: int, max_extra: int = 0) -> int:
        """Sequence length including up to `max_extra` extra queries (-1 = all)."""
        k = self.num_extra(i) if max_extra < 0 else min(max_extra, self.num_extra(i))
        return self.seq_len(i) + k * EXTRA_QUERY_LEN

    def subset(self, indices: Sequence[int]) -> "EncodedSplit":
        indices = list(indices)
        tokens, offsets = _flatten([self.seq(i) for i in indices], np.int16)
        extra, extra_offsets = _flatten([self.extra_pairs(i) for i in indices], np.int16, width=2)
        return EncodedSplit(
            tokens=tokens,
            offsets=offsets,
            prompt_lens=self.prompt_lens[indices],
            buckets=self.buckets[indices],
            tasks=self.tasks[indices],
            ids=[self.ids[i] for i in indices],
            extra=extra,
            extra_offsets=extra_offsets,
        )


def _flatten(arrays, dtype, width: Optional[int] = None):
    """Concatenate arrays and return (flat array, offsets)."""
    lens = np.array([len(a) for a in arrays], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    shape = (0,) if width is None else (0, width)
    flat = np.concatenate([np.asarray(a, dtype=dtype).reshape((-1,) + shape[1:]) for a in arrays]) \
        if arrays else np.zeros(shape, dtype)
    return flat.astype(dtype, copy=False), offsets


def _encode_file(path: Path, vocab: Vocab) -> EncodedSplit:
    seqs, extras, prompt_lens, buckets, tasks, ids = [], [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seq, plen, extra = encode_record(rec, vocab)
            seqs.append(np.asarray(seq, dtype=np.int16))
            extras.append(np.asarray(extra, dtype=np.int16).reshape(-1, 2))
            prompt_lens.append(plen)
            buckets.append(rec["context_length"])
            tasks.append(TASK_IDS[rec["task"]])
            ids.append(rec["id"])
    tokens, offsets = _flatten(seqs, np.int16)
    extra, extra_offsets = _flatten(extras, np.int16, width=2)
    return EncodedSplit(tokens, offsets, np.array(prompt_lens, np.int32), np.array(buckets, np.int32),
                        np.array(tasks, np.int8), ids, extra, extra_offsets)


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
                           z["tasks"], z["ids"].tolist(), z["extra"], z["extra_offsets"])
    else:
        enc = _encode_file(path, vocab)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez(tmp, tokens=enc.tokens, offsets=enc.offsets, prompt_lens=enc.prompt_lens,
                     buckets=enc.buckets, tasks=enc.tasks, ids=np.array(enc.ids),
                     extra=enc.extra, extra_offsets=enc.extra_offsets)
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


def build_training_sequence(split: EncodedSplit, i: int, vocab: Vocab, max_extra: int = 0,
                            rng: Optional[random.Random] = None):
    """Return (tokens, supervised, official): per-token lists, where `supervised[j]`
    means token j is a training target (predicted from position j-1) and
    `official[j]` marks the official answer token(s).

    max_extra: 0 = official query only; K > 0 = up to K extra queries about other
    projects, sampled without replacement in random order; -1 = all of them.
    """
    s = split.seq(i).astype(np.int64).tolist()
    p = int(split.prompt_lens[i])
    answer = s[p:-1]                                   # stored seq ends with <eos>
    tokens = s[:p] + answer
    supervised = [False] * p + [True] * len(answer)
    official = list(supervised)
    pairs = split.extra_pairs(i)
    if max_extra and len(pairs):
        k = len(pairs) if max_extra < 0 else min(max_extra, len(pairs))
        rng = rng or random
        for c in rng.sample(range(len(pairs)), k):
            entity, value = int(pairs[c][0]), int(pairs[c][1])
            q = [vocab.sep_id] + vocab.query_ids(entity) + [vocab.sep_id]
            tokens += q + [value]
            supervised += [False] * len(q) + [True]
            official += [False] * (len(q) + 1)
    tokens.append(vocab.eos_id)
    supervised.append(True)
    official.append(False)
    return tokens, supervised, official


def collate(split: EncodedSplit, indices: Sequence[int], vocab: Vocab, max_extra: int = 0,
            rng: Optional[random.Random] = None):
    """Teacher-forcing batch. Returns (x, y, official):
    x = tokens[:-1]; y = tokens[1:] on supervised tokens, IGNORE_INDEX elsewhere;
    official = bool mask of y positions holding the official answer."""
    built = [build_training_sequence(split, i, vocab, max_extra, rng) for i in indices]
    T = max(len(t) for t, _, _ in built) - 1
    x = torch.full((len(indices), T), vocab.pad_id, dtype=torch.long)
    y = torch.full((len(indices), T), IGNORE_INDEX, dtype=torch.long)
    official = torch.zeros((len(indices), T), dtype=torch.bool)
    for row, (tokens, sup, off) in enumerate(built):
        t = torch.tensor(tokens, dtype=torch.long)
        n = len(tokens) - 1
        x[row, :n] = t[:-1]
        mask = torch.tensor(sup[1:], dtype=torch.bool)
        y[row, :n] = torch.where(mask, t[1:], torch.full_like(t[1:], IGNORE_INDEX))
        official[row, :n] = torch.tensor(off[1:], dtype=torch.bool)
    return x, y, official


class BucketBatchSampler:
    """Infinite, step-based sampler. Each batch comes from one length bucket.

    Batch size is `max_tokens // bucket_seq_len`, so every step processes about
    the same number of tokens. Buckets are drawn in proportion to their number
    of batches, so in expectation every example is seen equally often.

    `curriculum_steps > 0` unlocks buckets from short to long: at step s the
    first 1 + floor((s / curriculum_steps) * (n_buckets - 1)) buckets are allowed.
    """

    def __init__(self, split: EncodedSplit, max_tokens: int, seed: int = 0,
                 curriculum_steps: int = 0, max_extra: int = 0):
        self.rng = random.Random(seed)
        self.curriculum_steps = curriculum_steps
        groups: Dict[int, List[int]] = {}
        for i, b in enumerate(split.buckets.tolist()):
            groups.setdefault(b, []).append(i)
        self.buckets = sorted(groups)
        self.groups = [groups[b] for b in self.buckets]
        self.batch_sizes = [max(1, max_tokens // max(split.train_len(i, max_extra) for i in g))
                            for g in self.groups]
        self.weights = [len(g) / bs for g, bs in zip(self.groups, self.batch_sizes)]
        self._perm = [[] for _ in self.groups]

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
