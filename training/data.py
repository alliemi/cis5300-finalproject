"""Turn dataset records into token ids and length-bucketed batches.

Each example is encoded as

    <bos> context <sep> query <sep> answer <eos>

The model reads everything up to and including the second <sep> and must
predict the answer token at that position. The answer and <eos> are not
fed to the model (it is causal, so they could not affect the prediction
anyway), so the longest model input is 2,060 tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from synthdata.tokenizer import tokenize

PAD, BOS, SEP, EOS = "<pad>", "<bos>", "<sep>", "<eos>"
TASK_IDS = {"recall": 0, "update": 1}
CACHE_VERSION = 1


def load_vocab(data_dir: str) -> List[str]:
    with open(os.path.join(data_dir, "vocab.txt"), encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f if line.strip()]
    for special in (PAD, BOS, SEP, EOS):
        if special not in vocab:
            raise ValueError(f"vocab.txt is missing {special}")
    return vocab


def encode_record(rec: Dict, tok2id: Dict[str, int]) -> Tuple[List[int], int]:
    """Full token ids for one record, and the index of the answer token."""
    ids = [tok2id[BOS]] + [tok2id[t] for t in tokenize(rec["context"])]
    ids += [tok2id[SEP]] + [tok2id[t] for t in tokenize(rec["query"])] + [tok2id[SEP]]
    answer_index = len(ids)
    ids += [tok2id[rec["answer"]], tok2id[EOS]]
    return ids, answer_index


class EncodedSplit:
    """All examples of one split as a flat int16 tensor plus offsets.

    Only the model input (ids up to and including the second <sep>) and
    the answer id are kept, with the metadata needed for batching and
    accuracy breakdowns.
    """

    def __init__(self, tensors: Dict[str, torch.Tensor], ids: List[str]):
        self.tokens = tensors["tokens"]                # int16 [total_tokens]
        self.offsets = tensors["offsets"]              # int64 [N + 1]
        self.answers = tensors["answers"]              # int64 [N]
        self.context_length = tensors["context_length"]  # int64 [N]
        self.task = tensors["task"]                    # int64 [N], 0 = recall, 1 = update
        self.num_entities = tensors["num_entities"]    # int64 [N]
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def input_ids(self, i: int) -> torch.Tensor:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]].long()

    def input_length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def num_tokens(self, indices: Optional[Sequence[int]] = None) -> int:
        lengths = self.offsets[1:] - self.offsets[:-1]
        return int(lengths.sum() if indices is None else lengths[list(indices)].sum())

    def select(self, max_context_length: Optional[int] = None,
               limit: Optional[int] = None) -> List[int]:
        """Indices of examples with context_length <= max_context_length,
        keeping the first `limit` of them (the files are already shuffled)."""
        keep = [i for i in range(len(self))
                if max_context_length is None or int(self.context_length[i]) <= max_context_length]
        return keep if limit is None else keep[:limit]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_split(data_dir: str, split: str, use_cache: bool = True) -> EncodedSplit:
    """Encode data/<split>.jsonl, caching the result in data/cache/<split>.pt.

    The cache is rebuilt automatically if the jsonl or vocab file changes.
    """
    path = os.path.join(data_dir, f"{split}.jsonl")
    vocab = load_vocab(data_dir)
    key = f"{CACHE_VERSION}:{_file_sha256(path)}:{hashlib.sha256(chr(10).join(vocab).encode()).hexdigest()}"
    cache_path = os.path.join(data_dir, "cache", f"{split}.pt")
    if use_cache and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("key") == key:
            return EncodedSplit(cached["tensors"], cached["ids"])

    tok2id = {t: i for i, t in enumerate(vocab)}
    if len(vocab) > torch.iinfo(torch.int16).max:
        raise ValueError("vocabulary too large for int16 storage")
    chunks: List[torch.Tensor] = []
    offsets = [0]
    answers, context_length, task, num_entities, ids = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            full, answer_index = encode_record(rec, tok2id)
            chunks.append(torch.tensor(full[:answer_index], dtype=torch.int16))
            offsets.append(offsets[-1] + answer_index)
            answers.append(full[answer_index])
            context_length.append(rec["context_length"])
            task.append(TASK_IDS[rec["task"]])
            num_entities.append(rec["num_entities"])
            ids.append(rec["id"])
    tensors = {
        "tokens": torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int16),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "answers": torch.tensor(answers, dtype=torch.int64),
        "context_length": torch.tensor(context_length, dtype=torch.int64),
        "task": torch.tensor(task, dtype=torch.int64),
        "num_entities": torch.tensor(num_entities, dtype=torch.int64),
    }
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        torch.save({"key": key, "tensors": tensors, "ids": ids}, tmp)
        os.replace(tmp, cache_path)
    return EncodedSplit(tensors, ids)


def collate(split: EncodedSplit, indices: Sequence[int], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad a batch. `last` is each example's final real position, where
    the answer is predicted (padding comes after it, so a causal model
    never sees it)."""
    lengths = [split.input_length(i) for i in indices]
    batch = torch.full((len(indices), max(lengths)), pad_id, dtype=torch.long)
    for row, i in enumerate(indices):
        batch[row, :lengths[row]] = split.input_ids(i)
    return {
        "input_ids": batch,
        "last": torch.tensor(lengths, dtype=torch.long) - 1,
        "answers": split.answers[list(indices)],
    }


class BucketBatcher:
    """Batches drawn from one context-length bucket at a time.

    The batch order for an epoch depends only on (seed, epoch), so training
    can resume mid-epoch by rebuilding the epoch and skipping finished batches.
    """

    def __init__(self, split: EncodedSplit, indices: Sequence[int], batch_size: int, seed: int):
        self.batch_size = batch_size
        self.seed = seed
        self.buckets: Dict[int, List[int]] = {}
        for i in indices:
            self.buckets.setdefault(int(split.context_length[i]), []).append(i)

    def epoch_batches(self, epoch: int) -> List[List[int]]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        batches: List[List[int]] = []
        for length in sorted(self.buckets):
            members = self.buckets[length]
            order = torch.randperm(len(members), generator=g).tolist()
            shuffled = [members[j] for j in order]
            batches += [shuffled[k:k + self.batch_size] for k in range(0, len(shuffled), self.batch_size)]
        order = torch.randperm(len(batches), generator=g).tolist()
        return [batches[j] for j in order]

    def sequential_batches(self) -> Iterator[List[int]]:
        """Deterministic batches for evaluation (no shuffling)."""
        for length in sorted(self.buckets):
            members = self.buckets[length]
            for k in range(0, len(members), self.batch_size):
                yield members[k:k + self.batch_size]
