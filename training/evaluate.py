"""Batched prediction and quick accuracy breakdowns (score.py is the official scorer)."""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch

from .data import BucketBatcher, EncodedSplit, collate

TASK_NAMES = ("recall", "update")


def autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def predict_ids(model: torch.nn.Module, split: EncodedSplit, indices: Sequence[int], batch_size: int,
                device: torch.device, pad_id: int, amp_dtype: Optional[torch.dtype] = None) -> Dict[int, int]:
    """Greedy answer-token prediction for each example index."""
    was_training = model.training
    model.eval()
    preds: Dict[int, int] = {}
    for batch_idx in BucketBatcher(split, indices, batch_size, seed=0).sequential_batches():
        batch = collate(split, batch_idx, pad_id)
        with autocast(device, amp_dtype):
            logits = model(batch["input_ids"].to(device), batch["last"].to(device))
        for i, p in zip(batch_idx, logits.argmax(-1).tolist()):
            preds[i] = p
    model.train(was_training)
    return preds


def accuracy_breakdown(split: EncodedSplit, preds: Dict[int, int]) -> Dict:
    """Accuracy overall, by task and by context length (for training logs)."""
    hits = {i: int(p == int(split.answers[i])) for i, p in preds.items()}
    groups: Dict[str, Dict] = {"task": defaultdict(list), "context_length": defaultdict(list)}
    for i, h in hits.items():
        groups["task"][TASK_NAMES[int(split.task[i])]].append(h)
        groups["context_length"][str(int(split.context_length[i]))].append(h)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "accuracy": mean(list(hits.values())),
        "n": len(hits),
        "by_task": {k: mean(v) for k, v in sorted(groups["task"].items())},
        "by_context_length": {k: mean(v) for k, v in sorted(groups["context_length"].items(), key=lambda kv: int(kv[0]))},
    }


def write_predictions(path: str, split: EncodedSplit, preds: Dict[int, int], vocab: List[str]) -> None:
    """{"id", "prediction"} lines in dataset order, the format score.py reads."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(preds):
            f.write(json.dumps({"id": split.ids[i], "prediction": vocab[preds[i]]}) + "\n")
