"""Greedy decoding, checkpoints and device helpers shared by train.py and predict.py."""

from __future__ import annotations

import contextlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import EncodedSplit, Vocab, eval_batches
from .model import LanguageModel, ModelConfig


def pick_device(name: str = "auto") -> torch.device:
    """auto: CUDA, then Apple-silicon GPU (MPS), then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def greedy_decode(model: LanguageModel, split: EncodedSplit, indices: List[int], vocab: Vocab,
                  device: torch.device, max_new_tokens: int = 2, amp: bool = True) -> List[List[int]]:
    """Generate after each prompt until <eos> or `max_new_tokens`.

    Answers are one token, so 2 new tokens (answer + <eos>) decide exact match:
    a prediction that does not end after one token is wrong either way.
    Decoding recomputes the full forward each step (no cache); it is only 2 steps.
    """
    B = len(indices)
    lens = torch.tensor([int(split.prompt_lens[i]) for i in indices], device=device)
    buf = torch.full((B, int(lens.max()) + max_new_tokens), vocab.pad_id, dtype=torch.long, device=device)
    for row, i in enumerate(indices):
        p = int(split.prompt_lens[i])
        buf[row, :p] = torch.from_numpy(split.seq(i)[:p].astype(np.int64))
    rows = torch.arange(B, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out: List[List[int]] = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        with autocast(device, amp):
            h = model.hidden(buf[:, :int(lens.max())])
            logits = model.head(h[rows, lens - 1])
        nxt = logits.float().argmax(-1)
        for row in (~done).nonzero().flatten().tolist():
            out[row].append(int(nxt[row]))
        nxt = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
        buf[rows, lens] = nxt
        done |= nxt == vocab.eos_id
        lens = lens + 1
        if bool(done.all()):
            break
    return out


@torch.no_grad()
def predict_split(model: LanguageModel, split: EncodedSplit, vocab: Vocab, device: torch.device,
                  max_tokens: int, max_new_tokens: int = 2, amp: bool = True,
                  progress: bool = False) -> Dict[str, str]:
    """Return {example id: predicted text} for every example in `split`."""
    model.eval()
    preds: Dict[str, str] = {}
    batches = list(eval_batches(split, max_tokens))
    for n, batch in enumerate(batches, 1):
        for i, ids in zip(batch, greedy_decode(model, split, batch, vocab, device, max_new_tokens, amp)):
            preds[split.ids[i]] = vocab.decode(ids)
        if progress and (n % 20 == 0 or n == len(batches)):
            print(f"  predicted {len(preds)}/{len(split)}", flush=True)
    return preds


def gold_answers(split: EncodedSplit, vocab: Vocab) -> Dict[str, str]:
    return {split.ids[i]: vocab.decode(split.seq(i)[int(split.prompt_lens[i]):]) for i in range(len(split))}


def accuracy_report(split: EncodedSplit, vocab: Vocab, preds: Dict[str, str]) -> Dict:
    """Quick exact-match accuracy overall, by task and by length (for model selection).
    The official breakdowns come from score.py."""
    gold = gold_answers(split, vocab)
    groups = defaultdict(lambda: [0, 0])
    task_names = {0: "recall", 1: "update"}
    for i, ex_id in enumerate(split.ids):
        ok = int(preds[ex_id].strip() == gold[ex_id].strip())
        for key in ("all", f"task={task_names[int(split.tasks[i])]}", f"len={int(split.buckets[i])}"):
            groups[key][0] += ok
            groups[key][1] += 1
    return {k: round(c / n, 4) for k, (c, n) in sorted(groups.items())}


def save_checkpoint(path, model: LanguageModel, vocab: Vocab, **extra):
    """Write atomically (temp file + rename), so a disconnect mid-save can't
    leave a corrupt checkpoint behind."""
    state = {k: v for k, v in model.state_dict().items()}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"config": model.cfg.to_dict(), "model": state, "vocab": vocab.itos, **extra}, tmp)
    os.replace(tmp, path)


def write_gold_subset(data_dir, split: str, ids, path) -> None:
    """Copy the gold records for `ids` (score.py needs a prediction for every gold id)."""
    keep = set(ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as src, \
            open(path, "w", encoding="utf-8") as dst:
        for line in src:
            if json.loads(line)["id"] in keep:
                dst.write(line)


def write_predictions(path, split: EncodedSplit, preds: Dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex_id in split.ids:
            f.write(json.dumps({"id": ex_id, "prediction": preds[ex_id]}) + "\n")


def load_checkpoint(path, device: torch.device, kernel: Optional[str] = None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    if kernel is not None:
        cfg_dict["kernel"] = kernel
    model = LanguageModel(ModelConfig(**cfg_dict)).to(device)
    state = ckpt["model"]
    state = {k.removeprefix("_orig_mod.") if hasattr(k, "removeprefix") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, Vocab(ckpt["vocab"]), ckpt
