"""Training loop shared by every architecture.

A run directory holds everything about one run:
  config.json            merged config + data filters + git commit + parameter count
  log.jsonl              one line per log/eval event (appended across resumes)
  last.pt                full state for resuming (model, optimizer, step, data position)
  best.pt                weights with the best dev-subset accuracy
  dev_predictions.jsonl  predictions on the full dev split (from best.pt)
  dev_scores.json        score.py report for those predictions

Rerunning with the same run directory resumes from last.pt.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from models import build_model, check_device, count_parameters
from .data import PAD, BucketBatcher, collate, encode_split, load_vocab
from .evaluate import accuracy_breakdown, autocast, predict_ids, write_predictions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_for(device: torch.device) -> Optional[torch.dtype]:
    """bf16 mixed precision on GPUs that support it (A100, L4); fp32 elsewhere."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def lr_at(step: int, t: Dict) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    progress = min(1.0, (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"]))
    floor = t["min_lr_ratio"]
    return t["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(config: Dict, run_dir: str, data_dir: str = "data", device: str = "auto",
          max_context_length: Optional[int] = None, train_subset: Optional[int] = None,
          final_eval: bool = True, log: Callable[[str], None] = print) -> Dict:
    t = config["train"]
    dev = pick_device(device)
    check_device(config, dev)
    amp = amp_dtype_for(dev)
    os.makedirs(run_dir, exist_ok=True)

    # One run directory = one configuration; refuse to silently mix two.
    run_info = {"config": config, "data": {"max_context_length": max_context_length,
                                           "train_subset": train_subset}}
    info_path = os.path.join(run_dir, "config.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            saved = json.load(f)
        if {k: saved.get(k) for k in run_info} != json.loads(json.dumps(run_info)):
            raise ValueError(f"{run_dir} was started with a different config or data filter; "
                             "use a new --run directory")

    vocab = load_vocab(data_dir)
    pad_id = vocab.index(PAD)
    train_split = encode_split(data_dir, "train")
    dev_split = encode_split(data_dir, "dev")
    train_idx = train_split.select(max_context_length, train_subset)
    eval_idx = dev_split.select(max_context_length, t["eval_examples"])
    batcher = BucketBatcher(train_split, train_idx, t["batch_size"], seed=t["seed"])
    epoch_tokens = train_split.num_tokens(train_idx)

    torch.manual_seed(t["seed"])
    model = build_model(config, len(vocab)).to(dev)
    n_params = count_parameters(model)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=t["lr"], betas=tuple(t["betas"]))
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "best_acc": -1.0}

    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        torch.set_rng_state(ckpt["rng"].cpu())
        log(f"resuming {run_dir} at step {state['step']}")
    else:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({**run_info, "params": n_params, "git_commit": _git_commit(),
                       "device": dev.type, "amp": str(amp)}, f, indent=2)
        log(f"{config['mixer']}: {n_params:,} parameters on {dev.type}"
            f"{' (bf16)' if amp else ''}; {len(train_idx):,} train examples, "
            f"{epoch_tokens / 1e6:.1f}M tokens per epoch")

    log_file = open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")

    def record(event: Dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    def evaluate_and_save() -> None:
        preds = predict_ids(model, dev_split, eval_idx, 2 * t["batch_size"], dev, pad_id, amp)
        result = accuracy_breakdown(dev_split, preds)
        record({"type": "eval", "step": state["step"], **result})
        log(f"step {state['step']}: dev acc {result['accuracy']:.4f} "
            f"(recall {result['by_task'].get('recall')}, update {result['by_task'].get('update')}) "
            f"by length {result['by_context_length']}")
        if result["accuracy"] > state["best_acc"]:
            state["best_acc"] = result["accuracy"]
            _save({"model": model.state_dict(), "config": config, "vocab_size": len(vocab),
                   "step": state["step"], "dev_subset_acc": result["accuracy"]}, best_path)
        _save({"model": model.state_dict(), "optimizer": opt.state_dict(), "state": state,
               "rng": torch.get_rng_state()}, last_path)

    model.train()
    window_start, window_tokens, window_loss, window_steps = time.time(), 0, 0.0, 0
    while state["step"] < t["max_steps"]:
        batches = batcher.epoch_batches(state["epoch"])
        for batch_idx in batches[state["batch_in_epoch"]:]:
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"], t)
            batch = collate(train_split, batch_idx, pad_id)
            with autocast(dev, amp):
                logits = model(batch["input_ids"].to(dev), batch["last"].to(dev))
            loss = F.cross_entropy(logits.float(), batch["answers"].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            state["step"] += 1
            state["batch_in_epoch"] += 1
            window_tokens += int(batch["input_ids"].numel())
            window_loss += loss.detach()   # summed on-device; read once per log
            window_steps += 1

            if state["step"] % t["log_every"] == 0 or state["step"] == t["max_steps"]:
                elapsed = time.time() - window_start
                tps = window_tokens / max(elapsed, 1e-9)
                mean_loss = float(window_loss) / window_steps
                record({"type": "train", "step": state["step"], "loss": round(mean_loss, 4),
                         "lr": lr_at(state["step"] - 1, t), "tokens_per_s": round(tps)})
                log(f"step {state['step']}: loss {mean_loss:.4f}, "
                    f"{tps:,.0f} tokens/s, ~{epoch_tokens / tps / 60:.1f} min per epoch")
                window_start, window_tokens, window_loss, window_steps = time.time(), 0, 0.0, 0
            if state["step"] % t["eval_every"] == 0 or state["step"] == t["max_steps"]:
                evaluate_and_save()
                window_start = time.time()   # don't count eval time as training speed
            if state["step"] >= t["max_steps"]:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0
    log_file.close()

    summary = {"step": state["step"], "best_dev_subset_acc": state["best_acc"], "params": n_params}
    if final_eval:
        summary.update(final_dev_evaluation(run_dir, data_dir, dev, log))
    return summary


def final_dev_evaluation(run_dir: str, data_dir: str, device: torch.device,
                         log: Callable[[str], None] = print) -> Dict:
    """Predict the full dev split with best.pt and score it with score.py."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from score import evaluate as score_evaluate

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    vocab = load_vocab(data_dir)
    model = build_model(ckpt["config"], len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    split = encode_split(data_dir, "dev")
    preds = predict_ids(model, split, range(len(split)), 2 * ckpt["config"]["train"]["batch_size"],
                        device, vocab.index(PAD), amp_dtype_for(device))
    pred_path = os.path.join(run_dir, "dev_predictions.jsonl")
    write_predictions(pred_path, split, preds, vocab)
    report = score_evaluate(os.path.join(data_dir, "dev.jsonl"), pred_path)
    with open(os.path.join(run_dir, "dev_scores.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    overall = report["overall"]
    log(f"full dev (best.pt from step {ckpt['step']}): accuracy {overall['accuracy']:.4f}; "
        f"errors {overall['error_counts']}")
    return {"dev_accuracy": overall["accuracy"], "dev_error_counts": overall["error_counts"]}
