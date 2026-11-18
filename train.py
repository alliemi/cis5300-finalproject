#!/usr/bin/env python3
"""Train one model on the Recall / Update data. Only --attention should differ
between the runs being compared.

CPU smoke test (a few minutes):
    python train.py --attention softmax --out runs/smoke-softmax --max-context-length 128 \
        --max-train-examples 2000 --eval-examples 200 --d-model 64 --n-layers 2 --n-heads 2 \
        --steps 300 --max-tokens 4096 --eval-every 100

GPU run (see README for the full recipe):
    python train.py --attention gated_deltanet --out runs/gdn

Rerunning the same command resumes from <out>/last.pt (e.g. after a Colab
disconnect). When training ends, best.pt predicts the full dev split and
score.py writes <out>/dev_scores.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.attention import ATTENTION
from models.data import IGNORE_INDEX, BucketBatchSampler, Vocab, collate, load_split
from models.inference import (accuracy_report, autocast, load_checkpoint, pick_device, predict_split,
                              save_checkpoint, write_gold_subset, write_predictions)
from models.model import LanguageModel, ModelConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attention", required=True, choices=sorted(ATTENTION))
    p.add_argument("--out", required=True, help="run directory, e.g. runs/softmax-s0")
    p.add_argument("--data-dir", default="data")
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--expand-v", type=float, default=1.0, help="GDN/KDA value expansion")
    p.add_argument("--kernel", default="auto", choices=["auto", "torch", "fla"],
                   help="GDN/KDA recurrence: fla Triton kernels (CUDA) or PyTorch reference")
    p.add_argument("--short-conv", type=int, default=0,
                   help="shared causal conv window before every mixer, for all attention types (0 = off)")
    # data
    p.add_argument("--max-context-length", type=int, default=None,
                   help="drop examples from longer buckets (train and dev)")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--eval-examples", type=int, default=2000, help="dev subset used during training")
    p.add_argument("--max-tokens", type=int, default=32768, help="tokens per training batch")
    p.add_argument("--eval-max-tokens", type=int, default=None, help="tokens per eval batch")
    p.add_argument("--multi-query", type=int, default=0,
                   help="extra training queries per example about other projects in the context "
                        "(0 = off, -1 = all); evaluation always uses only the official query")
    p.add_argument("--curriculum-steps", type=int, default=0,
                   help="unlock length buckets short-to-long over this many steps (0 = off)")
    # optimization
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # infrastructure
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many evals without dev improvement (0 = never)")
    p.add_argument("--no-final-eval", action="store_true",
                   help="skip predicting and scoring the full dev split at the end")
    return p.parse_args(argv)


# Flags that don't change what the model learns, so a resumed run may change them.
NON_RESULT_ARGS = {"out", "device", "compile", "no_final_eval", "log_every", "eval_max_tokens"}


def check_run_dir(out: Path, args) -> None:
    """Refuse to reuse a run directory with different training settings."""
    path = out / "config.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8")).get("args", {})
    current = json.loads(json.dumps(vars(args)))
    diff = sorted(k for k in current if k not in NON_RESULT_ARGS and k in saved and saved[k] != current[k])
    if diff:
        changes = ", ".join(f"{k}: {saved[k]!r} -> {current[k]!r}" for k in diff)
        sys.exit(f"{out} was started with different settings ({changes}). "
                 "Use a new --out, or rerun with the original flags to resume.")


def truncate_metrics(path: Path, step: int) -> None:
    """On resume, drop log lines written after the checkpoint (those steps are redone)."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            if json.loads(line).get("step", 0) <= step:
                keep.append(line)
        except json.JSONDecodeError:   # half-written last line from a disconnect
            pass
    path.write_text("".join(keep), encoding="utf-8")


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def final_dev_eval(args, out: Path, device: torch.device, amp: bool) -> None:
    """Predict the full dev split (or the --max-context-length part of it) with
    best.pt, and score it with score.py."""
    from score import evaluate as score_evaluate

    model, vocab, ckpt = load_checkpoint(out / "best.pt", device)
    dev = load_split(args.data_dir, "dev", vocab, max_context_length=args.max_context_length)
    preds = predict_split(model, dev, vocab, device, args.eval_max_tokens or args.max_tokens, amp=amp)
    write_predictions(out / "dev_pred.jsonl", dev, preds)
    gold = Path(args.data_dir) / "dev.jsonl"
    if args.max_context_length is not None:
        gold = out / "dev_gold.jsonl"
        write_gold_subset(args.data_dir, "dev", dev.ids, gold)
    report = score_evaluate(gold, out / "dev_pred.jsonl")
    (out / "dev_scores.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    o = report["overall"]
    print(f"full dev (best.pt from step {ckpt.get('step')}): accuracy {o['accuracy']:.4f} "
          f"({o['correct']}/{o['n']}); errors {o['error_counts']}; scores in {out}/dev_scores.json")


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    frac = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)


def make_optimizer(model, args, device):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or getattr(p, "_no_weight_decay", False) or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), **kwargs)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_run_dir(out, args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp = not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vocab = Vocab.load(Path(args.data_dir) / "vocab.txt")
    t0 = time.time()
    train = load_split(args.data_dir, "train", vocab, max_examples=args.max_train_examples,
                       max_context_length=args.max_context_length, seed=args.seed)
    dev = load_split(args.data_dir, "dev", vocab, max_examples=args.eval_examples,
                     max_context_length=args.max_context_length, seed=12345)
    print(f"data: {len(train)} train / {len(dev)} dev examples, "
          f"train buckets {sorted(set(train.buckets.tolist()))} ({time.time() - t0:.1f}s)")

    cfg = ModelConfig(vocab_size=len(vocab), attention=args.attention, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=args.n_heads, mlp_ratio=args.mlp_ratio,
                      expand_v=args.expand_v, kernel=args.kernel, short_conv=args.short_conv)
    model = LanguageModel(cfg).to(device)
    print(f"model: {args.attention}, {model.num_params():,} params "
          f"({model.num_mixer_params():,} in attention layers), device {device}")
    # dynamic=True: sequence length changes with the bucket
    hidden_fn = torch.compile(model.hidden, dynamic=True) if args.compile else model.hidden
    opt = make_optimizer(model, args, device)
    sampler = BucketBatchSampler(train, args.max_tokens, seed=args.seed,
                                 curriculum_steps=args.curriculum_steps, max_extra=args.multi_query)
    query_rng = random.Random(args.seed + 1)
    eval_max_tokens = args.eval_max_tokens or args.max_tokens

    (out / "config.json").write_text(json.dumps({"args": vars(args), "model": cfg.to_dict(),
                                                 "num_params": model.num_params()}, indent=2))
    best, evals_since_best, start_step, finished = -1.0, 0, 0, False
    tokens_seen, tokens_window, t_last = 0, 0, time.time()
    loss_acc, acc_acc, ans_acc, n_acc = 0.0, 0.0, 0.0, 0

    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        state = ckpt.get("train_state")
        if state is None:
            sys.exit(f"{out}/last.pt has no training state (written by an older train.py), "
                     "so it can't be resumed. Use a new --out.")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        set_rng_state(state["rng"])
        if "query_rng" in state:   # absent in runs started before --multi-query existed
            query_rng.setstate(state["query_rng"])
        best, evals_since_best = state["best"], state["evals_since_best"]
        start_step, finished, tokens_seen = state["step"], state["finished"], state["tokens_seen"]
        truncate_metrics(out / "metrics.jsonl", start_step)
        print(f"resuming {out} at step {start_step}" + (" (training already finished)" if finished else ""))
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def evaluate(step):
        nonlocal best, evals_since_best, finished
        t = time.time()
        preds = predict_split(model, dev, vocab, device, eval_max_tokens, amp=amp)
        rep = accuracy_report(dev, vocab, preds)
        model.train()
        improved = rep["all"] > best
        if improved:
            best, evals_since_best = rep["all"], 0
            save_checkpoint(out / "best.pt", model, vocab, step=step, dev=rep, args=vars(args))
        else:
            evals_since_best += 1
        finished = step >= args.steps or bool(args.patience and evals_since_best >= args.patience)
        train_state = {"step": step, "finished": finished, "optimizer": opt.state_dict(),
                       "sampler": sampler.state_dict(), "rng": rng_state(), "query_rng": query_rng.getstate(), "best": best,
                       "evals_since_best": evals_since_best, "tokens_seen": tokens_seen}
        save_checkpoint(out / "last.pt", model, vocab, step=step, dev=rep, args=vars(args),
                        train_state=train_state)
        print(f"eval step {step}: dev acc {rep['all']:.4f} (best {best:.4f}) "
              + " ".join(f"{k}={v:.3f}" for k, v in rep.items() if k != "all")
              + f" [{time.time() - t:.1f}s]", flush=True)
        log.write(json.dumps({"type": "eval", "step": step, **rep}) + "\n")
        log.flush()

    model.train()
    for step in range(start_step, 0 if finished else args.steps):
        lr = lr_at(step, args)
        for group in opt.param_groups:
            group["lr"] = lr
        idx = sampler.batch(step)
        x, y, official = collate(train, idx, vocab, args.multi_query, query_rng)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        official = official.to(device, non_blocking=True)
        with autocast(device, amp):
            h = hidden_fn(x)
            sel = y != IGNORE_INDEX
            logits = model.head(h[sel]).float()
            loss = F.cross_entropy(logits, y[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        tokens_seen += x.numel()
        tokens_window += x.numel()
        loss_acc += loss.item()
        # teacher-forced accuracy: tok_acc over all supervised tokens (answers, <eos>),
        # ans_acc over the official answers only (comparable with dev accuracy)
        hit = logits.argmax(-1) == y[sel]
        acc_acc += hit.float().mean().item()
        ans_acc += hit[official[sel]].float().mean().item()
        n_acc += 1
        if (step + 1) % args.log_every == 0:
            dt = time.time() - t_last
            rec = {"type": "train", "step": step + 1, "loss": loss_acc / n_acc, "tok_acc": acc_acc / n_acc,
                   "ans_acc": ans_acc / n_acc,
                   "lr": lr, "grad_norm": float(gnorm), "tokens": tokens_seen,
                   "max_bucket": sampler.buckets[sampler.allowed(step) - 1],
                   "tok_per_s": int(tokens_window / dt)}
            tokens_window = 0
            print(f"step {step + 1}/{args.steps} loss {rec['loss']:.4f} tok_acc {rec['tok_acc']:.3f} ans_acc {rec['ans_acc']:.3f} "
                  f"lr {lr:.2e} gnorm {rec['grad_norm']:.2f} max_len {rec['max_bucket']} "
                  f"~{rec['tok_per_s']:,} tok/s", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_last, loss_acc, acc_acc, ans_acc, n_acc = time.time(), 0.0, 0.0, 0.0, 0
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
            t_last = time.time()      # keep eval time out of the tok/s reading
            if args.patience and evals_since_best >= args.patience:
                print(f"early stop: no dev improvement in {args.patience} evals")
                break
    log.close()
    print(f"done. best dev acc {best:.4f}; checkpoints in {out}/best.pt and {out}/last.pt")
    if not args.no_final_eval:
        final_dev_eval(args, out, device, amp)


if __name__ == "__main__":
    main()
