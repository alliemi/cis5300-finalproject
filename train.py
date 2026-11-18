#!/usr/bin/env python3
"""Train one model on the Recall / Update data. Only --attention should differ
between the runs being compared.

CPU smoke test (a few minutes):
    python train.py --attention softmax --out runs/smoke-softmax --max-context-length 128 \
        --max-train-examples 2000 --eval-examples 200 --d-model 64 --n-layers 2 --n-heads 2 \
        --steps 300 --max-tokens 4096 --eval-every 100

GPU run (see README for the full recipe):
    python train.py --attention gated_deltanet --out runs/gdn
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.attention import ATTENTION
from models.data import IGNORE_INDEX, BucketBatchSampler, Vocab, collate, load_split
from models.inference import accuracy_report, autocast, pick_device, predict_split, save_checkpoint
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
    return p.parse_args(argv)


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
                      expand_v=args.expand_v, kernel=args.kernel)
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
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    best, evals_since_best = -1.0, 0
    tokens_seen, tokens_window, t_last = 0, 0, time.time()
    loss_acc, acc_acc, ans_acc, n_acc = 0.0, 0.0, 0.0, 0

    def evaluate(step):
        nonlocal best, evals_since_best
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
        save_checkpoint(out / "last.pt", model, vocab, step=step, dev=rep, args=vars(args))
        print(f"eval step {step}: dev acc {rep['all']:.4f} (best {best:.4f}) "
              + " ".join(f"{k}={v:.3f}" for k, v in rep.items() if k != "all")
              + f" [{time.time() - t:.1f}s]", flush=True)
        log.write(json.dumps({"type": "eval", "step": step, **rep}) + "\n")
        log.flush()

    model.train()
    for step in range(args.steps):
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


if __name__ == "__main__":
    main()
