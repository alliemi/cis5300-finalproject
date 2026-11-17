"""Validation checks for a generated data directory.

Every record is re-derived from its own context text with the independent
parser in `errors.py`, then compared with the stored answer and metadata.
Files are streamed, so this works on large splits without loading them.

The test split only gets the per-record correctness checks and duplicate
checks; no baseline or heuristic is ever scored on it.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from . import wordlists as W
from .errors import parse_facts, parse_query, value_history
from .generator import LENGTHS, NUM_BINS, SPLITS, design_marginals, example_key
from .tokenizer import tokenize

FIELDS = OrderedDict([
    ("id", str), ("task", str), ("context", str), ("query", str), ("answer", str),
    ("target_entity", str), ("stale_values", list), ("distractor_values", list),
    ("context_length", int), ("context_tokens", int), ("num_entities", int),
    ("num_updates", int), ("num_distractor_updates", int), ("num_facts", int),
    ("needle_position", float), ("needle_position_bin", int),
])
LENGTH_TOLERANCE = 2          # context_tokens must be in [L - 2, L]
BALANCE_FIELDS = ("task", "context_length", "num_entities", "num_updates", "needle_position_bin")
MAX_ERRORS_LISTED = 20

_NAMES = set(W.NAMES)
_VALUES = set(W.VALUES)


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_vocab(data_dir: str) -> set:
    with open(os.path.join(data_dir, "vocab.txt"), encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}


def check_record(rec: Dict, split: str, index: int, vocab: set) -> List[str]:
    """Return a list of problems with one record (empty list = OK)."""
    errs: List[str] = []
    for name, typ in FIELDS.items():
        if name not in rec:
            errs.append(f"missing field {name}")
        elif typ is float:
            if not isinstance(rec[name], (int, float)) or isinstance(rec[name], bool):
                errs.append(f"{name} has type {type(rec[name]).__name__}")
        elif not isinstance(rec[name], typ) or (typ is int and isinstance(rec[name], bool)):
            errs.append(f"{name} has type {type(rec[name]).__name__}")
    if errs:
        return errs
    if set(rec) != set(FIELDS):
        errs.append(f"unexpected fields {sorted(set(rec) - set(FIELDS))}")
    if rec["id"] != f"{split}-{index:06d}":
        errs.append(f"id {rec['id']} should be {split}-{index:06d}")
    if rec["task"] not in ("recall", "update"):
        errs.append(f"bad task {rec['task']}")

    # Tokens, length and vocabulary.
    ctx = rec["context"]
    toks = tokenize(ctx)
    L = rec["context_length"]
    if L not in LENGTHS:
        errs.append(f"context_length {L} not in {LENGTHS}")
    if len(toks) != rec["context_tokens"]:
        errs.append(f"context_tokens {rec['context_tokens']} != recount {len(toks)}")
    if not (L - LENGTH_TOLERANCE <= len(toks) <= L):
        errs.append(f"context has {len(toks)} tokens, outside [{L - LENGTH_TOLERANCE}, {L}]")
    oov = {t for t in toks + tokenize(rec["query"]) + [rec["answer"]] if t not in vocab}
    if oov:
        errs.append(f"out-of-vocabulary tokens {sorted(oov)[:5]}")

    # Query names the target.
    target = parse_query(rec["query"])
    if target is None or target != rec["target_entity"]:
        errs.append(f"query {rec['query']!r} does not ask about {rec['target_entity']}")
        return errs

    # Facts read back from the text.
    facts = parse_facts(ctx)
    n = len(facts)
    if sum(t == "Project" for t in toks) != n:
        errs.append("a 'Project' mention is not a well-formed fact")
    if sum(t in _VALUES for t in toks) != n:
        errs.append("a value word appears outside a fact statement")
    if sum(t in _NAMES for t in toks) != n:
        errs.append("a project name appears outside a fact statement")
    hist = value_history(facts)
    if target not in hist:
        errs.append("target project never stated in context")
        return errs
    tvals = hist[target]
    U = rec["num_updates"]

    if rec["answer"] != tvals[-1]:
        errs.append(f"answer {rec['answer']} != final stated value {tvals[-1]}")
    if rec["stale_values"] != tvals[:-1]:
        errs.append(f"stale_values {rec['stale_values']} != earlier values {tvals[:-1]}")
    if len(rec["answer"].split()) != 1 or rec["answer"] not in _VALUES:
        errs.append("answer is not a single value token")
    if rec["task"] == "recall":
        if U != 0 or len(tvals) != 1:
            errs.append(f"recall example states the target {len(tvals)} times (num_updates={U})")
    else:
        if U < 1 or len(tvals) != U + 1:
            errs.append(f"update example states target {len(tvals)} times, expected {U + 1}")
    if not any(len(v) == len(tvals) for e, v in hist.items() if e != target):
        errs.append("no other project is mentioned as often as the target (mention count leaks the target)")
    dvals = [f.value for f in facts if f.entity != target]
    if rec["distractor_values"] != dvals:
        errs.append("distractor_values do not match other projects' values in context")
    all_vals = [f.value for f in facts]
    if len(set(all_vals)) != len(all_vals):
        errs.append("a value is used more than once in the context")
    ans, stale, dist = {rec["answer"]}, set(rec["stale_values"]), set(rec["distractor_values"])
    if ans & stale or ans & dist or stale & dist:
        errs.append("answer / stale / distractor values overlap")

    # Counts.
    if rec["num_entities"] != len(hist):
        errs.append(f"num_entities {rec['num_entities']} != {len(hist)} projects in context")
    if rec["num_facts"] != n:
        errs.append(f"num_facts {rec['num_facts']} != {n}")
    if rec["num_distractor_updates"] != n - len(hist) - U:
        errs.append("num_distractor_updates inconsistent with fact counts")

    # Needle position = token offset of the final target statement / context tokens.
    last_target = [f for f in facts if f.entity == target][-1]
    offset = len(tokenize(ctx[:last_target.char_start]))
    pos = offset / max(len(toks), 1)
    if abs(pos - rec["needle_position"]) > 1e-3:
        errs.append(f"needle_position {rec['needle_position']} != recomputed {pos:.4f}")
    if rec["needle_position_bin"] != min(int(pos * NUM_BINS), NUM_BINS - 1):
        errs.append("needle_position_bin inconsistent with needle_position")
    return errs


# ---------------------------------------------------------------------------
# Shortcut baselines (dev only)
# ---------------------------------------------------------------------------
BASELINES = ("last_value_in_context", "first_value_in_context", "random_value_in_context",
             "latest_value_of_most_mentioned_project", "first_value_of_target",
             "memorized_train_answer_for_name")


def baseline_predictions(rec: Dict, rng: random.Random,
                         train_answers: Dict[str, Counter]) -> Dict[str, Optional[str]]:
    facts = parse_facts(rec["context"])
    hist = value_history(facts)
    most = max(hist, key=lambda e: len(hist[e]))   # ties -> first stated
    memo = train_answers.get(rec["target_entity"])
    return {
        "last_value_in_context": facts[-1].value,
        "first_value_in_context": facts[0].value,
        "random_value_in_context": rng.choice(facts).value,
        "latest_value_of_most_mentioned_project": hist[most][-1],
        "first_value_of_target": hist[rec["target_entity"]][0],
        "memorized_train_answer_for_name": min(memo, key=lambda v: (-memo[v], v)) if memo else None,
    }


# ---------------------------------------------------------------------------
# Whole directory
# ---------------------------------------------------------------------------
def _share_table(counter: Counter, n: int) -> Dict[str, float]:
    return OrderedDict((str(k), round(counter[k] / n, 4)) for k in sorted(counter))


def validate_dir(data_dir: str) -> Dict:
    """Run every check on data_dir; returns a JSON-serialisable report.

    report["ok"] is True only if no record error, duplicate or balance
    failure was found.
    """
    vocab = load_vocab(data_dir)
    expected = design_marginals()
    report: Dict = OrderedDict([("ok", True), ("splits", OrderedDict())])
    seen: Dict[str, str] = {}
    train_answers: Dict[str, Counter] = defaultdict(Counter)

    for split in SPLITS:
        path = os.path.join(data_dir, f"{split}.jsonl")
        errors: List[str] = []
        n_err = 0
        n = 0
        dup_within = dup_across = 0
        marg = {f: Counter() for f in BALANCE_FIELDS}
        joint_L_K: Counter = Counter()
        joint_L_U: Counter = Counter()
        tokens_by_L: Dict[int, List[int]] = defaultdict(list)
        base_hits = {task: Counter() for task in ("recall", "update", "all")}
        chance = {task: 0.0 for task in ("recall", "update", "all")}
        task_n: Counter = Counter()
        by_k_hits: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
        by_k_n: Counter = Counter()
        rng = random.Random(0)

        for i, rec in enumerate(iter_jsonl(path), start=1):
            n += 1
            errs = check_record(rec, split, i, vocab)
            if errs:
                n_err += 1
                if len(errors) < MAX_ERRORS_LISTED:
                    errors.append(f"{rec.get('id', i)}: {'; '.join(errs)}")
                continue
            key = example_key(rec)
            if key in seen:
                if seen[key] == split:
                    dup_within += 1
                else:
                    dup_across += 1
                if len(errors) < MAX_ERRORS_LISTED:
                    errors.append(f"{rec['id']}: duplicate of an example in {seen[key]}")
            else:
                seen[key] = split
            for f in BALANCE_FIELDS:
                marg[f][rec[f]] += 1
            joint_L_K[(rec["context_length"], rec["num_entities"])] += 1
            if rec["task"] == "update":
                joint_L_U[(rec["context_length"], rec["num_updates"])] += 1
            tokens_by_L[rec["context_length"]].append(rec["context_tokens"])
            if split == "train":
                train_answers[rec["target_entity"]][rec["answer"]] += 1
            if split == "dev":
                preds = baseline_predictions(rec, rng, train_answers)
                by_k_n[(rec["task"], rec["num_entities"])] += 1
                for b, p in preds.items():
                    by_k_hits[(rec["task"], rec["num_entities"])][b] += int(p == rec["answer"])
                for task in (rec["task"], "all"):
                    task_n[task] += 1
                    chance[task] += 1 / rec["num_facts"]
                    for b, p in preds.items():
                        base_hits[task][b] += int(p == rec["answer"])

        n_bad = n_err + dup_within + dup_across
        sr: Dict = OrderedDict([
            ("count", n), ("records_with_errors", n_err),
            ("duplicates_within_split", dup_within), ("duplicates_across_splits", dup_across),
            ("errors", errors),
        ])
        # Balance: observed share of each setting vs the design's expected share.
        balance = OrderedDict()
        balance_ok = True
        for f in BALANCE_FIELDS:
            rows = OrderedDict()
            for val, exp in sorted(expected[f].items(), key=lambda kv: (len(kv[0]), kv[0])):
                obs = marg[f][val if f == "task" else int(val)] / n if n else 0.0
                tol = max(0.02, 4 * math.sqrt(exp * (1 - exp) / max(n, 1)))
                ok = abs(obs - exp) <= tol
                balance_ok &= ok
                rows[val] = OrderedDict([("observed", round(obs, 4)), ("expected", round(exp, 4)), ("ok", ok)])
            balance[f] = rows
        sr["balance_ok"] = balance_ok
        sr["balance"] = balance
        sr["joint_context_length_x_num_entities"] = OrderedDict(
            (f"{L}x{K}", c) for (L, K), c in sorted(joint_L_K.items()))
        sr["joint_context_length_x_num_updates_update_task"] = OrderedDict(
            (f"{L}x{U}", c) for (L, U), c in sorted(joint_L_U.items()))
        sr["context_tokens_by_length"] = OrderedDict(
            (str(L), OrderedDict([("min", min(v)), ("mean", round(sum(v) / len(v), 1)), ("max", max(v))]))
            for L, v in sorted(tokens_by_L.items()))
        if split == "dev":
            sr["baselines_accuracy"] = OrderedDict(
                (task, OrderedDict([("n", task_n[task]),
                                    ("chance_random_fact", round(chance[task] / max(task_n[task], 1), 4))]
                                   + [(b, round(base_hits[task][b] / max(task_n[task], 1), 4))
                                      for b in BASELINES]))
                for task in ("all", "recall", "update"))
            sr["baselines_accuracy_by_num_entities"] = OrderedDict(
                (task, OrderedDict(
                    (str(K), OrderedDict([("n", by_k_n[(task, K)])] + [
                        (b, round(by_k_hits[(task, K)][b] / by_k_n[(task, K)], 4)) for b in BASELINES]))
                    for (t, K) in sorted(by_k_n) if t == task))
                for task in ("recall", "update"))
        report["splits"][split] = sr
        if n_bad or not balance_ok or n == 0:
            report["ok"] = False
    return report
