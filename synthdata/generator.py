"""Generate the Recall / Update long-context retrieval dataset.

Overview of one example (see data.md for the plain-language version):
  1. sample settings: task, context length L, #entities K, #updates U,
     needle-position bin, and per-distractor update counts (one "mirrored
     decoy" distractor always has exactly U updates)
  2. list the fact statements, pick a template for each, and give each a
     rough position; every project's final statement is placed by the same
     procedure as the target's needle, earlier statements go before it
  3. split statements into a prefix (before the needle) and a suffix,
     then top each part up with filler to an exact token budget
  4. order each part by position, assign distinct values in reading order
  5. render text and record metadata

All randomness goes through `Rng`, which only calls `random.Random.random()`
(whose output stream Python guarantees across versions), so a seed gives
the same data on any Python 3 install.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import wordlists as W
from .tokenizer import build_vocab, render, tokenize

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
TASKS = ("recall", "update")
LENGTHS = (128, 256, 512, 1024, 2048)
NUM_ENTITIES = (2, 4, 8, 16, 32, 64)
UPDATE_COUNTS = (1, 2, 4, 8)          # Update task; Recall always uses 0
NUM_BINS = 5                           # needle position bins of width 0.2
FACT_DENSITY = 0.6                     # facts may fill at most ~60% of the context
EST_FACT_TOKENS = 7                    # budget per statement (templates are 5/6/8 tokens)
MAX_FACT_TOKENS = max(len(t) for t in W.FACT_TEMPLATES)
BIN_MARGIN = 0.03                      # keep sampled positions away from bin edges
SPLITS = ("train", "dev", "test")
GENERATION_ORDER = ("test", "dev", "train")   # test first: never depends on train size
DEFAULT_SEED = 5300
MAX_ATTEMPTS = 50


class Rng:
    """Seeded RNG whose helpers are built only on `random()` (version-stable)."""

    def __init__(self, key: str):
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        self._r = random.Random(seed)

    def random(self) -> float:
        return self._r.random()

    def below(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        return min(int(self._r.random() * n), n - 1)

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b]."""
        return a + self.below(b - a + 1)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self._r.random()

    def choice(self, seq: Sequence):
        return seq[self.below(len(seq))]

    def shuffle(self, items: List) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, seq: Sequence, k: int) -> List:
        pool = list(seq)
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


# ---------------------------------------------------------------------------
# Sampling design: which settings fit together, and how they are drawn
# ---------------------------------------------------------------------------
def max_statements(L: int) -> int:
    return int(FACT_DENSITY * L) // EST_FACT_TOKENS


def entity_options(task: str, L: int) -> List[int]:
    min_u = 0 if task == "recall" else min(UPDATE_COUNTS)
    return [K for K in NUM_ENTITIES if K + 2 * min_u <= max_statements(L)]


def update_options(task: str, L: int, K: int) -> List[int]:
    # 2*U: the target's U updates plus the mirrored decoy's U updates.
    if task == "recall":
        return [0]
    return [U for U in UPDATE_COUNTS if K + 2 * U <= max_statements(L)]


def position_range(L: int, U: int, b: int) -> Optional[Tuple[float, float]]:
    """Allowed needle positions inside bin b, or None if the bin is infeasible.

    The U stale statements must fit before the needle, and the needle itself
    must fit before the end of the context.
    """
    lo = max(b / NUM_BINS + BIN_MARGIN, (U * MAX_FACT_TOKENS + 1) / L)
    hi = min((b + 1) / NUM_BINS - BIN_MARGIN, 1 - (MAX_FACT_TOKENS + 2) / L)
    return (lo, hi) if lo < hi else None


def bin_options(L: int, U: int) -> List[int]:
    return [b for b in range(NUM_BINS) if position_range(L, U, b) is not None]


def design_cells():
    """Yield (task, L, K, U, bin, probability) for every cell of the design."""
    for task in TASKS:
        for L in LENGTHS:
            ks = entity_options(task, L)
            for K in ks:
                us = update_options(task, L, K)
                for U in us:
                    bins = bin_options(L, U)
                    for b in bins:
                        p = 1 / len(TASKS) / len(LENGTHS) / len(ks) / len(us) / len(bins)
                        yield task, L, K, U, b, p


def design_marginals() -> Dict[str, Dict[str, float]]:
    """Expected share of each setting value under the sampling design."""
    names = ("task", "context_length", "num_entities", "num_updates", "needle_position_bin")
    out: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for cell in design_cells():
        p = cell[-1]
        for name, val in zip(names, cell[:-1]):
            key = str(val)
            out[name][key] = out[name].get(key, 0.0) + p
    return out


def distractor_update_counts(rng: Rng, L: int, K: int, U: int) -> List[int]:
    """Number of updates for each of the K-1 distractor projects.

    Entry 0 is the mirrored decoy: it gets exactly U updates, so at least
    one other project is mentioned as often as the target and "count the
    mentions" can't identify the queried project. Up to K-2 further
    distractors get update counts from the same menu as the target
    (UPDATE_COUNTS), truncated to the statement budget.
    """
    budget = max_statements(L) - K - 2 * U
    counts = [U] + [0] * (K - 2)
    for i in range(1, 1 + rng.randint(0, K - 2)):
        u = min(rng.choice(UPDATE_COUNTS), budget)
        if u <= 0:
            break
        counts[i] = u
        budget -= u
    return counts


def sample_position_bin(rng: Rng, L: int, U: int) -> Tuple[int, float]:
    """Final-statement position for a project with U updates: a uniform
    feasible bin, then a uniform point inside it."""
    b = rng.choice(bin_options(L, U))
    lo, hi = position_range(L, U, b)
    return b, rng.uniform(lo, hi)


def sample_position(rng: Rng, L: int, U: int) -> float:
    return sample_position_bin(rng, L, U)[1]


def sample_settings(rng: Rng) -> Dict:
    task = rng.choice(TASKS)
    L = rng.choice(LENGTHS)
    K = rng.choice(entity_options(task, L))
    U = rng.choice(update_options(task, L, K))
    b, p = sample_position_bin(rng, L, U)
    return {"task": task, "L": L, "K": K, "U": U, "bin": b, "p": p,
            "dist_updates": distractor_update_counts(rng, L, K, U)}


# ---------------------------------------------------------------------------
# Filler
# ---------------------------------------------------------------------------
_T = {k: tuple(tuple(tokenize(p)) for p in getattr(W, k))
      for k in ("FILLER_OPENERS", "FILLER_SUBJECTS", "FILLER_PREDICATES", "FILLER_TIMES",
                "FILLER_ITEMS", "FILLER_PLACES", "FILLER_ACTIONS")}
_PADS = {n: tuple(tuple(tokenize(p)) for p in pads) for n, pads in W.FILLER_PADS.items()}


def random_filler(rng: Rng) -> List[str]:
    opener = list(rng.choice(_T["FILLER_OPENERS"]))
    kind = rng.below(4)
    if kind <= 1:   # "The <subject> <predicate> [time] ." (most common)
        body = ["the" if opener else "The"] + list(rng.choice(_T["FILLER_SUBJECTS"]))
        body += list(rng.choice(_T["FILLER_PREDICATES"])) + list(rng.choice(_T["FILLER_TIMES"]))
    elif kind == 2:
        body = ["someone" if opener else "Someone", "left", "a"] + list(rng.choice(_T["FILLER_ITEMS"]))
        body += ["in", "the"] + list(rng.choice(_T["FILLER_PLACES"]))
    else:
        body = ["please" if opener else "Please", "remember", "to"] + list(rng.choice(_T["FILLER_ACTIONS"]))
    return opener + body + ["."]


def make_filler(rng: Rng, n_tokens: int) -> List[List[str]]:
    """Filler sentences totalling exactly n_tokens (n_tokens - 1 if n_tokens == 1)."""
    out: List[List[str]] = []
    remaining = n_tokens
    while remaining >= 2:
        if remaining <= 6:   # finish with short pads: 2, 3, 4, 3+2, 4+2
            n = remaining if remaining <= 4 else remaining - 2
            out.append(list(rng.choice(_PADS[n])))
            remaining -= n
            continue
        sent = None
        for _ in range(20):
            cand = random_filler(rng)
            if len(cand) == remaining or len(cand) <= remaining - 2:
                sent = cand
                break
        if sent is None:
            sent = list(rng.choice(_PADS[4]))
        out.append(sent)
        remaining -= len(sent)
    return out


# ---------------------------------------------------------------------------
# One example
# ---------------------------------------------------------------------------
class _Stmt:
    __slots__ = ("entity", "template", "is_target")

    def __init__(self, entity: str, template: int, is_target: bool):
        self.entity, self.template, self.is_target = entity, template, is_target

    def __len__(self):
        return len(W.FACT_TEMPLATES[self.template])


def _fill_template(template: Sequence[str], entity: str, value: Optional[str] = None) -> List[str]:
    return [entity if w == "{E}" else value if w == "{V}" else w for w in template]


def _ntok(items) -> int:
    return sum(len(x) for x in items)


def build_example(rng: Rng, s: Dict) -> Optional[Dict]:
    """Build one example from settings `s`; None if the random layout failed."""
    L, K, U = s["L"], s["K"], s["U"]
    names = rng.sample(W.NAMES, K)
    target, distractors = names[0], names[1:]
    n_templates = len(W.FACT_TEMPLATES)

    p = s["p"]
    needle = _Stmt(target, rng.below(n_templates), True)
    # Every item gets a sort key in [0, 1] (its rough relative position).
    # Each project's final statement is placed with the same procedure used
    # for the target's needle, and its earlier statements go uniformly before
    # it. So the target and the mirrored decoy follow identical placement
    # patterns, and position can't reveal which project is queried.
    keyed: List[Tuple[float, _Stmt]] = [(rng.uniform(0, p), _Stmt(target, rng.below(n_templates), True))
                                        for _ in range(U)]
    for e, u in zip(distractors, s["dist_updates"]):
        q = sample_position(rng, L, u)
        keyed.append((q, _Stmt(e, rng.below(n_templates), False)))
        keyed += [(rng.uniform(0, q), _Stmt(e, rng.below(n_templates), False)) for _ in range(u)]
    D = sum(s["dist_updates"])

    # Split around the needle with exact token budgets. If one side overflows,
    # move the distractor statement nearest the needle across it.
    prefix_budget = int(round(p * L))
    suffix_budget = L - prefix_budget - len(needle)
    prefix = [kv for kv in keyed if kv[0] < p]
    suffix = [kv for kv in keyed if kv[0] >= p]
    eps = 1e-9
    while _ntok(st for _, st in prefix) > prefix_budget:
        movable = [kv for kv in prefix if not kv[1].is_target]
        if not movable:
            return None
        kv = max(movable, key=lambda x: x[0])
        prefix.remove(kv)
        suffix.append((p + eps * (1 + kv[0]), kv[1]))      # keeps moved items in order
    while _ntok(st for _, st in suffix) > suffix_budget:
        kv = min(suffix, key=lambda x: x[0])
        suffix.remove(kv)
        prefix.append((p - eps * (2 - kv[0]), kv[1]))
    if _ntok(st for _, st in prefix) > prefix_budget:
        return None

    prefix += [(rng.uniform(0, p), f) for f in make_filler(rng, prefix_budget - _ntok(st for _, st in prefix))]
    suffix += [(rng.uniform(p, 1), f) for f in make_filler(rng, suffix_budget - _ntok(st for _, st in suffix))]
    prefix.sort(key=lambda x: x[0])
    suffix.sort(key=lambda x: x[0])
    sequence = [it for _, it in prefix] + [needle] + [it for _, it in suffix]

    # Values: all distinct within the example, assigned in reading order.
    n_stmts = sum(isinstance(x, _Stmt) for x in sequence)
    values = iter(rng.sample(W.VALUES, n_stmts))
    tokens: List[str] = []
    history: "OrderedDict[str, List[str]]" = OrderedDict()
    distractor_values: List[str] = []
    needle_offset = -1
    for item in sequence:
        if isinstance(item, _Stmt):
            v = next(values)
            history.setdefault(item.entity, []).append(v)
            if item is needle:
                needle_offset = len(tokens)
            if not item.is_target:
                distractor_values.append(v)
            tokens += _fill_template(W.FACT_TEMPLATES[item.template], item.entity, v)
        else:
            tokens += item

    target_values = history[target]
    pos = needle_offset / len(tokens)
    pos_bin = min(int(pos * NUM_BINS), NUM_BINS - 1)
    if pos_bin != s["bin"]:
        return None
    return {
        "task": s["task"],
        "context": render(tokens),
        "query": render(_fill_template(W.QUERY_TEMPLATE, target)),
        "answer": target_values[-1],
        "target_entity": target,
        "stale_values": target_values[:-1],
        "distractor_values": distractor_values,
        "context_length": L,
        "context_tokens": len(tokens),
        "num_entities": K,
        "num_updates": U,
        "num_distractor_updates": D,
        "num_facts": n_stmts,
        "needle_position": round(pos, 4),
        "needle_position_bin": pos_bin,
    }


def example_key(rec: Dict) -> str:
    """Content hash used to detect duplicates within and across splits."""
    return hashlib.sha256((rec["context"] + "\x00" + rec["query"]).encode("utf-8")).hexdigest()


def generate_split(split: str, n: int, seed: int, seen: Set[str]) -> Tuple[List[Dict], Dict]:
    """Generate n unique examples for `split`, rejecting anything already in `seen`.

    `seen` is updated in place, so passing the same set across splits keeps
    every example in exactly one split.
    """
    rng = Rng(f"{seed}-{split}")
    records: List[Dict] = []
    info = {"layout_retries": 0, "duplicates_rejected": 0}
    while len(records) < n:
        settings = sample_settings(rng)
        rec = None
        for _ in range(MAX_ATTEMPTS):
            rec = build_example(rng, settings)
            if rec is not None:
                break
            info["layout_retries"] += 1
        if rec is None:
            raise RuntimeError(f"could not build an example for settings {settings}")
        key = example_key(rec)
        if key in seen:
            info["duplicates_rejected"] += 1
            continue
        seen.add(key)
        records.append(OrderedDict([("id", f"{split}-{len(records) + 1:06d}")] + list(rec.items())))
    return records, info


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def generate_dataset(out_dir: str, n_train: int, n_dev: int, n_test: int,
                     seed: int = DEFAULT_SEED) -> Dict:
    """Write {train,dev,test}.jsonl, vocab.txt and generation_info.json to out_dir.

    Every output is a deterministic function of (seed, split sizes); the
    dev and test files depend only on their own size and the seed.
    """
    W.check_wordlists()
    os.makedirs(out_dir, exist_ok=True)
    sizes = {"train": n_train, "dev": n_dev, "test": n_test}
    seen: Set[str] = set()
    info = OrderedDict([("seed", seed), ("splits", OrderedDict())])
    for split in GENERATION_ORDER:
        records, split_info = generate_split(split, sizes[split], seed, seen)
        write_jsonl(os.path.join(out_dir, f"{split}.jsonl"), records)
        info["splits"][split] = OrderedDict([("count", len(records))] + list(split_info.items()))
    vocab = build_vocab()
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(vocab) + "\n")
    info["vocab_size"] = len(vocab)
    info["design"] = OrderedDict([
        ("tasks", TASKS), ("context_lengths", LENGTHS), ("num_entities", NUM_ENTITIES),
        ("num_updates_update_task", UPDATE_COUNTS), ("needle_position_bins", NUM_BINS),
        ("max_fact_density", FACT_DENSITY),
        ("distractor_updates", "one mirrored decoy with exactly num_updates updates; up to K-2 more "
                               "distractors with updates drawn from num_updates_update_task (budget-capped)"),
        ("num_names", len(W.NAMES)), ("num_values", len(W.VALUES)),
        ("tokenizer", "word-level: regex [A-Za-z]+|[0-9]+|[^\\sA-Za-z0-9]"),
    ])
    with open(os.path.join(out_dir, "generation_info.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info
