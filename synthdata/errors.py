"""Read facts back out of a context and classify model predictions.

This module deliberately does NOT import the generator: the validator uses
it as an independent second opinion on what each context says.

Error labels match score.py, which is the official scorer:
  correct      prediction == answer
  stale        an earlier (outdated) value of the queried project
  distractor   a value that belonged to a different project
  other        anything else: failed retrieval / made-up value
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional

FACT_RE = re.compile(r"(?:The status of )?Project ([A-Z][a-z]+) (?:status: |is )([a-z]+)\.")
QUERY_RE = re.compile(r"^What is the current status of Project ([A-Z][a-z]+)\?$")

LABELS = ("correct", "stale", "distractor", "other")


class Fact(NamedTuple):
    entity: str
    value: str
    char_start: int


def parse_facts(context: str) -> List[Fact]:
    """All fact statements in reading order."""
    return [Fact(m.group(1), m.group(2), m.start()) for m in FACT_RE.finditer(context)]


def parse_query(query: str) -> Optional[str]:
    m = QUERY_RE.match(query)
    return m.group(1) if m else None


def value_history(facts: List[Fact]) -> "OrderedDict[str, List[str]]":
    """entity -> its values in the order they were stated (last = current)."""
    hist: "OrderedDict[str, List[str]]" = OrderedDict()
    for f in facts:
        hist.setdefault(f.entity, []).append(f.value)
    return hist


def normalize(prediction: str) -> str:
    """Same rule as score.py: strip surrounding whitespace only, case-sensitive."""
    return prediction.strip()


def classify_prediction(prediction: str, record: Dict) -> str:
    """One of LABELS, using only the record's metadata."""
    p = normalize(prediction)
    if p == record["answer"]:
        return "correct"
    if p in record["stale_values"]:
        return "stale"
    if p in record["distractor_values"]:
        return "distractor"
    return "other"


def describe_value(value: str, context: str) -> Optional[Dict]:
    """Finer detail for a predicted value: which project it belonged to,
    which statement of that project it was (0 = first), and whether it was
    that project's current value. None if the value is not in the context."""
    facts = parse_facts(context)
    hist = value_history(facts)
    for f in facts:
        if f.value == value:
            vals = hist[f.entity]
            idx = vals.index(value)
            return {"entity": f.entity, "statement_index": idx,
                    "is_current": idx == len(vals) - 1, "num_statements": len(vals)}
    return None
