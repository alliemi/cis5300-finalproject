"""Word-level tokenizer used to measure (and optionally train on) the data.

One token = one word or one punctuation mark. For example
"Project Falcon status: shipped." -> ["Project", "Falcon", "status", ":", "shipped", "."]
"""

from __future__ import annotations

import re
from typing import Iterable, List

from . import wordlists as W

TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
_NO_SPACE_BEFORE = {".", ",", ":", "?"}
SPECIAL_TOKENS = ("<pad>", "<bos>", "<sep>", "<eos>")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def render(tokens: Iterable[str]) -> str:
    """Join tokens into text. Inverse of `tokenize` for our vocabulary."""
    out: List[str] = []
    for tok in tokens:
        if out and tok not in _NO_SPACE_BEFORE:
            out.append(" ")
        out.append(tok)
    return "".join(out)


def build_vocab() -> List[str]:
    """Special tokens followed by every token the generator can produce, sorted."""
    toks = set(W.NAMES) | set(W.VALUES)
    for t in W.FACT_TEMPLATES + (W.QUERY_TEMPLATE,):
        toks.update(w for w in t if "{" not in w)
    for phrase in W.filler_phrases():
        toks.update(tokenize(phrase))
    return list(SPECIAL_TOKENS) + sorted(toks)
