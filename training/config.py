"""Load a model config, merging it over its base config.

configs/<model>.json names its base ("base": "base.json") and only adds the
mixer and anything that differs, so all models share one training recipe.
Command-line overrides use dotted keys: --set train.lr=3e-4
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Iterable


def _merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_merged(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    base_name = config.pop("base", None)
    if base_name:
        config = _merge(_load_merged(os.path.join(os.path.dirname(path), base_name)), config)
    return config


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict:
    config = _load_merged(path)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    for section in ("mixer", "model", "train"):
        if section not in config:
            raise ValueError(f"config is missing {section!r}")
    return config
