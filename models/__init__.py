"""Model registry.

To add an attention architecture ("mixer"):
  1. Write an nn.Module in models/<name>.py, built as Mixer(d_model, n_heads, **mixer_cfg),
     with forward(x: [B, T, D]) -> [B, T, D]. It must be causal: position t
     may only use positions <= t. Set `requires_cuda = True` if it only runs on
     NVIDIA GPUs (e.g. Triton kernels).
  2. Import it below and add it to MIXERS.
  3. Add configs/<name>.json with {"base": "base.json", "mixer": "<name>", "mixer_cfg": {...}}.
  4. Run the tests: the contract tests in tests/test_harness.py check every mixer in MIXERS.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import RetrievalLM
from .linear_attention import LinearAttention
from .no_mixer import NoMixer

MIXERS: Dict[str, type] = {
    "none": NoMixer,
    "linear": LinearAttention,
}


def build_model(config: Dict, vocab_size: int) -> RetrievalLM:
    name = config["mixer"]
    if name not in MIXERS:
        raise ValueError(f"unknown mixer {name!r}; registered: {sorted(MIXERS)}")
    m = config["model"]
    mixers = [MIXERS[name](m["d_model"], m["n_heads"], **config.get("mixer_cfg", {}))
              for _ in range(m["n_layers"])]
    return RetrievalLM(vocab_size, m["d_model"], m["n_layers"], mixers,
                       mlp_ratio=m["mlp_ratio"], short_conv=m["short_conv"])


def check_device(config: Dict, device: torch.device) -> None:
    if getattr(MIXERS[config["mixer"]], "requires_cuda", False) and device.type != "cuda":
        raise RuntimeError(f"mixer {config['mixer']!r} needs an NVIDIA GPU (CUDA); "
                           f"this machine has {device.type}. Run it on Colab.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
