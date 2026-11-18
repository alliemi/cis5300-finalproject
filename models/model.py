"""Shared decoder-only backbone. Only the token-mixing layer changes between runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RMSNorm, ShortConv, build_attention


@dataclass
class ModelConfig:
    vocab_size: int
    attention: str = "softmax"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    rope_base: float = 10000.0     # softmax only
    linear_chunk: int = 64         # linear only
    conv_size: int = 4             # gated_deltanet / kda
    expand_v: float = 1.0          # gated_deltanet / kda: value dim = expand_v * head dim
    kernel: str = "auto"           # gated_deltanet / kda: auto | torch | fla
    short_conv: int = 0            # all: shared causal conv before every mixer (0 = off)

    def to_dict(self):
        return asdict(self)


class SwiGLU(nn.Module):
    def __init__(self, d: int, ratio: float):
        super().__init__()
        hidden = int(math.ceil(2 / 3 * ratio * d / 64) * 64)
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class Block(nn.Module):
    """Pre-norm block. With short_conv > 0, a causal depthwise conv (residual)
    runs before the mixer, giving every architecture the same local window for
    linking a project name to the value right after it."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.conv = ShortConv(cfg.d_model, cfg.short_conv) if cfg.short_conv > 0 else None
        self.attn = build_attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x):
        h = self.norm1(x)
        if self.conv is not None:
            h = h + self.conv(h)
        x = x + self.attn(h)
        return x + self.mlp(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():   # GPT-2 style residual scaling
            if name.endswith("o_proj.weight") or name.endswith("w_out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.hidden(ids))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_mixer_params(self) -> int:
        return sum(p.numel() for b in self.blocks for p in b.attn.parameters())
