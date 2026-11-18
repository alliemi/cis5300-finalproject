"""Shared model body. Every architecture uses this; only the mixer differs.

    token embedding
    N x Block:  x = x + mixer(shortconv(norm(x)))
                x = x + mlp(norm(x))
    final norm, output layer tied to the embedding

The short causal convolution (window 4) lets every model link a project
name to the value a few tokens later, so differences between models come
from long-range memory, not from local word linking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class ShortConv(nn.Module):
    """Causal depthwise convolution with a residual: y = x + conv(x).
    Position t only sees positions t-k+1 .. t."""

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        y = self.conv(x.transpose(1, 2))[..., :T]
        return x + y.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: int):
        super().__init__()
        self.up = nn.Linear(d_model, ratio * d_model)
        self.down = nn.Linear(ratio * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, mixer: nn.Module, mlp_ratio: int, short_conv: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.conv = ShortConv(d_model, short_conv) if short_conv > 0 else nn.Identity()
        self.mixer = mixer
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.conv(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class RetrievalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, mixers,
                 mlp_ratio: int = 4, short_conv: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, m, mlp_ratio, short_conv) for m in mixers)
        assert len(self.blocks) == n_layers
        self.norm = RMSNorm(d_model)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits [B, T, V], or [B, V] at one position per row if `positions` is given."""
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        if positions is not None:
            x = x[torch.arange(x.shape[0], device=x.device), positions]
        return self.norm(x) @ self.embed.weight.t()
