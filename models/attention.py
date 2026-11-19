"""Token-mixing layers, swappable by name. Every layer maps (B, T, D) -> (B, T, D)
and is causal: the output at position t depends only on inputs at positions <= t.

    softmax   causal softmax attention with RoPE (strong baseline)
    linear    linear attention, phi = elu + 1 (Katharopoulos et al., 2020) (simple baseline)
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(t):
    """Upcast to at least float32 (keeps float64 for tests)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def _full_precision(fn):
    """Run a recurrence with autocast off. Upcasting with _up() is not enough on
    its own: under bf16 autocast, `@` and einsum still compute in bf16."""
    @functools.wraps(fn)
    def wrapper(q, *args, **kwargs):
        with torch.autocast(device_type=q.device.type, enabled=False):
            return fn(q, *args, **kwargs)
    return wrapper


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = _up(x)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * _up(self.weight)).to(x.dtype)


# ---------------------------------------------------------------- softmax

def _rope(x, base: float):
    """Rotary position embedding (rotate-half form). x: (B, H, T, Dh)."""
    T, Dh = x.shape[-2], x.shape[-1]
    half = Dh // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv_freq[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = _up(x[..., :half]), _up(x[..., half:])
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class SoftmaxAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, D))


# ---------------------------------------------------------------- linear

@_full_precision
def linear_attention_recurrent(q, k, v, eps: float = 1e-6):
    """Reference: S_t = S_{t-1} + k_t v_t^T, z_t = z_{t-1} + k_t,
    o_t = q_t^T S_t / (q_t . z_t). Inputs (B, H, T, D) with q, k already feature-mapped."""
    dtype = v.dtype
    q, k, v = _up(q), _up(k), _up(v)
    B, H, T, Dk = q.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1])
    z = q.new_zeros(B, H, Dk)
    out = []
    for t in range(T):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + k[:, :, t]
        num = torch.einsum("bhk,bhkv->bhv", q[:, :, t], S)
        den = (q[:, :, t] * z).sum(-1, keepdim=True)
        out.append(num / (den + eps))
    return torch.stack(out, dim=2).to(dtype)


@_full_precision
def linear_attention_chunked(q, k, v, chunk: int = 64, eps: float = 1e-6):
    """Same result as `linear_attention_recurrent`, computed in parallel.

    Within a chunk: masked quadratic attention. Across chunks: running sums of
    k v^T and k carried from previous chunks. Memory is O(T * C + (T/C) * Dk * Dv).
    """
    dtype = v.dtype
    q, k, v = _up(q), _up(k), _up(v)
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:  # zero-padded keys contribute nothing; padded queries are dropped
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    N = q.shape[2] // chunk
    q, k, v = (t.view(B, H, N, chunk, -1) for t in (q, k, v))

    mask = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    A = (q @ k.transpose(-1, -2)).masked_fill(~mask, 0.0)          # (B,H,N,C,C)
    num = A @ v
    den = A.sum(-1)

    kv = k.transpose(-1, -2) @ v                                    # (B,H,N,Dk,Dv)
    S_prev = kv.cumsum(2) - kv                                      # exclusive prefix sum
    ksum = k.sum(-2)                                                # (B,H,N,Dk)
    z_prev = ksum.cumsum(2) - ksum
    num = num + q @ S_prev
    den = den + (q * z_prev.unsqueeze(-2)).sum(-1)

    o = num / (den.unsqueeze(-1) + eps)
    return o.reshape(B, H, N * chunk, Dv)[:, :, :T].to(dtype)


class LinearAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.chunk = cfg.linear_chunk
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        o = linear_attention_chunked(q, k, v, chunk=self.chunk)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, D))


# ---------------------------------------------------------------- short conv

class ShortConv(nn.Module):
    """Causal depthwise conv over time (the optional shared --short-conv)."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=False)

    def forward(self, x):                         # (B, T, C)
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)


ATTENTION = {
    "softmax": SoftmaxAttention,
    "linear": LinearAttention,
}


def build_attention(cfg) -> nn.Module:
    try:
        return ATTENTION[cfg.attention](cfg)
    except KeyError:
        raise ValueError(f"unknown attention {cfg.attention!r}; choose from {sorted(ATTENTION)}") from None
