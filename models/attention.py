"""Token-mixing layers, swappable by name. Every layer maps (B, T, D) -> (B, T, D)
and is causal: the output at position t depends only on inputs at positions <= t.

    softmax         causal softmax attention with RoPE (strong baseline)
    linear          linear attention, phi = elu + 1 (Katharopoulos et al., 2020)
    gated_deltanet  Gated DeltaNet (Yang et al., 2025): scalar decay per head + delta rule
    kda             Kimi Delta Attention (Kimi Team, 2025): per-channel decay + delta rule

The two delta-rule layers share one module; they differ only in the decay
gate and the output gate. Their recurrence runs either in pure PyTorch
(`delta_rule_recurrent`, a slow, exact reference used on CPU and in tests) or
in the flash-linear-attention (fla) Triton kernels on CUDA.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import Optional

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


# ---------------------------------------------------------------- delta rule

@_full_precision
def delta_rule_recurrent(q, k, v, g, beta, scale: Optional[float] = None):
    """Reference gated delta rule, step by step, in float32.

    Shapes: q, k (B, T, H, K); v (B, T, H, V); beta (B, T, H);
    g is the log-decay, (B, T, H) for Gated DeltaNet or (B, T, H, K) for KDA.

        S_t = Diag(exp(g_t)) S_{t-1}                       (decay; scalar for GDN)
        S_t = S_t + beta_t k_t (v_t - S_t^T k_t)^T         (delta-rule write)
        o_t = S_t^T (scale * q_t)

    Equivalent to S_t = S_{t-1} a_t (I - b_t k_t k_t^T) + b_t v_t k_t^T in the
    papers' notation (state transposed). Matches fla's convention.
    """
    dtype = v.dtype
    q, k, v, g, beta = (_up(t) for t in (q, k, v, g, beta))
    B, T, H, K = q.shape
    scale = K ** -0.5 if scale is None else scale
    if g.dim() == 3:
        g = g.unsqueeze(-1)                       # broadcast over key channels
    decay = g.exp().unsqueeze(-1)                 # (B, T, H, K|1, 1)
    S = q.new_zeros(B, H, K, v.shape[-1])
    out = []
    for t in range(T):
        S = S * decay[:, t]
        kt = k[:, t]
        v_old = torch.einsum("bhk,bhkv->bhv", kt, S)
        u = (v[:, t] - v_old) * beta[:, t, :, None]
        S = S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, dim=1).to(dtype)


def _fla_kernel(per_channel: bool):
    try:
        if per_channel:
            from fla.ops.kda import chunk_kda as fn
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn
        return fn
    except Exception:  # ImportError, or Triton problems at import time
        return None


def delta_rule_fla(q, k, v, g, beta, scale, per_channel: bool):
    fn = _fla_kernel(per_channel)
    if fn is None:
        raise RuntimeError("flash-linear-attention is not installed; `pip install flash-linear-attention`")
    out = fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale, output_final_state=False)
    return out[0] if isinstance(out, tuple) else out


class ShortConv(nn.Module):
    """Causal depthwise conv over time, as used by GDN / KDA before q, k, v."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=False)

    def forward(self, x):                         # (B, T, C)
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)


class DeltaRuleAttention(nn.Module):
    """Gated DeltaNet (per_channel=False) or Kimi Delta Attention (per_channel=True)."""

    def __init__(self, cfg, per_channel: bool):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        self.h, self.dk = H, D // H
        self.dv = int(self.dk * cfg.expand_v)
        self.per_channel = per_channel
        self.kernel = cfg.kernel
        key_dim, val_dim = H * self.dk, H * self.dv

        self.q_proj = nn.Linear(D, key_dim, bias=False)
        self.k_proj = nn.Linear(D, key_dim, bias=False)
        self.v_proj = nn.Linear(D, val_dim, bias=False)
        self.q_conv = ShortConv(key_dim, cfg.conv_size)
        self.k_conv = ShortConv(key_dim, cfg.conv_size)
        self.v_conv = ShortConv(val_dim, cfg.conv_size)
        self.b_proj = nn.Linear(D, H, bias=False)

        n_decay = key_dim if per_channel else H
        if per_channel:   # KDA: low-rank projection to one decay per key channel
            self.a_proj = nn.Sequential(nn.Linear(D, self.dk, bias=False),
                                        nn.Linear(self.dk, key_dim, bias=False))
            self.g_proj = nn.Sequential(nn.Linear(D, self.dv, bias=False),
                                        nn.Linear(self.dv, val_dim, bias=False))
        else:             # GDN: one decay per head, full-rank output gate
            self.a_proj = nn.Linear(D, H, bias=False)
            self.g_proj = nn.Linear(D, val_dim, bias=False)
        # Mamba-style decay parameterization: g = -exp(A_log) * softplus(a + dt_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt = torch.exp(torch.empty(n_decay).uniform_(math.log(1e-3), math.log(1e-1))).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        self.o_norm = RMSNorm(self.dv, cfg.norm_eps)
        self.o_proj = nn.Linear(val_dim, D, bias=False)

    def use_fla(self, x) -> bool:
        if self.kernel == "torch":
            return False
        if self.kernel == "fla":
            return True
        ok = x.is_cuda and _fla_kernel(self.per_channel) is not None
        if x.is_cuda and not ok and not getattr(self, "_warned", False):
            warnings.warn("fla not available: falling back to the slow PyTorch delta rule on GPU")
            self._warned = True
        return ok

    def forward(self, x):
        B, T, _ = x.shape
        H, K, V = self.h, self.dk, self.dv
        q = F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, K)
        k = F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, K)
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, V)
        q, k = F.normalize(_up(q), dim=-1).to(v.dtype), F.normalize(_up(k), dim=-1).to(v.dtype)
        beta = torch.sigmoid(_up(self.b_proj(x)))                     # (B, T, H)

        a = _up(self.a_proj(x))
        A = _up(self.A_log).exp()
        if self.per_channel:
            g = -A.view(H, 1) * F.softplus(a.view(B, T, H, K) + _up(self.dt_bias).view(H, K))
        else:
            g = -A * F.softplus(a + _up(self.dt_bias))                 # (B, T, H)

        if self.use_fla(x):
            o = delta_rule_fla(q, k, v, g, beta.to(v.dtype), K ** -0.5, self.per_channel)
        else:
            o = delta_rule_recurrent(q, k, v, g, beta, K ** -0.5)

        gate = self.g_proj(x).view(B, T, H, V)
        gate = torch.sigmoid(gate) if self.per_channel else F.silu(gate)
        o = self.o_norm(o.to(x.dtype)) * gate
        return self.o_proj(o.reshape(B, T, H * V))


class GatedDeltaNet(DeltaRuleAttention):
    def __init__(self, cfg):
        super().__init__(cfg, per_channel=False)


class KimiDeltaAttention(DeltaRuleAttention):
    def __init__(self, cfg):
        super().__init__(cfg, per_channel=True)


ATTENTION = {
    "softmax": SoftmaxAttention,
    "linear": LinearAttention,
    "gated_deltanet": GatedDeltaNet,
    "kda": KimiDeltaAttention,
}


def build_attention(cfg) -> nn.Module:
    try:
        return ATTENTION[cfg.attention](cfg)
    except KeyError:
        raise ValueError(f"unknown attention {cfg.attention!r}; choose from {sorted(ATTENTION)}") from None
