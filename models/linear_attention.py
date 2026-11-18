"""Vanilla causal linear attention (Katharopoulos et al., ICML 2020).

    out_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)
    S_t   = sum_{i<=t} phi(k_i) v_i^T        (one d_k x d_v memory per head)
    z_t   = sum_{i<=t} phi(k_i)
    phi(x) = elu(x) + 1

Every token adds into the memory; nothing is overwritten or forgotten.

Two implementations of the same math:
  linear_attention_chunked    used for training: chunks of C tokens use a
                              small masked matrix product inside the chunk
                              plus the memory carried in from earlier chunks
  linear_attention_recurrent  token by token, slow but obviously correct;
                              used by the tests to check the chunked version
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1


def linear_attention_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """q, k: [B, H, T, d_k] (already feature-mapped), v: [B, H, T, d_v]."""
    B, H, T, dk = q.shape
    S = q.new_zeros(B, H, dk, v.shape[-1])
    z = q.new_zeros(B, H, dk)
    out = []
    for t in range(T):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + k[:, :, t]
        num = torch.einsum("bhd,bhde->bhe", q[:, :, t], S)
        den = (q[:, :, t] * z).sum(-1, keepdim=True)
        out.append(num / (den + eps))
    return torch.stack(out, dim=2)


def linear_attention_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             chunk_size: int = 64, eps: float = 1e-6) -> torch.Tensor:
    """Same result as `linear_attention_recurrent`, computed chunk by chunk."""
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        # Zero keys/values add nothing to the memory; padded outputs are dropped.
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    N = (T + pad) // C
    qc, kc, vc = (t.reshape(B, H, N, C, -1) for t in (q, k, v))

    # Memory carried into each chunk = sum over all earlier chunks.
    kv = kc.transpose(-1, -2) @ vc                                   # [B,H,N,dk,dv]
    S_before = torch.cumsum(kv, dim=2) - kv
    ksum = kc.sum(dim=3)                                             # [B,H,N,dk]
    z_before = torch.cumsum(ksum, dim=2) - ksum

    # Within a chunk, position t sees positions <= t.
    mask = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    scores = (qc @ kc.transpose(-1, -2)).masked_fill(~mask, 0.0)     # [B,H,N,C,C]

    num = scores @ vc + qc @ S_before                                # [B,H,N,C,dv]
    den = scores.sum(-1) + (qc * z_before.unsqueeze(3)).sum(-1)      # [B,H,N,C]
    out = num / (den.unsqueeze(-1) + eps)
    return out.reshape(B, H, N * C, dv)[:, :, :T]


class LinearAttention(nn.Module):
    """Multi-head vanilla linear attention. Memory sums run in float32."""

    requires_cuda = False

    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, eps: float = 1e-6):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.eps = eps
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            o = linear_attention_chunked(feature_map(q), feature_map(k), v, self.chunk_size, self.eps)
        o = o.to(x.dtype).transpose(1, 2).reshape(B, T, D)
        return self.out(o)
