#!/usr/bin/env python3
"""Shared memory-efficient attention wrapper for the refactored DiT path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
except Exception:
    xops = None


class MemoryEfficientAttention(nn.Module):
    """Attention layer that prefers xformers on CUDA and falls back to SDPA elsewhere."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim <= 0 or self.num_heads <= 0 or (self.embed_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config dim={embed_dim} heads={num_heads}")
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None and key_padding_mask.shape != (bsz, k_len):
            raise ValueError(
                f"key_padding_mask must have shape {(bsz, k_len)}, got {tuple(key_padding_mask.shape)}"
            )

        if need_weights:
            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                logits = logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            out = torch.matmul(attn, v)
            weights = attn.mean(dim=1)
        else:
            out = None
            if xops is not None and q.device.type == "cuda":
                try:
                    q_x = q.transpose(1, 2).contiguous()
                    k_x = k.transpose(1, 2).contiguous()
                    v_x = v.transpose(1, 2).contiguous()
                    attn_bias = None
                    if key_padding_mask is not None and key_padding_mask.any():
                        attn_bias = torch.zeros((bsz, q_len, k_len), device=q.device, dtype=q.dtype)
                        attn_bias = attn_bias.masked_fill(key_padding_mask[:, None, :], float("-inf"))
                    out = xops.memory_efficient_attention(
                        q_x,
                        k_x,
                        v_x,
                        attn_bias=attn_bias,
                        p=0.0,
                    )
                    out = out.transpose(1, 2).contiguous()
                except Exception:
                    out = None
            if out is None:
                attn_mask = None
                if key_padding_mask is not None:
                    attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            weights = None

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        out = self.out_proj(out)
        return out, weights
