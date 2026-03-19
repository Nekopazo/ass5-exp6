#!/usr/bin/env python3
"""Shared SDPA attention wrapper for the refactored DiT path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_torch_sdpa_backends() -> None:
    """Keep all torch SDPA backends available so CUDA can pick flash when valid."""
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    for fn_name in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "enable_cudnn_sdp",
    ):
        fn = getattr(cuda_backends, fn_name, None)
        if callable(fn):
            fn(True)


def describe_torch_sdpa_backends() -> str:
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return "torch_sdpa"
    parts = []
    for attr_name, label in (
        ("flash_sdp_enabled", "flash"),
        ("mem_efficient_sdp_enabled", "mem_efficient"),
        ("math_sdp_enabled", "math"),
        ("cudnn_sdp_enabled", "cudnn"),
    ):
        fn = getattr(cuda_backends, attr_name, None)
        if callable(fn):
            parts.append(f"{label}={int(bool(fn()))}")
    if not parts:
        return "torch_sdpa"
    return "torch_sdpa(" + ", ".join(parts) + ")"


class SDPAAttention(nn.Module):
    """Attention layer backed by torch.scaled_dot_product_attention."""

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
                logits = logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            out = torch.matmul(attn, v)
            weights = attn.mean(dim=1)
        else:
            attn_mask = None
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.bool()
                if key_padding_mask.any():
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


enable_torch_sdpa_backends()
