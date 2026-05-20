#!/usr/bin/env python3
"""Shared SDPA attention wrapper for the refactored DiT path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class _HeadRMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.hidden_dim = int(hidden_dim)
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """JiT-style 2D RoPE for tensors shaped [B, H, T, Dh]."""

    def __init__(
        self,
        dim: int,
        *,
        pt_seq_len: int,
        ft_seq_len: int | None = None,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.pt_seq_len = int(pt_seq_len)
        self.ft_seq_len = self.pt_seq_len if ft_seq_len is None else int(ft_seq_len)
        if self.dim <= 0:
            raise ValueError(f"RoPE dim must be positive, got {dim}")
        if self.pt_seq_len <= 0 or self.ft_seq_len <= 0:
            raise ValueError(
                f"RoPE sequence lengths must be positive, got pt={pt_seq_len} ft={ft_seq_len}"
            )
        freqs = 1.0 / (
            float(theta)
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32)[: (self.dim // 2)] / float(self.dim))
        )
        positions = torch.arange(self.ft_seq_len, dtype=torch.float32) / float(self.ft_seq_len) * self.pt_seq_len
        freqs_1d = torch.einsum("i,j->ij", positions, freqs).repeat_interleave(2, dim=-1)
        freqs_h = freqs_1d[:, None, :].expand(self.ft_seq_len, self.ft_seq_len, self.dim)
        freqs_w = freqs_1d[None, :, :].expand(self.ft_seq_len, self.ft_seq_len, self.dim)
        freqs_2d = torch.cat((freqs_h, freqs_w), dim=-1).reshape(self.ft_seq_len * self.ft_seq_len, -1)
        self.register_buffer("freqs_cos", freqs_2d.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs_2d.sin(), persistent=False)

    @property
    def base_seq_len(self) -> int:
        return int(self.freqs_cos.size(0))

    @property
    def rot_dim(self) -> int:
        return int(self.freqs_cos.size(-1))

    def _position_buffers(
        self,
        token_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = int(token_count)
        if token_count == self.base_seq_len:
            cos = self.freqs_cos
            sin = self.freqs_sin
        elif token_count > 0 and token_count % self.base_seq_len == 0:
            repeat_count = token_count // self.base_seq_len
            cos = self.freqs_cos.repeat(repeat_count, 1)
            sin = self.freqs_sin.repeat(repeat_count, 1)
        else:
            raise ValueError(
                "RoPE token count must match the 2D grid or a whole-number repetition of it: "
                f"got {token_count}, base={self.base_seq_len}"
            )
        return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 4:
            raise ValueError(f"RoPE input must be 4D [B, H, T, D], got {tuple(t.shape)}")
        if self.rot_dim > t.size(-1):
            raise ValueError(f"RoPE rot_dim {self.rot_dim} exceeds head dim {t.size(-1)}")
        cos, sin = self._position_buffers(t.size(-2), device=t.device, dtype=t.dtype)
        cos = cos.view(1, 1, cos.size(0), cos.size(1))
        sin = sin.view(1, 1, sin.size(0), sin.size(1))
        rot = t[..., : self.rot_dim]
        right = t[..., self.rot_dim :]
        rot = (rot * cos) + (_rotate_half(rot) * sin)
        if right.numel() == 0:
            return rot
        return torch.cat((rot, right), dim=-1)


def enable_torch_sdpa_backends() -> None:
    """Force flash-only SDPA backend selection."""
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    for fn_name, enabled in (
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", False),
        ("enable_cudnn_sdp", False),
    ):
        fn = getattr(cuda_backends, fn_name, None)
        if callable(fn):
            fn(enabled)


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
        self.q_norm = _HeadRMSNorm(self.head_dim)
        self.k_norm = _HeadRMSNorm(self.head_dim)

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        projected_query = self.q_proj(query)
        return self.prepare_projected_query(projected_query)

    def prepare_projected_query(self, projected_query: torch.Tensor) -> torch.Tensor:
        if projected_query.dim() != 3:
            raise ValueError(f"projected_query must be 3D [B, T, D], got {tuple(projected_query.shape)}")
        bsz, q_len, embed_dim = projected_query.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"projected_query embed dim mismatch: expected {self.embed_dim}, got {embed_dim}"
            )
        q = projected_query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        return self.q_norm(q)

    def project_key_value(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, k_len, _ = key.shape
        v_bsz, v_len, _ = value.shape
        if (bsz, k_len) != (v_bsz, v_len):
            raise ValueError(
                "key/value shape mismatch before projection: "
                f"key={tuple(key.shape)} value={tuple(value.shape)}"
            )
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        return self.k_norm(k), v

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_valid_mask: Optional[torch.Tensor] = None,
        query_rope: Optional[nn.Module] = None,
        key_rope: Optional[nn.Module] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError(
                "projected q/k/v must be 4D tensors shaped [B, H, T, D], got "
                f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
            )
        if need_weights:
            raise RuntimeError("Flash-only SDPA does not support returning attention weights in this project.")
        if query.size(0) != key.size(0) or key.size(0) != value.size(0):
            raise ValueError(
                "projected q/k/v batch mismatch: "
                f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
            )
        if query.size(1) != self.num_heads or key.size(1) != self.num_heads or value.size(1) != self.num_heads:
            raise ValueError(
                "projected q/k/v head mismatch: "
                f"expected {self.num_heads}, got {query.size(1)}, {key.size(1)}, {value.size(1)}"
            )
        if query.size(3) != self.head_dim or key.size(3) != self.head_dim or value.size(3) != self.head_dim:
            raise ValueError(
                "projected q/k/v head_dim mismatch: "
                f"expected {self.head_dim}, got {query.size(3)}, {key.size(3)}, {value.size(3)}"
            )
        if query_rope is not None:
            query = query_rope(query)
        if key_rope is not None:
            key = key_rope(key)
        attn_mask = None
        if key_valid_mask is not None:
            expected_mask_shape = (key.size(0), key.size(2))
            if key_valid_mask.shape != expected_mask_shape:
                raise ValueError(
                    "key_valid_mask shape mismatch: "
                    f"expected {expected_mask_shape}, got {tuple(key_valid_mask.shape)}"
                )
            key_valid_mask = key_valid_mask.to(device=key.device, dtype=torch.bool)
            if bool((~key_valid_mask).all(dim=1).any()):
                raise ValueError("key_valid_mask must keep at least one key per attention batch")
            attn_mask = key_valid_mask[:, None, None, :]
        kernel_context = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if attn_mask is None
            else sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        )
        with kernel_context:
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        weights = None
        out = out.transpose(1, 2).contiguous().view(query.size(0), query.size(2), self.embed_dim)
        out = self.out_proj(out)
        return out, weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_rope: Optional[nn.Module] = None,
        key_rope: Optional[nn.Module] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.project_query(query)
        k, v = self.project_key_value(key, value)

        if key_padding_mask is not None and key_padding_mask.shape != (bsz, k_len):
            raise ValueError(
                f"key_padding_mask must have shape {(bsz, k_len)}, got {tuple(key_padding_mask.shape)}"
            )

        key_valid_mask = None
        if key_padding_mask is not None:
            key_valid_mask = ~key_padding_mask.to(device=key.device, dtype=torch.bool)
        return self.attend_projected(
            q,
            k,
            v,
            key_valid_mask=key_valid_mask,
            query_rope=query_rope,
            key_rope=key_rope,
            need_weights=need_weights,
        )


enable_torch_sdpa_backends()
