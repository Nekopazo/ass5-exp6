#!/usr/bin/env python3
"""Pixel-space DiT backbone for content+style glyph flow generation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdpa_attention import SDPAAttention


def _build_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, dtype=torch.float32) / float(max(1, half_dim))
    omega = 1.0 / (10000**omega)
    out = positions.reshape(-1, 1).float() * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    ys = torch.arange(grid_h, dtype=torch.float32)
    xs = torch.arange(grid_w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    emb_h = _build_1d_sincos_pos_embed(embed_dim // 2, grid_y.reshape(-1))
    emb_w = _build_1d_sincos_pos_embed(embed_dim // 2, grid_x.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / float(max(1, half))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlyphDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        content_cross_attn_heads: int,
        mlp_ratio: float,
        *,
        use_content_cross_attn: bool = True,
        use_style_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.use_content_cross_attn = bool(use_content_cross_attn)
        self.use_style_modulation = bool(use_style_modulation)
        self.norm_content_query = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_cross_attn
            else None
        )
        self.norm_content_kv = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_cross_attn
            else None
        )
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.content_cross_attn = (
            SDPAAttention(hidden_dim, content_cross_attn_heads) if self.use_content_cross_attn else None
        )
        self.content_modulation = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 3),
            )
            if self.use_content_cross_attn
            else None
        )
        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio)

        self.modulation_chunk_count = 6

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.modulation_chunk_count),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        if self.content_modulation is not None:
            nn.init.zeros_(self.content_modulation[-1].weight)
            nn.init.zeros_(self.content_modulation[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        self_cond: torch.Tensor,
        content_time_cond: torch.Tensor | None,
        content_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation_chunks = self.modulation(self_cond).chunk(self.modulation_chunk_count, dim=-1)
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = modulation_chunks

        x = patch_tokens
        if self.use_content_cross_attn:
            if (
                content_time_cond is None
                or content_tokens is None
                or self.norm_content_query is None
                or self.norm_content_kv is None
                or self.content_cross_attn is None
                or self.content_modulation is None
            ):
                raise RuntimeError(
                    "content_time_cond and content_tokens must be provided when content cross-attention is enabled"
                )
            if content_tokens.shape[:2] != x.shape[:2]:
                raise RuntimeError(
                    "content token shape mismatch: "
                    f"expected {tuple(x.shape[:2])}, got {tuple(content_tokens.shape[:2])}"
                )
            shift_content, scale_content, gate_content = self.content_modulation(content_time_cond).chunk(3, dim=-1)
            content_out, _ = self.content_cross_attn(
                modulate(self.norm_content_query(x), shift_content, scale_content),
                self.norm_content_kv(content_tokens),
                self.norm_content_kv(content_tokens),
                need_weights=False,
            )
            x = x + gate_content.unsqueeze(1) * content_out

        q = modulate(self.norm_self(x), shift_self, scale_self)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + gate_self.unsqueeze(1) * self_out

        mlp_out = self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiffusionTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        hidden_dim: int = 512,
        depth: int = 16,
        num_heads: int = 8,
        content_cross_attn_heads: int | None = None,
        mlp_ratio: float = 4.0,
        content_cross_attn_layers: Sequence[int] | None = None,
        style_modulation_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.content_cross_attn_heads = (
            self.num_heads if content_cross_attn_heads is None else int(content_cross_attn_heads)
        )
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.content_cross_attn_layers = self._normalize_layer_indices(
            content_cross_attn_layers,
            default_layers=range(1, min(self.depth, 6) + 1),
            depth=self.depth,
            field_name="content_cross_attn_layers",
        )
        self.style_modulation_layers = self._normalize_layer_indices(
            style_modulation_layers,
            default_layers=range(max(1, self.depth - 5), self.depth + 1),
            depth=self.depth,
            field_name="style_modulation_layers",
        )
        content_layer_set = set(self.content_cross_attn_layers)
        style_layer_set = set(self.style_modulation_layers)

        self.content_layer_mask = [
            (block_idx + 1) in content_layer_set
            for block_idx in range(self.depth)
        ]
        self.style_layer_mask = [
            (block_idx + 1) in style_layer_set
            for block_idx in range(self.depth)
        ]
        self.has_content_cross_attn = any(self.content_layer_mask)

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        pos_embed = build_2d_sincos_pos_embed(self.hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.style_cond_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.blocks = nn.ModuleList()
        for block_idx in range(self.depth):
            self.blocks.append(
                GlyphDiTBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.content_cross_attn_heads,
                    mlp_ratio,
                    use_content_cross_attn=self.content_layer_mask[block_idx],
                    use_style_modulation=self.style_layer_mask[block_idx],
                )
            )
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

    @staticmethod
    def _normalize_layer_indices(
        layers: Sequence[int] | None,
        *,
        default_layers: Sequence[int],
        depth: int,
        field_name: str,
    ) -> tuple[int, ...]:
        raw_layers = default_layers if layers is None else layers
        if isinstance(raw_layers, (str, bytes)):
            raise TypeError(f"{field_name} must be a sequence of integers, got {type(raw_layers).__name__}")
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_idx in raw_layers:
            layer_idx = int(raw_idx)
            if layer_idx < 1 or layer_idx > int(depth):
                raise ValueError(f"{field_name} entries must be in [1, {depth}], got {layer_idx}")
            if layer_idx in seen:
                continue
            normalized.append(layer_idx)
            seen.add(layer_idx)
        return tuple(normalized)

    def forward(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_global: torch.Tensor,
    ) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"image must be 4D, got {tuple(image.shape)}")
        expected_shape = (self.in_channels, self.image_size, self.image_size)
        if tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                "image shape mismatch: "
                f"expected (*, {self.in_channels}, {self.image_size}, {self.image_size}), "
                f"got {tuple(image.shape)}"
            )

        x = self.patch_embed(image).flatten(2).transpose(1, 2).contiguous()
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        time_cond = timestep_embedding(timesteps, self.hidden_dim).to(dtype=x.dtype)
        time_cond = self.time_mlp(time_cond)
        style_cond = self.style_cond_proj(style_global.to(device=x.device, dtype=x.dtype))

        if self.has_content_cross_attn:
            content_tokens = content_tokens.to(device=x.device, dtype=x.dtype)
            content_tokens = content_tokens + self.pos_embed.to(device=x.device, dtype=x.dtype)
        else:
            content_tokens = None

        for block in self.blocks:
            self_cond = time_cond
            if block.use_style_modulation:
                self_cond = self_cond + style_cond
            x = block(
                x,
                self_cond=self_cond,
                content_time_cond=time_cond if block.use_content_cross_attn else None,
                content_tokens=content_tokens if block.use_content_cross_attn else None,
            )

        return self.final_norm(x)
