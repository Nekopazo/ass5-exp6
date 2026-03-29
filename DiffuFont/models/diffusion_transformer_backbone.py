#!/usr/bin/env python3
"""Pixel-space DiT backbone for content+style glyph flow generation."""

from __future__ import annotations

import math

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
        mlp_ratio: float,
        *,
        use_content_fusion: bool = True,
        use_style_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.use_content_fusion = bool(use_content_fusion)
        self.use_style_modulation = bool(use_style_modulation)
        self.norm_content_x = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_fusion
            else None
        )
        self.norm_content_cond = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_fusion
            else None
        )
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.content_fuse = nn.Linear(hidden_dim * 2, hidden_dim) if self.use_content_fusion else None
        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio)

        self.modulation_chunk_count = 6

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.modulation_chunk_count),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        cond: torch.Tensor,
        content_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation_chunks = self.modulation(cond).chunk(self.modulation_chunk_count, dim=-1)
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = modulation_chunks

        x = patch_tokens
        if self.use_content_fusion:
            if (
                content_tokens is None
                or self.norm_content_x is None
                or self.norm_content_cond is None
                or self.content_fuse is None
            ):
                raise RuntimeError("content_tokens must be provided when content fusion is enabled")
            if content_tokens.shape[:2] != x.shape[:2]:
                raise RuntimeError(
                    "content token shape mismatch: "
                    f"expected {tuple(x.shape[:2])}, got {tuple(content_tokens.shape[:2])}"
                )
            fused = torch.cat(
                [self.norm_content_x(x), self.norm_content_cond(content_tokens)],
                dim=-1,
            )
            x = x + self.content_fuse(fused)

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
        mlp_ratio: float = 4.0,
        content_fusion_start: int | None = None,
        content_fusion_end: int | None = None,
        style_fusion_start: int | None = None,
        style_fusion_end: int | None = None,
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
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size
        if content_fusion_start is None and content_fusion_end is None:
            self.content_fusion_start = 0
            self.content_fusion_end = min(self.depth, 6)
        else:
            self.content_fusion_start = max(
                0,
                min(self.depth, 0 if content_fusion_start is None else int(content_fusion_start)),
            )
            self.content_fusion_end = max(
                self.content_fusion_start,
                min(self.depth, self.depth if content_fusion_end is None else int(content_fusion_end)),
            )
        self.content_fusion_layers = self.content_fusion_end - self.content_fusion_start

        if style_fusion_start is None and style_fusion_end is None:
            self.style_fusion_start = max(0, self.depth - 6)
            self.style_fusion_end = self.depth
        else:
            self.style_fusion_start = max(
                0,
                min(self.depth, 0 if style_fusion_start is None else int(style_fusion_start)),
            )
            self.style_fusion_end = max(
                self.style_fusion_start,
                min(self.depth, self.depth if style_fusion_end is None else int(style_fusion_end)),
            )

        self.content_layer_mask = [
            self.content_fusion_start <= block_idx < self.content_fusion_end
            for block_idx in range(self.depth)
        ]
        self.style_layer_mask = [
            self.style_fusion_start <= block_idx < self.style_fusion_end
            for block_idx in range(self.depth)
        ]
        self.has_content_fusion = any(self.content_layer_mask)
        self.has_style_modulation = any(self.style_layer_mask)

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
                    mlp_ratio,
                    use_content_fusion=self.content_layer_mask[block_idx],
                    use_style_modulation=self.style_layer_mask[block_idx],
                )
            )
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

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

        content_tokens = content_tokens.to(device=x.device, dtype=x.dtype) if self.has_content_fusion else None

        for block in self.blocks:
            block_cond = time_cond
            if block.use_style_modulation:
                block_cond = block_cond + style_cond
            x = block(
                x,
                cond=block_cond,
                content_tokens=content_tokens if block.use_content_fusion else None,
            )

        return self.final_norm(x)
