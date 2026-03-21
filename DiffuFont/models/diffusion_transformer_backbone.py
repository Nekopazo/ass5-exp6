#!/usr/bin/env python3
"""Latent DiT backbone for content+style glyph flow generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdpa_attention import SDPAAttention


def _build_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, dtype=torch.float32) / float(max(1, half_dim))
    omega = 1.0 / (10000 ** omega)
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
        use_content_cross_attn: bool = True,
        use_style_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_content_cross_attn = bool(use_content_cross_attn)
        self.use_style_cross_attn = bool(use_style_cross_attn)
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_content = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_cross_attn
            else None
        )
        self.norm_style = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_style_cross_attn
            else None
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.content_attn = SDPAAttention(hidden_dim, num_heads) if self.use_content_cross_attn else None
        self.style_attn = SDPAAttention(hidden_dim, num_heads) if self.use_style_cross_attn else None
        self.mlp = FeedForward(hidden_dim, mlp_ratio)
        self.style_residual_gate = (
            nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            if self.use_style_cross_attn
            else None
        )

        self.modulation_chunk_count = 6
        if self.use_content_cross_attn:
            self.modulation_chunk_count += 3
        if self.use_style_cross_attn:
            self.modulation_chunk_count += 3

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.modulation_chunk_count),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        *,
        cond: torch.Tensor,
        content_tokens: torch.Tensor | None,
        style_tokens: torch.Tensor | None,
        style_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation_chunks = self.modulation(cond).chunk(self.modulation_chunk_count, dim=-1)
        chunk_idx = 0

        shift_self = modulation_chunks[chunk_idx]
        scale_self = modulation_chunks[chunk_idx + 1]
        gate_self = modulation_chunks[chunk_idx + 2]
        chunk_idx += 3

        shift_content = scale_content = gate_content = None
        if self.use_content_cross_attn:
            shift_content = modulation_chunks[chunk_idx]
            scale_content = modulation_chunks[chunk_idx + 1]
            gate_content = modulation_chunks[chunk_idx + 2]
            chunk_idx += 3

        shift_style = scale_style = gate_style = None
        if self.use_style_cross_attn:
            shift_style = modulation_chunks[chunk_idx]
            scale_style = modulation_chunks[chunk_idx + 1]
            gate_style = modulation_chunks[chunk_idx + 2]
            chunk_idx += 3

        shift_mlp = modulation_chunks[chunk_idx]
        scale_mlp = modulation_chunks[chunk_idx + 1]
        gate_mlp = modulation_chunks[chunk_idx + 2]

        x = latent_tokens
        q = modulate(self.norm_self(x), shift_self, scale_self)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + gate_self.unsqueeze(1) * self_out

        if self.use_content_cross_attn:
            if content_tokens is None or self.norm_content is None or self.content_attn is None:
                raise RuntimeError("content_tokens must be provided when content cross-attention is enabled")
            q = modulate(self.norm_content(x), shift_content, scale_content)
            content_out, _ = self.content_attn(q, content_tokens, content_tokens, need_weights=False)
            x = x + gate_content.unsqueeze(1) * content_out

        if self.use_style_cross_attn:
            if style_tokens is None or self.norm_style is None or self.style_attn is None or self.style_residual_gate is None:
                raise RuntimeError("style_tokens must be provided when style cross-attention is enabled")
            q = modulate(self.norm_style(x), shift_style, scale_style)
            key_padding_mask = None
            if style_token_mask is not None:
                key_padding_mask = ~style_token_mask.bool()
            style_out, _ = self.style_attn(
                q,
                style_tokens,
                style_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            style_gate = gate_style.unsqueeze(1) + self.style_residual_gate.to(device=x.device, dtype=x.dtype)
            x = x + style_gate * style_out

        mlp_out = self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiffusionTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int = 4,
        latent_size: int = 16,
        hidden_dim: int = 512,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        content_cross_attn_layers: int | None = None,
        style_cross_attn_every_n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.num_tokens = self.latent_size * self.latent_size
        if content_cross_attn_layers is None:
            content_cross_attn_layers = self.depth
        self.content_cross_attn_layers = max(0, min(self.depth, int(content_cross_attn_layers)))
        self.style_cross_attn_every_n_layers = max(1, int(style_cross_attn_every_n_layers))
        self.has_content_cross_attn = self.content_cross_attn_layers > 0
        self.has_style_cross_attn = any(
            block_idx % self.style_cross_attn_every_n_layers == 0
            for block_idx in range(self.depth)
        )

        self.latent_proj = nn.Linear(self.latent_channels, self.hidden_dim)
        pos_embed = build_2d_sincos_pos_embed(self.hidden_dim, self.latent_size, self.latent_size)
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
                    use_content_cross_attn=(block_idx < self.content_cross_attn_layers),
                    use_style_cross_attn=(block_idx % self.style_cross_attn_every_n_layers == 0),
                )
            )
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.final_proj = nn.Linear(self.hidden_dim, self.latent_channels)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        style_global: torch.Tensor,
        style_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.dim() != 4:
            raise ValueError(f"latent must be 4D, got {tuple(latent.shape)}")
        if tuple(latent.shape[1:]) != (self.latent_channels, self.latent_size, self.latent_size):
            raise ValueError(
                "latent shape mismatch: "
                f"expected (*, {self.latent_channels}, {self.latent_size}, {self.latent_size}), "
                f"got {tuple(latent.shape)}"
            )

        x = latent.flatten(2).transpose(1, 2).contiguous()
        x = self.latent_proj(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        t_embed = timestep_embedding(timesteps, self.hidden_dim).to(dtype=x.dtype)
        t_embed = self.time_mlp(t_embed)
        cond = t_embed + self.style_cond_proj(style_global.to(dtype=x.dtype))

        content_tokens = (
            content_tokens.to(device=x.device, dtype=x.dtype)
            if self.has_content_cross_attn
            else None
        )
        style_tokens = (
            style_tokens.to(device=x.device, dtype=x.dtype)
            if self.has_style_cross_attn
            else None
        )
        if self.has_style_cross_attn and style_token_mask is not None:
            style_token_mask = style_token_mask.to(device=x.device, dtype=torch.bool)

        for block in self.blocks:
            x = block(
                x,
                cond=cond,
                content_tokens=content_tokens if block.use_content_cross_attn else None,
                style_tokens=style_tokens if block.use_style_cross_attn else None,
                style_token_mask=style_token_mask if block.use_style_cross_attn else None,
            )

        x = self.final_norm(x)
        x = self.final_proj(x)
        x = x.transpose(1, 2).contiguous().view(
            latent.size(0),
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        return x
