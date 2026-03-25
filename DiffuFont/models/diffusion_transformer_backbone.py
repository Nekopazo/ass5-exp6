#!/usr/bin/env python3
"""Pixel-space patch-level + PiT backbone for glyph generation."""

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


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * denom


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = max(hidden_dim, int(hidden_dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def modulate(x: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


def zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MultiReferenceStyleModule(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.attn = SDPAAttention(hidden_dim, num_heads)
        self.weight_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        style_tokens: torch.Tensor,
        style_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style_tokens.dim() != 4:
            raise ValueError(f"style_tokens must have shape [B, K, L, D], got {tuple(style_tokens.shape)}")
        batch_size, ref_count, token_count, hidden_dim = style_tokens.shape
        if x.shape != (batch_size, token_count, hidden_dim):
            raise ValueError(
                "style_tokens and x shape mismatch: "
                f"x={tuple(x.shape)} style={tuple(style_tokens.shape)}"
            )

        q = self.norm(x)
        q_per_ref = q.unsqueeze(1).expand(batch_size, ref_count, token_count, hidden_dim).reshape(
            batch_size * ref_count,
            token_count,
            hidden_dim,
        )
        kv = style_tokens.reshape(batch_size * ref_count, token_count, hidden_dim)
        out, _ = self.attn(q_per_ref, kv, kv, need_weights=False)
        out = out.view(batch_size, ref_count, token_count, hidden_dim).permute(0, 2, 1, 3).contiguous()

        ref_logits = self.weight_proj(out).squeeze(-1)
        if style_ref_mask is not None:
            if style_ref_mask.shape != (batch_size, ref_count):
                raise ValueError(
                    "style_ref_mask shape mismatch: "
                    f"expected {(batch_size, ref_count)}, got {tuple(style_ref_mask.shape)}"
                )
            valid_mask = style_ref_mask.to(device=ref_logits.device, dtype=torch.bool)
            ref_logits = ref_logits.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
            out = out * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=out.dtype)
        weights = torch.softmax(ref_logits, dim=-1).to(dtype=out.dtype)
        return (out * weights.unsqueeze(-1)).sum(dim=2)


class PatchConditionedBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        use_style: bool,
    ) -> None:
        super().__init__()
        self.use_style = bool(use_style)
        self.norm_attn = RMSNorm(hidden_dim)
        self.norm_mlp = RMSNorm(hidden_dim)
        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        self.style_module = MultiReferenceStyleModule(hidden_dim, num_heads) if self.use_style else None
        self.style_gate = nn.Parameter(torch.zeros(1)) if self.use_style else None
        zero_init_linear(self.cond_proj[-1])

    def forward(
        self,
        x: torch.Tensor,
        *,
        global_cond: torch.Tensor,
        style_tokens: torch.Tensor | None,
        style_ref_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = self.cond_proj(global_cond).chunk(6, dim=-1)
        attn_in = modulate(self.norm_attn(x), beta1, gamma1)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + alpha1.unsqueeze(1) * attn_out
        if self.use_style:
            if style_tokens is None or self.style_module is None or self.style_gate is None:
                raise RuntimeError("style_tokens must be provided for style-enabled patch blocks")
            style_out = self.style_module(x, style_tokens=style_tokens, style_ref_mask=style_ref_mask)
            x = x + self.style_gate.to(device=x.device, dtype=x.dtype) * style_out
        mlp_out = self.mlp(modulate(self.norm_mlp(x), beta2, gamma2))
        x = x + alpha2.unsqueeze(1) * mlp_out
        return x


class PixelTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        patch_area: int,
        patch_hidden_dim: int,
        pixel_hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        grid_size: int,
    ) -> None:
        super().__init__()
        self.patch_area = int(patch_area)
        self.patch_hidden_dim = int(patch_hidden_dim)
        self.pixel_hidden_dim = int(pixel_hidden_dim)

        self.norm_attn = RMSNorm(self.pixel_hidden_dim)
        self.norm_mlp = RMSNorm(self.pixel_hidden_dim)
        self.global_norm = RMSNorm(self.patch_hidden_dim)
        self.cond_proj = nn.Linear(self.patch_hidden_dim, self.patch_area * 6 * self.pixel_hidden_dim)
        self.compact_proj = nn.Linear(self.patch_area * self.pixel_hidden_dim, self.patch_hidden_dim)
        self.global_attn = SDPAAttention(self.patch_hidden_dim, num_heads)
        self.expand_proj = nn.Linear(self.patch_hidden_dim, self.patch_area * self.pixel_hidden_dim)
        self.mlp = FeedForward(self.pixel_hidden_dim, mlp_ratio)
        zero_init_linear(self.cond_proj)

        pos_embed = build_2d_sincos_pos_embed(self.patch_hidden_dim, grid_size, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, *, s_cond: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"pixel tokens must have shape [B, L, P, Dpix], got {tuple(x.shape)}")
        if s_cond.shape[:2] != x.shape[:2]:
            raise ValueError(f"s_cond shape mismatch: x={tuple(x.shape)} s_cond={tuple(s_cond.shape)}")

        batch_size, token_count, patch_area, pixel_hidden_dim = x.shape
        if patch_area != self.patch_area or pixel_hidden_dim != self.pixel_hidden_dim:
            raise ValueError(
                "pixel token inner shape mismatch: "
                f"expected {(self.patch_area, self.pixel_hidden_dim)}, got {(patch_area, pixel_hidden_dim)}"
            )

        x_flat = x.view(batch_size * token_count, patch_area, pixel_hidden_dim)
        cond = self.cond_proj(s_cond.reshape(batch_size * token_count, -1)).view(
            batch_size * token_count,
            patch_area,
            6,
            pixel_hidden_dim,
        )
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = cond.unbind(dim=2)

        attn_in = gamma1 * self.norm_attn(x_flat) + beta1
        compact = self.compact_proj(attn_in.reshape(batch_size, token_count, patch_area * pixel_hidden_dim))
        compact = compact + self.pos_embed.to(device=compact.device, dtype=compact.dtype)
        compact = self.global_norm(compact)
        attn_out, _ = self.global_attn(compact, compact, compact, need_weights=False)
        expand = self.expand_proj(attn_out).view(batch_size * token_count, patch_area, pixel_hidden_dim)
        x_flat = x_flat + alpha1 * expand

        mlp_in = gamma2 * self.norm_mlp(x_flat) + beta2
        mlp_out = self.mlp(mlp_in)
        x_flat = x_flat + alpha2 * mlp_out
        return x_flat.view(batch_size, token_count, patch_area, pixel_hidden_dim)


class PixelDiffusionTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        patch_size: int,
        patch_hidden_dim: int,
        patch_depth: int,
        patch_heads: int,
        patch_mlp_ratio: float,
        pixel_hidden_dim: int,
        pit_depth: int,
        pit_heads: int,
        pit_mlp_ratio: float,
        style_fusion_start: int,
        use_style_tokens: bool,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.patch_hidden_dim = int(patch_hidden_dim)
        self.patch_depth = int(patch_depth)
        self.patch_heads = int(patch_heads)
        self.pixel_hidden_dim = int(pixel_hidden_dim)
        self.pit_depth = int(pit_depth)
        self.pit_heads = int(pit_heads)

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size, got {self.image_size} and {self.patch_size}"
            )
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_area = self.patch_size * self.patch_size
        self.style_fusion_start = int(style_fusion_start)
        self.use_style_tokens = bool(use_style_tokens)
        if self.style_fusion_start < 0 or self.style_fusion_start > self.patch_depth:
            raise ValueError(
                f"style_fusion_start must be in [0, {self.patch_depth}], got {self.style_fusion_start}"
            )

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.patch_hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch_pos_embed = build_2d_sincos_pos_embed(self.patch_hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("patch_pos_embed", patch_pos_embed.unsqueeze(0), persistent=False)
        self.patch_input_proj = nn.Linear(self.patch_hidden_dim * 2, self.patch_hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim),
        )
        self.global_cond_proj = nn.Sequential(
            nn.Linear(self.patch_hidden_dim * 2, self.patch_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.patch_hidden_dim, self.patch_hidden_dim),
        )
        self.patch_blocks = nn.ModuleList(
            [
                PatchConditionedBlock(
                    self.patch_hidden_dim,
                    self.patch_heads,
                    patch_mlp_ratio,
                    use_style=(self.use_style_tokens and block_idx >= self.style_fusion_start),
                )
                for block_idx in range(self.patch_depth)
            ]
        )

        self.pixel_embed = nn.Conv2d(self.in_channels, self.pixel_hidden_dim, kernel_size=1, stride=1)
        self.pit_blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                    patch_area=self.patch_area,
                    patch_hidden_dim=self.patch_hidden_dim,
                    pixel_hidden_dim=self.pixel_hidden_dim,
                    num_heads=self.pit_heads,
                    mlp_ratio=pit_mlp_ratio,
                    grid_size=self.grid_size,
                )
                for _ in range(self.pit_depth)
            ]
        )
        self.out_proj = nn.Conv2d(self.pixel_hidden_dim, self.in_channels, kernel_size=1, stride=1)

    def _patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"input image size mismatch: expected {(self.image_size, self.image_size)}, got {(height, width)}"
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(
            batch_size,
            self.grid_size,
            self.patch_size,
            self.grid_size,
            self.patch_size,
            self.pixel_hidden_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, self.num_patches, self.patch_area, self.pixel_hidden_dim)

    def _unpatchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.pixel_hidden_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, self.image_size, self.image_size, self.pixel_hidden_dim)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor | None,
        style_global: torch.Tensor,
        style_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.dim() != 4:
            raise ValueError(f"x_t must be BCHW, got {tuple(x_t.shape)}")
        if tuple(x_t.shape[1:]) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                "x_t shape mismatch: "
                f"expected (*, {self.in_channels}, {self.image_size}, {self.image_size}), got {tuple(x_t.shape)}"
            )
        if content_tokens.shape != (x_t.size(0), self.num_patches, self.patch_hidden_dim):
            raise ValueError(
                "content_tokens shape mismatch: "
                f"expected {(x_t.size(0), self.num_patches, self.patch_hidden_dim)}, got {tuple(content_tokens.shape)}"
            )
        if self.use_style_tokens:
            if style_tokens is None:
                raise ValueError("style_tokens must be provided when use_style_tokens=True")
            if style_tokens.shape[:3] != (x_t.size(0), style_tokens.size(1), self.num_patches):
                raise ValueError(f"style_tokens shape mismatch: got {tuple(style_tokens.shape)}")
            if style_tokens.size(-1) != self.patch_hidden_dim:
                raise ValueError(
                    f"style_tokens hidden dim mismatch: expected {self.patch_hidden_dim}, got {style_tokens.size(-1)}"
                )
        if style_global.shape != (x_t.size(0), self.patch_hidden_dim):
            raise ValueError(
                f"style_global shape mismatch: expected {(x_t.size(0), self.patch_hidden_dim)}, got {tuple(style_global.shape)}"
            )

        patch_tokens = self.patch_embed(x_t).flatten(2).transpose(1, 2).contiguous()
        patch_tokens = patch_tokens + self.patch_pos_embed.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        content_tokens = content_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        patch_tokens = self.patch_input_proj(torch.cat([patch_tokens, content_tokens], dim=-1))

        t_embed = timestep_embedding(timesteps, self.patch_hidden_dim).to(dtype=patch_tokens.dtype)
        t_embed = self.time_mlp(t_embed)
        global_cond = self.global_cond_proj(torch.cat([t_embed, style_global.to(dtype=patch_tokens.dtype)], dim=-1))

        if style_tokens is not None:
            style_tokens = style_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        style_ref_mask = (
            None if style_ref_mask is None else style_ref_mask.to(device=patch_tokens.device, dtype=torch.bool)
        )
        for block in self.patch_blocks:
            patch_tokens = block(
                patch_tokens,
                global_cond=global_cond,
                style_tokens=style_tokens if block.use_style else None,
                style_ref_mask=style_ref_mask,
            )

        s_cond = patch_tokens + t_embed.unsqueeze(1)

        pixel_tokens = self.pixel_embed(x_t)
        pixel_tokens = self._patchify_pixels(pixel_tokens)
        for block in self.pit_blocks:
            pixel_tokens = block(pixel_tokens, s_cond=s_cond)

        pixel_map = self._unpatchify_pixels(pixel_tokens)
        return self.out_proj(pixel_map)
