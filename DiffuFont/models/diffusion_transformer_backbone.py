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


def _broadcast_modulation(
    hidden_states: torch.Tensor,
    modulation: torch.Tensor,
) -> torch.Tensor:
    if modulation.dim() == hidden_states.dim() - 1:
        return modulation.unsqueeze(1)
    if modulation.dim() != hidden_states.dim():
        raise ValueError(
            "modulation rank mismatch: "
            f"hidden_states={tuple(hidden_states.shape)} modulation={tuple(modulation.shape)}"
        )
    return modulation


def modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shift = _broadcast_modulation(hidden_states, shift)
    scale = _broadcast_modulation(hidden_states, scale)
    return hidden_states * (1.0 + scale) + shift


def _tensor_rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float().pow(2)) + 1e-12)


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


def _build_zero_linear(hidden_dim: int, out_dim: int | None = None) -> nn.Linear:
    out_dim = int(hidden_dim) if out_dim is None else int(out_dim)
    layer = nn.Linear(hidden_dim, out_dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio)
        self.ffn_time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.ffn_time_mod_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.ffn_time_modulation[-1].weight)
        nn.init.zeros_(self.ffn_time_modulation[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        self_attn_shift: torch.Tensor,
        self_attn_scale: torch.Tensor,
        self_attn_gate: torch.Tensor,
        ffn_time_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = patch_tokens
        q = modulate(self.norm_self(x), self_attn_shift, self_attn_scale)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + _broadcast_modulation(self_out, self_attn_gate) * self_out
        shift_mlp, scale_mlp, gate_mlp = self.ffn_time_modulation(
            self.ffn_time_mod_input_norm(ffn_time_cond)
        ).chunk(3, dim=-1)
        mlp_out = self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class GlyphDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        style_cond_dim: int,
        use_content_injection: bool = True,
        use_style_injection: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.style_cond_dim = int(style_cond_dim)
        self.use_content_injection = bool(use_content_injection)
        self.use_style_injection = bool(use_style_injection)
        self.content_control_norm = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.use_content_injection
            else None
        )
        self.attn_time_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn_time_to_token = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_joint_mod = (
            _build_zero_linear(hidden_dim, hidden_dim * 3)
        )
        self.style_cond_norm = (
            nn.LayerNorm(self.style_cond_dim, elementwise_affine=False, eps=1e-6)
            if self.use_style_injection
            else None
        )
        self.style_to_time = (
            _build_zero_linear(self.style_cond_dim, hidden_dim)
            if self.use_style_injection
            else None
        )
        self.dit_block = DiTBlock(hidden_dim, num_heads, mlp_ratio)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        time_cond: torch.Tensor,
        content_tokens: torch.Tensor | None,
        style_global: torch.Tensor | None,
        return_injection_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        x = patch_tokens
        block_stats: dict[str, float] = {}
        block_input_rms = None if not return_injection_stats else float(_tensor_rms(x.detach()).item())
        attn_time_tokens = self.attn_time_to_token(self.attn_time_norm(time_cond)).unsqueeze(1).expand_as(x)
        attn_joint_source = attn_time_tokens
        if self.use_content_injection:
            if (
                content_tokens is None
                or self.content_control_norm is None
            ):
                raise RuntimeError("content_tokens must be provided when content injection is enabled")
            if content_tokens.shape != x.shape:
                raise RuntimeError(
                    "content token shape mismatch: "
                    f"expected {tuple(x.shape)}, got {tuple(content_tokens.shape)}"
                )
            content_source = self.content_control_norm(content_tokens)
            attn_joint_source = attn_joint_source + content_source
            if return_injection_stats and block_input_rms is not None:
                with torch.no_grad():
                    normalized_tokens = self.dit_block.norm_self(x.detach())
                    content_attn_shift, content_attn_scale, _ = self.attn_joint_mod(content_source.detach()).chunk(3, dim=-1)
                    content_attn_delta = normalized_tokens * content_attn_scale + content_attn_shift
                    block_stats["content_ratio"] = float(
                        (_tensor_rms(content_attn_delta) / max(block_input_rms, 1e-12)).item()
                    )

        self_attn_shift, self_attn_scale, self_attn_gate = self.attn_joint_mod(attn_joint_source).chunk(3, dim=-1)

        ffn_time_cond = time_cond
        if self.use_style_injection:
            if (
                style_global is None
                or self.style_cond_norm is None
                or self.style_to_time is None
            ):
                raise RuntimeError("style_global must be provided when style injection is enabled")
            expected_style_shape = (x.size(0), self.style_cond_dim)
            if tuple(style_global.shape) != expected_style_shape:
                raise RuntimeError(
                    "style_global shape mismatch: "
                    f"expected {expected_style_shape}, got {tuple(style_global.shape)}"
                )
            ffn_time_cond = ffn_time_cond + self.style_to_time(self.style_cond_norm(style_global))

        x = self.dit_block(
            x,
            self_attn_shift=self_attn_shift,
            self_attn_scale=self_attn_scale,
            self_attn_gate=self_attn_gate,
            ffn_time_cond=ffn_time_cond,
        )
        if return_injection_stats:
            return x, block_stats
        return x


class DiffusionTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        hidden_dim: int = 512,
        style_cond_dim: int = 768,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        content_injection_layers: Sequence[int] | None = None,
        style_injection_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.style_cond_dim = int(style_cond_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.content_injection_layers = self._normalize_layer_indices(
            content_injection_layers,
            default_layers=range(1, self.depth + 1),
            depth=self.depth,
            field_name="content_injection_layers",
        )
        self.style_injection_layers = self._normalize_layer_indices(
            style_injection_layers,
            default_layers=range(1, self.depth + 1),
            depth=self.depth,
            field_name="style_injection_layers",
        )
        content_layer_set = set(self.content_injection_layers)
        style_layer_set = set(self.style_injection_layers)

        self.content_layer_mask = [
            (block_idx + 1) in content_layer_set
            for block_idx in range(self.depth)
        ]
        self.style_layer_mask = [
            (block_idx + 1) in style_layer_set
            for block_idx in range(self.depth)
        ]
        self.has_content_injection = any(self.content_layer_mask)
        self.has_style_injection = any(self.style_layer_mask)

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
        # Keep long-run time conditioning amplitude stable before it enters modulation heads.
        self.time_cond_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.blocks = nn.ModuleList()
        for block_idx in range(self.depth):
            self.blocks.append(
                GlyphDiTBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio,
                    style_cond_dim=self.style_cond_dim,
                    use_content_injection=self.content_layer_mask[block_idx],
                    use_style_injection=self.style_layer_mask[block_idx],
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
        return_injection_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
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
        time_cond = self.time_cond_norm(time_cond)

        if self.has_content_injection:
            content_tokens = content_tokens.to(device=x.device, dtype=x.dtype)
        else:
            content_tokens = None
        if self.has_style_injection:
            style_global = style_global.to(device=x.device, dtype=x.dtype)
        else:
            style_global = None

        injection_stats: dict[str, float] = {}
        for block_idx, block in enumerate(self.blocks):
            if return_injection_stats:
                x, block_stats = block(
                    x,
                    time_cond=time_cond,
                    content_tokens=content_tokens if block.use_content_injection else None,
                    style_global=style_global if block.use_style_injection else None,
                    return_injection_stats=True,
                )
                block_prefix = f"block_{block_idx + 1:02d}"
                for key, value in block_stats.items():
                    injection_stats[f"{block_prefix}_{key}"] = float(value)
            else:
                x = block(
                    x,
                    time_cond=time_cond,
                    content_tokens=content_tokens if block.use_content_injection else None,
                    style_global=style_global if block.use_style_injection else None,
                )

        x = self.final_norm(x)
        if return_injection_stats:
            return x, injection_stats
        return x
