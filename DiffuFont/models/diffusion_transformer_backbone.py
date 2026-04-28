#!/usr/bin/env python3
"""Pixel-space DiT backbone for content+style glyph x-pred generation."""

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


def _normalize_norm_variant(norm_variant: str) -> str:
    norm_variant = str(norm_variant)
    if norm_variant not in {"ln", "rms"}:
        raise ValueError(f"norm_variant must be 'ln' or 'rms', got {norm_variant!r}")
    return norm_variant


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, *, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x.pow(2), dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.weight is None:
            return x
        return x * self.weight


def _build_norm(hidden_dim: int, *, norm_variant: str) -> nn.Module:
    norm_variant = _normalize_norm_variant(norm_variant)
    if norm_variant == "rms":
        if hasattr(nn, "RMSNorm"):
            return nn.RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        return RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
    if norm_variant == "ln":
        return nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
    raise ValueError(f"norm_variant must be 'ln' or 'rms', got {norm_variant!r}")


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, inner_dim * 2)
        self.out = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, *, activation: str = "gelu") -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        activation = str(activation)
        if activation not in {"gelu", "swiglu"}:
            raise ValueError(f"FeedForward activation must be 'gelu' or 'swiglu', got {activation!r}")
        if activation == "swiglu":
            self.net = SwiGLU(hidden_dim, inner_dim)
        else:
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
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
    ) -> None:
        super().__init__()
        self.norm_self = _build_norm(hidden_dim, norm_variant=norm_variant)
        self.norm_mlp = _build_norm(hidden_dim, norm_variant=norm_variant)
        self.self_attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio, activation=ffn_activation)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        self_attn_shift: torch.Tensor,
        self_attn_scale: torch.Tensor,
        self_attn_gate: torch.Tensor,
        ffn_shift: torch.Tensor,
        ffn_scale: torch.Tensor,
        ffn_gate: torch.Tensor,
    ) -> torch.Tensor:
        x = patch_tokens
        q = modulate(self.norm_self(x), self_attn_shift, self_attn_scale)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + _broadcast_modulation(self_out, self_attn_gate) * self_out
        mlp_out = self.mlp(modulate(self.norm_mlp(x), ffn_shift, ffn_scale))
        x = x + _broadcast_modulation(mlp_out, ffn_gate) * mlp_out
        return x


class GlyphDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        use_content_injection: bool = True,
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        self.use_content_injection = bool(use_content_injection)
        self.norm_variant = _normalize_norm_variant(norm_variant)
        if self.use_content_injection and (self.condition_dim % 2) != 0:
            raise ValueError(
                f"conditioning dim must be even for content/style split, got {self.condition_dim}"
            )
        self.condition_half_dim = self.condition_dim // 2
        self.content_condition_norm = (
            _build_norm(self.condition_half_dim, norm_variant=self.norm_variant)
            if self.use_content_injection
            else None
        )
        self.style_condition_norm = (
            _build_norm(self.condition_half_dim, norm_variant=self.norm_variant)
            if self.use_content_injection
            else None
        )
        self.content_condition_to_hidden = (
            nn.Linear(self.condition_half_dim, hidden_dim)
            if self.use_content_injection
            else None
        )
        self.style_condition_to_hidden = (
            nn.Linear(self.condition_half_dim, hidden_dim)
            if self.use_content_injection
            else None
        )
        self.time_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.joint_mod = _build_zero_linear(hidden_dim, hidden_dim * 6)
        self.dit_block = DiTBlock(
            hidden_dim,
            num_heads,
            mlp_ratio,
            ffn_activation=ffn_activation,
            norm_variant=self.norm_variant,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        time_cond: torch.Tensor,
        conditioning_tokens: torch.Tensor | None,
        conditioning_norm_parts: tuple[torch.Tensor, torch.Tensor] | None = None,
        condition_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = patch_tokens
        if time_cond.dim() != 2 or time_cond.shape != (x.size(0), self.hidden_dim):
            raise RuntimeError(
                "time condition shape mismatch: "
                f"expected {(x.size(0), self.hidden_dim)}, got {tuple(time_cond.shape)}"
            )
        time_hidden = self.time_to_hidden(time_cond).unsqueeze(1).expand_as(x)
        joint_hidden = time_hidden
        if self.use_content_injection:
            if (
                condition_hidden is None
                and conditioning_tokens is None
                and conditioning_norm_parts is None
            ):
                raise RuntimeError("conditioning input must be provided when content injection is enabled")
            if condition_hidden is not None:
                if condition_hidden.shape != x.shape:
                    raise RuntimeError(
                        "condition_hidden shape mismatch: "
                        f"expected {tuple(x.shape)}, got {tuple(condition_hidden.shape)}"
                    )
            elif (
                self.content_condition_norm is None
                or self.style_condition_norm is None
                or self.content_condition_to_hidden is None
                or self.style_condition_to_hidden is None
            ):
                raise RuntimeError("conditioning_tokens must be provided when content injection is enabled")
            if condition_hidden is None:
                if conditioning_norm_parts is None:
                    expected_condition_shape = (x.size(0), x.size(1), self.condition_dim)
                    if conditioning_tokens is None or conditioning_tokens.shape != expected_condition_shape:
                        raise RuntimeError(
                            "conditioning token shape mismatch: "
                            f"expected {expected_condition_shape}, got "
                            f"{None if conditioning_tokens is None else tuple(conditioning_tokens.shape)}"
                        )
                    content_tokens, style_tokens = conditioning_tokens.split(self.condition_half_dim, dim=-1)
                    content_normed = self.content_condition_norm(content_tokens)
                    style_normed = self.style_condition_norm(style_tokens)
                else:
                    content_normed, style_normed = conditioning_norm_parts
                content_hidden = self.content_condition_to_hidden(content_normed)
                style_hidden = self.style_condition_to_hidden(style_normed)
                joint_hidden = time_hidden + content_hidden + style_hidden
            else:
                joint_hidden = time_hidden + condition_hidden
        modulation = self.joint_mod(F.silu(joint_hidden))

        (
            self_attn_shift,
            self_attn_scale,
            self_attn_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = modulation.chunk(6, dim=-1)

        x = self.dit_block(
            x,
            self_attn_shift=self_attn_shift,
            self_attn_scale=self_attn_scale,
            self_attn_gate=self_attn_gate,
            ffn_shift=ffn_shift,
            ffn_scale=ffn_scale,
            ffn_gate=ffn_gate,
        )
        return x


class DiffusionTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 8,
        patch_embed_bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        conditioning_dim: int | None = None,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        content_injection_layers: Sequence[int] | None = None,
        conditioning_injection_mode: str = "all",
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_embed_bottleneck_dim = int(patch_embed_bottleneck_dim)
        if self.patch_embed_bottleneck_dim <= 0:
            raise ValueError(
                f"patch_embed_bottleneck_dim must be > 0, got {patch_embed_bottleneck_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.conditioning_dim = self.hidden_dim if conditioning_dim is None else int(conditioning_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        if str(conditioning_injection_mode) != "all":
            raise ValueError(
                "conditioning_injection_mode is fixed to 'all' in the current model, "
                f"got {conditioning_injection_mode!r}"
            )
        self.ffn_activation = str(ffn_activation)
        self.norm_variant = _normalize_norm_variant(norm_variant)
        if self.ffn_activation != "swiglu":
            raise ValueError(
                "ffn_activation is fixed to 'swiglu' in the current model, "
                f"got {ffn_activation!r}"
            )
        if self.norm_variant != "rms":
            raise ValueError(
                "norm_variant is fixed to 'rms' in the current model, "
                f"got {norm_variant!r}"
            )
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.content_injection_layers = self._normalize_layer_indices(
            content_injection_layers,
            default_layers=range(1, self.depth + 1),
            depth=self.depth,
            field_name="content_injection_layers",
        )
        content_layer_set = set(self.content_injection_layers)

        self.content_layer_mask = [
            (block_idx + 1) in content_layer_set
            for block_idx in range(self.depth)
        ]
        self.has_content_injection = any(self.content_layer_mask)
        self.patch_embed_proj1 = nn.Conv2d(
            self.in_channels,
            self.patch_embed_bottleneck_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.patch_embed_proj2 = nn.Conv2d(
            self.patch_embed_bottleneck_dim,
            self.hidden_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        pos_embed = build_2d_sincos_pos_embed(self.hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Keep long-run time conditioning amplitude stable before it enters modulation heads.
        self.time_cond_norm = _build_norm(self.hidden_dim, norm_variant=self.norm_variant)
        self.blocks = nn.ModuleList()
        for block_idx in range(self.depth):
            self.blocks.append(
                GlyphDiTBlock(
                    self.hidden_dim,
                    self.conditioning_dim,
                    self.num_heads,
                    mlp_ratio,
                    use_content_injection=self.content_layer_mask[block_idx],
                    ffn_activation=self.ffn_activation,
                    norm_variant=self.norm_variant,
                )
            )
        self.final_norm = _build_norm(self.hidden_dim, norm_variant=self.norm_variant)

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

    def build_time_cond(
        self,
        timesteps: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time_cond = timestep_embedding(timesteps, self.hidden_dim).to(dtype=dtype)
        time_cond = self.time_mlp(time_cond)
        return self.time_cond_norm(time_cond)

    def normalize_conditioning_tokens(
        self,
        conditioning_tokens: torch.Tensor,
        *,
        batch_size: int,
        token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.has_content_injection:
            return None
        conditioning_tokens = conditioning_tokens.to(device=device, dtype=dtype)
        expected_condition_shape = (int(batch_size), int(token_count), self.conditioning_dim)
        if conditioning_tokens.shape != expected_condition_shape:
            raise RuntimeError(
                "conditioning token shape mismatch: "
                f"expected {expected_condition_shape}, got {tuple(conditioning_tokens.shape)}"
            )
        first_injection_block = next(block for block in self.blocks if block.use_content_injection)
        if first_injection_block.content_condition_norm is None or first_injection_block.style_condition_norm is None:
            raise RuntimeError("content injection block is missing condition norms")
        content_tokens, style_tokens = conditioning_tokens.split(
            first_injection_block.condition_half_dim,
            dim=-1,
        )
        return (
            first_injection_block.content_condition_norm(content_tokens),
            first_injection_block.style_condition_norm(style_tokens),
        )

    def build_condition_hidden_cache(
        self,
        conditioning_tokens: torch.Tensor,
        *,
        batch_size: int,
        token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor | None]:
        norm_parts = self.normalize_conditioning_tokens(
            conditioning_tokens,
            batch_size=batch_size,
            token_count=token_count,
            device=device,
            dtype=dtype,
        )
        if norm_parts is None:
            return [None for _ in self.blocks]
        content_normed, style_normed = norm_parts
        cache: list[torch.Tensor | None] = []
        for block in self.blocks:
            if not block.use_content_injection:
                cache.append(None)
                continue
            if block.content_condition_to_hidden is None or block.style_condition_to_hidden is None:
                raise RuntimeError("content injection block is missing condition projections")
            cache.append(
                block.content_condition_to_hidden(content_normed)
                + block.style_condition_to_hidden(style_normed)
            )
        return cache

    def forward(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        conditioning_tokens: torch.Tensor,
        condition_hidden_cache: list[torch.Tensor | None] | None = None,
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

        x = self.patch_embed_proj2(self.patch_embed_proj1(image)).flatten(2).transpose(1, 2).contiguous()
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        time_cond = self.build_time_cond(
            timesteps,
            dtype=x.dtype,
        )

        conditioning_norm_parts = None
        if self.has_content_injection:
            if condition_hidden_cache is not None:
                if len(condition_hidden_cache) != len(self.blocks):
                    raise RuntimeError(
                        "condition_hidden_cache length mismatch: "
                        f"expected {len(self.blocks)}, got {len(condition_hidden_cache)}"
                    )
            else:
                conditioning_norm_parts = self.normalize_conditioning_tokens(
                    conditioning_tokens,
                    batch_size=x.size(0),
                    token_count=x.size(1),
                    device=x.device,
                    dtype=x.dtype,
                )
        elif condition_hidden_cache is not None:
            raise RuntimeError("condition_hidden_cache was provided but this backbone has no content injection")

        for block_idx, block in enumerate(self.blocks):
            x = block(
                x,
                time_cond=time_cond,
                conditioning_tokens=None if condition_hidden_cache is not None else conditioning_tokens,
                conditioning_norm_parts=conditioning_norm_parts if block.use_content_injection else None,
                condition_hidden=None if condition_hidden_cache is None else condition_hidden_cache[block_idx],
            )

        x = self.final_norm(x)
        return x
