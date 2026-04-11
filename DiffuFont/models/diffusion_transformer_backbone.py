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


def modulate_scale_only(hidden_states: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = _broadcast_modulation(hidden_states, scale)
    return hidden_states * (1.0 + scale)


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
        ffn_activation: str = "gelu",
        norm_variant: str = "ln",
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
        self_attn_shift: torch.Tensor | None,
        self_attn_scale: torch.Tensor,
        self_attn_gate: torch.Tensor,
        ffn_shift: torch.Tensor | None,
        ffn_scale: torch.Tensor,
        ffn_gate: torch.Tensor,
    ) -> torch.Tensor:
        x = patch_tokens
        if self_attn_shift is None:
            q = modulate_scale_only(self.norm_self(x), self_attn_scale)
        else:
            q = modulate(self.norm_self(x), self_attn_shift, self_attn_scale)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + _broadcast_modulation(self_out, self_attn_gate) * self_out
        if ffn_shift is None:
            mlp_out = self.mlp(modulate_scale_only(self.norm_mlp(x), ffn_scale))
        else:
            mlp_out = self.mlp(modulate(self.norm_mlp(x), ffn_shift, ffn_scale))
        x = x + _broadcast_modulation(mlp_out, ffn_gate) * mlp_out
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
        conditioning_injection_mode: str = "all",
        ffn_activation: str = "gelu",
        norm_variant: str = "ln",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.style_cond_dim = int(style_cond_dim)
        self.use_content_injection = bool(use_content_injection)
        self.use_style_injection = bool(use_style_injection)
        self.conditioning_injection_mode = str(conditioning_injection_mode)
        self.norm_variant = _normalize_norm_variant(norm_variant)
        if self.conditioning_injection_mode not in {
            "all",
            "content_sa_style_ffn",
            "content_style_sa_style_ffn",
            "content_style_sa_t_ffn",
        }:
            raise ValueError(
                "conditioning_injection_mode must be one of "
                "{'all', 'content_sa_style_ffn', 'content_style_sa_style_ffn', 'content_style_sa_t_ffn'}, "
                f"got {conditioning_injection_mode!r}"
            )
        self.content_control_norm = (
            _build_norm(hidden_dim, norm_variant=self.norm_variant)
            if self.use_content_injection
            else None
        )
        self.attn_time_norm = _build_norm(hidden_dim, norm_variant=self.norm_variant)
        self.attn_time_to_token = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        joint_out_dim = hidden_dim * (4 if self.norm_variant == "rms" else 6)
        self.joint_mod = _build_zero_linear(hidden_dim, joint_out_dim)
        self.style_cond_norm = (
            _build_norm(self.style_cond_dim, norm_variant=self.norm_variant)
            if self.use_style_injection
            else None
        )
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
        content_tokens: torch.Tensor | None,
        style_global: torch.Tensor | None,
    ) -> torch.Tensor:
        x = patch_tokens
        attn_time_tokens = self.attn_time_to_token(self.attn_time_norm(time_cond)).unsqueeze(1).expand_as(x)
        sa_source = attn_time_tokens
        ffn_source: torch.Tensor = attn_time_tokens if self.conditioning_injection_mode == "all" else time_cond
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
            sa_source = sa_source + content_source
            if self.conditioning_injection_mode == "all":
                ffn_source = ffn_source + content_source

        if self.use_style_injection:
            if (
                style_global is None
                or self.style_cond_norm is None
            ):
                raise RuntimeError("style_global must be provided when style injection is enabled")
            expected_style_shape = (x.size(0), self.style_cond_dim)
            if tuple(style_global.shape) != expected_style_shape:
                raise RuntimeError(
                    "style_global shape mismatch: "
                    f"expected {expected_style_shape}, got {tuple(style_global.shape)}"
                )
            style_source = self.style_cond_norm(style_global).unsqueeze(1).expand_as(x)
            if self.conditioning_injection_mode == "all":
                ffn_source = ffn_source + style_source
                sa_source = sa_source + style_source
            elif self.conditioning_injection_mode == "content_sa_style_ffn":
                ffn_source = ffn_source + self.style_cond_norm(style_global)
            elif self.conditioning_injection_mode == "content_style_sa_style_ffn":
                sa_source = sa_source + style_source
                ffn_source = ffn_source + self.style_cond_norm(style_global)
            elif self.conditioning_injection_mode == "content_style_sa_t_ffn":
                sa_source = sa_source + style_source

        if self.norm_variant == "rms":
            if self.conditioning_injection_mode == "all":
                self_attn_scale, self_attn_gate, ffn_scale, ffn_gate = self.joint_mod(sa_source).chunk(4, dim=-1)
            else:
                self_attn_scale, self_attn_gate, _, _ = self.joint_mod(sa_source).chunk(4, dim=-1)
                _, _, ffn_scale, ffn_gate = self.joint_mod(ffn_source).chunk(4, dim=-1)
            self_attn_shift = None
            ffn_shift = None
        else:
            if self.conditioning_injection_mode == "all":
                (
                    self_attn_shift,
                    self_attn_scale,
                    self_attn_gate,
                    ffn_shift,
                    ffn_scale,
                    ffn_gate,
                ) = self.joint_mod(sa_source).chunk(6, dim=-1)
            else:
                (
                    self_attn_shift,
                    self_attn_scale,
                    self_attn_gate,
                    _,
                    _,
                    _,
                ) = self.joint_mod(sa_source).chunk(6, dim=-1)
                (
                    _,
                    _,
                    _,
                    ffn_shift,
                    ffn_scale,
                    ffn_gate,
                ) = self.joint_mod(ffn_source).chunk(6, dim=-1)

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
        patch_size: int = 16,
        hidden_dim: int = 512,
        style_cond_dim: int = 768,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        content_injection_layers: Sequence[int] | None = None,
        style_injection_layers: Sequence[int] | None = None,
        conditioning_injection_mode: str = "all",
        ffn_activation: str = "gelu",
        norm_variant: str = "ln",
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
        self.conditioning_injection_mode = str(conditioning_injection_mode)
        self.ffn_activation = str(ffn_activation)
        self.norm_variant = _normalize_norm_variant(norm_variant)
        if self.ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError(f"ffn_activation must be 'gelu' or 'swiglu', got {ffn_activation!r}")
        if self.conditioning_injection_mode not in {
            "all",
            "content_sa_style_ffn",
            "content_style_sa_style_ffn",
            "content_style_sa_t_ffn",
        }:
            raise ValueError(
                "conditioning_injection_mode must be one of "
                "{'all', 'content_sa_style_ffn', 'content_style_sa_style_ffn', 'content_style_sa_t_ffn'}, "
                f"got {conditioning_injection_mode!r}"
            )
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
        self.time_cond_norm = _build_norm(self.hidden_dim, norm_variant=self.norm_variant)
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
                    conditioning_injection_mode=self.conditioning_injection_mode,
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
        time_cond = self.time_cond_norm(time_cond)

        if self.has_content_injection:
            content_tokens = content_tokens.to(device=x.device, dtype=x.dtype)
        else:
            content_tokens = None
        if self.has_style_injection:
            style_global = style_global.to(device=x.device, dtype=x.dtype)
        else:
            style_global = None

        for block in self.blocks:
            x = block(
                x,
                time_cond=time_cond,
                content_tokens=content_tokens if block.use_content_injection else None,
                style_global=style_global if block.use_style_injection else None,
            )

        x = self.final_norm(x)
        return x
