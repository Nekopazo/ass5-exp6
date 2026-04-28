#!/usr/bin/env python3
"""Content+style diffusion transformer for Chinese glyph generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_transformer_backbone import (
    DiffusionTransformerBackbone,
    _build_norm,
    _build_zero_linear,
    modulate,
)
from .sdpa_attention import SDPAAttention


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DBlock(nn.Module):
    """Residual block with optional stage-level spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.downsample = bool(downsample)
        self.norm1 = nn.GroupNorm(_group_count(self.in_channels), self.in_channels)
        self.norm2 = nn.GroupNorm(_group_count(self.out_channels), self.out_channels)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=2 if self.downsample else 1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv_sc = (
            None
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        )
        self.pool = nn.AvgPool2d(2) if self.downsample else nn.Identity()

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.conv_sc is not None:
            residual = self.conv_sc(residual)
        return self.pool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class ConvPyramidEncoder(nn.Module):
    """Shared DBlock pyramid template used by both content and style encoders."""

    def __init__(
        self,
        *,
        in_channels: int,
        stage_channels: Sequence[int],
        block_depth: int = 1,
        output_grid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        normalized_stage_channels = [int(ch) for ch in stage_channels]
        if not normalized_stage_channels:
            raise ValueError("stage_channels must be non-empty")
        self.block_depth = max(1, int(block_depth))

        self.in_channels = int(in_channels)
        self.stage_channels = normalized_stage_channels
        self.output_grid_size = None if output_grid_size is None else int(output_grid_size)
        self.stages = nn.ModuleList()
        prev_channels = self.in_channels
        for out_channels in self.stage_channels:
            blocks = [DBlock(prev_channels, out_channels, downsample=True)]
            blocks.extend(
                DBlock(out_channels, out_channels, downsample=False)
                for _ in range(self.block_depth - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        for stage in self.stages:
            x = stage(x)
        if self.output_grid_size is not None and x.shape[-2:] != (self.output_grid_size, self.output_grid_size):
            x = F.interpolate(
                x,
                size=(self.output_grid_size, self.output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


def _build_pyramid_stage_channels(
    *,
    image_size: int,
    output_grid_size: int,
    hidden_dim: int,
    base_channels: int,
    max_channels: int,
) -> tuple[list[int], int]:
    if image_size % output_grid_size != 0:
        raise ValueError(
            f"image_size must be divisible by output_grid_size, got {image_size} vs {output_grid_size}"
        )
    downsample_factor = int(image_size) // int(output_grid_size)
    if downsample_factor < 1:
        raise ValueError(f"Invalid downsample factor: {downsample_factor}")
    downsample_depth = int(round(math.log2(downsample_factor))) if downsample_factor > 1 else 0
    if 2**downsample_depth != downsample_factor:
        raise ValueError(
            f"image_size/output_grid_size must be a power of two, got {image_size}/{output_grid_size}"
        )

    remaining_downsample_depth = max(0, downsample_depth - 1)
    stage_channels = [int(base_channels)]
    for stage_idx in range(remaining_downsample_depth):
        out_channels = hidden_dim if stage_idx == remaining_downsample_depth - 1 else min(
            base_channels * (2 ** (stage_idx + 1)),
            max_channels,
        )
        stage_channels.append(int(out_channels))
    return stage_channels, int(downsample_depth)


class ContentEncoder(ConvPyramidEncoder):
    """CNN content encoder that downsamples to the DiT patch grid."""

    def __init__(
        self,
        *,
        image_size: int = 128,
        output_grid_size: int = 16,
        hidden_dim: int = 256,
        base_channels: int = 64,
        max_channels: int = 256,
        block_depth: int = 2,
    ) -> None:
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.block_depth = max(1, int(block_depth))
        stage_channels, downsample_depth = _build_pyramid_stage_channels(
            image_size=int(image_size),
            output_grid_size=int(output_grid_size),
            hidden_dim=int(hidden_dim),
            base_channels=int(base_channels),
            max_channels=int(max_channels),
        )
        self.downsample_depth = int(downsample_depth)

        super().__init__(
            in_channels=1,
            stage_channels=stage_channels,
            block_depth=self.block_depth,
            output_grid_size=int(output_grid_size),
        )
        self.output_grid_size = int(output_grid_size)


class StyleEncoder(ConvPyramidEncoder):
    """CNN style encoder sharing the same spatial token lattice as content."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        output_grid_size: int = 16,
        hidden_dim: int = 256,
        base_channels: int = 64,
        max_channels: int = 256,
        block_depth: int = 2,
    ) -> None:
        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.block_depth = max(1, int(block_depth))
        stage_channels, self.downsample_depth = _build_pyramid_stage_channels(
            image_size=int(image_size),
            output_grid_size=int(output_grid_size),
            hidden_dim=int(hidden_dim),
            base_channels=int(base_channels),
            max_channels=int(max_channels),
        )
        self.local_hidden_dim = int(stage_channels[-1])
        super().__init__(
            in_channels=int(in_channels),
            stage_channels=stage_channels,
            block_depth=self.block_depth,
            output_grid_size=int(output_grid_size),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class ContentStyleCrossAttention(nn.Module):
    """External content<-style fusion utilities for concat cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.num_heads <= 0 or (self.embed_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config embed_dim={embed_dim} num_heads={num_heads}")

        self.query_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.token_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAAttention(self.embed_dim, self.num_heads)

    def _validate_style_inputs(
        self,
        style_tokens: torch.Tensor,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if style_tokens.dim() != 4:
            raise ValueError(f"style_tokens must be 4D [B, R, T, D], got {tuple(style_tokens.shape)}")
        if token_valid_mask is not None:
            if token_valid_mask.shape != style_tokens.shape[:3]:
                raise ValueError(
                    f"token_valid_mask shape mismatch: expected {tuple(style_tokens.shape[:3])}, "
                    f"got {tuple(token_valid_mask.shape)}"
                )

    def project_style_bank_kv(
        self,
        style_tokens: torch.Tensor,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._validate_style_inputs(style_tokens, token_valid_mask=token_valid_mask)
        batch_size, num_refs, tokens_per_ref, hidden_dim = style_tokens.shape
        normed_tokens = self.token_norm(style_tokens)
        style_key_valid_mask = None
        if token_valid_mask is not None:
            token_valid_mask = token_valid_mask.to(device=style_tokens.device, dtype=torch.bool)
            if bool((~token_valid_mask).all(dim=(1, 2)).any()):
                raise ValueError("token_valid_mask must keep at least one style token per sample")
            if not bool(token_valid_mask.all()):
                style_key_valid_mask = token_valid_mask.reshape(batch_size, num_refs * tokens_per_ref)[:, None, :]
        concat_tokens = normed_tokens.reshape(batch_size, num_refs * tokens_per_ref, hidden_dim)
        key, value = self.attn.project_key_value(concat_tokens, concat_tokens)
        concat_len = int(key.size(2))
        return (
            key.view(batch_size, 1, self.num_heads, concat_len, self.attn.head_dim),
            value.view(batch_size, 1, self.num_heads, concat_len, self.attn.head_dim),
            style_key_valid_mask,
        )

    def project_content_query(self, content_tokens: torch.Tensor) -> torch.Tensor:
        if content_tokens.dim() != 3:
            raise ValueError(f"content_tokens must be 3D [B, T, D], got {tuple(content_tokens.shape)}")
        return self.attn.project_query(self.query_norm(content_tokens))

    def fuse_content_style_tokens_from_projected(
        self,
        content_tokens: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
    ) -> torch.Tensor:
        content_query = self.project_content_query(content_tokens)
        return self.fuse_content_style_tokens_from_preprojected_query(
            content_tokens,
            content_query,
            style_key,
            style_value,
        )

    def fuse_content_style_tokens_from_preprojected_query(
        self,
        content_tokens: torch.Tensor,
        content_query: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
        style_key_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if content_tokens.dim() != 3:
            raise ValueError(f"content_tokens must be 3D [B, T, D], got {tuple(content_tokens.shape)}")
        if content_query.dim() != 4:
            raise ValueError(
                f"content_query must be 4D [B, H, T, Dh], got {tuple(content_query.shape)}"
            )
        if style_key.dim() != 5 or style_value.dim() != 5:
            raise ValueError(
                "style_key/style_value must be 5D [B, R, H, T, D], got "
                f"{tuple(style_key.shape)} and {tuple(style_value.shape)}"
            )
        if style_key.shape != style_value.shape:
            raise ValueError(
                "style_key/style_value shape mismatch: "
                f"{tuple(style_key.shape)} vs {tuple(style_value.shape)}"
            )
        batch_size, query_len, hidden_dim = content_tokens.shape
        key_batch, num_refs, num_heads, tokens_per_ref, head_dim = style_key.shape
        if key_batch != batch_size:
            raise ValueError(f"style_key batch mismatch: expected {batch_size}, got {key_batch}")
        expected_query_shape = (batch_size, num_heads, query_len, head_dim)
        if content_query.shape != expected_query_shape:
            raise ValueError(
                "content_query shape mismatch: "
                f"expected {expected_query_shape}, got {tuple(content_query.shape)}"
            )
        if style_key_valid_mask is not None:
            expected_mask_shape = (batch_size, num_refs, tokens_per_ref)
            if style_key_valid_mask.shape != expected_mask_shape:
                raise ValueError(
                    "style_key_valid_mask shape mismatch: "
                    f"expected {expected_mask_shape}, got {tuple(style_key_valid_mask.shape)}"
                )
            style_key_valid_mask = style_key_valid_mask.to(device=style_key.device, dtype=torch.bool)
            style_key_valid_mask = style_key_valid_mask.reshape(batch_size * num_refs, tokens_per_ref)
        expanded_query = (
            content_query.unsqueeze(1)
            .expand(batch_size, num_refs, num_heads, query_len, head_dim)
            .reshape(batch_size * num_refs, num_heads, query_len, head_dim)
        )
        flat_style_key = style_key.reshape(batch_size * num_refs, num_heads, tokens_per_ref, head_dim)
        flat_style_value = style_value.reshape(batch_size * num_refs, num_heads, tokens_per_ref, head_dim)
        style_context, _ = self.attn.attend_projected(
            expanded_query,
            flat_style_key,
            flat_style_value,
            key_valid_mask=style_key_valid_mask,
            need_weights=False,
        )
        style_context = style_context.view(batch_size, query_len, self.embed_dim)
        return torch.cat([content_tokens, style_context], dim=-1).contiguous()


class SourcePartRefDiT(nn.Module):
    """Pure DiT glyph generator with external content-style fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 8,
        patch_embed_bottleneck_dim: int = 128,
        encoder_hidden_dim: int = 256,
        content_encoder_block_depth: int = 2,
        style_encoder_block_depth: int = 2,
        dit_hidden_dim: int = 256,
        dit_depth: int = 12,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
        content_injection_layers: Sequence[int] | None = None,
        conditioning_injection_mode: str = "all",
        content_style_fusion_heads: int = 4,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(f"Only grayscale glyphs are supported, got in_channels={in_channels}")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_embed_bottleneck_dim = int(patch_embed_bottleneck_dim)
        self.patch_grid_size = self.image_size // self.patch_size
        self.num_patches = self.patch_grid_size * self.patch_grid_size
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.content_encoder_block_depth = max(1, int(content_encoder_block_depth))
        self.style_encoder_block_depth = max(1, int(style_encoder_block_depth))
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.dit_depth = int(dit_depth)
        self.dit_heads = int(dit_heads)
        self.dit_mlp_ratio = float(dit_mlp_ratio)
        self.ffn_activation = str(ffn_activation)
        self.norm_variant = str(norm_variant)
        if self.ffn_activation != "swiglu":
            raise ValueError(
                "ffn_activation is fixed to 'swiglu' in the refactored model, "
                f"got {ffn_activation!r}"
            )
        if self.norm_variant != "rms":
            raise ValueError(
                "norm_variant is fixed to 'rms' in the refactored model, "
                f"got {norm_variant!r}"
            )
        if str(conditioning_injection_mode) != "all":
            raise ValueError(
                "conditioning_injection_mode is fixed to 'all' in the refactored model, "
                f"got {conditioning_injection_mode!r}"
            )
        self.content_style_fusion_heads = int(content_style_fusion_heads)
        if self.content_style_fusion_heads <= 0:
            raise ValueError(f"content_style_fusion_heads must be > 0, got {content_style_fusion_heads}")
        self.content_injection_layers = DiffusionTransformerBackbone._normalize_layer_indices(
            content_injection_layers,
            default_layers=range(1, self.dit_depth + 1),
            depth=self.dit_depth,
            field_name="content_injection_layers",
        )
        self.output_patch_dim = self.in_channels * self.patch_size * self.patch_size

        self.content_encoder = ContentEncoder(
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.content_encoder_block_depth,
        )
        self.style_encoder = StyleEncoder(
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.style_encoder_block_depth,
        )
        self.style_token_hidden_dim = int(self.style_encoder.local_hidden_dim)
        self.style_token_proj = (
            nn.Identity()
            if self.style_token_hidden_dim == self.encoder_hidden_dim
            else nn.Linear(self.style_token_hidden_dim, self.encoder_hidden_dim)
        )
        self.content_style_attn = ContentStyleCrossAttention(
            embed_dim=self.encoder_hidden_dim,
            num_heads=self.content_style_fusion_heads,
        )
        self.conditioning_dim = self.encoder_hidden_dim * 2
        self.backbone = DiffusionTransformerBackbone(
            in_channels=self.in_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            patch_embed_bottleneck_dim=self.patch_embed_bottleneck_dim,
            hidden_dim=self.dit_hidden_dim,
            conditioning_dim=self.conditioning_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_injection_layers=self.content_injection_layers,
            ffn_activation=self.ffn_activation,
            norm_variant=self.norm_variant,
        )
        self.output_norm = _build_norm(self.dit_hidden_dim, norm_variant=self.norm_variant)
        self.output_condition_half_dim = self.encoder_hidden_dim
        self.output_content_condition_norm = _build_norm(
            self.output_condition_half_dim,
            norm_variant=self.norm_variant,
        )
        self.output_style_condition_norm = _build_norm(
            self.output_condition_half_dim,
            norm_variant=self.norm_variant,
        )
        self.output_content_condition_to_hidden = nn.Linear(
            self.output_condition_half_dim,
            self.dit_hidden_dim,
        )
        self.output_style_condition_to_hidden = nn.Linear(
            self.output_condition_half_dim,
            self.dit_hidden_dim,
        )
        self.output_time_to_hidden = nn.Linear(self.dit_hidden_dim, self.dit_hidden_dim)
        self.output_mod = _build_zero_linear(self.dit_hidden_dim, self.dit_hidden_dim * 2)
        self.output_proj = nn.Linear(self.dit_hidden_dim, self.output_patch_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        patch_proj1 = self.backbone.patch_embed_proj1.weight.data
        nn.init.xavier_uniform_(patch_proj1.view(patch_proj1.shape[0], -1))
        patch_proj2 = self.backbone.patch_embed_proj2.weight.data
        nn.init.xavier_uniform_(patch_proj2.view(patch_proj2.shape[0], -1))
        if self.backbone.patch_embed_proj2.bias is not None:
            nn.init.constant_(self.backbone.patch_embed_proj2.bias, 0)

        nn.init.normal_(self.backbone.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.backbone.time_mlp[2].weight, std=0.02)
        if self.backbone.time_mlp[0].bias is not None:
            nn.init.constant_(self.backbone.time_mlp[0].bias, 0)
        if self.backbone.time_mlp[2].bias is not None:
            nn.init.constant_(self.backbone.time_mlp[2].bias, 0)

        for block in self.backbone.blocks:
            nn.init.constant_(block.joint_mod.weight, 0)
            nn.init.constant_(block.joint_mod.bias, 0)

        nn.init.constant_(self.output_mod.weight, 0)
        nn.init.constant_(self.output_mod.bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def export_config(self) -> dict[str, int | float]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "patch_embed_bottleneck_dim": int(self.patch_embed_bottleneck_dim),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "content_encoder_block_depth": int(self.content_encoder_block_depth),
            "style_encoder_block_depth": int(self.style_encoder_block_depth),
            "dit_hidden_dim": int(self.dit_hidden_dim),
            "dit_depth": int(self.dit_depth),
            "dit_heads": int(self.dit_heads),
            "dit_mlp_ratio": float(self.dit_mlp_ratio),
            "content_injection_layers": list(self.content_injection_layers),
            "content_style_fusion_heads": int(self.content_style_fusion_heads),
        }

    def encode_content_tokens(self, content_img: torch.Tensor) -> torch.Tensor:
        content_features = self.content_encoder(content_img)
        return content_features.flatten(2).transpose(1, 2).contiguous()

    def _encode_style_features(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")

        batch, refs, channels, height, width = style_img.shape
        if style_ref_mask is None:
            ref_valid_mask = torch.ones((batch, refs), device=style_img.device, dtype=torch.bool)
        else:
            ref_valid_mask = style_ref_mask.to(device=style_img.device, dtype=torch.bool)
        if ref_valid_mask.shape != (batch, refs):
            raise RuntimeError(
                f"style_ref_mask shape mismatch: expected {(batch, refs)}, got {tuple(ref_valid_mask.shape)}"
            )
        if bool((~ref_valid_mask).all(dim=1).any()):
            raise RuntimeError("style_ref_mask must keep at least one reference per sample")

        flat_style = style_img.view(batch * refs, channels, height, width)
        style_features = self.style_encoder.forward_features(flat_style)
        style_tokens = style_features.flatten(2).transpose(1, 2).contiguous()
        tokens_per_ref = int(style_tokens.size(1))
        style_tokens = style_tokens.view(batch, refs, tokens_per_ref, self.style_token_hidden_dim)
        token_valid_mask = (
            ref_valid_mask[:, :, None]
            .expand(batch, refs, tokens_per_ref)
        )
        return style_tokens, token_valid_mask

    def encode_style_token_bank(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style_tokens, token_valid_mask = self._encode_style_features(
            style_img,
            style_ref_mask=style_ref_mask,
        )
        return self.style_token_proj(style_tokens).contiguous(), token_valid_mask

    def precompute_style_bank_kv(
        self,
        style_tokens: torch.Tensor,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.content_style_attn.project_style_bank_kv(
            style_tokens,
            token_valid_mask=token_valid_mask,
        )

    def precompute_content_query(
        self,
        content_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.content_style_attn.project_content_query(content_tokens)

    def build_conditioning_tokens(
        self,
        content_tokens: torch.Tensor,
        style_token_bank: Optional[torch.Tensor] = None,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
        content_query: Optional[torch.Tensor] = None,
        style_key: Optional[torch.Tensor] = None,
        style_value: Optional[torch.Tensor] = None,
        style_key_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if content_query is None:
            content_query = self.precompute_content_query(content_tokens)
        if style_key is None or style_value is None:
            if style_token_bank is None:
                raise ValueError("style_token_bank is required when precomputed style_key/style_value are not provided")
            style_key, style_value, style_key_valid_mask = self.precompute_style_bank_kv(
                style_token_bank,
                token_valid_mask=token_valid_mask,
            )
        return self.content_style_attn.fuse_content_style_tokens_from_preprojected_query(
            content_tokens,
            content_query,
            style_key,
            style_value,
            style_key_valid_mask=style_key_valid_mask,
        )

    def precompute_backbone_condition_hidden_cache(
        self,
        conditioning_tokens: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor | None]:
        return self.backbone.build_condition_hidden_cache(
            conditioning_tokens,
            batch_size=int(conditioning_tokens.size(0)),
            token_count=int(conditioning_tokens.size(1)),
            device=device,
            dtype=dtype,
        )

    def precompute_output_condition_hidden(
        self,
        conditioning_tokens: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_condition_shape = (conditioning_tokens.size(0), self.num_patches, self.conditioning_dim)
        if conditioning_tokens.shape != expected_condition_shape:
            raise ValueError(
                "conditioning token shape mismatch for final head: "
                f"expected {expected_condition_shape}, got {tuple(conditioning_tokens.shape)}"
            )
        conditioning_tokens = conditioning_tokens.to(device=device, dtype=dtype)
        content_tokens, style_tokens = conditioning_tokens.split(self.output_condition_half_dim, dim=-1)
        return self.output_content_condition_to_hidden(
            self.output_content_condition_norm(content_tokens)
        ) + self.output_style_condition_to_hidden(
            self.output_style_condition_norm(style_tokens)
        )

    def decode_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        output_condition_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be 3D, got {tuple(patch_tokens.shape)}")
        if tuple(patch_tokens.shape[1:]) != (self.num_patches, self.dit_hidden_dim):
            raise ValueError(
                "patch token shape mismatch: "
                f"expected (*, {self.num_patches}, {self.dit_hidden_dim}), got {tuple(patch_tokens.shape)}"
            )
        expected_condition_shape = (patch_tokens.size(0), self.num_patches, self.conditioning_dim)
        if conditioning_tokens.shape != expected_condition_shape:
            raise ValueError(
                "conditioning token shape mismatch for final head: "
                f"expected {expected_condition_shape}, got {tuple(conditioning_tokens.shape)}"
            )

        time_hidden = self.output_time_to_hidden(
            self.backbone.build_time_cond(
                timesteps,
                dtype=patch_tokens.dtype,
            )
        ).unsqueeze(1)
        if output_condition_hidden is None:
            conditioning_tokens = conditioning_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            content_tokens, style_tokens = conditioning_tokens.split(self.output_condition_half_dim, dim=-1)
            content_hidden = self.output_content_condition_to_hidden(
                self.output_content_condition_norm(content_tokens)
            )
            style_hidden = self.output_style_condition_to_hidden(
                self.output_style_condition_norm(style_tokens)
            )
            joint_hidden = time_hidden + content_hidden + style_hidden
        elif output_condition_hidden.shape != (patch_tokens.size(0), self.num_patches, self.dit_hidden_dim):
            raise ValueError(
                "output_condition_hidden shape mismatch: "
                f"expected {(patch_tokens.size(0), self.num_patches, self.dit_hidden_dim)}, "
                f"got {tuple(output_condition_hidden.shape)}"
            )
        else:
            joint_hidden = time_hidden + output_condition_hidden
        shift, scale = self.output_mod(F.silu(joint_hidden)).chunk(2, dim=-1)
        patch_pixels = self.output_proj(modulate(self.output_norm(patch_tokens), shift, scale))
        patch_pixels = patch_pixels.view(
            patch_tokens.size(0),
            self.patch_grid_size,
            self.patch_grid_size,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        return (
            patch_pixels.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(patch_tokens.size(0), self.in_channels, self.image_size, self.image_size)
        )

    def predict_x(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        conditioning_tokens: torch.Tensor,
        backbone_condition_hidden_cache: Optional[list[torch.Tensor | None]] = None,
        output_condition_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patch_tokens = self.backbone(
            x_t_image,
            timesteps,
            conditioning_tokens=conditioning_tokens,
            condition_hidden_cache=backbone_condition_hidden_cache,
        )
        return self.decode_patch_tokens(
            patch_tokens,
            timesteps=timesteps,
            conditioning_tokens=conditioning_tokens,
            output_condition_hidden=output_condition_hidden,
        )

    def forward(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        content_img: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content_tokens = self.encode_content_tokens(content_img)
        style_token_bank, token_valid_mask = self.encode_style_token_bank(
            style_img,
            style_ref_mask=style_ref_mask,
        )
        conditioning_tokens = self.build_conditioning_tokens(
            content_tokens,
            style_token_bank,
            token_valid_mask=token_valid_mask,
        )
        return self.predict_x(
            x_t_image,
            timesteps,
            conditioning_tokens=conditioning_tokens,
        )
