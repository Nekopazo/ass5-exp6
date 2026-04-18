#!/usr/bin/env python3
"""Content+style diffusion transformer for Chinese glyph generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_transformer_backbone import DiffusionTransformerBackbone
from .sdpa_attention import SDPAAttention


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    """Two-conv residual block with optional 1x1 skip projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class ConvPyramidEncoder(nn.Module):
    """Shared CNN pyramid template used by both content and style encoders."""

    def __init__(
        self,
        *,
        in_channels: int,
        stage_channels: Sequence[int],
        use_stem: bool,
        stem_stride: int = 1,
        output_grid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        normalized_stage_channels = [int(ch) for ch in stage_channels]
        if not normalized_stage_channels:
            raise ValueError("stage_channels must be non-empty")

        self.in_channels = int(in_channels)
        self.stage_channels = normalized_stage_channels
        self.output_grid_size = None if output_grid_size is None else int(output_grid_size)
        self.use_stem = bool(use_stem)
        self.stem_stride = int(stem_stride)
        if self.use_stem:
            if self.stem_stride <= 0:
                raise ValueError(f"stem_stride must be > 0, got {stem_stride}")
            stem_channels = self.stage_channels[0]
            self.stem = nn.Conv2d(
                self.in_channels,
                stem_channels,
                kernel_size=3,
                stride=self.stem_stride,
                padding=1,
            )
            self.stem_block = ResBlock(stem_channels, stem_channels)
            prev_channels = stem_channels
            stage_start_idx = 1
        else:
            self.stem = nn.Identity()
            self.stem_block = nn.Identity()
            prev_channels = self.in_channels
            stage_start_idx = 0

        self.stage_downsamples = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        for out_channels in self.stage_channels[stage_start_idx:]:
            self.stage_downsamples.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            self.stage_blocks.append(ResBlock(out_channels, out_channels))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")

        x = self.stem(x)
        x = self.stem_block(x)
        for downsample, resblock in zip(self.stage_downsamples, self.stage_blocks):
            x = downsample(x)
            x = resblock(x)
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
    ) -> None:
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
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
            use_stem=True,
            stem_stride=2 if self.downsample_depth > 0 else 1,
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
    ) -> None:
        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
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
            use_stem=True,
            stem_stride=2 if self.downsample_depth > 0 else 1,
            output_grid_size=int(output_grid_size),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class ContentStyleCrossAttention(nn.Module):
    """Cross-attention with content queries and style key/value tokens."""

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

    def forward(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if content_tokens.dim() != 3:
            raise ValueError(f"content_tokens must be 3D, got {tuple(content_tokens.shape)}")
        if style_tokens.dim() != 3:
            raise ValueError(f"style_tokens must be 3D, got {tuple(style_tokens.shape)}")
        if content_tokens.size(0) != style_tokens.size(0):
            raise ValueError(
                "content_tokens/style_tokens batch mismatch: "
                f"{tuple(content_tokens.shape)} vs {tuple(style_tokens.shape)}"
            )
        if token_valid_mask is not None:
            if token_valid_mask.shape != style_tokens.shape[:2]:
                raise ValueError(
                    f"token_valid_mask shape mismatch: expected {tuple(style_tokens.shape[:2])}, "
                    f"got {tuple(token_valid_mask.shape)}"
                )
            if bool((~token_valid_mask).any().item()):
                raise RuntimeError(
                    "ContentStyleCrossAttention requires all style refs to be valid in the current path."
                )

        style_context, _ = self.attn(
            self.query_norm(content_tokens),
            self.token_norm(style_tokens),
            style_tokens,
            key_padding_mask=None,
            need_weights=False,
        )
        return style_context


class SourcePartRefDiT(nn.Module):
    """Pure DiT glyph generator with external content-style fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 8,
        encoder_hidden_dim: int = 256,
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
        self.patch_grid_size = self.image_size // self.patch_size
        self.num_patches = self.patch_grid_size * self.patch_grid_size
        self.encoder_hidden_dim = int(encoder_hidden_dim)
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
        )
        self.style_encoder = StyleEncoder(
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
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
            hidden_dim=self.dit_hidden_dim,
            conditioning_dim=self.conditioning_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_injection_layers=self.content_injection_layers,
            ffn_activation=self.ffn_activation,
            norm_variant=self.norm_variant,
        )
        self.output_norm = nn.LayerNorm(self.dit_hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(self.dit_hidden_dim, self.output_patch_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def export_config(self) -> dict[str, int | float]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
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
        style_tokens = style_tokens.view(batch, refs * tokens_per_ref, self.style_token_hidden_dim)
        token_valid_mask = (
            ref_valid_mask[:, :, None]
            .expand(batch, refs, tokens_per_ref)
            .reshape(batch, refs * tokens_per_ref)
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
        return self.style_token_proj(style_tokens), token_valid_mask

    def fuse_content_style_tokens(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        *,
        token_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        style_context = self.content_style_attn(
            content_tokens,
            style_tokens,
            token_valid_mask=token_valid_mask,
        )
        fused_context = content_tokens + style_context
        return torch.cat([content_tokens, fused_context], dim=-1).contiguous()

    def decode_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be 3D, got {tuple(patch_tokens.shape)}")
        if tuple(patch_tokens.shape[1:]) != (self.num_patches, self.dit_hidden_dim):
            raise ValueError(
                "patch token shape mismatch: "
                f"expected (*, {self.num_patches}, {self.dit_hidden_dim}), got {tuple(patch_tokens.shape)}"
            )

        patch_pixels = self.output_proj(self.output_norm(patch_tokens))
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
    ) -> torch.Tensor:
        patch_tokens = self.backbone(
            x_t_image,
            timesteps,
            conditioning_tokens=conditioning_tokens,
        )
        return self.decode_patch_tokens(patch_tokens)

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
        conditioning_tokens = self.fuse_content_style_tokens(
            content_tokens,
            style_token_bank,
            token_valid_mask=token_valid_mask,
        )
        return self.predict_x(
            x_t_image,
            timesteps,
            conditioning_tokens=conditioning_tokens,
        )
