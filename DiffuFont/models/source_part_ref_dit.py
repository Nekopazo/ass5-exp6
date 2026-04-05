#!/usr/bin/env python3
"""Content+style pixel-space DiP model for Chinese glyph generation."""

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


class ConvSiLU(nn.Module):
    """Single convolution followed by SiLU, matching the paper's shallow U-Net blocks."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x))


class ContentEncoder(nn.Module):
    """CNN content encoder that downsamples to the DiT patch grid."""

    def __init__(
        self,
        *,
        image_size: int = 128,
        output_grid_size: int = 8,
        hidden_dim: int = 512,
        base_channels: int = 64,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
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

        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.downsample_depth = int(downsample_depth)

        self.stem_stride = 2 if self.downsample_depth > 0 else 1
        self.stem = nn.Conv2d(1, self.base_channels, kernel_size=3, stride=self.stem_stride, padding=1)
        self.stem_resblock = ResBlock(self.base_channels, self.base_channels)

        self.downsample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        in_channels = self.base_channels
        remaining_downsample_depth = max(0, self.downsample_depth - 1)
        if remaining_downsample_depth == 0:
            self.downsample_layers.append(nn.Identity())
            self.resblocks.append(ResBlock(in_channels, self.hidden_dim))
        else:
            stage_channels = []
            for stage_idx in range(remaining_downsample_depth):
                out_channels = self.hidden_dim if stage_idx == remaining_downsample_depth - 1 else min(
                    self.base_channels * (2 ** (stage_idx + 1)),
                    self.max_channels,
                )
                stage_channels.append(out_channels)
            if len(stage_channels) != remaining_downsample_depth:
                raise RuntimeError(
                    f"content stage construction mismatch: expected {remaining_downsample_depth}, got {len(stage_channels)}"
                )
            for stage_idx in range(remaining_downsample_depth):
                out_channels = int(stage_channels[stage_idx])
                self.downsample_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
                self.resblocks.append(ResBlock(out_channels, out_channels))
                in_channels = out_channels

        self.out_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")

        x = self.stem(x)
        x = self.stem_resblock(x)
        for downsample, resblock in zip(self.downsample_layers, self.resblocks):
            x = downsample(x)
            x = resblock(x)
        if x.shape[-2:] != (self.output_grid_size, self.output_grid_size):
            x = F.interpolate(
                x,
                size=(self.output_grid_size, self.output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return F.silu(self.out_norm(x))


class StyleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        stage_channels = [64, 128, 256, 384, self.hidden_dim]
        self.downsample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        prev_channels = int(in_channels)
        for out_channels in stage_channels:
            self.downsample_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            self.resblocks.append(ResBlock(out_channels, out_channels))
            prev_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        for downsample, resblock in zip(self.downsample_layers, self.resblocks):
            x = downsample(x)
            x = resblock(x)
        return x


class StyleAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if self.num_heads <= 0 or (self.hidden_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config hidden_dim={hidden_dim} num_heads={num_heads}")

        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) / math.sqrt(float(self.hidden_dim)))
        self.query_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.token_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAAttention(self.hidden_dim, self.num_heads)

    def forward(self, style_tokens: torch.Tensor, token_valid_mask: torch.Tensor) -> torch.Tensor:
        if style_tokens.dim() != 3:
            raise ValueError(f"style_tokens must be 3D, got {tuple(style_tokens.shape)}")
        if token_valid_mask.shape != style_tokens.shape[:2]:
            raise ValueError(
                f"token_valid_mask shape mismatch: expected {tuple(style_tokens.shape[:2])}, "
                f"got {tuple(token_valid_mask.shape)}"
            )

        query = self.query.to(device=style_tokens.device, dtype=style_tokens.dtype).expand(style_tokens.size(0), -1, -1)
        key_padding_mask = None
        need_weights = False
        if bool((~token_valid_mask).any().item()):
            key_padding_mask = ~token_valid_mask
            need_weights = True
        pooled_style, _ = self.attn(
            self.query_norm(query),
            self.token_norm(style_tokens),
            style_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return pooled_style.squeeze(1)


class PatchDetailerHead(nn.Module):
    """Per-patch local refiner conditioned on the final DiT token."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int = 16,
        context_dim: int = 512,
        base_channels: int = 32,
        max_channels: int = 256,
        bottleneck_channels: int = 384,
    ) -> None:
        super().__init__()
        if patch_size < 8:
            raise ValueError(f"patch_size must be >= 8, got {patch_size}")
        depth = int(round(math.log2(patch_size)))
        if 2**depth != int(patch_size):
            raise ValueError(f"patch_size must be a power of two, got {patch_size}")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.context_dim = int(context_dim)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.depth = int(depth)
        self.stage_channels = [
            min(self.base_channels * (2**idx), self.max_channels)
            for idx in range(self.depth)
        ]

        self.enc_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        prev_channels = self.in_channels
        for ch in self.stage_channels:
            self.enc_blocks.append(ConvSiLU(prev_channels, ch, kernel_size=3, padding=1))
            self.downsample_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = ch

        self.context_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.context_dim, self.bottleneck_channels),
        )
        self.bottleneck = ConvSiLU(
            self.stage_channels[-1] + self.bottleneck_channels,
            self.bottleneck_channels,
            kernel_size=3,
            padding=1,
        )

        self.upsample_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current_ch = self.bottleneck_channels
        for skip_ch in reversed(self.stage_channels):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(
                ConvSiLU(current_ch + skip_ch, skip_ch, kernel_size=3, padding=1)
            )
            current_ch = skip_ch

        self.out_proj = nn.Conv2d(current_ch, self.in_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, patch_tokens: torch.Tensor, noisy_patches: torch.Tensor) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be 3D, got {tuple(patch_tokens.shape)}")
        if noisy_patches.dim() != 5:
            raise ValueError(f"noisy_patches must be 5D, got {tuple(noisy_patches.shape)}")
        batch, patch_count, channels, patch_h, patch_w = noisy_patches.shape
        if channels != self.in_channels:
            raise ValueError(
                f"noisy_patches channel mismatch: expected {self.in_channels}, got {channels}"
            )
        if patch_h != self.patch_size or patch_w != self.patch_size:
            raise ValueError(
                f"noisy_patches spatial mismatch: expected ({self.patch_size}, {self.patch_size}), "
                f"got ({patch_h}, {patch_w})"
            )
        if tuple(patch_tokens.shape[:2]) != (batch, patch_count):
            raise ValueError(
                "patch_tokens and noisy_patches must share batch/patch dims: "
                f"patch_tokens={tuple(patch_tokens.shape)} noisy_patches={tuple(noisy_patches.shape)}"
            )

        flat_patches = noisy_patches.reshape(batch * patch_count, channels, patch_h, patch_w)
        flat_tokens = patch_tokens.reshape(batch * patch_count, patch_tokens.size(-1))

        skips: list[torch.Tensor] = []
        x = flat_patches
        for enc_block, downsample in zip(self.enc_blocks, self.downsample_layers):
            x = enc_block(x)
            skips.append(x)
            x = downsample(x)

        context = self.context_proj(flat_tokens).view(flat_tokens.size(0), -1, 1, 1)
        x = torch.cat([x, context], dim=1)
        x = self.bottleneck(x)

        for upsample, dec_block, skip in zip(self.upsample_layers, self.dec_blocks, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.out_proj(x)
        return x.view(batch, patch_count, self.in_channels, self.patch_size, self.patch_size)


class SourcePartRefDiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        encoder_hidden_dim: int = 512,
        style_hidden_dim: int = 684,
        dit_hidden_dim: int = 512,
        dit_depth: int = 12,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        content_injection_layers: Sequence[int] | None = None,
        style_injection_layers: Sequence[int] | None = None,
        detailer_base_channels: int = 32,
        detailer_max_channels: int = 256,
        detailer_bottleneck_channels: int = 384,
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
        self.style_hidden_dim = int(style_hidden_dim)
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.dit_depth = int(dit_depth)
        self.dit_heads = int(dit_heads)
        self.dit_mlp_ratio = float(dit_mlp_ratio)
        self.detailer_base_channels = int(detailer_base_channels)
        self.detailer_max_channels = int(detailer_max_channels)
        self.detailer_bottleneck_channels = int(detailer_bottleneck_channels)
        self.content_injection_layers = DiffusionTransformerBackbone._normalize_layer_indices(
            content_injection_layers,
            default_layers=range(1, min(self.dit_depth, 6) + 1),
            depth=self.dit_depth,
            field_name="content_injection_layers",
        )
        self.style_injection_layers = DiffusionTransformerBackbone._normalize_layer_indices(
            style_injection_layers,
            default_layers=range(max(1, self.dit_depth - 5), self.dit_depth + 1),
            depth=self.dit_depth,
            field_name="style_injection_layers",
        )

        self.content_encoder = ContentEncoder(
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
        )
        self.style_encoder = StyleEncoder(
            in_channels=self.in_channels,
            hidden_dim=self.style_hidden_dim,
        )
        self.style_pool_heads = self._resolve_style_pool_heads(self.style_hidden_dim, self.dit_heads)
        self.style_pool = StyleAttentionPool(
            hidden_dim=self.style_hidden_dim,
            num_heads=self.style_pool_heads,
        )
        self.backbone = DiffusionTransformerBackbone(
            in_channels=self.in_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.dit_hidden_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_injection_layers=self.content_injection_layers,
            style_injection_layers=self.style_injection_layers,
        )
        self.detailer = PatchDetailerHead(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            context_dim=self.dit_hidden_dim,
            base_channels=self.detailer_base_channels,
            max_channels=self.detailer_max_channels,
            bottleneck_channels=self.detailer_bottleneck_channels,
        )

        if self.encoder_hidden_dim != self.dit_hidden_dim:
            self.content_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
        else:
            self.content_proj = nn.Identity()
        if self.style_hidden_dim != self.dit_hidden_dim:
            self.style_proj = nn.Linear(self.style_hidden_dim, self.dit_hidden_dim)
        else:
            self.style_proj = nn.Identity()

    def export_config(self) -> dict[str, int | float]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "style_hidden_dim": int(self.style_hidden_dim),
            "dit_hidden_dim": int(self.dit_hidden_dim),
            "dit_depth": int(self.dit_depth),
            "dit_heads": int(self.dit_heads),
            "dit_mlp_ratio": float(self.dit_mlp_ratio),
            "content_injection_layers": list(self.content_injection_layers),
            "style_injection_layers": list(self.style_injection_layers),
            "detailer_base_channels": int(self.detailer_base_channels),
            "detailer_max_channels": int(self.detailer_max_channels),
            "detailer_bottleneck_channels": int(self.detailer_bottleneck_channels),
        }

    @staticmethod
    def _resolve_style_pool_heads(style_hidden_dim: int, max_heads: int) -> int:
        style_hidden_dim = int(style_hidden_dim)
        max_heads = max(1, int(max_heads))
        for num_heads in range(min(style_hidden_dim, max_heads), 0, -1):
            if style_hidden_dim % num_heads == 0:
                return num_heads
        return 1

    def encode_content_tokens(self, content_img: torch.Tensor) -> torch.Tensor:
        content_features = self.content_encoder(content_img)
        return content_features.flatten(2).transpose(1, 2).contiguous()

    def encode_style_global(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        style_features = self.style_encoder(flat_style)
        style_tokens = style_features.flatten(2).transpose(1, 2).contiguous()
        tokens_per_ref = int(style_tokens.size(1))
        style_tokens = style_tokens.view(batch, refs * tokens_per_ref, self.style_hidden_dim)
        token_valid_mask = (
            ref_valid_mask[:, :, None]
            .expand(batch, refs, tokens_per_ref)
            .reshape(batch, refs * tokens_per_ref)
        )
        pooled_style = self.style_pool(style_tokens, token_valid_mask=token_valid_mask)
        pooled_style = pooled_style.to(dtype=style_features.dtype)
        return self.style_proj(pooled_style)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2).contiguous()
        return patches.view(
            x.size(0),
            self.num_patches,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        if patch_values.dim() != 5:
            raise ValueError(f"patch_values must be 5D, got {tuple(patch_values.shape)}")
        batch, patch_count, channels, patch_h, patch_w = patch_values.shape
        if patch_count != self.num_patches:
            raise ValueError(f"patch_count mismatch: expected {self.num_patches}, got {patch_count}")
        if channels != self.in_channels:
            raise ValueError(f"channel mismatch: expected {self.in_channels}, got {channels}")
        if patch_h != self.patch_size or patch_w != self.patch_size:
            raise ValueError(
                f"patch spatial mismatch: expected ({self.patch_size}, {self.patch_size}), "
                f"got ({patch_h}, {patch_w})"
            )
        patch_cols = patch_values.view(batch, patch_count, -1).transpose(1, 2).contiguous()
        return F.fold(
            patch_cols,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def predict_flow(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_global: torch.Tensor,
        return_injection_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        backbone_output = self.backbone(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_global=style_global,
            return_injection_stats=return_injection_stats,
        )
        if return_injection_stats:
            patch_tokens, injection_stats = backbone_output
        else:
            patch_tokens = backbone_output
            injection_stats = {}
        noisy_patches = self._patchify(x_t_image)
        pred_patches = self.detailer(patch_tokens, noisy_patches)
        pred_flow = self._unpatchify(pred_patches)
        if return_injection_stats:
            return pred_flow, injection_stats
        return pred_flow

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
        content_tokens = self.content_proj(content_tokens)
        style_global = self.encode_style_global(
            style_img,
            style_ref_mask=style_ref_mask,
        )
        return self.predict_flow(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_global=style_global,
        )
