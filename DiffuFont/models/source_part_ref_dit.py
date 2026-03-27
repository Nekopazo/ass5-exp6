#!/usr/bin/env python3
"""Content+style pixel-space DiP model for Chinese glyph generation."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_transformer_backbone import DiffusionTransformerBackbone, build_2d_sincos_pos_embed
from .sdpa_attention import SDPAAttention


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
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


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SDPAAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GlyphTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 128,
        patch_size: int = 8,
        hidden_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        add_cls_token: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.add_cls_token = bool(add_cls_token)

        self.patch_embed = nn.Conv2d(1, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        pos_embed = build_2d_sincos_pos_embed(self.hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("patch_pos_embed", pos_embed.unsqueeze(0), persistent=False)

        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        else:
            self.cls_token = None
            self.cls_pos_embed = None

        self.blocks = nn.ModuleList(
            [EncoderBlock(self.hidden_dim, int(num_heads), float(mlp_ratio)) for _ in range(int(depth))]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.patch_pos_embed.to(device=tokens.device, dtype=tokens.dtype)

        if self.add_cls_token:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            cls = cls + self.cls_pos_embed.to(device=tokens.device, dtype=tokens.dtype)
            tokens = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        if self.add_cls_token:
            return tokens[:, 1:], tokens[:, 0]
        return tokens, None


class StyleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResBlock(256, 256),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            ResBlock(384, 384),
            nn.Conv2d(384, self.hidden_dim, kernel_size=3, stride=2, padding=1),
            ResBlock(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        return self.net(x)


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
        self.depth = int(depth)
        self.stage_channels = [
            min(self.base_channels * (2**idx), self.max_channels)
            for idx in range(self.depth)
        ]

        self.stem = nn.Conv2d(self.in_channels, self.stage_channels[0], kernel_size=3, padding=1)
        self.enc_blocks = nn.ModuleList([ResBlock(ch, ch) for ch in self.stage_channels])
        self.downsamples = nn.ModuleList()
        for idx, ch in enumerate(self.stage_channels):
            next_ch = self.stage_channels[min(idx + 1, len(self.stage_channels) - 1)]
            self.downsamples.append(nn.Conv2d(ch, next_ch, kernel_size=3, stride=2, padding=1))

        bottleneck_channels = self.stage_channels[-1]
        self.context_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.context_dim, bottleneck_channels),
        )
        self.bottleneck = ResBlock(bottleneck_channels * 2, bottleneck_channels)

        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current_ch = bottleneck_channels
        for skip_ch in reversed(self.stage_channels):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_ch, skip_ch, kernel_size=3, padding=1),
                )
            )
            self.dec_blocks.append(ResBlock(skip_ch + skip_ch, skip_ch))
            current_ch = skip_ch

        self.out_norm = nn.GroupNorm(_group_count(current_ch), current_ch)
        self.out_proj = nn.Conv2d(current_ch, self.in_channels, kernel_size=3, padding=1)

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

        x = self.stem(flat_patches)
        skips: list[torch.Tensor] = []
        for enc_block, downsample in zip(self.enc_blocks, self.downsamples):
            x = enc_block(x)
            skips.append(x)
            x = downsample(x)

        context = self.context_proj(flat_tokens).view(flat_tokens.size(0), -1, 1, 1)
        x = torch.cat([x, context], dim=1)
        x = self.bottleneck(x)

        for upsample, dec_block, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = F.silu(self.out_norm(x))
        x = self.out_proj(x)
        return x.view(batch, patch_count, self.in_channels, self.patch_size, self.patch_size)


class SourcePartRefDiT(nn.Module):
    STYLE_STATE_PREFIXES = (
        "style_encoder.",
        "style_proj.",
        "style_global_proj.",
        "style_contrastive_head.",
    )

    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        encoder_patch_size: int = 8,
        encoder_hidden_dim: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        dit_hidden_dim: int = 512,
        dit_depth: int = 12,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        content_fusion_start: int | None = None,
        content_fusion_end: int | None = None,
        style_fusion_start: int | None = None,
        style_fusion_end: int | None = None,
        content_cross_attn_layers: int | None = None,
        style_cross_attn_every_n_layers: int | None = None,
        contrastive_proj_dim: int = 128,
        detailer_base_channels: int = 32,
        detailer_max_channels: int = 256,
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
        self.encoder_patch_size = int(encoder_patch_size)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.encoder_depth = int(encoder_depth)
        self.encoder_heads = int(encoder_heads)
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.dit_depth = int(dit_depth)
        self.dit_heads = int(dit_heads)
        self.dit_mlp_ratio = float(dit_mlp_ratio)
        self.contrastive_proj_dim = int(contrastive_proj_dim)
        self.detailer_base_channels = int(detailer_base_channels)
        self.detailer_max_channels = int(detailer_max_channels)

        if content_fusion_start is None and content_fusion_end is None:
            if content_cross_attn_layers is None:
                self.content_fusion_start = 0
                self.content_fusion_end = min(self.dit_depth, 8)
            else:
                self.content_fusion_start = 0
                self.content_fusion_end = max(0, min(self.dit_depth, int(content_cross_attn_layers)))
        else:
            self.content_fusion_start = max(
                0,
                min(self.dit_depth, 0 if content_fusion_start is None else int(content_fusion_start)),
            )
            self.content_fusion_end = max(
                self.content_fusion_start,
                min(self.dit_depth, self.dit_depth if content_fusion_end is None else int(content_fusion_end)),
            )

        if style_fusion_start is None and style_fusion_end is None:
            if style_cross_attn_every_n_layers is None:
                self.style_fusion_start = max(0, self.dit_depth - 6)
                self.style_fusion_end = self.dit_depth
            else:
                self.style_fusion_start = 0
                self.style_fusion_end = self.dit_depth
        else:
            self.style_fusion_start = max(
                0,
                min(self.dit_depth, 0 if style_fusion_start is None else int(style_fusion_start)),
            )
            self.style_fusion_end = max(
                self.style_fusion_start,
                min(self.dit_depth, self.dit_depth if style_fusion_end is None else int(style_fusion_end)),
            )

        self.content_cross_attn_layers = self.content_fusion_end - self.content_fusion_start
        self.style_cross_attn_every_n_layers = (
            None if style_cross_attn_every_n_layers is None else max(1, int(style_cross_attn_every_n_layers))
        )

        self.content_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=False,
        )
        self.style_encoder = StyleEncoder(
            in_channels=self.in_channels,
            hidden_dim=self.encoder_hidden_dim,
        )
        self.backbone = DiffusionTransformerBackbone(
            in_channels=self.in_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.dit_hidden_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_fusion_start=self.content_fusion_start,
            content_fusion_end=self.content_fusion_end,
            style_fusion_start=self.style_fusion_start,
            style_fusion_end=self.style_fusion_end,
            content_cross_attn_layers=self.content_cross_attn_layers,
            style_cross_attn_every_n_layers=self.style_cross_attn_every_n_layers,
        )
        self.detailer = PatchDetailerHead(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            context_dim=self.dit_hidden_dim,
            base_channels=self.detailer_base_channels,
            max_channels=self.detailer_max_channels,
        )

        if self.encoder_hidden_dim != self.dit_hidden_dim:
            self.content_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
            self.style_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
            self.style_global_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
        else:
            self.content_proj = nn.Identity()
            self.style_proj = nn.Identity()
            self.style_global_proj = nn.Identity()
        self.style_contrastive_head = nn.Sequential(
            nn.Linear(self.dit_hidden_dim, self.dit_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dit_hidden_dim, self.contrastive_proj_dim),
        )

    def export_config(self) -> dict[str, int | float]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "encoder_patch_size": int(self.encoder_patch_size),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "encoder_depth": int(self.encoder_depth),
            "encoder_heads": int(self.encoder_heads),
            "dit_hidden_dim": int(self.dit_hidden_dim),
            "dit_depth": int(self.dit_depth),
            "dit_heads": int(self.dit_heads),
            "dit_mlp_ratio": float(self.dit_mlp_ratio),
            "content_fusion_start": int(self.content_fusion_start),
            "content_fusion_end": int(self.content_fusion_end),
            "style_fusion_start": int(self.style_fusion_start),
            "style_fusion_end": int(self.style_fusion_end),
            "contrastive_proj_dim": int(self.contrastive_proj_dim),
            "detailer_base_channels": int(self.detailer_base_channels),
            "detailer_max_channels": int(self.detailer_max_channels),
        }

    @classmethod
    def _is_style_state_key(cls, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in cls.STYLE_STATE_PREFIXES)

    def style_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if self._is_style_state_key(key)
        }

    def load_style_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict) or "style_state" not in checkpoint:
            raise RuntimeError(f"Style checkpoint must contain 'style_state': {path}")
        state_dict = checkpoint["style_state"]

        if not state_dict:
            raise RuntimeError(f"No style encoder weights found in checkpoint: {path}")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        unexpected_style_keys = [key for key in unexpected_keys if self._is_style_state_key(key)]
        if unexpected_style_keys:
            raise RuntimeError(
                "Unexpected style checkpoint keys: "
                f"{unexpected_style_keys}. The style encoder architecture likely changed."
            )
        missing_style_keys = [
            key
            for key in missing_keys
            if self._is_style_state_key(key)
        ]
        if missing_style_keys:
            raise RuntimeError(
                "Missing style checkpoint keys: "
                f"{missing_style_keys}. The style encoder architecture likely changed; "
                "rerun style pretraining for the new aggregator."
            )

    def freeze_style(self) -> None:
        for name, param in self.named_parameters():
            if self._is_style_state_key(name):
                param.requires_grad_(False)

    def encode_content(self, content_img: torch.Tensor) -> torch.Tensor:
        return self.content_proj(self.encode_content_features(content_img))

    def encode_content_features(self, content_img: torch.Tensor) -> torch.Tensor:
        content_tokens, _ = self.content_encoder(content_img)
        return content_tokens

    def encode_style(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        return_contrastive: bool = False,
        detach_style_encoder: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")

        batch, refs, channels, height, width = style_img.shape
        flat_style = style_img.view(batch * refs, channels, height, width)
        if style_ref_mask is None:
            ref_valid_mask = torch.ones((batch, refs), device=style_img.device, dtype=torch.bool)
        else:
            ref_valid_mask = style_ref_mask.to(device=style_img.device) > 0.5
        empty_rows = ~ref_valid_mask.any(dim=1)
        if empty_rows.any():
            ref_valid_mask = ref_valid_mask.clone()
            ref_valid_mask[empty_rows, 0] = True

        style_context = torch.no_grad() if detach_style_encoder else nullcontext()
        with style_context:
            style_features = self.style_encoder(flat_style)
            style_vectors = F.adaptive_avg_pool2d(style_features, output_size=1).flatten(1)
            style_vectors = F.normalize(style_vectors, dim=-1, eps=1e-6)
            style_vectors = style_vectors.view(batch, refs, style_vectors.size(-1))
            ref_weights = ref_valid_mask.to(device=style_vectors.device, dtype=style_vectors.dtype)
            ref_weight_sum = ref_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_style = (style_vectors * ref_weights.unsqueeze(-1)).sum(dim=1) / ref_weight_sum
            style_tokens = self.style_proj(pooled_style).unsqueeze(1)
            style_global = self.style_global_proj(pooled_style)
            style_token_mask = torch.ones((batch, 1), device=style_tokens.device, dtype=torch.bool)

        contrastive_style = None
        if return_contrastive:
            contrastive_style = F.normalize(self.style_contrastive_head(style_global), dim=-1)
        return {
            "style_tokens": style_tokens,
            "style_global": style_global,
            "style_token_mask": style_token_mask,
            "contrastive_style": contrastive_style,
        }

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
        style_tokens: torch.Tensor,
        style_global: torch.Tensor,
        style_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patch_tokens = self.backbone(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
            style_token_mask=style_token_mask,
        )
        noisy_patches = self._patchify(x_t_image)
        pred_patches = self.detailer(patch_tokens, noisy_patches)
        return self._unpatchify(pred_patches)

    def forward(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        content_img: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content_features = self.encode_content_features(content_img)
        content_tokens = self.content_proj(content_features)
        style_pack = self.encode_style(
            style_img,
            style_ref_mask=style_ref_mask,
        )
        return self.predict_flow(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_pack["style_tokens"],
            style_global=style_pack["style_global"],
            style_token_mask=style_pack["style_token_mask"],
        )
