#!/usr/bin/env python3
"""Content+style diffusion transformer for Chinese glyph generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_transformer_backbone import (
    DiffusionTransformerBackbone,
    FeedForward,
    _build_norm,
    _build_zero_linear,
    modulate,
)
from .sdpa_attention import SDPAAttention, VisionRotaryEmbeddingFast


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _build_encoder_norm(channels: int, *, norm_type: str) -> nn.Module:
    if str(norm_type) != "group":
        raise ValueError(f"Unsupported encoder norm_type: {norm_type!r}")
    return nn.GroupNorm(_group_count(channels), channels)


class ConvBlock(nn.Module):
    """Convolutional encoder block with optional spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_type: str,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.main = nn.Sequential(
            _build_encoder_norm(self.in_channels, norm_type=norm_type),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=2 if downsample else 1,
                padding=1,
                bias=False,
            ),
            _build_encoder_norm(self.out_channels, norm_type=norm_type),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ResBlock(nn.Module):
    """Residual block with optional spatial downsampling on the skip path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_type: str,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.downsample = bool(downsample)
        stride = 2 if self.downsample else 1
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _build_encoder_norm(self.in_channels, norm_type=norm_type)
        self.norm2 = _build_encoder_norm(self.out_channels, norm_type=norm_type)
        if self.downsample or self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block matching the DiT backbone's attention/MLP style."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_variant: str = "rms",
    ) -> None:
        super().__init__()
        self.norm_attn = _build_norm(hidden_dim, norm_variant=norm_variant)
        self.norm_mlp = _build_norm(hidden_dim, norm_variant=norm_variant)
        self.attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio, activation="swiglu")

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: VisionRotaryEmbeddingFast,
    ) -> torch.Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, query_rope=rope, key_rope=rope, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CustomResidualGlyphEncoder(nn.Module):
    """Fixed RGB glyph transformer encoder producing a 16x16 map of 256-dim tokens."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 256,
        output_grid_size: int = 16,
        hidden_dim: int = 256,
        block_depth: int = 4,
        norm_type: str = "group",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(f"CustomResidualGlyphEncoder requires RGB input, got {in_channels}")
        if int(image_size) != 256:
            raise ValueError(f"CustomResidualGlyphEncoder is fixed to image_size=256, got {image_size}")
        if int(output_grid_size) != 16:
            raise ValueError(f"CustomResidualGlyphEncoder is fixed to output_grid_size=16, got {output_grid_size}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(block_depth) <= 0:
            raise ValueError(f"block_depth must be positive, got {block_depth}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.block_depth = int(block_depth)
        self.downsample_depth = 4
        self.local_hidden_dim = self.hidden_dim
        self.norm_type = str(norm_type)
        self.num_heads = int(num_heads)
        self.patch_size = self.image_size // self.output_grid_size
        if self.patch_size * self.output_grid_size != self.image_size:
            raise ValueError(
                "CustomResidualGlyphEncoder requires image_size divisible by output_grid_size, "
                f"got image_size={self.image_size}, output_grid_size={self.output_grid_size}"
            )
        self.num_tokens = self.output_grid_size * self.output_grid_size
        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_embed = nn.Linear(patch_dim, self.hidden_dim)
        head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0 or head_dim % 4 != 0:
            raise ValueError(
                "Encoder requires hidden_dim divisible by num_heads and head dim divisible by 4 for RoPE, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"
            )
        self.encoder_rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.output_grid_size,
        )
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                norm_variant="rms",
            )
            for _ in range(self.block_depth)
        )
        self.final_norm = _build_norm(self.hidden_dim, norm_variant="rms")

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        return patch_tokens.contiguous()

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        if x.shape[1:] != (3, 256, 256):
            raise ValueError(f"expected RGB 256x256 glyph tensor, got {tuple(x.shape)}")
        patch_tokens = self.patch_embed(self._patchify(x))
        for block in self.blocks:
            patch_tokens = block(patch_tokens, rope=self.encoder_rope)
        patch_tokens = self.final_norm(patch_tokens)
        return [patch_tokens]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[-1]


class ContentEncoder(CustomResidualGlyphEncoder):
    """Content glyph encoder with the shared architecture and separate weights."""

    def __init__(self, **kwargs) -> None:
        if "hidden_dim" not in kwargs:
            kwargs["hidden_dim"] = 512
        if "block_depth" not in kwargs:
            kwargs["block_depth"] = 4
        if "num_heads" not in kwargs:
            kwargs["num_heads"] = 8
        super().__init__(norm_type="group", **kwargs)


class StyleEncoder(CustomResidualGlyphEncoder):
    """Style glyph encoder with the shared architecture and separate weights."""

    def __init__(self, **kwargs) -> None:
        if "hidden_dim" not in kwargs:
            kwargs["hidden_dim"] = 512
        if "block_depth" not in kwargs:
            kwargs["block_depth"] = 4
        if "num_heads" not in kwargs:
            kwargs["num_heads"] = 8
        super().__init__(norm_type="group", **kwargs)


class ContentStyleCrossAttention(nn.Module):
    """External content<-style fusion utilities for concat cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, *, grid_size: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.grid_size = int(grid_size)
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.num_heads <= 0 or (self.embed_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config embed_dim={embed_dim} num_heads={num_heads}")
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        head_dim = self.embed_dim // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "JiT-style 2D RoPE requires cross-attention head dim divisible by 4, "
                f"got embed_dim={embed_dim} num_heads={num_heads}"
            )
        self.attn = SDPAAttention(self.embed_dim, self.num_heads)
        self.fusion_norm = _build_norm(self.embed_dim, norm_variant="rms")
        self.fusion_mlp = FeedForward(self.embed_dim, 4.0, activation="swiglu")
        self.self_attn_norm = _build_norm(self.embed_dim, norm_variant="rms")
        self.self_attn = SDPAAttention(self.embed_dim, self.num_heads)
        self.self_mlp_norm = _build_norm(self.embed_dim, norm_variant="rms")
        self.self_mlp = FeedForward(self.embed_dim, 4.0, activation="swiglu")
        self.rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.grid_size,
        )

    def _post_fusion(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        fused_tokens = fused_tokens + self.fusion_mlp(self.fusion_norm(fused_tokens))
        self_attn_in = self.self_attn_norm(fused_tokens)
        self_attn_out, _ = self.self_attn(
            self_attn_in,
            self_attn_in,
            self_attn_in,
            query_rope=self.rope,
            key_rope=self.rope,
            need_weights=False,
        )
        fused_tokens = fused_tokens + self_attn_out
        return fused_tokens + self.self_mlp(self.self_mlp_norm(fused_tokens))

    def _validate_style_inputs(self, style_tokens: torch.Tensor) -> None:
        if style_tokens.dim() != 4:
            raise ValueError(f"style_tokens must be 4D [B, R, T, D], got {tuple(style_tokens.shape)}")

    def project_style_bank_kv(
        self,
        style_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_style_inputs(style_tokens)
        batch_size, num_refs, tokens_per_ref, hidden_dim = style_tokens.shape
        concat_tokens = style_tokens.reshape(batch_size, num_refs * tokens_per_ref, hidden_dim)
        key, value = self.attn.project_key_value(concat_tokens, concat_tokens)
        key = self.rope(key)
        concat_len = int(key.size(2))
        return (
            key.view(batch_size, 1, self.num_heads, concat_len, self.attn.head_dim),
            value.view(batch_size, 1, self.num_heads, concat_len, self.attn.head_dim),
        )

    def project_content_query(self, content_tokens: torch.Tensor) -> torch.Tensor:
        if content_tokens.dim() != 3:
            raise ValueError(f"content_tokens must be 3D [B, T, D], got {tuple(content_tokens.shape)}")
        query = self.attn.project_query(content_tokens)
        return self.rope(query)

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
            need_weights=False,
        )
        fused_tokens = style_context.view(batch_size, query_len, self.embed_dim).contiguous()
        return self._post_fusion(fused_tokens)


class SourcePartRefDiT(nn.Module):
    """Pure DiT glyph generator with external content-style fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 256,
        patch_size: int = 16,
        patch_embed_bottleneck_dim: int = 0,
        encoder_hidden_dim: int = 512,
        content_encoder_block_depth: int = 4,
        style_encoder_block_depth: int = 4,
        dit_hidden_dim: int = 256,
        dit_depth: int = 12,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
        content_injection_layers: Sequence[int] | None = None,
        conditioning_injection_mode: str = "all",
        content_style_fusion_heads: int = 8,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(f"Only RGB glyphs are supported, got in_channels={in_channels}")
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
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.content_encoder_block_depth,
            num_heads=self.content_style_fusion_heads,
        )
        self.style_encoder = StyleEncoder(
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.style_encoder_block_depth,
            num_heads=self.content_style_fusion_heads,
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
            grid_size=self.patch_grid_size,
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
        if self.backbone.patch_embed_proj1.bias is not None:
            nn.init.constant_(self.backbone.patch_embed_proj1.bias, 0)
        if self.backbone.use_patch_embed_bottleneck:
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
        return self.content_encoder(content_img)

    def _encode_style_features(self, style_img: torch.Tensor) -> torch.Tensor:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")

        batch, refs, channels, height, width = style_img.shape
        if refs <= 0:
            raise RuntimeError("style_img must contain at least one reference per sample")

        flat_style = style_img.view(batch * refs, channels, height, width)
        style_tokens = self.style_encoder(flat_style)
        tokens_per_ref = int(style_tokens.size(1))
        style_tokens = style_tokens.view(batch, refs, tokens_per_ref, self.style_token_hidden_dim)
        return style_tokens

    def encode_style_token_bank(self, style_img: torch.Tensor) -> torch.Tensor:
        style_tokens = self._encode_style_features(style_img)
        return self.style_token_proj(style_tokens).contiguous()

    def precompute_style_bank_kv(
        self,
        style_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.content_style_attn.project_style_bank_kv(style_tokens)

    def precompute_content_query(
        self,
        content_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.content_style_attn.project_content_query(content_tokens)

    def build_style_condition_tokens(
        self,
        content_tokens: torch.Tensor,
        style_token_bank: Optional[torch.Tensor] = None,
        *,
        content_query: Optional[torch.Tensor] = None,
        style_key: Optional[torch.Tensor] = None,
        style_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if content_query is None:
            content_query = self.precompute_content_query(content_tokens)
        if style_key is None or style_value is None:
            if style_token_bank is None:
                raise ValueError("style_token_bank is required when precomputed style_key/style_value are not provided")
            style_key, style_value = self.precompute_style_bank_kv(style_token_bank)
        return self.content_style_attn.fuse_content_style_tokens_from_preprojected_query(
            content_tokens,
            content_query,
            style_key,
            style_value,
        )

    def precompute_backbone_condition_hidden_cache(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor | None]:
        return self.backbone.build_condition_hidden_cache(
            content_tokens,
            style_tokens,
            batch_size=int(content_tokens.size(0)),
            token_count=int(content_tokens.size(1)),
            device=device,
            dtype=dtype,
        )

    def precompute_backbone_unique_content_hidden_cache(
        self,
        unique_content_tokens: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor | None]:
        return self.backbone.build_unique_content_hidden_cache(
            unique_content_tokens,
            token_count=self.num_patches,
            device=device,
            dtype=dtype,
        )

    def precompute_output_condition_hidden(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_part_shape = (content_tokens.size(0), self.num_patches, self.output_condition_half_dim)
        if content_tokens.shape != expected_part_shape:
            raise ValueError(
                "content token shape mismatch for final head: "
                f"expected {expected_part_shape}, got {tuple(content_tokens.shape)}"
            )
        if style_tokens.shape != expected_part_shape:
            raise ValueError(
                "style token shape mismatch for final head: "
                f"expected {expected_part_shape}, got {tuple(style_tokens.shape)}"
            )
        content_tokens = content_tokens.to(device=device, dtype=dtype)
        style_tokens = style_tokens.to(device=device, dtype=dtype)
        return self.output_content_condition_to_hidden(content_tokens) + self.output_style_condition_to_hidden(style_tokens)

    def decode_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        output_condition_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be 3D, got {tuple(patch_tokens.shape)}")
        if tuple(patch_tokens.shape[1:]) != (self.num_patches, self.dit_hidden_dim):
            raise ValueError(
                "patch token shape mismatch: "
                f"expected (*, {self.num_patches}, {self.dit_hidden_dim}), got {tuple(patch_tokens.shape)}"
            )
        expected_part_shape = (patch_tokens.size(0), self.num_patches, self.output_condition_half_dim)
        if content_tokens.shape != expected_part_shape:
            raise ValueError(
                "content token shape mismatch for final head: "
                f"expected {expected_part_shape}, got {tuple(content_tokens.shape)}"
            )
        if style_tokens.shape != expected_part_shape:
            raise ValueError(
                "style token shape mismatch for final head: "
                f"expected {expected_part_shape}, got {tuple(style_tokens.shape)}"
            )

        time_hidden = self.output_time_to_hidden(
            self.backbone.build_time_cond(
                timesteps,
                dtype=patch_tokens.dtype,
            )
        ).unsqueeze(1)
        if output_condition_hidden is None:
            content_tokens = content_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            style_tokens = style_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            content_hidden = self.output_content_condition_to_hidden(content_tokens)
            style_hidden = self.output_style_condition_to_hidden(style_tokens)
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

    def predict(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        backbone_condition_hidden_cache: Optional[list[torch.Tensor | None]] = None,
        backbone_unique_content_hidden_cache: Optional[list[torch.Tensor | None]] = None,
        content_index: Optional[torch.Tensor] = None,
        output_condition_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patch_tokens = self.backbone(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            condition_hidden_cache=backbone_condition_hidden_cache,
            unique_content_hidden_cache=backbone_unique_content_hidden_cache,
            content_index=content_index,
        )
        return self.decode_patch_tokens(
            patch_tokens,
            timesteps=timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            output_condition_hidden=output_condition_hidden,
        )

    def forward(
        self,
        x_t_image: torch.Tensor,
        timesteps: torch.Tensor,
        content_img: torch.Tensor,
        *,
        style_img: torch.Tensor,
    ) -> torch.Tensor:
        content_tokens = self.encode_content_tokens(content_img)
        style_token_bank = self.encode_style_token_bank(style_img)
        style_tokens = self.build_style_condition_tokens(
            content_tokens,
            style_token_bank,
        )
        return self.predict(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
        )
