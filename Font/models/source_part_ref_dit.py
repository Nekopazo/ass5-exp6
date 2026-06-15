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


class ResDownBlock(nn.Module):
    """A simple conv block with conv downsampling followed by one regular conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.main = nn.Sequential(
            nn.GroupNorm(_group_count(self.in_channels), self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(self.out_channels), self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=int(stride), padding=1, bias=False),
            nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ResDownBlockResidualAdd(nn.Module):
    """Same conv stack as ResDownBlock, but add the second conv output back to the first conv result."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_norm = nn.GroupNorm(_group_count(int(in_channels)), int(in_channels))
        self.in_act = nn.SiLU()
        self.down_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=2, padding=1, bias=False)
        self.out_norm = nn.GroupNorm(_group_count(int(out_channels)), int(out_channels))
        self.out_act = nn.SiLU()
        self.refine_conv = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down_conv(self.in_act(self.in_norm(x)))
        refined = self.refine_conv(self.out_act(self.out_norm(hidden)))
        return hidden + refined


class ResNetDownBlock(nn.Module):
    """ResNet-style downsampling block with a projected residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_norm = nn.GroupNorm(_group_count(int(in_channels)), int(in_channels))
        self.in_act = nn.SiLU()
        self.main_conv1 = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.out_norm = nn.GroupNorm(_group_count(int(out_channels)), int(out_channels))
        self.out_act = nn.SiLU()
        self.main_conv2 = nn.Conv2d(
            int(out_channels),
            int(out_channels),
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.shortcut = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=1,
            stride=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        hidden = self.main_conv1(self.in_act(self.in_norm(x)))
        hidden = self.main_conv2(self.out_act(self.out_norm(hidden)))
        return shortcut + hidden


class DWResBlock(nn.Module):
    """Depthwise residual block that preserves spatial size and channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.main = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)


class LiteDWResDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, use_dwres: bool) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, stride=2)
        if use_dwres:
            self.refine = DWResBlock(out_channels)
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
                nn.SiLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class CnnGlyphEncoder(nn.Module):
    """CNN glyph encoder with the original 32-channel stem and three downsample-first blocks."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 128,
        output_grid_size: int = 16,
        hidden_dim: int = 256,
        block_depth: int = 3,
        encoder_variant: str = "full_conv",
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(f"CnnGlyphEncoder requires RGB input, got {in_channels}")
        if int(image_size) != 128:
            raise ValueError(f"CnnGlyphEncoder is fixed to image_size=128, got {image_size}")
        if int(output_grid_size) != 16:
            raise ValueError(f"CnnGlyphEncoder is fixed to output_grid_size=16, got {output_grid_size}")
        if int(hidden_dim) != 256:
            raise ValueError(f"CnnGlyphEncoder is fixed to hidden_dim=256, got {hidden_dim}")
        base_channels: int = 64
        max_channels: int = 256
        if int(base_channels) != 64 or int(max_channels) != 256:
            raise ValueError(
                "CnnGlyphEncoder is fixed to base_channels=64 and max_channels=256, "
                f"got {base_channels} and {max_channels}"
            )
        if int(block_depth) != 3:
            raise ValueError(f"CnnGlyphEncoder is fixed to 3 ResDownBlocks, got {block_depth}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.encoder_variant = str(encoder_variant)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.block_depth = int(block_depth)
        self.downsample_depth = 3
        self.local_hidden_dim = self.hidden_dim
        self.num_tokens = self.output_grid_size * self.output_grid_size
        if self.encoder_variant in {"full_conv", "full_conv_stage_res", "full_resnet"}:
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(32), 32),
                nn.SiLU(),
            )
            if self.encoder_variant == "full_conv":
                self.blocks = nn.ModuleList((ResDownBlock(32, 64), ResDownBlock(64, 128), ResDownBlock(128, 256)))
            elif self.encoder_variant == "full_resnet":
                self.blocks = nn.ModuleList(
                    (
                        ResNetDownBlock(32, 64),
                        ResNetDownBlock(64, 128),
                        ResNetDownBlock(128, 256),
                    )
                )
            else:
                self.blocks = nn.ModuleList(
                    (
                        ResDownBlockResidualAdd(32, 64),
                        ResDownBlockResidualAdd(64, 128),
                        ResDownBlockResidualAdd(128, 256),
                    )
                )
        elif self.encoder_variant in {"lite_dwres", "lite_dwres_all"}:
            self.stem = ConvNormAct(self.in_channels, 32)
            self.blocks = nn.ModuleList(
                (
                    LiteDWResDownBlock(32, 64, use_dwres=self.encoder_variant == "lite_dwres_all"),
                    LiteDWResDownBlock(64, 128, use_dwres=True),
                    LiteDWResDownBlock(128, 256, use_dwres=True),
                )
            )
        else:
            raise ValueError(
                "CnnGlyphEncoder encoder_variant must be 'full_conv', 'full_conv_stage_res', 'full_resnet', 'lite_dwres', or 'lite_dwres_all', "
                f"got {self.encoder_variant!r}"
            )

    def _encode_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        if x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"expected RGB {self.image_size}x{self.image_size} glyph tensor, got {tuple(x.shape)}")
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        if x.shape[1:] != (3, 128, 128):
            raise ValueError(f"expected RGB 128x128 glyph tensor, got {tuple(x.shape)}")
        features: list[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self._encode_map(x)
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block for glyph patch tokens."""

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
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            query_rope=rope,
            key_rope=rope,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ClsPoolingBlock(nn.Module):
    """Cross-attend a learnable CLS token over per-reference token grids."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        use_rope_for_keys: bool,
        grid_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.use_rope_for_keys = bool(use_rope_for_keys)
        self.grid_size = int(grid_size)
        self.norm_query = _build_norm(self.hidden_dim, norm_variant="rms")
        self.norm_context = _build_norm(self.hidden_dim, norm_variant="rms")
        self.attn = SDPAAttention(self.hidden_dim, self.num_heads)
        self.mlp = FeedForward(self.hidden_dim, 4.0, activation="swiglu")
        head_dim = self.hidden_dim // self.num_heads
        if self.use_rope_for_keys:
            if head_dim % 4 != 0:
                raise ValueError(
                    "ClsPoolingBlock requires head dim divisible by 4 when using RoPE, "
                    f"got hidden_dim={hidden_dim}, num_heads={num_heads}"
                )
            self.key_rope = VisionRotaryEmbeddingFast(
                dim=head_dim // 2,
                pt_seq_len=self.grid_size,
            )
        else:
            self.key_rope = None

    def forward(self, cls_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if cls_tokens.dim() != 3 or cls_tokens.size(1) != 1:
            raise ValueError(f"cls_tokens must be [B, 1, D], got {tuple(cls_tokens.shape)}")
        if context_tokens.dim() != 3:
            raise ValueError(f"context_tokens must be [B, T, D], got {tuple(context_tokens.shape)}")
        query_in = self.norm_query(cls_tokens)
        context_in = self.norm_context(context_tokens)
        attn_out, _ = self.attn(
            query_in,
            context_in,
            context_in,
            key_rope=self.key_rope,
            need_weights=False,
        )
        cls_tokens = cls_tokens + attn_out
        cls_tokens = cls_tokens + self.mlp(self.norm_query(cls_tokens))
        return cls_tokens


class ViTGlyphEncoder(nn.Module):
    """ViT glyph encoder producing patch tokens and optional CLS-pooled features."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 128,
        output_grid_size: int = 16,
        hidden_dim: int = 256,
        block_depth: int = 4,
        num_heads: int = 4,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(f"ViTGlyphEncoder requires RGB input, got {in_channels}")
        if int(image_size) <= 0 or int(output_grid_size) <= 0:
            raise ValueError(f"image_size/output_grid_size must be positive, got {image_size} and {output_grid_size}")
        if int(hidden_dim) != 256:
            raise ValueError(f"ViTGlyphEncoder is fixed to hidden_dim=256, got {hidden_dim}")
        if int(block_depth) != 4:
            raise ValueError(f"ViTGlyphEncoder is fixed to block_depth=4, got {block_depth}")
        if int(num_heads) != 4:
            raise ValueError(f"ViTGlyphEncoder is fixed to num_heads=4, got {num_heads}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.output_grid_size = int(output_grid_size)
        self.hidden_dim = int(hidden_dim)
        self.block_depth = int(block_depth)
        self.num_heads = int(num_heads)
        self.use_cls_token = bool(use_cls_token)
        self.patch_size = self.image_size // self.output_grid_size
        if self.patch_size * self.output_grid_size != self.image_size:
            raise ValueError(
                "ViTGlyphEncoder requires image_size divisible by output_grid_size, "
                f"got image_size={self.image_size}, output_grid_size={self.output_grid_size}"
            )
        self.num_tokens = self.output_grid_size * self.output_grid_size
        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_embed = nn.Linear(patch_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if self.use_cls_token else None
        head_dim = self.hidden_dim // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "ViTGlyphEncoder requires head dim divisible by 4 for RoPE, "
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
        return F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2).contiguous()

    def _encode_tokens_and_cls(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        expected_shape = (3, self.image_size, self.image_size)
        if x.shape[1:] != expected_shape:
            raise ValueError(f"expected RGB {self.image_size}x{self.image_size} glyph tensor, got {tuple(x.shape)}")
        tokens = self.patch_embed(self._patchify(x))
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            joint_tokens = torch.cat([cls_tokens, tokens], dim=1)
        else:
            cls_tokens = None
            joint_tokens = tokens
        for block in self.blocks:
            if self.cls_token is None:
                joint_tokens = block(joint_tokens, rope=self.encoder_rope)
                continue
            cls_part = joint_tokens[:, :1, :]
            token_part = joint_tokens[:, 1:, :]
            cls_in = block.norm_attn(cls_part)
            token_in = block.norm_attn(token_part)
            attn_out, _ = block.attn(
                cls_in,
                token_in,
                token_in,
                key_rope=self.encoder_rope,
                need_weights=False,
            )
            cls_part = cls_part + attn_out
            cls_part = cls_part + block.mlp(block.norm_mlp(cls_part))
            token_attn_out, _ = block.attn(
                token_in,
                token_in,
                token_in,
                query_rope=self.encoder_rope,
                key_rope=self.encoder_rope,
                need_weights=False,
            )
            token_part = token_part + token_attn_out
            token_part = token_part + block.mlp(block.norm_mlp(token_part))
            joint_tokens = torch.cat([cls_part, token_part], dim=1)
        joint_tokens = self.final_norm(joint_tokens)
        if cls_tokens is not None:
            return joint_tokens[:, 1:, :], joint_tokens[:, 0, :]
        return joint_tokens, None

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.forward_tokens(x)]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._encode_tokens_and_cls(x)
        return tokens

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_tokens = self._encode_tokens_and_cls(x)
        if cls_tokens is None:
            raise RuntimeError("CLS pooling is not enabled for this ViTGlyphEncoder")
        return cls_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


class ContentEncoder(nn.Module):
    """Content glyph encoder with switchable CNN or ViT backbone."""

    def __init__(self, *, encoder_type: str, **kwargs) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type)
        if self.encoder_type == "cnn":
            kwargs.pop("num_heads", None)
            kwargs["block_depth"] = 3
            self.encoder = CnnGlyphEncoder(**kwargs)
        elif self.encoder_type == "vit":
            self.encoder = ViTGlyphEncoder(use_cls_token=False, **kwargs)
        else:
            raise ValueError(f"Unsupported content encoder_type: {encoder_type!r}")

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder.forward_features(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_tokens(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


class StyleEncoder(nn.Module):
    """Style glyph encoder with switchable CNN or ViT backbone."""

    def __init__(
        self,
        *,
        encoder_type: str,
        hidden_dim: int,
        output_grid_size: int,
        use_cls_pool: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder_type = str(encoder_type)
        self.hidden_dim = int(hidden_dim)
        self.output_grid_size = int(output_grid_size)
        self.use_cls_pool = bool(use_cls_pool)
        if self.encoder_type == "cnn":
            kwargs.pop("num_heads", None)
            kwargs["block_depth"] = 3
            self.encoder = CnnGlyphEncoder(hidden_dim=self.hidden_dim, output_grid_size=self.output_grid_size, **kwargs)
            if self.use_cls_pool:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
                self.cls_pool = ClsPoolingBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=4,
                    use_rope_for_keys=True,
                    grid_size=self.output_grid_size,
                )
                self.cls_pool_norm = _build_norm(self.hidden_dim, norm_variant="rms")
            else:
                self.cls_token = None
                self.cls_pool = None
                self.cls_pool_norm = None
        elif self.encoder_type == "vit":
            self.encoder = ViTGlyphEncoder(
                hidden_dim=self.hidden_dim,
                output_grid_size=self.output_grid_size,
                use_cls_token=self.use_cls_pool,
                **kwargs,
            )
            self.cls_token = None
            self.cls_pool = None
            self.cls_pool_norm = None
        else:
            raise ValueError(f"Unsupported style encoder_type: {encoder_type!r}")

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.forward_tokens(x)]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_tokens(x)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_cls_pool:
            raise RuntimeError("CLS pooling is disabled for this StyleEncoder")
        if self.encoder_type == "vit":
            return self.encoder.forward_pooled(x)
        feature_map = self.encoder._encode_map(x)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        if self.cls_token is None or self.cls_pool is None or self.cls_pool_norm is None:
            raise RuntimeError("CNN CLS pooling modules are not initialized")
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        cls_tokens = self.cls_pool(cls_tokens, tokens)
        cls_tokens = self.cls_pool_norm(cls_tokens)
        return cls_tokens[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


class ContentStyleCrossAttention(nn.Module):
    """External content<-style fusion utilities for concat cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, *, grid_size: int, fusion_mode: str = "cross") -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.grid_size = int(grid_size)
        self.fusion_mode = str(fusion_mode)
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.num_heads <= 0 or (self.embed_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config embed_dim={embed_dim} num_heads={num_heads}")
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if self.fusion_mode not in {"cross", "cross_mlp", "cross_mlp_residual"}:
            raise ValueError(
                "content_style_fusion_mode must be 'cross', 'cross_mlp', or 'cross_mlp_residual', "
                f"got {self.fusion_mode!r}"
            )
        head_dim = self.embed_dim // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "JiT-style 2D RoPE requires cross-attention head dim divisible by 4, "
                f"got embed_dim={embed_dim} num_heads={num_heads}"
            )
        self.attn = SDPAAttention(self.embed_dim, self.num_heads)
        self.rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.grid_size,
        )
        if self.fusion_mode in {"cross_mlp", "cross_mlp_residual"}:
            self.fused_norm = _build_norm(self.embed_dim, norm_variant="rms")
            self.fused_mlp = FeedForward(self.embed_dim, 4.0, activation="swiglu")
        else:
            self.fused_norm = None
            self.fused_mlp = None
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
        if self.fusion_mode in {"cross_mlp", "cross_mlp_residual"}:
            if self.fused_norm is None or self.fused_mlp is None:
                raise RuntimeError(f"{self.fusion_mode} fusion is missing its MLP modules")
            mlp_tokens = self.fused_mlp(self.fused_norm(fused_tokens))
            if self.fusion_mode == "cross_mlp":
                fused_tokens = mlp_tokens
            else:
                fused_tokens = fused_tokens + mlp_tokens
        return fused_tokens


class SourcePartRefDiT(nn.Module):
    """Pure DiT glyph generator with tokenwise or CLS-global style conditioning."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        image_size: int = 128,
        patch_size: int = 8,
        encoder_type: str = "vit",
        encoder_hidden_dim: int = 256,
        content_encoder_block_depth: int = 4,
        style_encoder_block_depth: int = 4,
        encoder_variant: str = "full_conv",
        dit_hidden_dim: int = 512,
        dit_depth: int = 8,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        ffn_activation: str = "swiglu",
        norm_variant: str = "rms",
        content_injection_layers: Sequence[int] | None = None,
        conditioning_injection_mode: str = "all",
        content_style_fusion_heads: int = 4,
        style_condition_mode: str = "tokenwise_cross",
        content_style_fusion_mode: str = "cross",
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(f"Only RGB glyphs are supported, got in_channels={in_channels}")
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} vs {patch_size}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_grid_size = self.image_size // self.patch_size
        self.num_patches = self.patch_grid_size * self.patch_grid_size
        self.encoder_type = str(encoder_type)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.encoder_variant = str(encoder_variant)
        self.content_encoder_block_depth = max(1, int(content_encoder_block_depth))
        self.style_encoder_block_depth = max(1, int(style_encoder_block_depth))
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.dit_depth = int(dit_depth)
        self.dit_heads = int(dit_heads)
        self.dit_mlp_ratio = float(dit_mlp_ratio)
        self.style_condition_mode = str(style_condition_mode)
        self.content_style_fusion_mode = str(content_style_fusion_mode)
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
        if self.encoder_type != "cnn":
            raise ValueError(f"encoder_type is fixed to 'cnn' for this experiment, got {self.encoder_type!r}")
        if self.style_condition_mode != "tokenwise_cross":
            raise ValueError(
                "style_condition_mode is fixed to 'tokenwise_cross' for this experiment, "
                f"got {self.style_condition_mode!r}"
            )
        if self.content_style_fusion_mode not in {"cross", "cross_mlp", "cross_mlp_residual"}:
            raise ValueError(
                "content_style_fusion_mode must be 'cross', 'cross_mlp', or 'cross_mlp_residual', "
                f"got {self.content_style_fusion_mode!r}"
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
            encoder_type=self.encoder_type,
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.content_encoder_block_depth,
            encoder_variant=self.encoder_variant,
            num_heads=4,
        )
        self.style_encoder = StyleEncoder(
            encoder_type=self.encoder_type,
            in_channels=self.in_channels,
            image_size=self.image_size,
            output_grid_size=self.patch_grid_size,
            hidden_dim=self.encoder_hidden_dim,
            block_depth=self.style_encoder_block_depth,
            encoder_variant=self.encoder_variant,
            num_heads=4,
            use_cls_pool=self.style_condition_mode == "global_cls",
        )
        self.content_style_attn = (
            ContentStyleCrossAttention(
                embed_dim=self.encoder_hidden_dim,
                num_heads=self.content_style_fusion_heads,
                grid_size=self.patch_grid_size,
                fusion_mode=self.content_style_fusion_mode,
            )
            if self.style_condition_mode == "tokenwise_cross"
            else None
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
            use_style_tokenwise_condition=self.style_condition_mode == "tokenwise_cross",
            use_style_global_condition=self.style_condition_mode == "global_cls",
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
        self.output_style_global_to_hidden = nn.Linear(self.output_condition_half_dim, self.dit_hidden_dim)
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
            "encoder_type": str(self.encoder_type),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "encoder_variant": str(self.encoder_variant),
            "content_encoder_block_depth": int(self.content_encoder_block_depth),
            "style_encoder_block_depth": int(self.style_encoder_block_depth),
            "dit_hidden_dim": int(self.dit_hidden_dim),
            "dit_depth": int(self.dit_depth),
            "dit_heads": int(self.dit_heads),
            "dit_mlp_ratio": float(self.dit_mlp_ratio),
            "content_injection_layers": list(self.content_injection_layers),
            "content_style_fusion_heads": int(self.content_style_fusion_heads),
            "style_condition_mode": str(self.style_condition_mode),
            "content_style_fusion_mode": str(self.content_style_fusion_mode),
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
        style_tokens = style_tokens.view(batch, refs, tokens_per_ref, self.encoder_hidden_dim)
        return style_tokens

    def _encode_style_cls(self, style_img: torch.Tensor) -> torch.Tensor:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")
        batch, refs, channels, height, width = style_img.shape
        flat_style = style_img.view(batch * refs, channels, height, width)
        style_cls = self.style_encoder.forward_cls(flat_style)
        return style_cls.view(batch, refs, self.encoder_hidden_dim)

    def encode_style_token_bank(self, style_img: torch.Tensor) -> torch.Tensor:
        return self._encode_style_features(style_img).contiguous()

    def encode_style_global_vectors(self, style_img: torch.Tensor) -> torch.Tensor:
        return self._encode_style_cls(style_img).contiguous()

    def precompute_style_bank_kv(
        self,
        style_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.content_style_attn is None:
            raise RuntimeError("tokenwise style conditioning is disabled")
        return self.content_style_attn.project_style_bank_kv(style_tokens)

    def precompute_content_query(
        self,
        content_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.content_style_attn is None:
            raise RuntimeError("tokenwise style conditioning is disabled")
        return self.content_style_attn.project_content_query(content_tokens)

    def build_style_global_condition(self, style_token_bank: torch.Tensor) -> torch.Tensor | None:
        if self.style_condition_mode != "global_cls":
            return None
        if style_token_bank.dim() != 3:
            raise ValueError(f"style_global_bank must be 3D [B, R, D], got {tuple(style_token_bank.shape)}")
        expected_style_shape = (
            int(style_token_bank.size(0)),
            int(style_token_bank.size(1)),
            self.output_condition_half_dim,
        )
        if style_token_bank.shape != expected_style_shape:
            raise ValueError(
                f"style_global_bank shape mismatch: expected {expected_style_shape}, got {tuple(style_token_bank.shape)}"
            )
        return style_token_bank.mean(dim=1).contiguous()

    def build_style_condition_tokens(
        self,
        content_tokens: torch.Tensor,
        style_token_bank: Optional[torch.Tensor] = None,
        *,
        content_query: Optional[torch.Tensor] = None,
        style_key: Optional[torch.Tensor] = None,
        style_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | None:
        if self.style_condition_mode != "tokenwise_cross":
            return None
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
        style_tokens: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor | None]:
        return self.backbone.build_condition_hidden_cache(
            content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
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
        style_tokens: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
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
        content_tokens = content_tokens.to(device=device, dtype=dtype)
        joint_hidden = self.output_content_condition_to_hidden(content_tokens)
        if style_tokens is not None:
            if style_tokens.shape != expected_part_shape:
                raise ValueError(
                    "style token shape mismatch for final head: "
                    f"expected {expected_part_shape}, got {tuple(style_tokens.shape)}"
                )
            style_tokens = style_tokens.to(device=device, dtype=dtype)
            joint_hidden = joint_hidden + self.output_style_condition_to_hidden(style_tokens)
        if style_global is not None:
            expected_style_global_shape = (content_tokens.size(0), self.output_condition_half_dim)
            if style_global.shape != expected_style_global_shape:
                raise ValueError(
                    "style global shape mismatch for final head: "
                    f"expected {expected_style_global_shape}, got {tuple(style_global.shape)}"
                )
            style_global = style_global.to(device=device, dtype=dtype)
            joint_hidden = joint_hidden + self.output_style_global_to_hidden(style_global).unsqueeze(1)
        return joint_hidden

    def decode_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
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
        if style_tokens is not None and style_tokens.shape != expected_part_shape:
            raise ValueError(
                "style token shape mismatch for final head: "
                f"expected {expected_part_shape}, got {tuple(style_tokens.shape)}"
            )
        expected_style_global_shape = (patch_tokens.size(0), self.output_condition_half_dim)
        if style_global is not None and style_global.shape != expected_style_global_shape:
            raise ValueError(
                "style global shape mismatch for final head: "
                f"expected {expected_style_global_shape}, got {tuple(style_global.shape)}"
            )

        time_hidden = self.output_time_to_hidden(
            self.backbone.build_time_cond(
                timesteps,
                dtype=patch_tokens.dtype,
            )
        ).unsqueeze(1)
        if output_condition_hidden is None:
            content_tokens = content_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            content_hidden = self.output_content_condition_to_hidden(content_tokens)
            joint_hidden = time_hidden + content_hidden
            if style_tokens is not None:
                style_tokens = style_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
                joint_hidden = joint_hidden + self.output_style_condition_to_hidden(style_tokens)
            if style_global is not None:
                style_global = style_global.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
                joint_hidden = joint_hidden + self.output_style_global_to_hidden(style_global).unsqueeze(1)
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
        style_tokens: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
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
            style_global=style_global,
            condition_hidden_cache=backbone_condition_hidden_cache,
            unique_content_hidden_cache=backbone_unique_content_hidden_cache,
            content_index=content_index,
        )
        return self.decode_patch_tokens(
            patch_tokens,
            timesteps=timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
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
        style_token_bank = self.encode_style_token_bank(style_img) if self.style_condition_mode == "tokenwise_cross" else None
        style_tokens = None
        style_global = None
        if self.style_condition_mode == "tokenwise_cross":
            style_tokens = self.build_style_condition_tokens(
                content_tokens,
                style_token_bank,
            )
        else:
            style_global = self.build_style_global_condition(self.encode_style_global_vectors(style_img))
        return self.predict(
            x_t_image,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
        )
