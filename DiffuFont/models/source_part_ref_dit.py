#!/usr/bin/env python3
"""Top-level content + unified style-memory PixelDiT model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion_transformer_backbone import PixelDiffusionTransformerBackbone, build_2d_sincos_pos_embed
from .sdpa_attention import SDPAAttention


def _group_count(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


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


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(hidden_dim)
        self.norm_mlp = RMSNorm(hidden_dim)
        self.attn = SDPAAttention(hidden_dim, num_heads)
        self.mlp = FeedForward(hidden_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class SharedPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size, got {self.image_size} and {self.patch_size}"
            )
        self.grid_size = self.image_size // self.patch_size
        self.num_tokens = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(1, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        pos_embed = build_2d_sincos_pos_embed(self.hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"encoder input must be BCHW, got {tuple(x.shape)}")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        return tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)


class TokenEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.blocks = nn.ModuleList(
            [EncoderBlock(self.hidden_dim, num_heads, mlp_ratio) for _ in range(max(0, int(depth)))]
        )
        self.final_norm = RMSNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"token encoder input must be BLD, got {tuple(tokens.shape)}")
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedStyleMemoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        feature_dim: int,
        hidden_dim: int,
        memory_slots: int,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_slots = int(memory_slots)
        if self.image_size % 4 != 0:
            raise ValueError(f"style image_size must be divisible by 4, got {self.image_size}")
        self.grid_size = self.image_size // 4

        stem_dim = max(32, self.feature_dim // 4)
        mid_dim = max(stem_dim, self.feature_dim // 2)
        self.cnn = nn.Sequential(
            ConvNormAct(1, stem_dim, stride=1),
            ConvNormAct(stem_dim, mid_dim, stride=2),
            ConvNormAct(mid_dim, self.feature_dim, stride=2),
            ConvNormAct(self.feature_dim, self.feature_dim, stride=1),
        )
        self.token_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.token_norm = RMSNorm(self.hidden_dim)
        self.memory_score = nn.Linear(self.hidden_dim, self.memory_slots)
        self.memory_norm = RMSNorm(self.hidden_dim)

    def _encode_dense_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)
        if feats.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError(
                f"style feature size mismatch: expected {(self.grid_size, self.grid_size)}, got {tuple(feats.shape[-2:])}"
            )
        tokens = feats.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_proj(tokens)
        return self.token_norm(tokens)

    def forward(
        self,
        style_img: torch.Tensor,
    ) -> torch.Tensor:
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BRCHW, got {tuple(style_img.shape)}")
        batch_size, ref_count, channels, height, width = style_img.shape
        if channels != 1 or height != self.image_size or width != self.image_size:
            raise ValueError(
                "style_img shape mismatch: "
                f"expected (*, *, 1, {self.image_size}, {self.image_size}), got {tuple(style_img.shape)}"
            )

        flat_style = style_img.view(batch_size * ref_count, channels, height, width)
        dense_tokens = self._encode_dense_tokens(flat_style)
        dense_tokens = dense_tokens.view(batch_size, ref_count, dense_tokens.size(1), dense_tokens.size(2))
        dense_tokens = dense_tokens.reshape(batch_size, ref_count * dense_tokens.size(2), dense_tokens.size(3))
        logits = self.memory_score(dense_tokens)
        weights = torch.softmax(logits.float(), dim=1).to(dtype=dense_tokens.dtype)
        style_memory = torch.einsum("bnk,bnd->bkd", weights, dense_tokens)
        return self.memory_norm(style_memory)


class SourcePartRefDiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 16,
        encoder_hidden_dim: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        style_feature_dim: int = 256,
        style_memory_k: int = 4,
        patch_hidden_dim: int = 512,
        patch_depth: int = 12,
        patch_heads: int = 8,
        patch_mlp_ratio: float = 4.0,
        pixel_hidden_dim: int = 32,
        pit_depth: int = 2,
        pit_heads: int = 8,
        pit_mlp_ratio: float = 4.0,
        style_fusion_start: int = 4,
        style_fusion_end: int = 8,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(f"Only grayscale glyphs are supported, got in_channels={in_channels}")

        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.encoder_depth = int(encoder_depth)
        self.encoder_heads = int(encoder_heads)
        self.encoder_mlp_ratio = float(encoder_mlp_ratio)
        self.style_feature_dim = int(style_feature_dim)
        self.style_memory_k = int(style_memory_k)
        self.patch_hidden_dim = int(patch_hidden_dim)
        self.patch_depth = int(patch_depth)
        self.patch_heads = int(patch_heads)
        self.patch_mlp_ratio = float(patch_mlp_ratio)
        self.pixel_hidden_dim = int(pixel_hidden_dim)
        self.pit_depth = int(pit_depth)
        self.pit_heads = int(pit_heads)
        self.pit_mlp_ratio = float(pit_mlp_ratio)
        self.style_fusion_start = int(style_fusion_start)
        self.style_fusion_end = int(style_fusion_end)

        self.shared_patch_embed = SharedPatchEmbed(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.encoder_hidden_dim,
        )
        self.content_token_encoder = TokenEncoderBackbone(
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            mlp_ratio=self.encoder_mlp_ratio,
        )
        self.style_encoder = UnifiedStyleMemoryEncoder(
            image_size=self.image_size,
            feature_dim=self.style_feature_dim,
            hidden_dim=self.patch_hidden_dim,
            memory_slots=self.style_memory_k,
        )

        if self.encoder_hidden_dim != self.patch_hidden_dim:
            self.content_proj = nn.Linear(self.encoder_hidden_dim, self.patch_hidden_dim)
        else:
            self.content_proj = nn.Identity()

        self.backbone = PixelDiffusionTransformerBackbone(
            image_size=self.image_size,
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            patch_hidden_dim=self.patch_hidden_dim,
            patch_depth=self.patch_depth,
            patch_heads=self.patch_heads,
            patch_mlp_ratio=self.patch_mlp_ratio,
            pixel_hidden_dim=self.pixel_hidden_dim,
            pit_depth=self.pit_depth,
            pit_heads=self.pit_heads,
            pit_mlp_ratio=self.pit_mlp_ratio,
            style_fusion_start=self.style_fusion_start,
            style_fusion_end=self.style_fusion_end,
            style_memory_k=self.style_memory_k,
        )

    def export_config(self) -> dict[str, object]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "encoder_depth": int(self.encoder_depth),
            "encoder_heads": int(self.encoder_heads),
            "encoder_mlp_ratio": float(self.encoder_mlp_ratio),
            "style_feature_dim": int(self.style_feature_dim),
            "style_memory_k": int(self.style_memory_k),
            "patch_hidden_dim": int(self.patch_hidden_dim),
            "patch_depth": int(self.patch_depth),
            "patch_heads": int(self.patch_heads),
            "patch_mlp_ratio": float(self.patch_mlp_ratio),
            "pixel_hidden_dim": int(self.pixel_hidden_dim),
            "pit_depth": int(self.pit_depth),
            "pit_heads": int(self.pit_heads),
            "pit_mlp_ratio": float(self.pit_mlp_ratio),
            "style_fusion_start": int(self.style_fusion_start),
            "style_fusion_end": int(self.style_fusion_end),
        }

    def _embed_patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        return self.shared_patch_embed(img)

    def encode_content_features(self, content_img: torch.Tensor) -> torch.Tensor:
        return self.content_token_encoder(self._embed_patch_tokens(content_img))

    def encode_content(self, content_img: torch.Tensor) -> torch.Tensor:
        return self.content_proj(self.encode_content_features(content_img))

    def encode_style(self, style_img: torch.Tensor) -> torch.Tensor:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")
        return self.style_encoder(style_img)

    def predict_flow(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_memory: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(
            x_t,
            timesteps,
            content_tokens=content_tokens,
            style_memory=style_memory,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        content_img: torch.Tensor,
        *,
        style_img: torch.Tensor,
    ) -> torch.Tensor:
        content_tokens = self.encode_content(content_img)
        style_memory = self.encode_style(style_img)
        return self.predict_flow(
            x_t,
            timesteps,
            content_tokens=content_tokens,
            style_memory=style_memory,
        )
