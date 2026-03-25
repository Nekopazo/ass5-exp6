#!/usr/bin/env python3
"""Top-level content + multi-style PixelDiT model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .diffusion_transformer_backbone import PixelDiffusionTransformerBackbone, build_2d_sincos_pos_embed
from .sdpa_attention import SDPAAttention


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


class GlobalStyleEncoder(nn.Module):
    def __init__(self, *, out_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)


class SourcePartRefDiT(nn.Module):
    GLOBAL_STYLE_STATE_PREFIXES = (
        "global_style_encoder.",
        "style_global_proj.",
    )

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
        style_global_dim: int = 256,
        patch_hidden_dim: int = 512,
        patch_depth: int = 12,
        patch_heads: int = 8,
        patch_mlp_ratio: float = 4.0,
        pixel_hidden_dim: int = 32,
        pit_depth: int = 2,
        pit_heads: int = 8,
        pit_mlp_ratio: float = 4.0,
        style_fusion_start: int = 8,
        use_style_tokens: bool = True,
        contrastive_proj_dim: int = 128,
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
        self.style_global_dim = int(style_global_dim)
        self.patch_hidden_dim = int(patch_hidden_dim)
        self.patch_depth = int(patch_depth)
        self.patch_heads = int(patch_heads)
        self.patch_mlp_ratio = float(patch_mlp_ratio)
        self.pixel_hidden_dim = int(pixel_hidden_dim)
        self.pit_depth = int(pit_depth)
        self.pit_heads = int(pit_heads)
        self.pit_mlp_ratio = float(pit_mlp_ratio)
        self.style_fusion_start = int(style_fusion_start)
        self.use_style_tokens = bool(use_style_tokens)
        self.contrastive_proj_dim = int(contrastive_proj_dim)

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
        self.style_token_encoder = (
            TokenEncoderBackbone(
                hidden_dim=self.encoder_hidden_dim,
                depth=self.encoder_depth,
                num_heads=self.encoder_heads,
                mlp_ratio=self.encoder_mlp_ratio,
            )
            if self.use_style_tokens
            else None
        )
        self.global_style_encoder = GlobalStyleEncoder(out_dim=self.style_global_dim)

        if self.encoder_hidden_dim != self.patch_hidden_dim:
            self.content_proj = nn.Linear(self.encoder_hidden_dim, self.patch_hidden_dim)
            self.style_proj = (
                nn.Linear(self.encoder_hidden_dim, self.patch_hidden_dim) if self.use_style_tokens else None
            )
        else:
            self.content_proj = nn.Identity()
            self.style_proj = nn.Identity() if self.use_style_tokens else None
        if self.style_global_dim != self.patch_hidden_dim:
            self.style_global_proj = nn.Linear(self.style_global_dim, self.patch_hidden_dim)
        else:
            self.style_global_proj = nn.Identity()

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
            use_style_tokens=self.use_style_tokens,
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
            "style_global_dim": int(self.style_global_dim),
            "patch_hidden_dim": int(self.patch_hidden_dim),
            "patch_depth": int(self.patch_depth),
            "patch_heads": int(self.patch_heads),
            "patch_mlp_ratio": float(self.patch_mlp_ratio),
            "pixel_hidden_dim": int(self.pixel_hidden_dim),
            "pit_depth": int(self.pit_depth),
            "pit_heads": int(self.pit_heads),
            "pit_mlp_ratio": float(self.pit_mlp_ratio),
            "style_fusion_start": int(self.style_fusion_start),
            "use_style_tokens": bool(self.use_style_tokens),
            "contrastive_proj_dim": int(self.contrastive_proj_dim),
        }

    @classmethod
    def _is_global_style_state_key(cls, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in cls.GLOBAL_STYLE_STATE_PREFIXES)

    def global_style_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if self._is_global_style_state_key(key)
        }

    def load_style_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Malformed style checkpoint: {path}")
        raw_state_dict = checkpoint.get("global_style_state")
        if not isinstance(raw_state_dict, dict):
            raise RuntimeError(f"Style checkpoint must contain 'global_style_state': {path}")
        missing_keys, unexpected_keys = self.load_state_dict(raw_state_dict, strict=False)
        unexpected_global = [key for key in unexpected_keys if self._is_global_style_state_key(key)]
        if unexpected_global:
            raise RuntimeError(f"Unexpected global style keys: {unexpected_global}")
        missing_global = [key for key in missing_keys if self._is_global_style_state_key(key)]
        if missing_global:
            raise RuntimeError(f"Missing global style keys: {missing_global}")

    def freeze_style_global(self) -> None:
        for name, param in self.named_parameters():
            if self._is_global_style_state_key(name):
                param.requires_grad_(False)

    def _embed_patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        return self.shared_patch_embed(img)

    def encode_content_features(self, content_img: torch.Tensor) -> torch.Tensor:
        return self.content_token_encoder(self._embed_patch_tokens(content_img))

    def encode_content(self, content_img: torch.Tensor) -> torch.Tensor:
        return self.content_proj(self.encode_content_features(content_img))

    def _masked_mean(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def encode_style(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        *,
        need_style_tokens: bool = True,
        need_style_global: bool = True,
        detach_global_style_encoder: bool = False,
        detach_global_style: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        if style_img.dim() == 4:
            style_img = style_img.unsqueeze(1)
        if style_img.dim() != 5:
            raise ValueError(f"style_img must be BCHW or BRCHW, got {tuple(style_img.shape)}")

        batch_size, ref_count, channels, height, width = style_img.shape
        flat_style = style_img.view(batch_size * ref_count, channels, height, width)
        if style_ref_mask is None:
            valid_mask = torch.ones((batch_size, ref_count), device=style_img.device, dtype=torch.bool)
        else:
            valid_mask = style_ref_mask.to(device=style_img.device) > 0.5
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = True

        style_tokens = None
        if need_style_tokens and self.use_style_tokens:
            if self.style_token_encoder is None or self.style_proj is None:
                raise RuntimeError("style token branch is disabled or not initialized")
            local_tokens = self.style_token_encoder(self._embed_patch_tokens(flat_style))
            local_tokens = local_tokens.view(batch_size, ref_count, local_tokens.size(1), local_tokens.size(2))
            style_tokens = self.style_proj(local_tokens)
            style_tokens = style_tokens * valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=style_tokens.dtype)

        style_global = None
        if need_style_global:
            encoder_context = torch.no_grad() if detach_global_style_encoder else nullcontext()
            with encoder_context:
                global_vecs = self.global_style_encoder(flat_style).view(batch_size, ref_count, -1)
            pool_context = torch.no_grad() if detach_global_style else nullcontext()
            with pool_context:
                pooled = self._masked_mean(global_vecs, valid_mask)
                style_global = self.style_global_proj(pooled)

        return {
            "style_tokens": style_tokens,
            "style_global": style_global,
            "style_ref_mask": valid_mask,
        }

    def predict_flow(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_tokens: Optional[torch.Tensor],
        style_global: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone(
            x_t,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
            style_ref_mask=style_ref_mask,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        content_img: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content_tokens = self.encode_content(content_img)
        style_pack = self.encode_style(style_img, style_ref_mask=style_ref_mask)
        if style_pack["style_global"] is None:
            raise RuntimeError("style encoding did not return required tensors")
        if self.use_style_tokens and style_pack["style_tokens"] is None:
            raise RuntimeError("style token branch is enabled but style_tokens are missing")
        return self.predict_flow(
            x_t,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_pack["style_tokens"],
            style_global=style_pack["style_global"],
            style_ref_mask=style_pack["style_ref_mask"],
        )
