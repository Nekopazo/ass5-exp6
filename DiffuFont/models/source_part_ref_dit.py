#!/usr/bin/env python3
"""Content+style latent DiT for Chinese glyph generation."""

from __future__ import annotations

from contextlib import nullcontext
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


class GlyphVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        bottleneck_channels: int = 192,
        encoder_16x16_blocks: int = 2,
        decoder_16x16_blocks: int = 2,
        decoder_tail_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.encoder_16x16_blocks = max(1, int(encoder_16x16_blocks))
        self.decoder_16x16_blocks = max(1, int(decoder_16x16_blocks))
        self.decoder_tail_blocks = max(0, int(decoder_tail_blocks))

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            ResBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64),
            nn.Conv2d(64, self.bottleneck_channels, kernel_size=3, stride=2, padding=1),
            ResBlock(self.bottleneck_channels, self.bottleneck_channels),
        )
        self.enc_block16_extra = nn.ModuleList(
            [
                ResBlock(self.bottleneck_channels, self.bottleneck_channels)
                for _ in range(max(0, self.encoder_16x16_blocks - 1))
            ]
        )
        self.to_stats = nn.Conv2d(self.bottleneck_channels, self.latent_channels * 2, kernel_size=1)

        self.decoder_in = nn.Conv2d(self.latent_channels, self.bottleneck_channels, kernel_size=3, padding=1)
        self.dec_block16 = ResBlock(self.bottleneck_channels, self.bottleneck_channels)
        self.dec_block16_extra = nn.ModuleList(
            [
                ResBlock(self.bottleneck_channels, self.bottleneck_channels)
                for _ in range(max(0, self.decoder_16x16_blocks - 1))
            ]
        )
        self.dec_up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.bottleneck_channels, 64, kernel_size=3, padding=1),
            ResBlock(64, 64),
        )
        self.dec_up64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            ResBlock(32, 32),
        )
        self.dec_up128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
        )
        self.dec_block128 = ResBlock(16, 16) if self.decoder_tail_blocks >= 1 else None
        self.dec_block128_extra = nn.ModuleList(
            [ResBlock(16, 16) for _ in range(max(0, self.decoder_tail_blocks - 1))]
        )
        self.dec_norm128 = nn.GroupNorm(_group_count(16), 16)
        self.dec_act128 = nn.SiLU()
        self.decoder_out = nn.Conv2d(16, self.in_channels, kernel_size=3, padding=1)

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        for block in self.enc_block16_extra:
            h = block(h)
        mu, logvar = self.to_stats(h).chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor, *, sample_posterior: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(z)
        x = self.dec_block16(x)
        for block in self.dec_block16_extra:
            x = block(x)
        x = self.dec_up32(x)
        x = self.dec_up64(x)
        x = self.dec_up128(x)
        if self.dec_block128 is not None:
            x = self.dec_block128(x)
        for block in self.dec_block128_extra:
            x = block(x)
        x = self.dec_act128(self.dec_norm128(x))
        return torch.tanh(self.decoder_out(x))

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x, sample_posterior=sample_posterior)
        recon = self.decode(z)
        return recon, z, mu, logvar


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


class LocalStyleAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        grid_size: int,
        *,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        inner_dim = int(self.hidden_dim * float(mlp_ratio))
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.dwconv = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.pwconv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, self.hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, hidden_dim = tokens.shape
        expected_tokens = self.grid_size * self.grid_size
        if num_tokens != expected_tokens:
            raise ValueError(
                f"LocalStyleAdapter expected {expected_tokens} tokens for grid_size={self.grid_size}, "
                f"got {num_tokens}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"LocalStyleAdapter expected hidden_dim={self.hidden_dim}, got {hidden_dim}"
            )

        residual = tokens
        x = self.norm1(tokens)
        x = x.transpose(1, 2).contiguous().view(batch, hidden_dim, self.grid_size, self.grid_size)
        x = self.dwconv(x)
        x = self.pwconv(F.silu(x))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionTokenPool(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, output_tokens: int) -> None:
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.query = nn.Parameter(torch.randn(1, self.output_tokens, hidden_dim) * 0.02)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.attn = SDPAAttention(hidden_dim, num_heads)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = self.query.expand(tokens.size(0), -1, -1)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        pooled, _ = self.attn(
            queries,
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_norm(pooled)


class SourcePartRefDiT(nn.Module):
    GLOBAL_STYLE_STATE_PREFIXES = (
        "global_style_encoder.",
        "global_style_pool.",
        "style_global_proj.",
    )
    LEGACY_GLOBAL_STYLE_PREFIX_MAP = {
        "style_encoder.": "global_style_encoder.",
    }
    def __init__(
        self,
        *,
        in_channels: int = 1,
        image_size: int = 128,
        latent_channels: int = 4,
        latent_size: int = 16,
        encoder_patch_size: int = 8,
        encoder_hidden_dim: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        dit_hidden_dim: int = 512,
        dit_depth: int = 16,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        style_tokens_per_ref: int = 8,
        content_cross_attn_indices: list[int] | tuple[int, ...] | None = None,
        style_token_cross_attn_indices: list[int] | tuple[int, ...] | None = None,
        contrastive_proj_dim: int = 128,
        vae_bottleneck_channels: int = 192,
        vae_encoder_16x16_blocks: int = 2,
        vae_decoder_16x16_blocks: int = 2,
        vae_decoder_tail_blocks: int = 1,
        latent_normalize_for_dit: bool = True,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(f"Only grayscale glyphs are supported, got in_channels={in_channels}")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.encoder_patch_size = int(encoder_patch_size)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.encoder_depth = int(encoder_depth)
        self.encoder_heads = int(encoder_heads)
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.dit_depth = int(dit_depth)
        self.dit_heads = int(dit_heads)
        self.dit_mlp_ratio = float(dit_mlp_ratio)
        self.style_tokens_per_ref = max(1, int(style_tokens_per_ref))
        if content_cross_attn_indices is None:
            self.content_cross_attn_indices = tuple(range(self.dit_depth))
        else:
            self.content_cross_attn_indices = tuple(sorted({int(index) for index in content_cross_attn_indices}))
        if style_token_cross_attn_indices is None:
            start_idx = max(0, self.dit_depth - 6)
            self.style_token_cross_attn_indices = tuple(range(start_idx, self.dit_depth))
        else:
            self.style_token_cross_attn_indices = tuple(sorted({int(index) for index in style_token_cross_attn_indices}))
        self.contrastive_proj_dim = int(contrastive_proj_dim)
        self.vae_bottleneck_channels = int(vae_bottleneck_channels)
        self.vae_encoder_16x16_blocks = max(1, int(vae_encoder_16x16_blocks))
        self.vae_decoder_16x16_blocks = max(1, int(vae_decoder_16x16_blocks))
        self.vae_decoder_tail_blocks = max(0, int(vae_decoder_tail_blocks))
        self.latent_normalize_for_dit = bool(latent_normalize_for_dit)
        self.register_buffer(
            "latent_norm_mean",
            torch.zeros(1, self.latent_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_norm_std",
            torch.ones(1, self.latent_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_norm_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )

        self.vae = GlyphVAE(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            bottleneck_channels=self.vae_bottleneck_channels,
            encoder_16x16_blocks=self.vae_encoder_16x16_blocks,
            decoder_16x16_blocks=self.vae_decoder_16x16_blocks,
            decoder_tail_blocks=self.vae_decoder_tail_blocks,
        )
        self.content_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=False,
        )
        self.global_style_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=True,
        )
        self.token_style_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=False,
        )
        self.local_style_adapter = LocalStyleAdapter(
            hidden_dim=self.encoder_hidden_dim,
            grid_size=self.image_size // self.encoder_patch_size,
        )
        self.global_style_pool = AttentionTokenPool(
            hidden_dim=self.encoder_hidden_dim,
            num_heads=self.encoder_heads,
            output_tokens=1,
        )
        self.per_ref_style_pool = AttentionTokenPool(
            hidden_dim=self.encoder_hidden_dim,
            output_tokens=self.style_tokens_per_ref,
            num_heads=self.encoder_heads,
        )
        self.backbone = DiffusionTransformerBackbone(
            latent_channels=self.latent_channels,
            latent_size=self.latent_size,
            hidden_dim=self.dit_hidden_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_cross_attn_indices=self.content_cross_attn_indices,
            style_token_cross_attn_indices=self.style_token_cross_attn_indices,
        )

        if self.encoder_hidden_dim != self.dit_hidden_dim:
            self.content_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
            self.style_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
            self.style_global_proj = nn.Linear(self.encoder_hidden_dim, self.dit_hidden_dim)
        else:
            self.content_proj = nn.Identity()
            self.style_proj = nn.Identity()
            self.style_global_proj = nn.Identity()

    def export_config(self) -> dict[str, object]:
        return {
            "in_channels": int(self.in_channels),
            "image_size": int(self.image_size),
            "latent_channels": int(self.latent_channels),
            "latent_size": int(self.latent_size),
            "encoder_patch_size": int(self.encoder_patch_size),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "encoder_depth": int(self.encoder_depth),
            "encoder_heads": int(self.encoder_heads),
            "dit_hidden_dim": int(self.dit_hidden_dim),
            "dit_depth": int(self.dit_depth),
            "dit_heads": int(self.dit_heads),
            "dit_mlp_ratio": float(self.dit_mlp_ratio),
            "style_tokens_per_ref": int(self.style_tokens_per_ref),
            "content_cross_attn_indices": list(self.content_cross_attn_indices),
            "style_token_cross_attn_indices": list(self.style_token_cross_attn_indices),
            "contrastive_proj_dim": int(self.contrastive_proj_dim),
            "vae_bottleneck_channels": int(self.vae_bottleneck_channels),
            "vae_encoder_16x16_blocks": int(self.vae_encoder_16x16_blocks),
            "vae_decoder_16x16_blocks": int(self.vae_decoder_16x16_blocks),
            "vae_decoder_tail_blocks": int(self.vae_decoder_tail_blocks),
            "latent_normalize_for_dit": int(self.latent_normalize_for_dit),
        }

    def latent_norm_state_dict(self) -> dict[str, torch.Tensor | bool]:
        return {
            "mean": self.latent_norm_mean.detach().cpu(),
            "std": self.latent_norm_std.detach().cpu(),
            "initialized": bool(self.latent_norm_initialized.item()),
        }

    def load_latent_norm_state(self, state: Optional[dict[str, torch.Tensor | bool]]) -> None:
        if not state:
            return
        mean = state.get("mean")
        std = state.get("std")
        initialized = bool(state.get("initialized", False))
        if mean is None or std is None:
            return
        mean = torch.as_tensor(mean, dtype=self.latent_norm_mean.dtype, device=self.latent_norm_mean.device)
        std = torch.as_tensor(std, dtype=self.latent_norm_std.dtype, device=self.latent_norm_std.device)
        if mean.shape != self.latent_norm_mean.shape or std.shape != self.latent_norm_std.shape:
            raise RuntimeError(
                "Latent normalization stats shape mismatch: "
                f"expected mean/std {tuple(self.latent_norm_mean.shape)}, got "
                f"{tuple(mean.shape)} / {tuple(std.shape)}"
            )
        self.latent_norm_mean.copy_(mean)
        self.latent_norm_std.copy_(std.clamp_min(1e-6))
        self.latent_norm_initialized.fill_(initialized)

    def set_latent_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.latent_norm_mean.dtype, device=self.latent_norm_mean.device)
        std = torch.as_tensor(std, dtype=self.latent_norm_std.dtype, device=self.latent_norm_std.device)
        if mean.shape != self.latent_norm_mean.shape or std.shape != self.latent_norm_std.shape:
            raise RuntimeError(
                "Latent normalization stats shape mismatch: "
                f"expected mean/std {tuple(self.latent_norm_mean.shape)}, got "
                f"{tuple(mean.shape)} / {tuple(std.shape)}"
            )
        self.latent_norm_mean.copy_(mean)
        self.latent_norm_std.copy_(std.clamp_min(1e-6))
        self.latent_norm_initialized.fill_(True)

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.latent_normalize_for_dit or not bool(self.latent_norm_initialized.item()):
            return z
        mean = self.latent_norm_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_norm_std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.latent_normalize_for_dit or not bool(self.latent_norm_initialized.item()):
            return z
        mean = self.latent_norm_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_norm_std.to(device=z.device, dtype=z.dtype)
        return z * std + mean

    def _normalize_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        if not self.latent_normalize_for_dit or not bool(self.latent_norm_initialized.item()):
            return logvar
        std = self.latent_norm_std.to(device=logvar.device, dtype=logvar.dtype)
        return logvar - 2.0 * std.log()

    def encode_to_latent(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.vae.encode(x, sample_posterior=sample_posterior)
        z = self.normalize_latent(z)
        if return_stats:
            return z, self.normalize_latent(mu), self._normalize_logvar(logvar)
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(self.denormalize_latent(z))

    def vae_forward(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vae(x, sample_posterior=sample_posterior)

    def load_vae_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict) or "vae_state" not in checkpoint:
            raise RuntimeError(f"VAE checkpoint must contain 'vae_state': {path}")
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, dict) and "latent_normalize_for_dit" in model_config:
            self.latent_normalize_for_dit = bool(model_config["latent_normalize_for_dit"])
        state_dict = checkpoint["vae_state"]
        self.vae.load_state_dict(state_dict, strict=True)
        self.load_latent_norm_state(checkpoint.get("latent_norm_state"))

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
        if "global_style_state" in checkpoint:
            raw_state_dict = checkpoint["global_style_state"]
        elif "style_state" in checkpoint:
            raw_state_dict = checkpoint["style_state"]
        else:
            raise RuntimeError(f"Style checkpoint must contain 'global_style_state' or 'style_state': {path}")

        state_dict = self._normalize_global_style_state_dict(raw_state_dict)

        if not state_dict:
            raise RuntimeError(f"No global style weights found in checkpoint: {path}")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        unexpected_global_keys = [key for key in unexpected_keys if self._is_global_style_state_key(key)]
        if unexpected_global_keys:
            raise RuntimeError(
                "Unexpected style checkpoint keys: "
                f"{unexpected_global_keys}. The global style architecture likely changed."
            )
        missing_global_keys = [
            key
            for key in missing_keys
            if self._is_global_style_state_key(key)
        ]
        if missing_global_keys:
            raise RuntimeError(
                "Missing style checkpoint keys: "
                f"{missing_global_keys}. The global style architecture likely changed; "
                "rerun style pretraining for the new global extractor."
            )

    @classmethod
    def _normalize_global_style_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            mapped_key = key
            for legacy_prefix, current_prefix in cls.LEGACY_GLOBAL_STYLE_PREFIX_MAP.items():
                if key.startswith(legacy_prefix):
                    mapped_key = current_prefix + key[len(legacy_prefix):]
                    break
            if cls._is_global_style_state_key(mapped_key):
                normalized[mapped_key] = value
        return normalized

    def freeze_vae(self) -> None:
        for param in self.vae.parameters():
            param.requires_grad_(False)

    def freeze_style_global(self) -> None:
        for name, param in self.named_parameters():
            if self._is_global_style_state_key(name):
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

        style_global = None
        if need_style_global:
            global_encoder_context = torch.no_grad() if detach_global_style_encoder else nullcontext()
            with global_encoder_context:
                _, global_tokens = self.global_style_encoder(flat_style)
                if global_tokens is None:
                    raise RuntimeError("global_style_encoder did not produce global tokens")
                global_tokens = global_tokens.view(batch, refs, global_tokens.size(-1))
            global_context = torch.no_grad() if detach_global_style else nullcontext()
            with global_context:
                global_style = self.global_style_pool(global_tokens, valid_mask=ref_valid_mask).squeeze(1)
                style_global = self.style_global_proj(global_style)

        style_tokens = None
        style_token_mask = None
        if need_style_tokens:
            local_tokens, _ = self.token_style_encoder(flat_style)
            local_tokens = local_tokens.view(batch, refs, local_tokens.size(1), local_tokens.size(2))
            adapted_local_tokens = self.local_style_adapter(
                local_tokens.view(batch * refs, local_tokens.size(2), local_tokens.size(3))
            )
            pooled_per_ref = self.per_ref_style_pool(adapted_local_tokens)
            style_tokens = pooled_per_ref.view(
                batch,
                refs,
                self.style_tokens_per_ref,
                pooled_per_ref.size(-1),
            )
            style_token_mask = ref_valid_mask.unsqueeze(-1).expand(
                batch,
                refs,
                self.style_tokens_per_ref,
            )
            style_tokens = self.style_proj(style_tokens)
            style_tokens = style_tokens * style_token_mask.unsqueeze(-1).to(dtype=style_tokens.dtype)
            style_token_mask = style_token_mask.to(device=style_tokens.device, dtype=torch.bool)

        return {
            "style_tokens": style_tokens,
            "style_global": style_global,
            "style_token_mask": style_token_mask,
        }

    def predict_flow(
        self,
        x_t_latent: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
        style_global: torch.Tensor,
        style_token_mask: Optional[torch.Tensor] = None,
        return_style_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        return self.backbone(
            x_t_latent,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
            style_token_mask=style_token_mask,
            return_style_attn_weights=return_style_attn_weights,
        )

    def forward(
        self,
        x_t_latent: torch.Tensor,
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
            x_t_latent,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_pack["style_tokens"],
            style_global=style_pack["style_global"],
            style_token_mask=style_pack["style_token_mask"],
        )
