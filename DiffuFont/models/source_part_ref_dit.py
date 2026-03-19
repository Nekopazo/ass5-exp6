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
    def __init__(self, in_channels: int = 1, latent_channels: int = 4) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            ResBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 128),
        )
        self.to_stats = nn.Conv2d(128, self.latent_channels * 2, kernel_size=1)

        self.decoder_in = nn.Conv2d(self.latent_channels, 128, kernel_size=3, padding=1)
        self.dec_block16 = ResBlock(128, 128)
        self.dec_up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
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
            nn.GroupNorm(_group_count(16), 16),
            nn.SiLU(),
        )
        self.decoder_out = nn.Conv2d(16, self.in_channels, kernel_size=3, padding=1)

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
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
        x = self.dec_up32(x)
        x = self.dec_up64(x)
        x = self.dec_up128(x)
        return torch.tanh(self.decoder_out(x))

    def forward(self, x: torch.Tensor, *, sample_posterior: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    STYLE_STATE_PREFIXES = (
        "style_encoder.",
        "global_style_pool.",
        "local_style_pool.",
        "style_proj.",
        "style_global_proj.",
        "style_contrastive_head.",
    )
    STYLE_FINETUNE_PREFIXES = (
        "global_style_pool.",
        "local_style_pool.",
        "style_global_proj.",
    )
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
        dit_depth: int = 12,
        dit_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        local_style_tokens_per_ref: int = 16,
        content_cross_attn_layers: int = 8,
        style_cross_attn_every_n_layers: int = 1,
        contrastive_proj_dim: int = 128,
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
        self.local_style_tokens_per_ref = max(1, int(local_style_tokens_per_ref))
        self.content_cross_attn_layers = max(0, min(self.dit_depth, int(content_cross_attn_layers)))
        self.style_cross_attn_every_n_layers = max(1, int(style_cross_attn_every_n_layers))
        self.contrastive_proj_dim = int(contrastive_proj_dim)

        self.vae = GlyphVAE(in_channels=self.in_channels, latent_channels=self.latent_channels)
        self.content_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=False,
        )
        self.style_encoder = GlyphTokenEncoder(
            image_size=self.image_size,
            patch_size=self.encoder_patch_size,
            hidden_dim=self.encoder_hidden_dim,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            add_cls_token=True,
        )
        self.global_style_pool = AttentionTokenPool(
            hidden_dim=self.encoder_hidden_dim,
            num_heads=self.encoder_heads,
            output_tokens=1,
        )
        self.local_style_pool = AttentionTokenPool(
            hidden_dim=self.encoder_hidden_dim,
            output_tokens=self.local_style_tokens_per_ref,
            num_heads=self.encoder_heads,
        )
        self.backbone = DiffusionTransformerBackbone(
            latent_channels=self.latent_channels,
            latent_size=self.latent_size,
            hidden_dim=self.dit_hidden_dim,
            depth=self.dit_depth,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            content_cross_attn_layers=self.content_cross_attn_layers,
            style_cross_attn_every_n_layers=self.style_cross_attn_every_n_layers,
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
            "local_style_tokens_per_ref": int(self.local_style_tokens_per_ref),
            "content_cross_attn_layers": int(self.content_cross_attn_layers),
            "style_cross_attn_every_n_layers": int(self.style_cross_attn_every_n_layers),
            "contrastive_proj_dim": int(self.contrastive_proj_dim),
        }

    def encode_to_latent(
        self,
        x: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.vae.encode(x, sample_posterior=sample_posterior)
        if return_stats:
            return z, mu, logvar
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

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
        state_dict = checkpoint["vae_state"]
        self.vae.load_state_dict(state_dict, strict=True)

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
            raise RuntimeError(f"Unexpected style checkpoint keys: {unexpected_style_keys}")
        missing_style_keys = [
            key
            for key in missing_keys
            if self._is_style_state_key(key)
        ]
        if missing_style_keys:
            raise RuntimeError(f"Missing style checkpoint keys: {missing_style_keys}")

    def freeze_vae(self) -> None:
        for param in self.vae.parameters():
            param.requires_grad_(False)

    def freeze_style(self) -> None:
        for name, param in self.named_parameters():
            if self._is_style_state_key(name):
                param.requires_grad_(False)

    def enable_partial_style_finetune(
        self,
        *,
        train_last_encoder_blocks: int = 1,
    ) -> list[tuple[str, nn.Parameter]]:
        train_last_encoder_blocks = max(0, int(train_last_encoder_blocks))
        first_train_block = max(0, len(self.style_encoder.blocks) - train_last_encoder_blocks)
        trainable_params: list[tuple[str, nn.Parameter]] = []
        for name, param in self.named_parameters():
            should_train = any(name.startswith(prefix) for prefix in self.STYLE_FINETUNE_PREFIXES)
            if not should_train and train_last_encoder_blocks > 0:
                if name.startswith("style_encoder.blocks."):
                    parts = name.split(".")
                    if len(parts) >= 4:
                        block_idx = int(parts[2])
                        should_train = block_idx >= first_train_block
                elif name.startswith("style_encoder.norm."):
                    should_train = True
            if should_train and not param.requires_grad:
                param.requires_grad_(True)
                trainable_params.append((name, param))
        return trainable_params

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

        style_token_mask = ref_valid_mask.unsqueeze(-1).expand(batch, refs, self.local_style_tokens_per_ref).reshape(batch, -1)
        style_context = torch.no_grad() if detach_style_encoder else nullcontext()
        with style_context:
            local_tokens, global_tokens = self.style_encoder(flat_style)
            local_tokens = local_tokens.view(batch, refs, local_tokens.size(1), local_tokens.size(2))
            global_tokens = global_tokens.view(batch, refs, global_tokens.size(-1))
            global_style = self.global_style_pool(global_tokens, valid_mask=ref_valid_mask).squeeze(1)
            style_global = self.style_global_proj(global_style)
            pooled_local = self.local_style_pool(local_tokens.reshape(batch * refs, local_tokens.size(2), local_tokens.size(3)))
            pooled_local = pooled_local.view(batch, refs, self.local_style_tokens_per_ref, self.encoder_hidden_dim)
            projected_local = self.style_proj(pooled_local)
            projected_local = projected_local * ref_valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=projected_local.dtype)
            style_tokens = projected_local.reshape(batch, -1, projected_local.size(-1))
            style_tokens = style_tokens * style_token_mask.unsqueeze(-1).to(dtype=style_tokens.dtype)

        contrastive_style = None
        if return_contrastive:
            contrastive_style = F.normalize(self.style_contrastive_head(style_global), dim=-1)
        return {
            "style_tokens": style_tokens,
            "style_global": style_global,
            "style_token_mask": style_token_mask,
            "contrastive_style": contrastive_style,
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
    ) -> torch.Tensor:
        return self.backbone(
            x_t_latent,
            timesteps,
            content_tokens=content_tokens,
            style_tokens=style_tokens,
            style_global=style_global,
            style_token_mask=style_token_mask,
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
