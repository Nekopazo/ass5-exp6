#!/usr/bin/env python3
"""Source-aligned FontDiffuser model with online-retrieved PartBank style tokens.

Model inputs: noisy image x_t, timestep t, content image, part images + mask.
No style reference image (RSI) branch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .source_fontdiffuser import ContentEncoder, UNet


class SourcePartRefUNet(nn.Module):
    _MODE_ALIASES = {
        "parts_vector_only": "part_only",
    }
    _VALID_MODES = {"baseline", "part_only", "style_only", "part_style"}

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        content_start_channel: int = 64,
        style_start_channel: int = 64,  # kept for backward arg compatibility
        unet_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        content_encoder_downsample_size: int = 4,
        channel_attn: bool = True,
        conditioning_profile: str = "parts_vector_only",
        attn_scales: Optional[tuple[int, ...]] = None,
        style_token_count: int = 8,
        style_token_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.unet_input_size = self.image_size // 2
        if self.unet_input_size * 2 != self.image_size:
            raise ValueError(f"image_size must be even for stem/head path, got {self.image_size}")

        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.style_token_count = int(style_token_count)
        self.style_token_dim = int(style_token_dim)
        if self.style_token_count <= 0 or self.style_token_dim <= 0:
            raise ValueError("style_token_count and style_token_dim must be > 0")

        self.conditioning_profile = self._normalize_conditioning_mode(conditioning_profile)

        # Content encoder runs on 256 and we explicitly pick c128/c64/c32/c16 for UNet down blocks.
        self.content_encoder = ContentEncoder(
            G_ch=content_start_channel, resolution=self.image_size, input_nc=self.in_channels,
        )

        # 256 → 128 / 128 → 256 via bilinear interpolation.
        # UNet operates directly on 1-channel 128×128 images (grayscale).
        self.unet_in_channels = self.in_channels  # 1

        # Part set encoder + DeepSets(mean+LN) -> fixed M style tokens.
        self.part_patch_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.style_token_dim),
        )
        self.part_set_norm = nn.LayerNorm(self.style_token_dim)
        self.style_token_mlp = nn.Sequential(
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_count * self.style_token_dim),
        )
        # Style image encoder -> tokens (same token geometry as part tokens).
        self.style_img_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.style_token_dim),
        )
        self.style_img_norm = nn.LayerNorm(self.style_token_dim)
        self.style_img_token_mlp = nn.Sequential(
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_count * self.style_token_dim),
        )
        # Gate controls how much part tokens are injected into style tokens.
        self.style_part_gate = nn.Sequential(
            nn.Linear(self.style_token_dim * 2 + 1, self.style_token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim, 1),
        )
        self.style_part_fuse_norm = nn.LayerNorm(self.style_token_dim)
        # Contrastive projection head: tokens -> mean -> LN -> MLP -> L2 norm
        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim, self.style_token_dim),
        )

        self.unet = UNet(
            sample_size=self.unet_input_size,
            in_channels=self.unet_in_channels,
            out_channels=self.unet_in_channels,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D"),
            up_block_types=("StyleUpBlock2D", "StyleUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=unet_channels,
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn="silu",
            norm_num_groups=32,
            norm_eps=1e-5,
            cross_attention_dim=self.style_token_dim,
            attention_head_dim=1,
            channel_attn=channel_attn,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
            content_start_channel=content_start_channel,
            reduction=32,
            attn_scales=attn_scales,
            mid_enable_content_attn=False,
        )

    @classmethod
    def _normalize_conditioning_mode(cls, mode: str) -> str:
        m = str(mode).strip().lower()
        m = cls._MODE_ALIASES.get(m, m)
        if m not in cls._VALID_MODES:
            raise ValueError(
                f"conditioning mode must be one of: {sorted(cls._VALID_MODES)} "
                f"(aliases: {sorted(cls._MODE_ALIASES.keys())}), got '{mode}'"
            )
        return m

    def load_part_vector_pretrained(self, ckpt_path: str) -> None:
        """Optional warm-start for the part encoder/token MLP branch."""
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_path, map_location="cpu")

        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported part-vector checkpoint format: {type(obj)}")
        if "state_dict" in obj and isinstance(obj.get("state_dict"), dict):
            self.load_state_dict(obj["state_dict"], strict=False)
            return
        self.load_state_dict(obj, strict=False)

    def _check_hw(self, x: torch.Tensor, name: str) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"{name} must be {self.image_size}x{self.image_size}, got {h}x{w}. "
                "Online resize is disabled."
            )

    @staticmethod
    def _check_part_mask_shape(mask: torch.Tensor, b: int, p: int) -> torch.Tensor:
        if mask.shape != (b, p):
            raise ValueError(f"part_mask shape must be {(b, p)}, got {tuple(mask.shape)}")
        return mask

    def _part_vector(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")

        x = part_imgs.view(b * p, c, h, w)
        z = self.part_patch_encoder(x).view(b, p, self.style_token_dim)

        if part_mask is None:
            mask = torch.ones((b, p), dtype=z.dtype, device=z.device)
        else:
            mask_bool = self._check_part_mask_shape(part_mask, b, p).to(device=z.device) > 0
            mask = mask_bool.to(dtype=z.dtype)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.part_set_norm(pooled)
        return pooled

    def encode_part_vector(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._part_vector(part_imgs, part_mask)

    def encode_part_tokens(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._parts_vector_hidden_states(part_imgs, part_mask)

    def encode_style_tokens(self, style_img: torch.Tensor) -> torch.Tensor:
        self._check_hw(style_img, "style_img")
        b, c, _, _ = style_img.shape
        if c != self.in_channels:
            raise ValueError(f"style_img channels mismatch: got {c}, expected {self.in_channels}")
        style_vec = self.style_img_encoder(style_img).view(b, self.style_token_dim)
        style_vec = self.style_img_norm(style_vec)
        tokens = self.style_img_token_mlp(style_vec)
        return tokens.view(b, self.style_token_count, self.style_token_dim)

    def _fuse_style_part_tokens(
        self,
        style_tokens: torch.Tensor,
        part_tokens: torch.Tensor,
        has_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = int(style_tokens.shape[0])
        style_pool = style_tokens.mean(dim=1)
        part_pool = part_tokens.mean(dim=1)
        if has_parts is None:
            has = torch.ones((b, 1), dtype=style_tokens.dtype, device=style_tokens.device)
        else:
            has = has_parts.to(device=style_tokens.device, dtype=style_tokens.dtype).view(b, 1)
        gate_in = torch.cat([style_pool, part_pool, has], dim=1)
        gate = torch.sigmoid(self.style_part_gate(gate_in)).view(b, 1, 1) * has.view(b, 1, 1)
        fused = style_tokens + gate * part_tokens
        return self.style_part_fuse_norm(fused)

    def encode_contrastive_z(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode a part set into an L2-normalised contrastive embedding.

        parts -> CNN -> DeepSets mean-pool -> MLP -> M tokens -> mean -> proj -> L2.
        """
        tokens = self._parts_vector_hidden_states(part_imgs, part_mask)  # (B, M, D)
        pooled = tokens.mean(dim=1)                                      # (B, D)
        z = self.contrastive_head(pooled)                                # (B, D)
        z = torch.nn.functional.normalize(z, dim=-1)                     # L2
        return z

    def _parts_vector_hidden_states(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        part_style_vec = self._part_vector(part_imgs, part_mask)
        tokens = self.style_token_mlp(part_style_vec)
        return tokens.view(int(part_style_vec.shape[0]), self.style_token_count, self.style_token_dim)

    def _content_features_for_unet(self, content_img: torch.Tensor) -> list[torch.Tensor]:
        content_img_feature, content_residual_features = self.content_encoder(content_img)
        # ContentEncoder(256) residuals: [x256, c128, c64, c32, c16, c8]
        if len(content_residual_features) < 5:
            raise RuntimeError(
                f"ContentEncoder residual feature count too small: {len(content_residual_features)}"
            )
        _ = content_img_feature  # keep parity with previous call pattern
        return [
            content_residual_features[1],
            content_residual_features[2],
            content_residual_features[3],
            content_residual_features[4],
        ]

    def _resolve_style_hidden_states(
        self,
        mode: str,
        style_img: Optional[torch.Tensor],
        part_imgs: Optional[torch.Tensor],
        part_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        mode_norm = self._normalize_conditioning_mode(mode)

        style_tokens: Optional[torch.Tensor] = None
        part_tokens: Optional[torch.Tensor] = None
        has_parts: Optional[torch.Tensor] = None

        if style_img is not None:
            style_tokens = self.encode_style_tokens(style_img)
        if part_imgs is not None:
            part_tokens = self._parts_vector_hidden_states(part_imgs, part_mask)
            if part_mask is None:
                b = int(part_imgs.shape[0])
                has_parts = torch.ones((b,), device=part_imgs.device, dtype=part_imgs.dtype)
            else:
                has_parts = (part_mask.to(device=part_imgs.device) > 0).any(dim=1).to(dtype=part_imgs.dtype)

        if mode_norm == "baseline":
            return None
        if mode_norm == "part_only":
            return part_tokens
        if mode_norm == "style_only":
            return style_tokens
        if mode_norm == "part_style":
            if style_tokens is None:
                return part_tokens
            if part_tokens is None:
                return style_tokens
            return self._fuse_style_part_tokens(style_tokens, part_tokens, has_parts=has_parts)
        return None

    # ------------------------------------------------------------------ #
    #  Pixel ↔ Latent helpers (used by Trainer, NOT inside forward)
    # ------------------------------------------------------------------ #
    def encode_to_latent(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """Pixel (B,1,256,256) → latent (B,1,128,128) via bilinear resize."""
        return torch.nn.functional.interpolate(
            x_pixel, size=self.unet_input_size, mode="bilinear", align_corners=False,
        )

    def decode_from_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Latent (B,1,128,128) → pixel (B,1,256,256) via bilinear resize."""
        return torch.nn.functional.interpolate(
            z_latent, size=self.image_size, mode="bilinear", align_corners=False,
        )

    def forward(
        self,
        x_t_latent: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: Optional[torch.Tensor] = None,
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        condition_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Predict noise (or velocity) at 128×128.

        Args:
            x_t_latent: noisy image (B, 1, 128, 128) — resized from 256.
            t:          timestep indices (B,).
            content_img: pixel-space content glyph (B, 1, 256, 256).
            style_img:  style reference image (B, 1, 256, 256) or None.
            part_imgs:  part images (B, P, 1, 64, 64) or None.
            part_mask:  (B, P) or None.
            condition_mode: one of baseline/part_only/style_only/part_style.

        Returns:
            eps_hat / v_hat (B, 1, 128, 128).
        """
        self._check_hw(content_img, "content_img")
        if style_img is not None:
            self._check_hw(style_img, "style_img")

        content_residual_features = self._content_features_for_unet(content_img)
        mode = self.conditioning_profile if condition_mode is None else self._normalize_conditioning_mode(condition_mode)
        style_hidden_states = self._resolve_style_hidden_states(mode, style_img, part_imgs, part_mask)

        encoder_hidden_states = [
            None,
            content_residual_features,
            style_hidden_states,
        ]

        (out_latent,) = self.unet(
            x_t_latent,
            t,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
        )
        return out_latent
