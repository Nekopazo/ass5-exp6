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

        profile = str(conditioning_profile).strip().lower()
        if profile not in {"baseline", "parts_vector_only"}:
            raise ValueError(
                "conditioning_profile must be one of: baseline, parts_vector_only"
            )
        self.conditioning_profile = profile
        self.enable_parts_vector_condition = profile == "parts_vector_only"

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
        if not self.enable_parts_vector_condition:
            raise RuntimeError("encode_part_vector is unavailable when parts_vector conditioning is disabled.")
        return self._part_vector(part_imgs, part_mask)

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
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise (or velocity) at 128×128.

        Args:
            x_t_latent: noisy image (B, 1, 128, 128) — resized from 256.
            t:          timestep indices (B,).
            content_img: pixel-space content glyph (B, 1, 256, 256).
            part_imgs:  part images (B, P, 1, 64, 64) or None.
            part_mask:  (B, P) or None.

        Returns:
            eps_hat / v_hat (B, 1, 128, 128).
        """
        self._check_hw(content_img, "content_img")

        content_residual_features = self._content_features_for_unet(content_img)

        style_hidden_states = None
        if self.enable_parts_vector_condition:
            if part_imgs is not None:
                style_hidden_states = self._parts_vector_hidden_states(part_imgs, part_mask)
            else:
                b = int(x_t_latent.shape[0])
                dummy_parts = torch.zeros(
                    b, 1, self.in_channels, 64, 64,
                    device=x_t_latent.device, dtype=x_t_latent.dtype,
                )
                dummy_mask = torch.zeros(b, 1, device=x_t_latent.device, dtype=x_t_latent.dtype)
                style_hidden_states = self._parts_vector_hidden_states(dummy_parts, dummy_mask)

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
