#!/usr/bin/env python3
"""Source-aligned FontDiffuser model with online-retrieved PartBank style tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .source_fontdiffuser import ContentEncoder, UNet


class SourcePartRefUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
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

        # Kept for runtime interface compatibility with existing callsites.
        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.style_token_count = int(style_token_count)
        self.style_token_dim = int(style_token_dim)
        if self.style_token_count <= 0 or self.style_token_dim <= 0:
            raise ValueError("style_token_count and style_token_dim must be > 0")

        profile = str(conditioning_profile).strip().lower()
        if profile not in {"baseline", "parts_vector_only", "rsi_only", "full"}:
            raise ValueError(
                "conditioning_profile must be one of: baseline, parts_vector_only, rsi_only, full"
            )
        self.conditioning_profile = profile
        self.enable_parts_vector_condition = profile in {"parts_vector_only", "full"}
        self.enable_rsi_condition = profile in {"rsi_only", "full"}

        # Content encoder runs on 256 and we explicitly pick c128/c64/c32/c16 for UNet down blocks.
        self.content_encoder = ContentEncoder(G_ch=content_start_channel, resolution=self.image_size)

        # 256 -> 128 stem and 128 -> 256 head, as requested.
        self.input_stem = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1)
        self.output_head = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
        )

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
        self.null_part_vector = nn.Parameter(torch.zeros(self.style_token_dim))
        self.null_style_tokens = nn.Parameter(torch.zeros(self.style_token_count, self.style_token_dim))

        self.unet = UNet(
            sample_size=self.unet_input_size,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D"),
            up_block_types=("StyleRSIUpBlock2D", "StyleRSIUpBlock2D", "UpBlock2D", "UpBlock2D"),
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
        self.last_offset_loss = torch.tensor(0.0)

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

    def encode_style_feature(self, glyph_img: torch.Tensor) -> torch.Tensor:
        """Encode a glyph image into style feature for style-consistency loss."""
        if glyph_img.dim() != 4:
            raise ValueError(f"glyph_img must be 4D (B,C,H,W), got {tuple(glyph_img.shape)}")
        if int(glyph_img.shape[1]) != self.in_channels:
            raise ValueError(
                f"glyph_img channel mismatch: got {int(glyph_img.shape[1])}, expected {self.in_channels}"
            )
        feat = self.part_patch_encoder(glyph_img)
        return self.part_set_norm(feat)

    def _parts_vector_hidden_states(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        part_style_vec = self._part_vector(part_imgs, part_mask)
        tokens = self.style_token_mlp(part_style_vec)
        return tokens.view(int(part_style_vec.shape[0]), self.style_token_count, self.style_token_dim)

    def _null_parts_vector_hidden_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        null_tokens = self.null_style_tokens.to(device=device, dtype=dtype)
        return null_tokens.view(1, self.style_token_count, self.style_token_dim).expand(batch_size, -1, -1)

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

    def _rsi_content_features(
        self,
        style_img: torch.Tensor,
        batch_size: int,
    ) -> list[torch.Tensor]:
        # use reference style glyph image(s) -> content encoder -> structure features.
        if style_img.dim() != 4:
            raise ValueError(f"style_img must be 4D, got {tuple(style_img.shape)}")
        sb, sck, sh, sw = style_img.shape
        if sb != batch_size:
            raise ValueError(f"style_img batch mismatch: got {sb}, expected {batch_size}")
        if sck == self.in_channels:
            style_ref = style_img
        elif sck % self.in_channels == 0:
            sk = sck // self.in_channels
            style_ref = style_img.view(batch_size, sk, self.in_channels, sh, sw)[:, 0]
        else:
            raise ValueError(
                f"style_img channels mismatch: got {sck}, expected {self.in_channels} or multiple of it"
            )

        self._check_hw(style_ref, "style_img")
        style_content_feature, style_content_res_features = self.content_encoder(style_ref)
        style_content_res_features.append(style_content_feature)
        return style_content_res_features

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: Optional[torch.Tensor],
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_h, out_w = int(x_t.shape[-2]), int(x_t.shape[-1])
        self._check_hw(x_t, "x_t")
        self._check_hw(content_img, "content_img")

        x_t_stem = self.input_stem(x_t)
        content_residual_features = self._content_features_for_unet(content_img)

        if self.enable_rsi_condition and style_img is None:
            raise ValueError("style_img is required when RSI conditioning is enabled.")

        style_hidden_states = None
        style_content_res_features = None
        if self.enable_parts_vector_condition:
            if part_imgs is not None:
                style_hidden_states = self._parts_vector_hidden_states(part_imgs, part_mask)
            else:
                style_hidden_states = self._null_parts_vector_hidden_states(
                    batch_size=int(x_t.shape[0]),
                    device=x_t.device,
                    dtype=x_t.dtype,
                )

        if self.enable_rsi_condition and style_img is not None:
            style_content_res_features = self._rsi_content_features(style_img, int(x_t.shape[0]))

        encoder_hidden_states = [
            None,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out_stem, offset_out_sum = self.unet(
            x_t_stem,
            t,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
        )
        out = self.output_head(out_stem)
        self.last_offset_loss = offset_out_sum

        if out.shape[-2] != out_h or out.shape[-1] != out_w:
            raise RuntimeError(
                f"UNet output size mismatch: got {tuple(out.shape[-2:])}, expected {(out_h, out_w)}"
            )
        return out
