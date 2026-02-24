#!/usr/bin/env python3
"""Source-aligned FontDiffuser model with PartBank reference tokens.

This module keeps the original FontDiffuser UNet/RSI/offset path and only
replaces reference style-token construction with PartBank-derived features.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .source_fontdiffuser import ContentEncoder, UNet


class SourcePartRefUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 96,
        style_k: int = 3,
        content_start_channel: int = 64,
        style_start_channel: int = 64,
        unet_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        content_encoder_downsample_size: int = 3,
        channel_attn: bool = True,
        conditioning_profile: str = "full",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.style_k = int(style_k)
        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.cross_attention_dim = style_start_channel * 16

        profile = str(conditioning_profile).strip().lower()
        if profile not in {"baseline", "token_only", "rsi_only", "full"}:
            raise ValueError(
                "conditioning_profile must be one of: baseline, token_only, rsi_only, full"
            )
        self.conditioning_profile = profile
        self.enable_token_condition = profile in {"token_only", "full"}
        self.enable_rsi_condition = profile in {"rsi_only", "full"}

        self.content_encoder = ContentEncoder(G_ch=content_start_channel, resolution=self.image_size)
        self.part_token_count = 8
        self.part_token_heads = 8
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
            nn.Linear(256, self.cross_attention_dim),
            nn.LayerNorm(self.cross_attention_dim),
        )
        self.part_queries = nn.Parameter(torch.randn(self.part_token_count, self.cross_attention_dim) * 0.02)
        self.part_tokenizer = nn.MultiheadAttention(
            embed_dim=self.cross_attention_dim,
            num_heads=self.part_token_heads,
            batch_first=True,
        )
        self.part_token_refine = nn.Sequential(
            nn.LayerNorm(self.cross_attention_dim),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim),
            nn.GELU(),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim),
        )
        self.unet = UNet(
            sample_size=self.image_size,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("DownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "StyleRSIUpBlock2D", "StyleRSIUpBlock2D", "UpBlock2D"),
            block_out_channels=unet_channels,
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn="silu",
            norm_num_groups=32,
            norm_eps=1e-5,
            cross_attention_dim=self.cross_attention_dim,
            attention_head_dim=1,
            channel_attn=channel_attn,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
            content_start_channel=content_start_channel,
            reduction=32,
        )
        self.last_offset_loss = torch.tensor(0.0)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.image_size and x.shape[-2] == self.image_size:
            return x
        return F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _part_tokens(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")
        x = self._resize(part_imgs.view(b * p, c, h, w))
        z = self.part_patch_encoder(x).view(b, p, self.cross_attention_dim)

        if part_mask is None:
            mask = torch.ones((b, p), dtype=torch.bool, device=z.device)
        else:
            if part_mask.shape != (b, p):
                raise ValueError(f"part_mask shape must be {(b, p)}, got {tuple(part_mask.shape)}")
            mask = part_mask.to(device=z.device) > 0
        # MultiheadAttention uses True for padded (invalid) positions.
        key_padding_mask = ~mask
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False

        q = self.part_queries.unsqueeze(0).expand(b, -1, -1)
        tokens, _ = self.part_tokenizer(q, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + self.part_token_refine(tokens)
        return tokens

    def encode_part_tokens(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enable_token_condition:
            raise RuntimeError("encode_part_tokens is unavailable when token conditioning is disabled.")
        return self._part_tokens(part_imgs, part_mask)

    def _part_style_states(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        # part_imgs: (B,P,C,H,W)
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")

        style_hidden_states = self._part_tokens(part_imgs, part_mask) if self.enable_token_condition else None
        style_img_feature = None

        if part_mask is None:
            w_mask = torch.ones((b, p), dtype=part_imgs.dtype, device=part_imgs.device)
        else:
            w_mask = part_mask.to(dtype=part_imgs.dtype, device=part_imgs.device)
            if w_mask.shape != (b, p):
                raise ValueError(f"part_mask shape must be {(b, p)}, got {tuple(w_mask.shape)}")

        denom = w_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        w5 = w_mask.view(b, p, 1, 1, 1)

        style_content_res_features = None
        if self.enable_rsi_condition:
            part_proxy = (part_imgs * w5).sum(dim=1) / denom.view(b, 1, 1, 1)
            part_proxy = self._resize(part_proxy)
            style_content_feature, style_content_res_features = self.content_encoder(part_proxy)
            style_content_res_features.append(style_content_feature)

        return style_img_feature, style_hidden_states, style_content_res_features

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: Optional[torch.Tensor],
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del style_img, class_ids

        out_h, out_w = int(x_t.shape[-2]), int(x_t.shape[-1])
        x_t = self._resize(x_t)
        content_img = self._resize(content_img)
        content_img_feature, content_residual_features = self.content_encoder(content_img)
        content_residual_features.append(content_img_feature)

        style_img_feature = None
        style_hidden_states = None
        style_content_res_features = None
        if part_imgs is not None and (self.enable_token_condition or self.enable_rsi_condition):
            style_img_feature, style_hidden_states, style_content_res_features = self._part_style_states(part_imgs, part_mask)

        encoder_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]

        out, offset_out_sum = self.unet(
            x_t,
            t,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
        )
        self.last_offset_loss = offset_out_sum
        if out.shape[-2] != out_h or out.shape[-1] != out_w:
            out = F.interpolate(out, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return out
