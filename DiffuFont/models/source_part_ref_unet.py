#!/usr/bin/env python3
"""Source-aligned FontDiffuser model with PartBank reference part vectors.

This module keeps the original FontDiffuser UNet/RSI/offset path and only
replaces reference style conditioning with PartBank-derived part vectors.
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
        image_size: int = 256,
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
        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.cross_attention_dim = style_start_channel * 16

        profile = str(conditioning_profile).strip().lower()
        if profile not in {"baseline", "parts_vector_only", "rsi_only", "full"}:
            raise ValueError(
                "conditioning_profile must be one of: baseline, parts_vector_only, rsi_only, full"
            )
        self.conditioning_profile = profile
        self.enable_parts_vector_condition = profile in {"parts_vector_only", "full"}
        self.enable_rsi_condition = profile in {"rsi_only", "full"}

        self.content_encoder = ContentEncoder(G_ch=content_start_channel, resolution=self.image_size)
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
        # Null parts_vector used by strict unconditional branch in CFG sampling.
        self.null_part_vector = nn.Parameter(torch.zeros(self.cross_attention_dim))
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

    def _copy_overlap(self, dst: torch.Tensor, src: torch.Tensor) -> int:
        slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(dst.shape, src.shape))
        if any(s.stop <= 0 for s in slices):
            return 0
        with torch.no_grad():
            dst[slices].copy_(src[slices].to(dtype=dst.dtype, device=dst.device))
        n = 1
        for s in slices:
            n *= int(s.stop - s.start)
        return int(n)

    def load_part_vector_pretrained(self, ckpt_path: str) -> None:
        """Load part-vector branch pretrain weights.

        Supported formats:
        1) split component checkpoint: {"component":"trainable_vector_cnn","state_dict": ...}
        2) part_style pretrain checkpoint: {"part_style_encoder": {"part_cnn": ..., "part_fc": ...}}
        """
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_path, map_location="cpu")

        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported part-vector checkpoint format: {type(obj)}")

        # Case 1: exact state dict from split-save component.
        if "state_dict" in obj and isinstance(obj.get("state_dict"), dict):
            state = obj["state_dict"]
            self.load_state_dict(state, strict=False)
            return

        # Case 2: checkpoint directly stores current vector-branch keys.
        if any(str(k).startswith("part_patch_encoder.") for k in obj.keys()):
            self.load_state_dict(obj, strict=False)
            return

        # Case 3: legacy part_style pretrain (partial warm-start by overlap copy).
        pstate = obj.get("part_style_encoder", obj)
        if not isinstance(pstate, dict):
            raise RuntimeError("part-vector checkpoint missing dict state")
        pc = pstate.get("part_cnn")
        if not isinstance(pc, dict):
            raise RuntimeError("part-vector checkpoint missing 'part_style_encoder.part_cnn'")

        # map conv weights by overlap: old conv0/2/4 -> new conv0/3/6
        mapping = [
            ("0.weight", "part_patch_encoder.0.weight"),
            ("2.weight", "part_patch_encoder.3.weight"),
            ("4.weight", "part_patch_encoder.6.weight"),
        ]
        dst_state = self.state_dict()
        copied = 0
        for src_k, dst_k in mapping:
            src_w = pc.get(src_k)
            dst_w = dst_state.get(dst_k)
            if isinstance(src_w, torch.Tensor) and isinstance(dst_w, torch.Tensor):
                copied += self._copy_overlap(dst_w, src_w)
        self.load_state_dict(dst_state, strict=False)
        if copied <= 0:
            raise RuntimeError("No overlapping conv weights copied from part-style pretrain checkpoint.")

    def _check_hw(self, x: torch.Tensor, name: str) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"{name} must be {self.image_size}x{self.image_size}, got {h}x{w}. "
                "Online resize is disabled."
            )

    def _part_vector(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")
        x = part_imgs.view(b * p, c, h, w)
        z = self.part_patch_encoder(x).view(b, p, self.cross_attention_dim)

        if part_mask is None:
            mask = torch.ones((b, p), dtype=torch.bool, device=z.device)
        else:
            if part_mask.shape != (b, p):
                raise ValueError(f"part_mask shape must be {(b, p)}, got {tuple(part_mask.shape)}")
            mask = part_mask.to(device=z.device) > 0
        # DeepSets-style set aggregation: masked sum + unit-norm projection.
        g = (z * mask.to(dtype=z.dtype).unsqueeze(-1)).sum(dim=1)
        g = F.normalize(g, dim=-1)
        return g

    def encode_part_vector(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enable_parts_vector_condition:
            raise RuntimeError("encode_part_vector is unavailable when parts_vector conditioning is disabled.")
        return self._part_vector(part_imgs, part_mask)

    def _parts_vector_hidden_states(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")
        part_style_vec = self._part_vector(part_imgs, part_mask)
        # Keep UNet interface unchanged: pass a length-1 condition sequence.
        return part_style_vec.unsqueeze(1)

    def _null_parts_vector_hidden_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        null_vec = self.null_part_vector.to(device=device, dtype=dtype)
        return null_vec.view(1, 1, -1).expand(batch_size, 1, -1)

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
        content_img_feature, content_residual_features = self.content_encoder(content_img)
        content_residual_features.append(content_img_feature)

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

        out, offset_out_sum = self.unet(
            x_t,
            t,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=self.content_encoder_downsample_size,
        )
        self.last_offset_loss = offset_out_sum
        if out.shape[-2] != out_h or out.shape[-1] != out_w:
            raise RuntimeError(
                f"UNet output size mismatch: got {tuple(out.shape[-2:])}, expected {(out_h, out_w)}"
            )
        return out
