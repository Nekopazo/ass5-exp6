#!/usr/bin/env python3
"""Source-aligned DiffuFont model with hard-routed low/mid/high style tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_style_encoder import HierarchicalStyleEncoderMixin
from .source_fontdiffuser import ContentEncoder, UNet


class SourcePartRefUNet(nn.Module, HierarchicalStyleEncoderMixin):
    _MODE_ALIASES = {
        "parts_vector_only": "part_only",
    }
    _VALID_MODES = {"baseline", "part_only", "style_only"}

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        content_start_channel: int = 64,
        style_start_channel: int = 16,
        unet_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        content_encoder_downsample_size: int = 4,
        channel_attn: bool = True,
        conditioning_profile: str = "parts_vector_only",
        attn_scales: Optional[tuple[int, ...]] = None,
        style_token_dim: int = 256,
        style_token_count: int = 3,
        local_token_count: int = 3,
    ):
        super().__init__()
        _ = local_token_count
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.unet_input_size = self.image_size // 2
        if self.unet_input_size * 2 != self.image_size:
            raise ValueError(f"image_size must be even for stem/head path, got {self.image_size}")

        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.style_token_dim = int(style_token_dim)
        self.style_token_count = int(style_token_count)
        if self.style_token_dim <= 0:
            raise ValueError("style_token_dim must be > 0")
        if self.style_token_count != 3:
            raise ValueError(f"style_token_count must be 3, got {self.style_token_count}")

        self.conditioning_profile = self._normalize_conditioning_mode(conditioning_profile)
        self.content_encoder = ContentEncoder(
            G_ch=content_start_channel,
            resolution=self.unet_input_size,
            input_nc=self.in_channels,
        )
        self.unet_in_channels = self.in_channels
        self.style_start_channel = int(style_start_channel)

        self._init_hierarchical_style_encoder(
            in_channels=self.in_channels,
            style_token_dim=self.style_token_dim,
            style_token_count=self.style_token_count,
            local_token_count=3,
        )

        self.inject_mid = self._build_inject_head()
        self.inject_up16 = self._build_inject_head()
        self.inject_up32 = self._build_inject_head()
        self.inject_up64 = self._build_inject_head()

        if attn_scales is None:
            attn_scales = (16, 32, 64)

        self.unet = UNet(
            sample_size=self.unet_input_size,
            in_channels=self.unet_in_channels,
            out_channels=self.unet_in_channels,
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=("MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D", "MCADownBlock2D"),
            up_block_types=("StyleUpBlock2D", "StyleUpBlock2D", "StyleUpBlock2D", "StyleUpBlock2D"),
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
            content_start_channel=content_start_channel // 2,
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

    def _build_inject_head(self) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim, self.style_token_dim),
        )

    def load_style_pretrained(self, ckpt_path: str) -> None:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and isinstance(obj.get("style_encoder"), dict):
            sd = obj["style_encoder"]
        elif isinstance(obj, dict) and isinstance(obj.get("model_state"), dict):
            sd = obj["model_state"]
        elif isinstance(obj, dict):
            sd = obj
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(obj)}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[load_style_pretrained] loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}")

    def _check_hw(self, x: torch.Tensor, name: str) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        expected = self.unet_input_size
        if h != expected or w != expected:
            raise ValueError(f"{name} must be {expected}x{expected}, got {h}x{w}. Online resize is disabled.")

    def set_attention_logging(self, enabled: bool) -> None:
        if hasattr(self.unet, "set_attention_logging"):
            self.unet.set_attention_logging(bool(enabled))

    def reset_attention_logging(self) -> None:
        if hasattr(self.unet, "reset_attention_logging"):
            self.unet.reset_attention_logging()

    def collect_attention_logging(self) -> dict[str, float]:
        if not hasattr(self.unet, "collect_attention_logging"):
            return {}
        raw = self.unet.collect_attention_logging()
        out: dict[str, float] = {}
        for klen, mass in raw.items():
            if len(mass) <= 0:
                continue
            probs = torch.tensor(mass, dtype=torch.float64)
            probs = probs.clamp_min(1e-12)
            probs = probs / probs.sum().clamp_min(1e-12)
            out[f"attn_klen_{int(klen)}_max"] = float(probs.max().item())
            out[f"attn_klen_{int(klen)}_entropy"] = float(
                (-(probs * probs.log()).sum() / torch.log(torch.tensor(float(max(2, klen))))).item()
            )
        return out

    def _normalize_style_inputs(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if style_img.dim() == 4:
            b, c, _, _ = style_img.shape
            if c != self.in_channels:
                raise ValueError(f"style_img channels mismatch: got {c}, expected {self.in_channels}")
            self._check_hw(style_img, "style_img")
            style_img = style_img.unsqueeze(1)
            if style_ref_mask is not None and style_ref_mask.shape != (b, 1):
                raise ValueError(f"style_ref_mask must be {(b, 1)}, got {tuple(style_ref_mask.shape)}")
            if style_ref_mask is None:
                style_ref_mask = torch.ones((b, 1), device=style_img.device, dtype=torch.float32)
            return style_img, style_ref_mask

        if style_img.dim() != 5:
            raise ValueError(f"style_img must be 4D or 5D, got shape={tuple(style_img.shape)}")

        b, r, c, h, w = style_img.shape
        if c != self.in_channels:
            raise ValueError(f"style_img channels mismatch: got {c}, expected {self.in_channels}")
        if h != self.unet_input_size or w != self.unet_input_size:
            raise ValueError(f"style_img refs must be {self.unet_input_size}x{self.unet_input_size}, got {h}x{w}")

        if style_ref_mask is None:
            style_ref_mask = torch.ones((b, r), device=style_img.device, dtype=torch.float32)
        else:
            if style_ref_mask.shape != (b, r):
                raise ValueError(f"style_ref_mask must be {(b, r)}, got {tuple(style_ref_mask.shape)}")
            style_ref_mask = style_ref_mask.to(device=style_img.device, dtype=torch.float32)
        return style_img, style_ref_mask

    def _apply_style_token_dropout(self, style_tokens: torch.Tensor, drop_prob: float) -> torch.Tensor:
        p = float(max(0.0, min(1.0, drop_prob)))
        if p <= 0.0:
            return style_tokens
        if style_tokens.dim() != 3 or int(style_tokens.size(1)) != 3:
            raise ValueError(f"style_tokens must be (B,3,D), got {tuple(style_tokens.shape)}")

        b, t, _ = style_tokens.shape
        drop_rows = torch.rand((b,), device=style_tokens.device) < p
        if not bool(drop_rows.any().item()):
            return style_tokens
        keep = torch.ones((b, t), device=style_tokens.device, dtype=torch.bool)
        rows = torch.nonzero(drop_rows, as_tuple=False).flatten()
        picked = torch.randint(0, t, (rows.numel(),), device=style_tokens.device)
        keep[rows, picked] = False
        return style_tokens * keep.unsqueeze(-1).to(dtype=style_tokens.dtype)

    def _content_features_for_unet(self, content_img: torch.Tensor) -> list[torch.Tensor]:
        content_img_feature, content_residual_features = self.content_encoder(content_img)
        if len(content_residual_features) < 4:
            raise RuntimeError(f"ContentEncoder residual feature count too small: {len(content_residual_features)}")
        _ = content_img_feature
        return [
            None,
            content_residual_features[1],
            content_residual_features[2],
            content_residual_features[3],
        ]

    def _build_style_site_contexts(self, style_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        t_low = style_tokens[:, 0]
        t_mid = style_tokens[:, 1]
        t_high = style_tokens[:, 2]
        return {
            "mid": self.inject_mid(t_low).unsqueeze(1),
            "up_16": self.inject_up16(t_mid).unsqueeze(1),
            "up_32": self.inject_up32(t_mid).unsqueeze(1),
            "up_64": self.inject_up64(t_high).unsqueeze(1),
        }

    def _resolve_style_hidden_states(
        self,
        mode: str,
        style_img: Optional[torch.Tensor],
        style_ref_mask: Optional[torch.Tensor],
        style_token_drop_prob: float = 0.0,
    ) -> Optional[dict[str, torch.Tensor]]:
        mode_norm = self._normalize_conditioning_mode(mode)
        if mode_norm == "baseline" or style_img is None:
            return None

        tokens = self.encode_style_tokens(style_img, style_ref_mask=style_ref_mask)
        if self.training and float(style_token_drop_prob) > 0.0:
            tokens = self._apply_style_token_dropout(tokens, drop_prob=float(style_token_drop_prob))
        return self._build_style_site_contexts(tokens)

    def encode_to_latent(self, x_pixel: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x_pixel, size=self.unet_input_size, mode="bilinear", align_corners=False)

    def decode_from_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        return F.interpolate(z_latent, size=self.image_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        x_t_latent: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: Optional[torch.Tensor] = None,
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        condition_mode: Optional[str] = None,
        style_ref_mask: Optional[torch.Tensor] = None,
        style_token_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        _ = part_imgs
        _ = part_mask
        self._check_hw(content_img, "content_img")

        content_residual_features = self._content_features_for_unet(content_img)
        mode = self.conditioning_profile if condition_mode is None else self._normalize_conditioning_mode(condition_mode)
        style_hidden_states = self._resolve_style_hidden_states(
            mode,
            style_img,
            style_ref_mask=style_ref_mask,
            style_token_drop_prob=float(style_token_drop_prob),
        )

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
