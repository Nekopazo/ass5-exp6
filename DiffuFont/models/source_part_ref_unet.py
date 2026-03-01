#!/usr/bin/env python3
"""Source-aligned FontDiffuser model — 1 token per branch.

Model inputs: noisy image x_t, timestep t, content image, part images + mask.
Part images → CNN each → DeepSets sum-pool → 1 token (B, 1, D).
Style image → CNN → 1 token (B, 1, D).
part_style is temporarily aligned with part_only (single part token).

Contrastive learning and cross-attention share the *same* aggregated vector,
ensuring optimisation target consistency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        style_start_channel: int = 16,
        unet_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        content_encoder_downsample_size: int = 4,
        channel_attn: bool = True,
        conditioning_profile: str = "parts_vector_only",
        attn_scales: Optional[tuple[int, ...]] = None,
        style_token_dim: int = 256,
        part_encode_chunk_size: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.unet_input_size = self.image_size // 2
        if self.unet_input_size * 2 != self.image_size:
            raise ValueError(f"image_size must be even for stem/head path, got {self.image_size}")

        self.content_encoder_downsample_size = int(content_encoder_downsample_size)
        self.style_token_dim = int(style_token_dim)
        self.style_start_channel = int(style_start_channel)
        self.part_encode_chunk_size = max(0, int(part_encode_chunk_size))
        if self.style_token_dim <= 0:
            raise ValueError("style_token_dim must be > 0")
        if self.style_start_channel <= 0:
            raise ValueError("style_start_channel must be > 0")

        self.conditioning_profile = self._normalize_conditioning_mode(conditioning_profile)

        # Content encoder runs on 128 (half-res) and we inject into Down-1/2/3 only.
        # Down-0 (128×128) is skipped to avoid holding 256×256 tensors in VRAM.
        self.content_encoder = ContentEncoder(
            G_ch=content_start_channel, resolution=self.unet_input_size, input_nc=self.in_channels,
        )

        # 256 → 128 / 128 → 256 via bilinear interpolation.
        # UNet operates directly on 1-channel 128×128 images (grayscale).
        self.unet_in_channels = self.in_channels  # 1

        # Part/style encoders: 1x128x128 or 1x40x40 grayscale -> style_start*16 channels.
        # With style_start_channel=16 this gives 256 channels as requested.
        c1 = self.style_start_channel
        c2 = self.style_start_channel * 4
        c3 = self.style_start_channel * 16
        self._style_feat_channels = c3

        self.part_patch_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.part_feat_to_token = (
            nn.Identity() if c3 == self.style_token_dim else nn.Linear(c3, self.style_token_dim)
        )

        # Style image encoder → 1 token (style_token_dim).  Same arch as part.
        self.style_img_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.style_feat_to_token = (
            nn.Identity() if c3 == self.style_token_dim else nn.Linear(c3, self.style_token_dim)
        )

        # Contrastive projection head: aggregated part token -> LN -> MLP -> L2 norm
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
            content_start_channel=content_start_channel // 2,  # halved: encoder runs at 128, features are half-channel vs 256 mode
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
        """Optional warm-start for the part encoder / contrastive head.

        Supports three checkpoint layouts:
          1. {"part_encoder": state_dict, ...}  (from pretrain_part_style_encoder)
          2. {"state_dict": state_dict, ...}    (legacy full-model snapshot)
          3. plain state_dict
        """
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_path, map_location="cpu")

        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported part-vector checkpoint format: {type(obj)}")
        # Prefer "part_encoder" key (pretrain script output).
        if "part_encoder" in obj and isinstance(obj["part_encoder"], dict):
            sd = obj["part_encoder"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            sd = obj
        missing, unexpected = self.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(
            f"[load_part_vector_pretrained] loaded={loaded} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def _check_hw(self, x: torch.Tensor, name: str) -> None:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        expected = self.unet_input_size  # content/style images are now resized to 128 on CPU
        if h != expected or w != expected:
            raise ValueError(
                f"{name} must be {expected}x{expected}, got {h}x{w}. "
                "Online resize is disabled."
            )

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
        if 2 in raw and len(raw[2]) >= 2:
            out["attn_part"] = float(raw[2][0])
            out["attn_style"] = float(raw[2][1])
        if 1 in raw and len(raw[1]) >= 1:
            out["attn_single"] = float(raw[1][0])
        return out

    @staticmethod
    def _check_part_mask_shape(mask: torch.Tensor, b: int, p: int) -> torch.Tensor:
        if mask.shape != (b, p):
            raise ValueError(f"part_mask shape must be {(b, p)}, got {tuple(mask.shape)}")
        return mask

    # ------------------------------------------------------------------ #
    #  Part encoding: P images → 1 aggregated token
    # ------------------------------------------------------------------ #
    def _encode_per_part_vectors(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode each part image independently into 1 vector each.

        Internal helper — callers should use encode_part_tokens() or
        encode_part_vector() instead.

        Args:
            part_imgs: (B, P, C, H, W)
            part_mask: (B, P) or None.  0 = padding, 1 = valid.

        Returns:
            vectors: (B, P, D) — per-image vectors.
                     Padding positions are zeroed out.
        """
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(f"part_imgs channels mismatch: got {c}, expected {self.in_channels}")

        x = part_imgs.view(b * p, c, h, w)
        if self.part_encode_chunk_size > 0 and x.size(0) > self.part_encode_chunk_size:
            z_chunks = []
            for xc in x.split(self.part_encode_chunk_size, dim=0):
                z_chunks.append(self.part_patch_encoder(xc))
            z_flat = torch.cat(z_chunks, dim=0)
        else:
            z_flat = self.part_patch_encoder(x)
        vec_flat = F.adaptive_avg_pool2d(z_flat, 1).flatten(1)
        vec_flat = self.part_feat_to_token(vec_flat)
        vectors = vec_flat.view(b, p, self.style_token_dim)  # (B, P, D)

        # Zero out padding positions
        if part_mask is not None:
            mask_bool = self._check_part_mask_shape(part_mask, b, p).to(device=vectors.device) > 0
            vectors = vectors * mask_bool.unsqueeze(-1).to(dtype=vectors.dtype)

        return vectors

    def _masked_sum_pool(
        self,
        vectors: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """DeepSets-style sum-pool (B, P, D) → (B, D) respecting mask."""
        if part_mask is not None:
            mask_f = part_mask.to(device=vectors.device, dtype=vectors.dtype).unsqueeze(-1)
            return (vectors * mask_f).sum(dim=1)  # (B, D)
        return vectors.sum(dim=1)  # (B, D)

    def encode_part_vector(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Part images → 1 aggregated vector (B, D)."""
        vectors = self._encode_per_part_vectors(part_imgs, part_mask)
        pooled = self._masked_sum_pool(vectors, part_mask)
        return F.normalize(pooled, dim=-1, eps=1e-12)

    def encode_part_tokens(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Part images → 1 aggregated token (B, 1, D)."""
        return self.encode_part_vector(part_imgs, part_mask).unsqueeze(1)

    def encode_style_tokens(self, style_img: torch.Tensor) -> torch.Tensor:
        """Encode 1 style image into 1 token: (B, 1, D)."""
        self._check_hw(style_img, "style_img")
        b, c, _, _ = style_img.shape
        if c != self.in_channels:
            raise ValueError(f"style_img channels mismatch: got {c}, expected {self.in_channels}")
        style_feat = self.style_img_encoder(style_img)
        style_vec = F.adaptive_avg_pool2d(style_feat, 1).flatten(1)
        style_vec = self.style_feat_to_token(style_vec).view(b, self.style_token_dim)
        return style_vec.unsqueeze(1)  # (B, 1, D)

    def encode_contrastive_z(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode a part set into an L2-normalised contrastive embedding.

        Aggregated part vector (same as cross-attention token) → proj → L2.
        Contrastive loss directly shapes the token that UNet cross-attention uses.
        """
        pooled = self.encode_part_vector(part_imgs, part_mask)  # (B, D)
        z = self.contrastive_head(pooled)                       # (B, D)
        z = torch.nn.functional.normalize(z, dim=-1)            # L2
        return z

    def _content_features_for_unet(self, content_img: torch.Tensor) -> list[torch.Tensor]:
        content_img_feature, content_residual_features = self.content_encoder(content_img)
        # ContentEncoder(128) residuals: [x128, c64, c32, c16, c8]
        # Down-0 (128×128) receives None → injection skipped.
        # Down-1/2/3 receive feat[1]/[2]/[3] with matching spatial and channel sizes.
        if len(content_residual_features) < 4:
            raise RuntimeError(
                f"ContentEncoder residual feature count too small: {len(content_residual_features)}"
            )
        _ = content_img_feature  # keep parity with previous call pattern
        return [
            None,                              # Down-0: no injection (128×128 level skipped)
            content_residual_features[1],      # Down-1: 64×64
            content_residual_features[2],      # Down-2: 32×32
            content_residual_features[3],      # Down-3: 16×16
        ]

    def _resolve_style_hidden_states(
        self,
        mode: str,
        style_img: Optional[torch.Tensor],
        part_imgs: Optional[torch.Tensor],
        part_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Build the cross-attention token sequence for the UNet.

        part_only  → (B, 1, D)    1 aggregated part token
        style_only → (B, 1, D)    1 style token
        part_style → (B, 2, D)    concat(part_token, style_token)
        baseline   → None
        """
        mode_norm = self._normalize_conditioning_mode(mode)

        style_tokens: Optional[torch.Tensor] = None
        part_tokens: Optional[torch.Tensor] = None

        if style_img is not None:
            style_tokens = self.encode_style_tokens(style_img)           # (B, 1, D)
        if part_imgs is not None:
            part_tokens = self.encode_part_tokens(part_imgs, part_mask)   # (B, 1, D)

        if mode_norm == "baseline":
            return None
        if mode_norm == "part_only":
            return part_tokens
        if mode_norm == "style_only":
            return style_tokens
        if mode_norm == "part_style":
            # Temporary alignment requested: make part_style branch identical to part_only.
            return part_tokens
        return None

    # ------------------------------------------------------------------ #
    #  Pixel ↔ Latent helpers (used by Trainer, NOT inside forward)
    # ------------------------------------------------------------------ #
    def encode_to_latent(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """Pixel (B,1,H,W) → latent (B,1,unet_input_size,unet_input_size) via bilinear resize.

        Since dataset now resizes glyph images to unet_input_size (128) on CPU,
        this is effectively a no-op during training (128→128).
        """
        return torch.nn.functional.interpolate(
            x_pixel, size=self.unet_input_size, mode="bilinear", align_corners=False,
        )

    def decode_from_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Latent (B,1,128,128) → pixel (B,1,image_size,image_size) via bilinear resize.

        image_size=256 (default): output is 256×256 for visualization/saving only.
        This tensor is produced only at inference time, not during the training forward pass.
        """
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
            x_t_latent: noisy image (B, 1, 128, 128) — resized from target.
            t:          timestep indices (B,).
            content_img: content glyph (B, 1, 128, 128) — resized to half-res on CPU.
            style_img:  style reference image (B, 1, 128, 128) or None.
            part_imgs:  part images (B, P, 1, 40, 40) or None.
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
