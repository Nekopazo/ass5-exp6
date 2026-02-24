#!/usr/bin/env python3
"""
FontDiffusionUNet
=================

Implementation of the main network described in the paper.  The network
integrates the following modules:

1. **Content encoder**  – a stack of Conv-BN-ReLU blocks that extract content features.
2. **Style encoder**    – identical architecture for style feature extraction.
3. **U-Net backbone**   – DACA blocks are inserted in the down-sampling path, FGSA
   blocks in the up-sampling path.
4. **AdaLN**            – Adaptive LayerNorm injecting the diffusion timestep *t*.
5. **Sinusoidal timestep embedding** passed through an MLP.

Forward signature
-----------------
```python
out = model(
    x_t,         # noisy image  (B,C,H,W)
    t,           # timestep tensor (B,)
    content_img, # content image (B,C,H,W)
    style_img,   # reference style images (B,C*k,H,W), optional when use_global_style=False
)
# returns  \hat x_0   (B,C,H,W)
```
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.daca import DACA
from models.fgsa import FGSA
from models.style_encoders import GlyphStyleEncoder, encode_style_stack


# ------------------------- 基础工具 ------------------------- #


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal timestep embeddings (same formulation as in the Transformer).

    Args:
        timesteps:  `(B,)` timestep indices
        dim:        embedding dimension

    Returns
    -------
    Tensor of shape `(B, dim)`
    """
    half_dim = dim // 2
    exponent = -math.log(10000.0) / (half_dim - 1)
    exponents = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * exponent)
    angles = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization.
    GroupNorm over spatial dimensions followed by a timestep-conditioned affine transform.
    """

    def __init__(self, channels: int, t_embed_dim: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(1, channels, eps=1e-6, affine=False)  # LayerNorm over (H, W)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, channels * 2),  # γ, β
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W), t_emb: (B, t_embed_dim)"""
        B, C, _, _ = x.shape
        h = self.norm(x)
        params = self.mlp(t_emb).view(B, 2, C, 1, 1)  # (B,2,C,1,1)
        gamma, beta = params[:, 0], params[:, 1]
        return gamma * h + beta + x  # 残差加入，以稳健训练


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ------------------------- ResNet Block ------------------------- #


class ResBlock(nn.Module):
    """2×Conv + (optional) AdaLN residual block"""

    def __init__(self, channels: int, t_embed_dim: int, use_adaln: bool = True):
        super().__init__()
        self.use_adaln = use_adaln
        self.conv1 = ConvBNReLU(channels, channels)
        if use_adaln:
            self.adaln1 = AdaLN(channels, t_embed_dim)
        self.conv2 = ConvBNReLU(channels, channels)
        if use_adaln:
            self.adaln2 = AdaLN(channels, t_embed_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        if self.use_adaln:
            x = self.adaln1(x, t_emb)
        x = self.conv2(x)
        if self.use_adaln:
            x = self.adaln2(x, t_emb)
        return x + residual


# Decoder-side ResNet block with FGSA + AdaLN
class DecoderResBlock(nn.Module):
    def __init__(self, channels: int, t_embed_dim: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.fgsa = FGSA(channels)
        self.adaln = AdaLN(channels, t_embed_dim)
        self.conv2 = ConvBNReLU(channels, channels)

    def forward(self, x: torch.Tensor, style_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.fgsa(x, style_feat)
        x = self.adaln(x, t_emb)
        x = self.conv2(x)
        return x + residual


class AttnXGatedFusion(nn.Module):
    """AttnX-X1: lightweight gated residual style fusion."""

    def __init__(self, channels: int, style_channels: int, t_embed_dim: int):
        super().__init__()
        self.style_proj = nn.Conv2d(style_channels, channels, kernel_size=1, bias=True)
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, channels),
        )

    def forward(self, x: torch.Tensor, style_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        style_proj = self.style_proj(style_feat)
        gate = torch.sigmoid(self.gate_mlp(t_emb)).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return x + gate * style_proj


class GlobalStyleFusion(nn.Module):
    """Inject global style parts_vector conditions by cross-attention + FiLM."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        t_embed_dim: int,
        use_cross_attn: bool = True,
        use_film: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_cross_attn = bool(use_cross_attn)
        self.use_film = bool(use_film)

        if self.use_cross_attn:
            self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True)
            self.k_proj = nn.Linear(style_dim, self.channels)
            self.v_proj = nn.Linear(style_dim, self.channels)
            self.out_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True)
            self.attn_scale = self.channels ** -0.5

        if self.use_film:
            self.film_norm = nn.GroupNorm(1, self.channels, eps=1e-6, affine=False)
            self.film_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(style_dim + t_embed_dim, self.channels * 2),
            )

    def forward(
        self,
        x: torch.Tensor,
        style_tokens: Optional[torch.Tensor],
        style_global: Optional[torch.Tensor],
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_cross_attn:
            if style_tokens is None:
                raise ValueError("style_tokens is required when use_cross_attn=True")
            b, c, h, w = x.shape
            q = self.q_proj(x).view(b, c, h * w).transpose(1, 2)          # (B,HW,C)
            k = self.k_proj(style_tokens)                                  # (B,K,C)
            v = self.v_proj(style_tokens)                                  # (B,K,C)
            attn = torch.softmax((q @ k.transpose(1, 2)) * self.attn_scale, dim=-1)  # (B,HW,K)
            ctx = attn @ v                                                 # (B,HW,C)
            ctx = ctx.transpose(1, 2).contiguous().view(b, c, h, w)
            x = x + self.out_proj(ctx)

        if self.use_film:
            if style_global is None:
                raise ValueError("style_global is required when use_film=True")
            cond = torch.cat([style_global, t_emb], dim=-1)
            params = self.film_mlp(cond).view(x.size(0), 2, self.channels, 1, 1)
            gamma = torch.tanh(params[:, 0])
            beta = params[:, 1]
            x = x + gamma * self.film_norm(x) + beta

        return x


class PartStyleEncoder(nn.Module):
    """Few-part style encoder: edge-aware patch selection + DeepSets-style aggregation."""

    def __init__(
        self,
        in_channels: int,
        style_dim: int = 256,
        patch_size: int = 32,
        patch_stride: int = 16,
        min_patches_per_style: int = 1,
        max_patches_per_style: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.style_dim = style_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.min_patches_per_style = max(1, int(min_patches_per_style))
        self.max_patches_per_style = max(self.min_patches_per_style, int(max_patches_per_style))

        self.part_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.part_fc = nn.Linear(128, style_dim)

        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _select_num_patches(self, num_candidates: int) -> int:
        min_k = min(self.min_patches_per_style, num_candidates)
        max_k = min(self.max_patches_per_style, num_candidates)
        if max_k <= 0:
            return 0
        if self.training and max_k > min_k:
            return int(torch.randint(min_k, max_k + 1, (1,)).item())
        return max_k

    def _extract_top_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*K, C, H, W) -> patches: (B*K, N, C, p, p)."""
        bk, c, h, w = x.shape
        p = min(self.patch_size, h, w)
        s = min(self.patch_stride, p)

        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)

        edge_unfold = F.unfold(edge, kernel_size=p, stride=s)  # (B*K, p*p, L)
        img_unfold = F.unfold(x, kernel_size=p, stride=s)      # (B*K, C*p*p, L)
        num_locs = edge_unfold.size(-1)
        topk = self._select_num_patches(num_locs)
        if topk <= 0:
            return x.new_zeros((bk, 1, c, p, p))

        patch_scores = edge_unfold.mean(dim=1)                 # (B*K, L)
        _, idx = torch.topk(patch_scores, k=topk, dim=1)       # (B*K, topk)
        gather_idx = idx.unsqueeze(1).expand(-1, c * p * p, -1)
        selected = torch.gather(img_unfold, dim=2, index=gather_idx)  # (B*K, C*p*p, topk)
        selected = selected.transpose(1, 2).contiguous().view(bk, topk, c, p, p)
        return selected

    def _encode_patch_batch(self, patch_batch: torch.Tensor) -> torch.Tensor:
        feat = self.part_cnn(patch_batch).flatten(1)  # (N, 128)
        feat = self.part_fc(feat)  # (N, D)
        return feat

    def _forward_from_style_image(self, style_img: torch.Tensor) -> torch.Tensor:
        """style_img: (B, C*K, H, W) -> style_vec: (B, D)."""
        b, ck, h, w = style_img.shape
        if ck % self.in_channels != 0:
            raise ValueError(
                f"style_img channels ({ck}) must be divisible by in_channels ({self.in_channels})"
            )

        k = ck // self.in_channels
        style_stack = style_img.view(b, k, self.in_channels, h, w).reshape(b * k, self.in_channels, h, w)
        patches = self._extract_top_patches(style_stack)  # (B*K, N, C, p, p)
        bk, n, c, p, _ = patches.shape
        patch_batch = patches.view(bk * n, c, p, p)
        feat = self._encode_patch_batch(patch_batch)  # (B*K*N, D)
        feat = feat.view(b, k * n, self.style_dim)         # set of part features

        # DeepSets-style aggregation: sum then L2 normalization.
        style_vec = feat.sum(dim=1)
        style_vec = F.normalize(style_vec, dim=-1)
        return style_vec

    def _forward_from_part_set(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """part_imgs: (B, P, C, H, W) -> style_vec: (B, D)."""
        if part_imgs.dim() != 5:
            raise ValueError(f"part_imgs must be 5D (B,P,C,H,W), got shape={tuple(part_imgs.shape)}")
        b, p, c, h, w = part_imgs.shape
        if c != self.in_channels:
            raise ValueError(
                f"part_imgs channel mismatch: got C={c}, expected in_channels={self.in_channels}"
            )
        patch_batch = part_imgs.view(b * p, c, h, w)
        feat = self._encode_patch_batch(patch_batch).view(b, p, self.style_dim)

        if part_mask is not None:
            if part_mask.dim() != 2 or part_mask.shape[0] != b or part_mask.shape[1] != p:
                raise ValueError(
                    f"part_mask must be shape (B,P)=({b},{p}), got {tuple(part_mask.shape)}"
                )
            mask = part_mask.to(dtype=feat.dtype, device=feat.device).unsqueeze(-1)
            feat = feat * mask

        style_vec = feat.sum(dim=1)
        style_vec = F.normalize(style_vec, dim=-1)
        return style_vec

    def forward(
        self,
        part_imgs: torch.Tensor,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_from_part_set(part_imgs=part_imgs, part_mask=part_mask)


# ------------------------- 编码器 ------------------------- #

class Encoder(nn.Module):
    """A simple Conv-BN-ReLU stack.  Each block downsamples with `stride=2`.  All
    intermediate feature maps are returned for skip connections.
    """

    def __init__(self, in_ch: int, base_ch: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = base_ch * (2 ** i)
            layers.append(ConvBNReLU(ch, out_ch, stride=2))
            ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats  # list len==num_layers, 分辨率从 H/2 到 H/2^{n}


# ------------------------- U-Net Blocks ------------------------- #

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_embed_dim: int, with_daca: bool, use_adaln: bool):
        super().__init__()
        self.with_daca = with_daca
        self.use_adaln = use_adaln
        self.conv1 = ConvBNReLU(in_ch, out_ch, stride=2)
        if with_daca:
            self.daca = DACA(in_channels=out_ch)
        if use_adaln:
            self.adaln = AdaLN(out_ch, t_embed_dim)

    def forward(self, x: torch.Tensor, content_feat: torch.Tensor, t: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.with_daca:
            x, _, _ = self.daca(x, content_feat, t)
        if self.use_adaln:
            x = self.adaln(x, t_emb)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_embed_dim: int,
                 use_fgsa: bool, use_adaln: bool, need_upsample: bool, use_attnx: bool = False):
        super().__init__()
        self.need_upsample = need_upsample
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.use_fgsa = use_fgsa
        self.use_adaln = use_adaln
        self.use_attnx = use_attnx
        if use_attnx:
            self.attnx = AttnXGatedFusion(out_ch, out_ch, t_embed_dim)
        if use_fgsa:
            self.fgsa = FGSA(out_ch)
        if use_adaln:
            self.adaln = AdaLN(out_ch, t_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        style_feat: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # concat first (assumes same spatial size)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        if self.use_attnx:
            x = self.attnx(x, style_feat, t_emb)
        if self.use_fgsa:
            x = self.fgsa(x, style_feat)
        if self.use_adaln:
            x = self.adaln(x, t_emb)
        # upsample for next scale if needed
        if self.need_upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return x


# ------------------------- Font Diffusion U-Net ------------------------- #

class FontDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        time_embed_dim: int = 256,
        style_k: int = 1,  # 参考风格字符数量
        daca_layers: Sequence[bool] | None = None,
        fgsa_layers: Sequence[bool] | None = None,
        attnx_enabled: bool = False,
        attnx_positions: Sequence[str] | None = None,
        use_global_style: bool = True,
        use_global_style_vector: bool = False,
        style_vector_only: bool = False,
        global_style_dim: int = 256,
        global_style_positions: Sequence[str] | None = None,
        global_style_use_cross_attn: bool = True,
        global_style_use_film: bool = True,
        use_part_style: bool = False,
        num_classes: int = 4096,
        class_embed_dim: int = 256,
        use_part_condition_cross_attn: bool = True,
        part_condition_positions: Sequence[str] | None = None,
        part_condition_use_cross_attn: bool = True,
        part_condition_use_film: bool = False,
        part_patch_size: int = 32,
        part_patch_stride: int = 16,
        part_min_patches_per_style: int = 1,
        part_max_patches_per_style: int = 4,
        part_style_dim: int = 256,
        part_fuse_scales: Sequence[int] | None = None,
        part_fuse_scale_gains: Sequence[float] | None = None,
        part_fuse_strength: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.style_k = style_k
        self.num_layers = num_layers
        self.use_global_style = bool(use_global_style)
        self.use_global_style_vector = bool(use_global_style_vector)
        self.style_vector_only = bool(style_vector_only)
        self.global_style_dim = int(global_style_dim)
        self.global_style_use_cross_attn = bool(global_style_use_cross_attn)
        self.global_style_use_film = bool(global_style_use_film)
        self.use_part_style = use_part_style
        self.num_classes = int(max(2, num_classes))
        self.class_embed_dim = int(class_embed_dim)
        self.use_part_condition_cross_attn = bool(use_part_condition_cross_attn)
        self.part_condition_use_cross_attn = bool(part_condition_use_cross_attn)
        self.part_condition_use_film = bool(part_condition_use_film)
        self.part_fuse_strength = float(part_fuse_strength)
        if global_style_positions is None:
            global_style_positions = ("bottleneck", "decoder_32", "decoder_64")
        self.global_style_positions = set(global_style_positions)
        if part_condition_positions is None:
            part_condition_positions = ("bottleneck", "decoder_32", "decoder_64")
        self.part_condition_positions = set(part_condition_positions)

        if part_fuse_scales is None:
            # low-resolution style fusion is usually the most stable for Chinese glyphs
            part_fuse_scales = (max(0, num_layers - 2), max(0, num_layers - 1))
        fuse_scales_ordered: List[int] = []
        seen_scales = set()
        for raw_scale in part_fuse_scales:
            scale = int(raw_scale)
            if scale < 0 or scale >= num_layers:
                raise ValueError(f"part_fuse_scales contains out-of-range index {scale}, valid=[0,{num_layers - 1}]")
            if scale in seen_scales:
                continue
            seen_scales.add(scale)
            fuse_scales_ordered.append(scale)
        self.part_fuse_scales = set(fuse_scales_ordered)

        if part_fuse_scale_gains is None:
            self.part_fuse_scale_gains = {scale: 1.0 for scale in fuse_scales_ordered}
        else:
            gains = [float(x) for x in part_fuse_scale_gains]
            if len(gains) != len(fuse_scales_ordered):
                raise ValueError(
                    "part_fuse_scale_gains length must match part_fuse_scales length, "
                    f"got gains={len(gains)} scales={len(fuse_scales_ordered)}"
                )
            self.part_fuse_scale_gains = {
                scale: gain for scale, gain in zip(fuse_scales_ordered, gains)
            }

        if daca_layers is None:
            daca_layers = [True] * (num_layers - 1) + [False]
        if len(daca_layers) != num_layers:
            raise ValueError(f"daca_layers length must be {num_layers}, got {len(daca_layers)}")
        self.daca_layers = list(daca_layers)

        if fgsa_layers is None:
            fgsa_layers = [True] * (num_layers - 1) + [False]
        if len(fgsa_layers) != num_layers:
            raise ValueError(f"fgsa_layers length must be {num_layers}, got {len(fgsa_layers)}")
        self.fgsa_layers = list(fgsa_layers)

        if attnx_enabled:
            if attnx_positions is None:
                attnx_positions = ("bottleneck_16", "up0_16to32")
            self.attnx_positions = set(attnx_positions)
        else:
            self.attnx_positions = set()
        self.use_attnx_bottleneck = "bottleneck_16" in self.attnx_positions
        self.use_attnx_up0 = "up0_16to32" in self.attnx_positions

        # timestep embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Encoders for content and style (style 输入通道 = in_channels * style_k)
        self.content_encoder = Encoder(in_channels, base_channels, num_layers)
        self.style_encoder = Encoder(in_channels * style_k, base_channels, num_layers) if self.use_global_style else None
        self.global_style_encoder = (
            GlyphStyleEncoder(in_channels=in_channels, embedding_dim=self.global_style_dim)
            if self.use_global_style_vector
            else None
        )
        if self.use_part_style:
            self.part_style_encoder = PartStyleEncoder(
                in_channels=in_channels,
                style_dim=part_style_dim,
                patch_size=part_patch_size,
                patch_stride=part_patch_stride,
                min_patches_per_style=part_min_patches_per_style,
                max_patches_per_style=part_max_patches_per_style,
            )
            self.class_embed = nn.Embedding(self.num_classes, self.class_embed_dim)
            self.part_to_class_cond = nn.Linear(part_style_dim, self.class_embed_dim)

        # U-Net Down & Up
        down_blocks = []
        up_blocks = []
        ch = in_channels  # 直接从原始输入开始
        down_channels: List[int] = []  # record feature dims for skip connections
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)
            # 最后一层 down block 不使用 DACA/AdaLN
            is_last = i == num_layers - 1
            down_blocks.append(
                DownBlock(
                    ch,
                    out_ch,
                    time_embed_dim,
                    with_daca=self.daca_layers[i],
                    use_adaln=not is_last,
                )
            )
            down_channels.append(out_ch)
            ch = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)

        if self.use_part_style:
            self.part_to_style = nn.ModuleList([nn.Linear(part_style_dim, c) for c in down_channels])
            self.part_gate = nn.ModuleList([nn.Linear(time_embed_dim, 1) for _ in down_channels])

        # 编码器底部 ResBlock，无 AdaLN
        self.bottom_resblock = ResBlock(ch, time_embed_dim, use_adaln=False)

        # decoder first ResNet layer with FGSA & AdaLN
        self.decoder_resblock = DecoderResBlock(ch, time_embed_dim)
        if self.use_attnx_bottleneck:
            self.attnx_bottleneck = AttnXGatedFusion(ch, ch, time_embed_dim)

        # build up blocks (reverse order)
        rev_channels = list(reversed(down_channels))
        in_ch = ch  # after decoder_resblock channels unchanged
        for i, skip_ch in enumerate(rev_channels):
            out_ch = skip_ch
            is_last = i == len(rev_channels) - 1
            up_blocks.append(
                UpBlock(
                    in_ch + skip_ch,
                    out_ch,
                    time_embed_dim,
                    use_fgsa=self.fgsa_layers[i],
                    use_adaln=not is_last,
                    need_upsample=True,
                    use_attnx=(self.use_attnx_up0 and i == 0),
                )
            )  # 最后一层也上采样
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)

        self.global_style_fusions = nn.ModuleDict()
        if self.use_global_style_vector:
            if "bottleneck" in self.global_style_positions:
                self.global_style_fusions["bottleneck"] = GlobalStyleFusion(
                    channels=ch,
                    style_dim=self.global_style_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.global_style_use_cross_attn,
                    use_film=self.global_style_use_film,
                )
            if len(rev_channels) >= 1 and "decoder_32" in self.global_style_positions:
                self.global_style_fusions["decoder_32"] = GlobalStyleFusion(
                    channels=rev_channels[0],
                    style_dim=self.global_style_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.global_style_use_cross_attn,
                    use_film=self.global_style_use_film,
                )
            if len(rev_channels) >= 2 and "decoder_64" in self.global_style_positions:
                self.global_style_fusions["decoder_64"] = GlobalStyleFusion(
                    channels=rev_channels[1],
                    style_dim=self.global_style_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.global_style_use_cross_attn,
                    use_film=self.global_style_use_film,
                )

        self.part_condition_fusions = nn.ModuleDict()
        if self.use_part_style and self.use_part_condition_cross_attn:
            if "bottleneck" in self.part_condition_positions:
                self.part_condition_fusions["bottleneck"] = GlobalStyleFusion(
                    channels=ch,
                    style_dim=self.class_embed_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.part_condition_use_cross_attn,
                    use_film=self.part_condition_use_film,
                )
            if len(rev_channels) >= 1 and "decoder_32" in self.part_condition_positions:
                self.part_condition_fusions["decoder_32"] = GlobalStyleFusion(
                    channels=rev_channels[0],
                    style_dim=self.class_embed_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.part_condition_use_cross_attn,
                    use_film=self.part_condition_use_film,
                )
            if len(rev_channels) >= 2 and "decoder_64" in self.part_condition_positions:
                self.part_condition_fusions["decoder_64"] = GlobalStyleFusion(
                    channels=rev_channels[1],
                    style_dim=self.class_embed_dim,
                    t_embed_dim=time_embed_dim,
                    use_cross_attn=self.part_condition_use_cross_attn,
                    use_film=self.part_condition_use_film,
                )

        # final output conv
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def load_part_style_pretrained(self, ckpt_path: str, strict: bool = True) -> None:
        if not self.use_part_style:
            raise RuntimeError("use_part_style=False, cannot load part-style pretrained weights")

        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch versions may not support weights_only
            obj = torch.load(ckpt_path, map_location="cpu")
        state = obj.get("part_style_encoder", obj) if isinstance(obj, dict) else obj

        if isinstance(state, dict) and "part_cnn" in state and "part_fc" in state:
            self.part_style_encoder.part_cnn.load_state_dict(state["part_cnn"], strict=strict)
            self.part_style_encoder.part_fc.load_state_dict(state["part_fc"], strict=strict)
            return

        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

        # fallback: load from a full model state dict that contains prefixed keys
        cnn_prefix = "part_style_encoder.part_cnn."
        fc_prefix = "part_style_encoder.part_fc."
        cnn_state = {k[len(cnn_prefix):]: v for k, v in state.items() if k.startswith(cnn_prefix)}
        fc_state = {k[len(fc_prefix):]: v for k, v in state.items() if k.startswith(fc_prefix)}
        if not cnn_state or not fc_state:
            raise RuntimeError(
                "Could not find part-style weights in checkpoint. "
                "Expected keys 'part_style_encoder.part_cnn.*' and 'part_style_encoder.part_fc.*'."
            )
        self.part_style_encoder.part_cnn.load_state_dict(cnn_state, strict=strict)
        self.part_style_encoder.part_fc.load_state_dict(fc_state, strict=strict)

    def load_global_style_pretrained(self, ckpt_path: str, strict: bool = True) -> None:
        if not self.use_global_style_vector:
            raise RuntimeError("use_global_style_vector=False, cannot load global-style encoder weights")
        if self.global_style_encoder is None:
            raise RuntimeError("global_style_encoder is not initialized")

        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt_path, map_location="cpu")
        state = obj if isinstance(obj, dict) else {}
        if isinstance(state, dict) and "e_s" in state:
            state = state["e_s"]
        if isinstance(state, dict) and "global_style_encoder" in state:
            state = state["global_style_encoder"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format for global style encoder: {type(state)}")

        # accept either plain backbone keys or prefixed model keys
        pref = "global_style_encoder."
        if any(k.startswith(pref) for k in state.keys()):
            state = {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}

        self.global_style_encoder.load_state_dict(state, strict=strict)

    # -------------------- forward -------------------- #

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # 1) 生成时间步嵌入
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        return t_emb

    def encode_conditions(
        self,
        content_img: torch.Tensor,
        style_img: Optional[torch.Tensor],
        part_imgs: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor] | List[torch.Tensor]]:
        content_feats = self.content_encoder(content_img)
        if self.use_global_style:
            if style_img is None:
                raise ValueError("style_img is required when use_global_style=True")
            if self.style_encoder is None:
                raise RuntimeError("style_encoder is not initialized while use_global_style=True")
            style_feats = self.style_encoder(style_img)
        else:
            style_feats = [torch.zeros_like(feat) for feat in content_feats]

        style_tokens: torch.Tensor | None = None
        style_global: torch.Tensor | None = None
        if self.use_global_style_vector:
            if style_img is None:
                raise ValueError("style_img is required when use_global_style_vector=True")
            if self.global_style_encoder is None:
                raise RuntimeError("global_style_encoder is not initialized while use_global_style_vector=True")
            style_tokens, style_global = encode_style_stack(
                self.global_style_encoder,
                style_img=style_img,
                in_channels=self.in_channels,
            )

        if self.style_vector_only:
            style_feats = [torch.zeros_like(feat) for feat in content_feats]

        part_style_vec: torch.Tensor | None = None
        part_condition_tokens: torch.Tensor | None = None
        if self.use_part_style:
            if part_imgs is None:
                raise ValueError("part_imgs is required when use_part_style=True")
            part_style_vec = self.part_style_encoder(part_imgs=part_imgs, part_mask=part_mask)
            if self.use_part_condition_cross_attn:
                if class_ids is None:
                    raise ValueError("class_ids is required when use_part_condition_cross_attn=True")
                if class_ids.dim() != 1:
                    raise ValueError(f"class_ids must be 1D tensor (B,), got {tuple(class_ids.shape)}")
                class_tok = self.class_embed(class_ids.clamp(0, self.num_classes - 1))
                part_tok = self.part_to_class_cond(part_style_vec)
                part_condition_tokens = torch.stack([class_tok, part_tok], dim=1)  # (B,2,D)

        return {
            "content_feats": content_feats,
            "style_feats": style_feats,
            "part_style_vec": part_style_vec,
            "style_tokens": style_tokens,
            "style_global": style_global,
            "part_condition_tokens": part_condition_tokens,
        }

    def _fuse_part_style_feats(
        self,
        style_feats: Sequence[torch.Tensor],
        part_style_vec: torch.Tensor | None,
        t_emb: torch.Tensor,
    ) -> List[torch.Tensor]:
        if not self.use_part_style or part_style_vec is None:
            return list(style_feats)

        fused: List[torch.Tensor] = []
        for i, feat in enumerate(style_feats):
            if i not in self.part_fuse_scales:
                fused.append(feat)
                continue
            part_bias = self.part_to_style[i](part_style_vec).unsqueeze(-1).unsqueeze(-1)
            gate = torch.sigmoid(self.part_gate[i](t_emb)).unsqueeze(-1).unsqueeze(-1)
            scale_gain = float(self.part_fuse_scale_gains.get(i, 1.0))
            fused.append(feat + self.part_fuse_strength * scale_gain * gate * part_bias)
        return fused

    def _apply_global_style_fusion(
        self,
        key: str,
        x: torch.Tensor,
        style_tokens: torch.Tensor | None,
        style_global: torch.Tensor | None,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_global_style_vector:
            return x
        if key not in self.global_style_fusions:
            return x
        if style_tokens is None or style_global is None:
            return x
        return self.global_style_fusions[key](x, style_tokens=style_tokens, style_global=style_global, t_emb=t_emb)

    def _apply_part_condition_fusion(
        self,
        key: str,
        x: torch.Tensor,
        part_condition_tokens: torch.Tensor | None,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_part_style or not self.use_part_condition_cross_attn:
            return x
        if key not in self.part_condition_fusions:
            return x
        if part_condition_tokens is None:
            return x
        part_global = part_condition_tokens.mean(dim=1)
        return self.part_condition_fusions[key](
            x,
            style_tokens=part_condition_tokens,
            style_global=part_global,
            t_emb=t_emb,
        )

    def forward_with_feats(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_feats: Sequence[torch.Tensor],
        style_feats: Sequence[torch.Tensor],
        part_style_vec: torch.Tensor | None = None,
        style_tokens: torch.Tensor | None = None,
        style_global: torch.Tensor | None = None,
        part_condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(content_feats) < self.num_layers:
            raise ValueError(f"content_feats length must be >= {self.num_layers}, got {len(content_feats)}")
        if len(style_feats) < self.num_layers:
            raise ValueError(f"style_feats length must be >= {self.num_layers}, got {len(style_feats)}")

        t_emb = self._time_embedding(t)
        fused_style_feats = self._fuse_part_style_feats(style_feats, part_style_vec, t_emb)

        # U-Net encoder path (含 DACA)
        skips: List[torch.Tensor] = []
        h = x_t
        for i, down in enumerate(self.down_blocks):
            h = down(h, content_feats[i], t, t_emb)
            skips.append(h)

        # bottom without AdaLN/DACA
        h = self.bottom_resblock(h, t_emb)

        # decoder first ResNet layer with FGSA & AdaLN
        style_feat_bottom = fused_style_feats[-1]
        h = self.decoder_resblock(h, style_feat_bottom, t_emb)
        if self.use_attnx_bottleneck:
            h = self.attnx_bottleneck(h, style_feat_bottom, t_emb)
        h = self._apply_global_style_fusion(
            key="bottleneck",
            x=h,
            style_tokens=style_tokens,
            style_global=style_global,
            t_emb=t_emb,
        )
        h = self._apply_part_condition_fusion(
            key="bottleneck",
            x=h,
            part_condition_tokens=part_condition_tokens,
            t_emb=t_emb,
        )

        # decoder path (含 FGSA)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            style_feat = fused_style_feats[-(i + 1)]
            h = up(h, skip, style_feat, t_emb)
            if i == 0:
                h = self._apply_global_style_fusion(
                    key="decoder_32",
                    x=h,
                    style_tokens=style_tokens,
                    style_global=style_global,
                    t_emb=t_emb,
                )
                h = self._apply_part_condition_fusion(
                    key="decoder_32",
                    x=h,
                    part_condition_tokens=part_condition_tokens,
                    t_emb=t_emb,
                )
            elif i == 1:
                h = self._apply_global_style_fusion(
                    key="decoder_64",
                    x=h,
                    style_tokens=style_tokens,
                    style_global=style_global,
                    t_emb=t_emb,
                )
                h = self._apply_part_condition_fusion(
                    key="decoder_64",
                    x=h,
                    part_condition_tokens=part_condition_tokens,
                    t_emb=t_emb,
                )
        # output
        return self.out_conv(h)

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
        """推断一个时间步的 \hat{x}_0。"""
        cond = self.encode_conditions(
            content_img,
            style_img,
            part_imgs=part_imgs,
            part_mask=part_mask,
            class_ids=class_ids,
        )
        return self.forward_with_feats(
            x_t,
            t,
            cond["content_feats"],  # type: ignore[arg-type]
            cond["style_feats"],    # type: ignore[arg-type]
            part_style_vec=cond["part_style_vec"],          # type: ignore[arg-type]
            style_tokens=cond["style_tokens"],              # type: ignore[arg-type]
            style_global=cond["style_global"],              # type: ignore[arg-type]
            part_condition_tokens=cond["part_condition_tokens"],  # type: ignore[arg-type]
        )


# ------------------------- quick sanity test ------------------------- #

if __name__ == "__main__":
    B, C, H, W = 2, 3, 256, 256
    K = 3  # 风格字符数量
    net = FontDiffusionUNet(
        in_channels=C,
        style_k=K,
        daca_layers=[False, True, True, False],
        fgsa_layers=[True, True, True, False],
        attnx_enabled=True,
        use_part_style=True,
    )
    x_t = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    content = torch.randn(B, C, H, W)
    style = torch.randn(B, C*K, H, W)  # 3 张风格图像拼通道
    part_imgs = torch.randn(B, 4, C, 64, 64)
    part_mask = torch.ones(B, 4)
    class_ids = torch.randint(0, 2000, (B,))
    cond = net.encode_conditions(
        content,
        style,
        part_imgs=part_imgs,
        part_mask=part_mask,
        class_ids=class_ids,
    )
    out = net.forward_with_feats(
        x_t,
        t,
        cond["content_feats"],  # type: ignore[arg-type]
        cond["style_feats"],    # type: ignore[arg-type]
        part_style_vec=cond["part_style_vec"],          # type: ignore[arg-type]
        style_tokens=cond["style_tokens"],              # type: ignore[arg-type]
        style_global=cond["style_global"],              # type: ignore[arg-type]
        part_condition_tokens=cond["part_condition_tokens"],  # type: ignore[arg-type]
    )
    print("Output:", out.shape)  # Expected (B, C, H, W)
