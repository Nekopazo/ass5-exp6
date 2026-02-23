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
    style_img,   # reference style images (B,C*k,H,W)
)
# returns  \hat x_0   (B,C,H,W)
```
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.daca import DACA
from models.fgsa import FGSA


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

    def forward(self, style_img: torch.Tensor) -> torch.Tensor:
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

        feat = self.part_cnn(patch_batch).flatten(1)      # (B*K*N, 128)
        feat = self.part_fc(feat)                          # (B*K*N, D)
        feat = feat.view(b, k * n, self.style_dim)         # set of part features

        # DeepSets-style aggregation: sum then L2 normalization.
        style_vec = feat.sum(dim=1)
        style_vec = F.normalize(style_vec, dim=-1)
        return style_vec


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
        use_part_style: bool = False,
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
        self.use_part_style = use_part_style
        self.part_fuse_strength = float(part_fuse_strength)

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
        self.style_encoder = Encoder(in_channels * style_k, base_channels, num_layers)
        if self.use_part_style:
            self.part_style_encoder = PartStyleEncoder(
                in_channels=in_channels,
                style_dim=part_style_dim,
                patch_size=part_patch_size,
                patch_stride=part_patch_stride,
                min_patches_per_style=part_min_patches_per_style,
                max_patches_per_style=part_max_patches_per_style,
            )

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

    # -------------------- forward -------------------- #

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # 1) 生成时间步嵌入
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        return t_emb

    def encode_conditions(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        return_part: bool = False,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]] | tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor | None]:
        content_feats = self.content_encoder(content_img)
        style_feats = self.style_encoder(style_img)
        part_style_vec: torch.Tensor | None = None
        if self.use_part_style:
            part_style_vec = self.part_style_encoder(style_img)
        if return_part:
            return content_feats, style_feats, part_style_vec
        return content_feats, style_feats

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

    def forward_with_feats(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_feats: Sequence[torch.Tensor],
        style_feats: Sequence[torch.Tensor],
        part_style_vec: torch.Tensor | None = None,
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

        # decoder path (含 FGSA)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            style_feat = fused_style_feats[-(i + 1)]
            h = up(h, skip, style_feat, t_emb)
        # output
        return self.out_conv(h)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
    ) -> torch.Tensor:
        """推断一个时间步的 \hat{x}_0。"""
        content_feats, style_feats, part_style_vec = self.encode_conditions(
            content_img,
            style_img,
            return_part=True,
        )
        return self.forward_with_feats(x_t, t, content_feats, style_feats, part_style_vec=part_style_vec)


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
    content_feats, style_feats, part_style_vec = net.encode_conditions(content, style, return_part=True)
    out = net.forward_with_feats(x_t, t, content_feats, style_feats, part_style_vec=part_style_vec)
    print("Output:", out.shape)  # Expected (B, C, H, W)
