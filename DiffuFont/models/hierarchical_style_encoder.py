#!/usr/bin/env python3
"""Three-token style encoder with semantic low/mid/high reconstruction targets."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class ResNet18FeaturePyramid(nn.Module):
    """ResNet18 stem truncated at layer3, exposing 32/16/8 feature maps."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        net = tv_models.resnet18(weights=None)
        if int(in_channels) != 3:
            net.conv1 = nn.Conv2d(
                int(in_channels),
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_high = self.layer1(x)  # 32x32, fine detail
        feat_mid = self.layer2(feat_high)  # 16x16
        feat_low = self.layer3(feat_mid)  # 8x8, coarse/global
        return feat_low, feat_mid, feat_high


class CrossAttentionWithBias(nn.Module):
    """Cross-attention with optional padding mask and averaged weights output."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim <= 0 or self.num_heads <= 0 or (self.embed_dim % self.num_heads) != 0:
            raise ValueError(f"invalid attention config dim={embed_dim} heads={num_heads}")
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, k_len):
                raise ValueError(
                    f"key_padding_mask must have shape {(bsz, k_len)}, got {tuple(key_padding_mask.shape)}"
                )
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        out = self.out_proj(out)
        weights = attn.mean(dim=1) if need_weights else None
        return out, weights


class QueryPoolingBlock(nn.Module):
    """Pool a sequence of features into a small learned query set."""

    def __init__(self, embed_dim: int, num_heads: int, num_queries: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")
        self.query = nn.Parameter(torch.randn(self.num_queries, self.embed_dim) * 0.02)
        self.attn = CrossAttentionWithBias(embed_dim=self.embed_dim, num_heads=int(num_heads))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be 3D, got {tuple(patch_tokens.shape)}")
        bsz = int(patch_tokens.size(0))
        q = self.query.unsqueeze(0).expand(bsz, -1, -1)
        out, weights = self.attn(
            query=q,
            key=patch_tokens,
            value=patch_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        out = self.norm(out + q)
        out = self.out_norm(out + self.ffn(out))
        return out, weights


class HierarchicalStyleEncoderMixin:
    """Reusable low/mid/high token encoder for pretrain and generation."""

    style_token_names = ("t_low", "t_mid", "t_high")

    def _init_hierarchical_style_encoder(
        self,
        *,
        in_channels: int,
        style_token_dim: int,
        style_token_count: int,
        local_token_count: int = 3,
        style_memory_mid_count: int = 4,
        style_memory_up16_count: int = 6,
        style_memory_up32_count: int = 6,
        style_memory_mid_pool_hw: int = 8,
        style_memory_up16_pool_hw: int = 16,
        style_memory_up32_pool_hw: int = 16,
    ) -> None:
        _ = local_token_count
        self.in_channels = int(in_channels)
        self.style_token_dim = int(style_token_dim)
        self.style_token_count = int(style_token_count)
        self.style_memory_mid_count = int(style_memory_mid_count)
        self.style_memory_up16_count = int(style_memory_up16_count)
        self.style_memory_up32_count = int(style_memory_up32_count)
        self.style_memory_mid_pool_hw = int(style_memory_mid_pool_hw)
        self.style_memory_up16_pool_hw = int(style_memory_up16_pool_hw)
        self.style_memory_up32_pool_hw = int(style_memory_up32_pool_hw)
        if self.style_token_dim <= 0:
            raise ValueError("style_token_dim must be > 0")
        if self.style_token_count != 3:
            raise ValueError(f"style_token_count must be 3 for low/mid/high routing, got {self.style_token_count}")
        if self.style_memory_mid_count <= 0 or self.style_memory_up16_count <= 0 or self.style_memory_up32_count <= 0:
            raise ValueError("style memory token counts must be > 0")
        if self.style_memory_mid_pool_hw <= 0 or self.style_memory_up16_pool_hw <= 0 or self.style_memory_up32_pool_hw <= 0:
            raise ValueError("style memory pool hw must be > 0")

        self.style_backbone = ResNet18FeaturePyramid(in_channels=self.in_channels)

        self.t_low_feat_to_token = nn.Identity() if self.style_token_dim == 256 else nn.Linear(256, self.style_token_dim)
        self.t_mid_feat_to_token = nn.Linear(128, self.style_token_dim)
        self.t_high_feat_to_token = nn.Linear(64, self.style_token_dim)

        heads = 8 if (self.style_token_dim % 8 == 0) else (4 if (self.style_token_dim % 4 == 0) else 1)

        self.t_low_query = nn.Parameter(torch.randn(1, self.style_token_dim) * 0.02)
        self.t_mid_query = nn.Parameter(torch.randn(1, self.style_token_dim) * 0.02)
        self.t_high_query = nn.Parameter(torch.randn(1, self.style_token_dim) * 0.02)

        self.t_low_attn = CrossAttentionWithBias(embed_dim=self.style_token_dim, num_heads=heads)
        self.t_mid_attn = CrossAttentionWithBias(embed_dim=self.style_token_dim, num_heads=heads)
        self.t_high_attn = CrossAttentionWithBias(embed_dim=self.style_token_dim, num_heads=heads)

        self.t_low_norm = nn.LayerNorm(self.style_token_dim)
        self.t_mid_norm = nn.LayerNorm(self.style_token_dim)
        self.t_high_norm = nn.LayerNorm(self.style_token_dim)

        self.t_low_ffn = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_dim),
        )
        self.t_mid_ffn = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_dim),
        )
        self.t_high_ffn = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_dim),
        )

        self.t_low_out_norm = nn.LayerNorm(self.style_token_dim)
        self.t_mid_out_norm = nn.LayerNorm(self.style_token_dim)
        self.t_high_out_norm = nn.LayerNorm(self.style_token_dim)

        self.t_low_proxy_head = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 2, 256),
        )
        self.t_mid_proxy_head = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 2, 128),
        )
        self.t_high_proxy_head = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 2, 64),
        )
        self.memory_mid_feat_to_token = nn.Identity() if self.style_token_dim == 256 else nn.Linear(256, self.style_token_dim)
        self.memory_up16_feat_to_token = nn.Linear(128, self.style_token_dim)
        self.memory_up32_feat_to_token = nn.Linear(64, self.style_token_dim)
        self.memory_mid_pool = QueryPoolingBlock(
            embed_dim=self.style_token_dim,
            num_heads=heads,
            num_queries=self.style_memory_mid_count,
        )
        self.memory_up16_pool = QueryPoolingBlock(
            embed_dim=self.style_token_dim,
            num_heads=heads,
            num_queries=self.style_memory_up16_count,
        )
        self.memory_up32_pool = QueryPoolingBlock(
            embed_dim=self.style_token_dim,
            num_heads=heads,
            num_queries=self.style_memory_up32_count,
        )

    def iter_style_backbone_low_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.style_backbone.conv1,
            self.style_backbone.bn1,
            self.style_backbone.layer1,
        )

    def iter_style_backbone_high_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.style_backbone.layer2,
            self.style_backbone.layer3,
        )

    def iter_style_backbone_low_parameters(self):
        for module in self.iter_style_backbone_low_modules():
            yield from module.parameters()

    def iter_style_backbone_high_parameters(self):
        for module in self.iter_style_backbone_high_modules():
            yield from module.parameters()

    def _masked_ref_average(self, feat: torch.Tensor, style_ref_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, r, c, h, w = feat.shape
        if style_ref_mask is None:
            return feat.mean(dim=(1, 3, 4))
        weights = style_ref_mask.to(device=feat.device, dtype=feat.dtype).clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (feat * weights.view(b, r, 1, 1, 1)).sum(dim=1).mean(dim=(2, 3))

    def _build_key_padding_mask(
        self,
        feat: torch.Tensor,
        *,
        b: int,
        r: int,
        ref_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        _, _, h, w = feat.shape
        if ref_mask is None:
            return None
        valid = ref_mask.to(device=feat.device, dtype=torch.float32) > 0
        return ~valid.unsqueeze(-1).expand(b, r, h * w).reshape(b, r * h * w)

    def _flatten_projected_tokens(
        self,
        feat: torch.Tensor,
        *,
        b: int,
        r: int,
        projector: nn.Module,
    ) -> torch.Tensor:
        _, c, h, w = feat.shape
        patch = feat.view(b * r, c, h * w).transpose(1, 2).contiguous()
        patch = projector(patch)
        return patch.view(b, r * h * w, self.style_token_dim)

    def _aggregate_scale_token(
        self,
        feat: torch.Tensor,
        *,
        b: int,
        r: int,
        ref_mask: Optional[torch.Tensor],
        projector: nn.Module,
        query: torch.Tensor,
        attn: CrossAttentionWithBias,
        norm: nn.Module,
        ffn: nn.Module,
        out_norm: nn.Module,
        return_attention: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, _, h, w = feat.shape
        patch = self._flatten_projected_tokens(
            feat,
            b=b,
            r=r,
            projector=projector,
        )
        key_padding_mask = self._build_key_padding_mask(feat, b=b, r=r, ref_mask=ref_mask)

        q = query.unsqueeze(0).expand(b, -1, -1)
        out, weights = attn(
            query=q,
            key=patch,
            value=patch,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )
        tok = norm(out + q)
        tok = out_norm(tok + ffn(tok)).squeeze(1)

        if weights is None:
            return tok, None
        attn_map = weights.view(b, 1, r, h, w).squeeze(1)
        return tok, attn_map

    def _aggregate_style_memory(
        self,
        feat: torch.Tensor,
        *,
        b: int,
        r: int,
        ref_mask: Optional[torch.Tensor],
        projector: nn.Module,
        pooler: QueryPoolingBlock,
    ) -> torch.Tensor:
        patch = self._flatten_projected_tokens(
            feat,
            b=b,
            r=r,
            projector=projector,
        )
        key_padding_mask = self._build_key_padding_mask(feat, b=b, r=r, ref_mask=ref_mask)
        memory, _ = pooler(
            patch,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return memory

    @staticmethod
    def _pool_to_token_grid(feat: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if feat.shape[-2:] == (target_h, target_w):
            return feat
        return F.adaptive_avg_pool2d(feat, output_size=(target_h, target_w))

    def _encode_style_impl(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        return_token_attention: bool = False,
        return_proxy: bool = False,
    ) -> dict[str, torch.Tensor]:
        style_img, style_ref_mask = self._normalize_style_inputs(style_img, style_ref_mask)
        b, r, c, h, w = style_img.shape
        x = style_img.view(b * r, c, h, w)

        feat_low_raw, feat_mid_raw, feat_high_raw = self.style_backbone(x)
        feat_low = feat_low_raw.view(b, r, 256, feat_low_raw.shape[-2], feat_low_raw.shape[-1])
        feat_mid = feat_mid_raw.view(b, r, 128, feat_mid_raw.shape[-2], feat_mid_raw.shape[-1])
        feat_high = feat_high_raw.view(b, r, 64, feat_high_raw.shape[-2], feat_high_raw.shape[-1])
        token_grid_hw = feat_low_raw.shape[-2:]
        feat_mid_tok_raw = self._pool_to_token_grid(feat_mid_raw, token_grid_hw)
        feat_high_tok_raw = self._pool_to_token_grid(feat_high_raw, token_grid_hw)
        feat_low_mem_raw = self._pool_to_token_grid(
            feat_low_raw,
            (self.style_memory_mid_pool_hw, self.style_memory_mid_pool_hw),
        )
        feat_mid_mem_raw = self._pool_to_token_grid(
            feat_mid_raw,
            (self.style_memory_up16_pool_hw, self.style_memory_up16_pool_hw),
        )
        feat_high_mem_raw = self._pool_to_token_grid(
            feat_high_raw,
            (self.style_memory_up32_pool_hw, self.style_memory_up32_pool_hw),
        )

        t_low, attn_low = self._aggregate_scale_token(
            feat_low_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.t_low_feat_to_token,
            query=self.t_low_query,
            attn=self.t_low_attn,
            norm=self.t_low_norm,
            ffn=self.t_low_ffn,
            out_norm=self.t_low_out_norm,
            return_attention=return_token_attention,
        )
        t_mid, attn_mid = self._aggregate_scale_token(
            feat_mid_tok_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.t_mid_feat_to_token,
            query=self.t_mid_query,
            attn=self.t_mid_attn,
            norm=self.t_mid_norm,
            ffn=self.t_mid_ffn,
            out_norm=self.t_mid_out_norm,
            return_attention=return_token_attention,
        )
        t_high, attn_high = self._aggregate_scale_token(
            feat_high_tok_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.t_high_feat_to_token,
            query=self.t_high_query,
            attn=self.t_high_attn,
            norm=self.t_high_norm,
            ffn=self.t_high_ffn,
            out_norm=self.t_high_out_norm,
            return_attention=return_token_attention,
        )

        tokens = torch.stack([t_low, t_mid, t_high], dim=1)
        memory_mid_8 = self._aggregate_style_memory(
            feat_low_mem_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.memory_mid_feat_to_token,
            pooler=self.memory_mid_pool,
        )
        memory_up_16 = self._aggregate_style_memory(
            feat_mid_mem_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.memory_up16_feat_to_token,
            pooler=self.memory_up16_pool,
        )
        memory_up_32 = self._aggregate_style_memory(
            feat_high_mem_raw,
            b=b,
            r=r,
            ref_mask=style_ref_mask,
            projector=self.memory_up32_feat_to_token,
            pooler=self.memory_up32_pool,
        )

        out: dict[str, torch.Tensor] = {
            "tokens": tokens,
            "t_low": t_low,
            "t_mid": t_mid,
            "t_high": t_high,
            "memory_mid_8": memory_mid_8,
            "memory_up_16": memory_up_16,
            "memory_up_32": memory_up_32,
        }

        if return_token_attention:
            maps = []
            for attn_map in (attn_low, attn_mid, attn_high):
                if attn_map is None:
                    raise RuntimeError("requested token attention but attention maps are missing")
                up = F.interpolate(
                    attn_map.view(b * r, 1, int(attn_map.size(-2)), int(attn_map.size(-1))),
                    size=(32, 32),
                    mode="bilinear",
                    align_corners=False,
                ).view(b, r, 32, 32)
                maps.append(up)
            out["token_attn"] = torch.stack(maps, dim=1)  # (B,3,R,32,32)

        if return_proxy:
            # Supervise all three routed bands through the memory pathways that
            # actually feed the injected style contexts.
            out["pred_low"] = self.t_low_proxy_head(memory_mid_8.mean(dim=1))
            out["pred_mid"] = self.t_mid_proxy_head(memory_up_16.mean(dim=1))
            out["pred_high"] = self.t_high_proxy_head(memory_up_32.mean(dim=1))
            out["target_low"] = self._masked_ref_average(feat_low.detach(), style_ref_mask)
            out["target_mid"] = self._masked_ref_average(feat_mid.detach(), style_ref_mask)
            out["target_high"] = self._masked_ref_average(feat_high.detach(), style_ref_mask)

        return out

    def encode_style_tokens(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=False,
            return_proxy=False,
        )["tokens"]

    def encode_style_tokens_with_attention(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=True,
            return_proxy=False,
        )
        return out["tokens"], out["token_attn"]

    def encode_style_tokens_with_proxy(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=False,
            return_proxy=True,
        )
        proxy = {
            "pred_low": out["pred_low"],
            "pred_mid": out["pred_mid"],
            "pred_high": out["pred_high"],
            "target_low": out["target_low"],
            "target_mid": out["target_mid"],
            "target_high": out["target_high"],
        }
        return out["tokens"], proxy

    def encode_style_tokens_full(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        out = self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=True,
            return_proxy=True,
        )
        proxy = {
            "pred_low": out["pred_low"],
            "pred_mid": out["pred_mid"],
            "pred_high": out["pred_high"],
            "target_low": out["target_low"],
            "target_mid": out["target_mid"],
            "target_high": out["target_high"],
        }
        return out["tokens"], proxy, out["token_attn"]

    def encode_style_pack(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=False,
            return_proxy=False,
        )

    def encode_style_pack_full(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        out = self._encode_style_impl(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            return_token_attention=True,
            return_proxy=True,
        )
        proxy = {
            "pred_low": out["pred_low"],
            "pred_mid": out["pred_mid"],
            "pred_high": out["pred_high"],
            "target_low": out["target_low"],
            "target_mid": out["target_mid"],
            "target_high": out["target_high"],
        }
        pack = {
            "tokens": out["tokens"],
            "t_low": out["t_low"],
            "t_mid": out["t_mid"],
            "t_high": out["t_high"],
            "memory_mid_8": out["memory_mid_8"],
            "memory_up_16": out["memory_up_16"],
            "memory_up_32": out["memory_up_32"],
        }
        return pack, proxy, out["token_attn"]

    def encode_style_embedding(
        self,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.encode_style_tokens(style_img, style_ref_mask=style_ref_mask)
        return F.normalize(tokens.mean(dim=1), dim=-1)
