#!/usr/bin/env python3
"""Source-aligned DiffuFont model with fixed low/mid/high style routing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_style_encoder import CrossAttentionWithBias, HierarchicalStyleEncoderMixin
from .source_fontdiffuser import ContentEncoder, UNet


FIXED_STYLE_TRANSFORMER_SCALES = (16, 32)
FIXED_STYLE_LOCAL_MOD_SCALES = ()
FIXED_STYLE_TOKEN_CONSUMER_MAP = {
    "t_low": ("mid",),
    "t_mid": ("up_16",),
    "t_high": ("up_32",),
}
FIXED_STYLE_SITE_ARCH = {
    "mid": "transformer_cross_attn",
    "up_16": "transformer_cross_attn_content_guided_tokens",
    "up_32": "transformer_cross_attn_content_guided_tokens",
    "up_64": "self_attention_only",
}
ACTIVE_STYLE_SITES = ("mid", "up_16", "up_32")


class ContentQueryPool(nn.Module):
    """Pool a content feature map into a small set of learned queries."""

    def __init__(self, in_channels: int, token_dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")
        self.proj = nn.Conv2d(int(in_channels), self.token_dim, kernel_size=1)
        self.query = nn.Parameter(torch.randn(self.num_queries, self.token_dim) * 0.02)
        self.attn = CrossAttentionWithBias(embed_dim=self.token_dim, num_heads=int(num_heads))
        self.norm = nn.LayerNorm(self.token_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim * 4, self.token_dim),
        )
        self.out_norm = nn.LayerNorm(self.token_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(f"content feature must be 4D, got {tuple(feat.shape)}")
        x = self.proj(feat).flatten(2).transpose(1, 2).contiguous()
        q = self.query.unsqueeze(0).expand(int(x.size(0)), -1, -1)
        out, _ = self.attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=None,
            need_weights=False,
        )
        out = self.norm(out + q)
        out = self.out_norm(out + self.ffn(out))
        return out


class ContentGuidedRetriever(nn.Module):
    """Retrieve style tokens from style memory using content queries."""

    def __init__(self, token_dim: int, num_heads: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.attn = CrossAttentionWithBias(embed_dim=self.token_dim, num_heads=int(num_heads))
        self.norm = nn.LayerNorm(self.token_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim * 4, self.token_dim),
        )
        self.out_norm = nn.LayerNorm(self.token_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, content_queries: torch.Tensor, style_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if content_queries.dim() != 3 or style_memory.dim() != 3:
            raise ValueError(
                f"content_queries/style_memory must be 3D, got {tuple(content_queries.shape)} and {tuple(style_memory.shape)}"
            )
        out, weights = self.attn(
            query=content_queries,
            key=style_memory,
            value=style_memory,
            key_padding_mask=None,
            need_weights=True,
        )
        out = self.norm(out + content_queries)
        out = self.out_norm(out + self.ffn(out))
        return self.gate.tanh() * out, weights


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
        style_token_dim: int = 256,
        style_token_count: int = 3,
        local_token_count: int = 3,
        style_memory_up16_count: int = 6,
        style_memory_up32_count: int = 6,
        style_memory_up16_pool_hw: int = 16,
        style_memory_up32_pool_hw: int = 16,
        content_query_up16_count: int = 2,
        content_query_up32_count: int = 4,
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
        self.content_query_up16_count = int(content_query_up16_count)
        self.content_query_up32_count = int(content_query_up32_count)
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
            style_memory_up16_count=int(style_memory_up16_count),
            style_memory_up32_count=int(style_memory_up32_count),
            style_memory_up16_pool_hw=int(style_memory_up16_pool_hw),
            style_memory_up32_pool_hw=int(style_memory_up32_pool_hw),
        )

        self.inject_mid = self._build_inject_head()
        self.inject_up16 = self._build_inject_head()
        self.inject_up32 = self._build_inject_head()

        heads = 8 if (self.style_token_dim % 8 == 0) else (4 if (self.style_token_dim % 4 == 0) else 1)
        content_query_up16_channels = int(content_start_channel) * 4
        content_query_up32_channels = int(content_start_channel) * 2
        self.content_query_up16 = (
            ContentQueryPool(
                in_channels=content_query_up16_channels,
                token_dim=self.style_token_dim,
                num_queries=self.content_query_up16_count,
                num_heads=heads,
            )
            if self.content_query_up16_count > 0
            else None
        )
        self.content_query_up32 = (
            ContentQueryPool(
                in_channels=content_query_up32_channels,
                token_dim=self.style_token_dim,
                num_queries=self.content_query_up32_count,
                num_heads=heads,
            )
            if self.content_query_up32_count > 0
            else None
        )
        self.retriever_up16 = ContentGuidedRetriever(self.style_token_dim, heads) if self.content_query_up16 is not None else None
        self.retriever_up32 = ContentGuidedRetriever(self.style_token_dim, heads) if self.content_query_up32 is not None else None

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
            mid_enable_content_attn=False,
        )
        self.style_site_drop_prob = 0.0
        self.style_site_drop_min_keep = 1
        self._style_site_force_keep: frozenset[str] | None = None

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

    def project_style_sites(self, style_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if style_tokens.dim() != 3 or int(style_tokens.size(1)) != 3:
            raise ValueError(f"style_tokens must be (B,3,D), got {tuple(style_tokens.shape)}")
        t_low = style_tokens[:, 0]
        t_mid = style_tokens[:, 1]
        t_high = style_tokens[:, 2]
        return {
            "mid": self.inject_mid(t_low),
            "up_16": self.inject_up16(t_mid),
            "up_32": self.inject_up32(t_high),
        }

    def project_style_sites_from_pack(self, style_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "t_low" not in style_pack:
            if "tokens" not in style_pack:
                raise ValueError("style_pack must contain either t_low or tokens")
            return self.project_style_sites(style_pack["tokens"])
        memory_up16 = style_pack.get("memory_up_16", None)
        memory_up32 = style_pack.get("memory_up_32", None)
        if memory_up16 is None or memory_up32 is None:
            if "tokens" not in style_pack:
                raise ValueError("style_pack missing memory tensors and tokens fallback")
            return self.project_style_sites(style_pack["tokens"])
        return {
            "mid": self.inject_mid(style_pack["t_low"]),
            "up_16": self.inject_up16(memory_up16.mean(dim=1)),
            "up_32": self.inject_up32(memory_up32.mean(dim=1)),
        }

    def stack_style_site_contexts(self, style_tokens: torch.Tensor) -> torch.Tensor:
        site_contexts = self.project_style_sites(style_tokens)
        return torch.stack([site_contexts[name] for name in ACTIVE_STYLE_SITES], dim=1)

    def stack_style_site_contexts_from_pack(self, style_pack: dict[str, torch.Tensor]) -> torch.Tensor:
        site_contexts = self.project_style_sites_from_pack(style_pack)
        return torch.stack([site_contexts[name] for name in ACTIVE_STYLE_SITES], dim=1)

    def load_style_pretrained(self, ckpt_path: str) -> None:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict) and isinstance(obj.get("extra"), dict):
            expected_route = {k: list(v) for k, v in FIXED_STYLE_TOKEN_CONSUMER_MAP.items()}
            route_meta = obj["extra"].get("token_consumer_map", {}) or {}
            if route_meta != expected_route:
                print(
                    "[load_style_pretrained] warning: routing metadata mismatch "
                    f"ckpt={route_meta or '<missing>'} current={expected_route}; "
                    "loading style encoder weights non-strictly anyway.",
                    flush=True,
                )
            consumer_arch_meta = obj["extra"].get("style_site_arch", {}) or {}
            expected_arch = dict(FIXED_STYLE_SITE_ARCH)
            if consumer_arch_meta != expected_arch:
                print(
                    "[load_style_pretrained] warning: consumer metadata mismatch "
                    f"ckpt={consumer_arch_meta or '<missing>'} current={expected_arch}; "
                    "loading style encoder weights non-strictly anyway.",
                    flush=True,
                )
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

    def set_style_site_dropout(self, prob: float, min_keep: int = 1) -> None:
        self.style_site_drop_prob = float(max(0.0, min(1.0, prob)))
        self.style_site_drop_min_keep = max(1, int(min_keep))

    def set_style_site_force_keep(self, keep_sites: tuple[str, ...] | list[str] | None) -> None:
        if keep_sites is None:
            self._style_site_force_keep = None
            return
        self._style_site_force_keep = frozenset(str(x) for x in keep_sites)

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

    def _build_content_guided_style_contexts(
        self,
        style_pack: dict[str, torch.Tensor],
        content_residual_features: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mid_ctx = self.inject_mid(style_pack["t_low"]).unsqueeze(1)

        memory_up16 = style_pack["memory_up_16"]
        memory_up32 = style_pack["memory_up_32"]
        up16_ctx = self.inject_up16(memory_up16.mean(dim=1)).unsqueeze(1)
        up32_ctx = self.inject_up32(memory_up32.mean(dim=1)).unsqueeze(1)

        if self.content_query_up16 is not None and self.retriever_up16 is not None:
            content_feat_up16 = content_residual_features[3]
            guided_q_up16 = self.content_query_up16(content_feat_up16)
            guided_up16, _ = self.retriever_up16(guided_q_up16, memory_up16)
            up16_ctx = self.inject_up16(guided_up16)

        if self.content_query_up32 is not None and self.retriever_up32 is not None:
            content_feat_up32 = content_residual_features[2]
            guided_q_up32 = self.content_query_up32(content_feat_up32)
            guided_up32, _ = self.retriever_up32(guided_q_up32, memory_up32)
            up32_ctx = self.inject_up32(guided_up32)

        style_contexts = {
            "mid": mid_ctx,
            "up_16": up16_ctx,
            "up_32": up32_ctx,
        }
        style_contexts = self._apply_style_site_dropout(style_contexts)
        style_contexts = self._apply_style_site_force_mask(style_contexts)
        return style_contexts

    def _apply_style_site_dropout(self, style_contexts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if (not self.training) or self.style_site_drop_prob <= 0.0:
            return style_contexts

        active_sites = [site for site in ACTIVE_STYLE_SITES if site in style_contexts]
        if len(active_sites) <= self.style_site_drop_min_keep:
            return style_contexts

        keep_mask = torch.rand((len(active_sites),), device=next(iter(style_contexts.values())).device)
        keep_flags = keep_mask >= self.style_site_drop_prob
        min_keep = min(len(active_sites), self.style_site_drop_min_keep)
        if int(keep_flags.sum().item()) < min_keep:
            keep_flags[:] = False
            chosen = torch.randperm(len(active_sites), device=keep_flags.device)[:min_keep]
            keep_flags[chosen] = True

        out = dict(style_contexts)
        for idx, site in enumerate(active_sites):
            if not bool(keep_flags[idx].item()):
                out[site] = torch.zeros_like(out[site])
        return out

    def _apply_style_site_force_mask(self, style_contexts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._style_site_force_keep is None:
            return style_contexts
        out = dict(style_contexts)
        for site, tensor in out.items():
            if site not in self._style_site_force_keep:
                out[site] = torch.zeros_like(tensor)
        return out

    def _resolve_style_hidden_states(
        self,
        mode: str,
        style_img: Optional[torch.Tensor],
        style_ref_mask: Optional[torch.Tensor],
        content_residual_features: list[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        mode_norm = self._normalize_conditioning_mode(mode)
        if mode_norm == "baseline" or style_img is None:
            return None

        style_pack = self.encode_style_pack(style_img, style_ref_mask=style_ref_mask)
        return self._build_content_guided_style_contexts(style_pack, content_residual_features)

    def get_content_guided_gate_values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.retriever_up16 is not None:
            out["gate_up_16"] = float(self.retriever_up16.gate.detach().tanh().item())
        if self.retriever_up32 is not None:
            out["gate_up_32"] = float(self.retriever_up32.gate.detach().tanh().item())
        return out

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
            content_residual_features=content_residual_features,
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
