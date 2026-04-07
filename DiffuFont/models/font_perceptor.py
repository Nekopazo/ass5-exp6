#!/usr/bin/env python3
"""Custom grayscale font perceptor."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_stage_names(raw_names: Sequence[str] | None) -> List[str]:
    valid = {"stage1", "stage2", "stage3", "stage4"}
    if raw_names is None:
        return ["stage1", "stage2", "stage3", "stage4"]
    names = [str(name).strip() for name in raw_names if str(name).strip()]
    if not names:
        raise ValueError("feature_stage_names must contain at least one stage.")
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown feature stage names: {unknown}")
    return names


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        padding = kernel_size // 2
        if activation == "silu":
            act = nn.SiLU(inplace=True)
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(padding),
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(out_channels)),
            act,
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = max(int(in_channels), int(in_channels) * int(expansion))
        self.use_residual = int(stride) == 1 and int(in_channels) == int(out_channels)
        self.expand = ConvNormAct(in_channels, hidden_channels, kernel_size=1, activation="silu")
        self.depthwise = ConvNormAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=int(stride),
            groups=hidden_channels,
            activation="silu",
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, int(out_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
        )
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x = x + residual
        return self.out_act(x)


class FontPerceptor(nn.Module):
    """Lightweight grayscale CNN for font/char perceptual supervision."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 32,
        num_fonts: int = 512,
        num_chars: int = 1000,
        dropout: float = 0.0,
        feature_stage_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(f"FontPerceptor expects grayscale input, got in_channels={in_channels}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_fonts = int(num_fonts)
        self.num_chars = int(num_chars)
        self.dropout = max(0.0, float(dropout))
        self.feature_stage_names = _parse_stage_names(feature_stage_names)

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 3
        c4 = self.base_channels * 4
        c5 = self.base_channels * 6
        c6 = self.base_channels * 8

        self.stem = nn.Sequential(
            ConvNormAct(self.in_channels, c1, kernel_size=3, stride=2, activation="silu"),
            DepthwiseSeparableBlock(c1, c1, stride=1, expansion=2),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(c1, c2, stride=2, expansion=2),
            DepthwiseSeparableBlock(c2, c2, stride=1, expansion=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(c2, c3, stride=2, expansion=2),
            DepthwiseSeparableBlock(c3, c4, stride=1, expansion=2),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(c4, c5, stride=2, expansion=2),
            DepthwiseSeparableBlock(c5, c5, stride=1, expansion=2),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(c5, c6, stride=2, expansion=2),
            DepthwiseSeparableBlock(c6, c6, stride=1, expansion=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(c6, c6),
            nn.LayerNorm(c6),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout),
        )
        self.font_head = nn.Sequential(
            nn.Linear(c6, c6),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(c6, self.num_fonts),
        )
        self.char_head = nn.Sequential(
            nn.Linear(c6, c6),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(c6, self.num_chars),
        )

    def export_config(self) -> dict[str, int | float | list[str]]:
        return {
            "in_channels": int(self.in_channels),
            "base_channels": int(self.base_channels),
            "num_fonts": int(self.num_fonts),
            "num_chars": int(self.num_chars),
            "dropout": float(self.dropout),
            "feature_stage_names": list(self.feature_stage_names),
        }

    def _forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
        stage_features = self._forward_backbone(x)
        final_feature = stage_features["stage4"]
        pooled = self.global_pool(final_feature).flatten(1)
        global_feat = self.global_proj(pooled)
        font_logits = self.font_head(global_feat)
        char_logits = self.char_head(global_feat)
        return {
            "feature_maps": [stage_features[name] for name in self.feature_stage_names],
            "global_feat": global_feat,
            "font_logits": font_logits,
            "char_logits": char_logits,
        }


def style_similarity_stats(
    style_embed: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_pairs: int = 64,
) -> Dict[str, float]:
    if labels.dim() != 1:
        labels = labels.view(-1)
    if style_embed.size(0) != labels.size(0):
        raise ValueError(f"style_embed/labels batch mismatch: {tuple(style_embed.shape)} vs {tuple(labels.shape)}")
    if style_embed.dim() != 2:
        raise ValueError(f"style_embed must be 2D, got {tuple(style_embed.shape)}")

    labels = labels.to(device=style_embed.device)
    style_embed = F.normalize(style_embed, dim=-1, eps=1e-6)

    pair_budget = max(1, int(max_pairs))
    first_index_by_label: dict[int, int] = {}
    positive_pairs: list[tuple[int, int]] = []
    label_ids = labels.detach().cpu().tolist()
    for sample_idx, label_value in enumerate(label_ids):
        label_value = int(label_value)
        first_idx = first_index_by_label.get(label_value)
        if first_idx is None:
            first_index_by_label[label_value] = int(sample_idx)
            continue
        if len(positive_pairs) < pair_budget:
            positive_pairs.append((first_idx, int(sample_idx)))

    first_indices = list(first_index_by_label.values())
    negative_pairs = [
        (int(first_indices[idx]), int(first_indices[idx + 1]))
        for idx in range(min(len(first_indices) - 1, pair_budget))
    ]

    def _pair_cos_mean(index_pairs: list[tuple[int, int]]) -> float:
        if not index_pairs:
            return 0.0
        left_index = torch.tensor([pair[0] for pair in index_pairs], device=style_embed.device, dtype=torch.long)
        right_index = torch.tensor([pair[1] for pair in index_pairs], device=style_embed.device, dtype=torch.long)
        pair_cos = (
            style_embed.index_select(0, left_index) * style_embed.index_select(0, right_index)
        ).sum(dim=1)
        return float(pair_cos.mean().item())

    pos_mean = _pair_cos_mean(positive_pairs)
    neg_mean = _pair_cos_mean(negative_pairs)
    return {
        "style_pos_cos": pos_mean,
        "style_neg_cos": neg_mean,
        "style_cos_margin": pos_mean - neg_mean,
        "style_pos_pairs": float(len(positive_pairs)),
        "style_neg_pairs": float(len(negative_pairs)),
    }


def load_font_perceptor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[FontPerceptor, dict, dict | None]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "model_state" not in checkpoint:
        raise RuntimeError(f"Font perceptor checkpoint missing 'model_state': {checkpoint_path}")
    model_config = checkpoint.get("model_config", {})
    model = FontPerceptor(**model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    report = checkpoint.get("qualification")
    return model, checkpoint, report


class FrozenFontPerceptorGuidance(nn.Module):
    def __init__(self, model: FontPerceptor, *, checkpoint_path: str | Path, qualification_report: dict | None = None) -> None:
        super().__init__()
        self.model = model.eval()
        self.checkpoint_path = str(checkpoint_path)
        self.qualification_report = qualification_report
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
    ) -> "FrozenFontPerceptorGuidance":
        model, _, report = load_font_perceptor_from_checkpoint(checkpoint_path, map_location="cpu")
        model = model.to(device)
        return cls(model, checkpoint_path=checkpoint_path, qualification_report=report)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_type = "cuda" if pred.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            pred = pred.float()
            target = target.float()
            pred_outputs = self.model(pred)
            with torch.no_grad():
                target_outputs = self.model(target)

            pred_feature_maps = pred_outputs["feature_maps"]
            target_feature_maps = target_outputs["feature_maps"]
            if not isinstance(pred_feature_maps, list) or not isinstance(target_feature_maps, list):
                raise RuntimeError("FontPerceptor must return feature_maps as a list.")

            perceptual_per_sample = pred.new_zeros(pred.size(0))
            for pred_feat, target_feat in zip(pred_feature_maps, target_feature_maps):
                perceptual_per_sample = perceptual_per_sample + (pred_feat - target_feat).abs().flatten(1).mean(dim=1)
            perceptual = perceptual_per_sample.mean()
        return {
            "loss_perceptual": perceptual,
            "loss_perceptual_per_sample": perceptual_per_sample,
        }
