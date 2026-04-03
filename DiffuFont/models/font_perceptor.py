#!/usr/bin/env python3
"""Custom grayscale font perceptor and related losses."""

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
    """Lightweight grayscale CNN for font perceptual/style supervision."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 32,
        proj_dim: int = 128,
        num_chars: int = 1000,
        dropout: float = 0.0,
        feature_stage_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError(f"FontPerceptor expects grayscale input, got in_channels={in_channels}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.proj_dim = int(proj_dim)
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
        self.style_proj_head = nn.Sequential(
            nn.Linear(c6, c6),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(c6, self.proj_dim),
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
            "proj_dim": int(self.proj_dim),
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
        style_embed = F.normalize(self.style_proj_head(global_feat), dim=-1, eps=1e-6)
        char_logits = self.char_head(global_feat)
        return {
            "feature_maps": [stage_features[name] for name in self.feature_stage_names],
            "global_feat": global_feat,
            "style_embed": style_embed,
            "char_logits": char_logits,
        }


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
    if labels.dim() != 1:
        labels = labels.view(-1)
    if features.size(0) != labels.size(0):
        raise ValueError(f"features/labels batch mismatch: {tuple(features.shape)} vs {tuple(labels.shape)}")

    features = F.normalize(features, dim=-1, eps=1e-6)
    labels = labels.to(device=features.device)
    logits = torch.matmul(features, features.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = features.size(0)
    identity_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~identity_mask)
    logits_mask = (~identity_mask).to(dtype=features.dtype)

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid_rows = positive_count > 0
    if not valid_rows.any():
        return features.new_tensor(0.0)

    mean_log_prob_pos = (log_prob * positive_mask.to(dtype=log_prob.dtype)).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid_rows].mean()


def style_similarity_stats(
    style_embed: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples: int = 64,
    max_samples_per_label: int = 2,
) -> Dict[str, float]:
    if labels.dim() != 1:
        labels = labels.view(-1)
    if style_embed.size(0) != labels.size(0):
        raise ValueError(f"style_embed/labels batch mismatch: {tuple(style_embed.shape)} vs {tuple(labels.shape)}")
    if style_embed.dim() != 2:
        raise ValueError(f"style_embed must be 2D, got {tuple(style_embed.shape)}")

    labels = labels.to(device=style_embed.device)
    batch_size = int(style_embed.size(0))
    sample_budget = max(2, int(max_samples))
    per_label_budget = max(1, int(max_samples_per_label))
    if batch_size > sample_budget:
        unique_labels, counts = torch.unique(labels, sorted=True, return_counts=True)
        prioritized_labels = torch.cat([unique_labels[counts > 1], unique_labels[counts <= 1]], dim=0)
        sample_index_chunks = []
        sample_count = 0
        for label_value in prioritized_labels.tolist():
            label_indices = torch.nonzero(labels.eq(label_value), as_tuple=False).flatten()
            take_count = min(int(label_indices.numel()), per_label_budget, sample_budget - sample_count)
            if take_count <= 0:
                break
            sample_index_chunks.append(label_indices[:take_count])
            sample_count += take_count
            if sample_count >= sample_budget:
                break
        sample_indices = torch.cat(sample_index_chunks, dim=0)
        style_embed = style_embed.index_select(0, sample_indices)
        labels = labels.index_select(0, sample_indices)
    style_embed = F.normalize(style_embed, dim=-1, eps=1e-6)
    sim = torch.matmul(style_embed, style_embed.t())
    identity_mask = torch.eye(style_embed.size(0), device=style_embed.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~identity_mask)
    negative_mask = (~labels.unsqueeze(0).eq(labels.unsqueeze(1))) & (~identity_mask)

    pos_values = sim[positive_mask]
    neg_values = sim[negative_mask]
    pos_mean = float(pos_values.mean().item()) if pos_values.numel() > 0 else 0.0
    neg_mean = float(neg_values.mean().item()) if neg_values.numel() > 0 else 0.0
    return {
        "style_pos_cos": pos_mean,
        "style_neg_cos": neg_mean,
        "style_cos_margin": pos_mean - neg_mean,
        "style_pos_pairs": float(pos_values.numel()),
        "style_neg_pairs": float(neg_values.numel()),
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
            style_per_sample = 1.0 - F.cosine_similarity(
                pred_outputs["style_embed"],
                target_outputs["style_embed"],
                dim=-1,
                eps=1e-6,
            )
            perceptual = perceptual_per_sample.mean()
            style = style_per_sample.mean()
        return {
            "loss_perceptual": perceptual,
            "loss_perceptual_per_sample": perceptual_per_sample,
            "loss_style_embed": style,
            "loss_style_embed_per_sample": style_per_sample,
        }
