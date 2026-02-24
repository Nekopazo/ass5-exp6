#!/usr/bin/env python3
"""Style encoders and E_p font classifier backbones."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + self.shortcut(x)
        h = self.act(h)
        return h


class SmallResNetEmbedding(nn.Module):
    """Compact ResNet encoder returning a normalized embedding."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResidualBlock(32, 32, stride=1), ResidualBlock(32, 32, stride=1))
        self.layer2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64, stride=1))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128, stride=1))
        self.layer4 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256, stride=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, self.embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).flatten(1)
        h = self.head(h)
        if normalize:
            h = F.normalize(h, dim=-1)
        return h


class FontRetrievalEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.backbone = SmallResNetEmbedding(in_channels=in_channels, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.backbone(x, normalize=normalize)


class PartPatchEncoder(FontRetrievalEncoder):
    """Backward-compatible alias for old checkpoints/scripts."""


class GlyphStyleEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.backbone = SmallResNetEmbedding(in_channels=in_channels, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.backbone(x, normalize=normalize)


def encode_style_stack(
    encoder: GlyphStyleEncoder,
    style_img: torch.Tensor,
    in_channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode style image stack.

    Args:
        encoder: glyph style encoder
        style_img: tensor of shape (B, C*K, H, W)
        in_channels: channels per style image (usually 3)

    Returns:
        style_tokens: (B, K, D)
        style_global: (B, D), mean-pooled and normalized
    """
    if style_img.dim() != 4:
        raise ValueError(f"style_img must be 4D (B,C*K,H,W), got {tuple(style_img.shape)}")
    b, ck, h, w = style_img.shape
    if ck % int(in_channels) != 0:
        raise ValueError(f"style_img channels ({ck}) must be divisible by in_channels ({in_channels})")
    k = ck // int(in_channels)
    x = style_img.view(b, k, in_channels, h, w).reshape(b * k, in_channels, h, w)
    tokens = encoder(x, normalize=True).view(b, k, -1)
    style_global = F.normalize(tokens.mean(dim=1), dim=-1)
    return tokens, style_global


class LightCNNBackbone(nn.Module):
    """Lightweight CNN backbone with global average pooling."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, int(out_dim))
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class FontClassifier(nn.Module):
    """E_p font classifier: Backbone -> GAP -> Linear logits."""

    def __init__(
        self,
        in_channels: int = 3,
        num_fonts: int = 170,
        backbone: str = "resnet18",
        light_cnn_dim: int = 256,
    ):
        super().__init__()
        if int(num_fonts) <= 1:
            raise ValueError(f"num_fonts must be > 1, got {num_fonts}")
        self.num_fonts = int(num_fonts)
        self.backbone_name = str(backbone).strip().lower()
        self.in_channels = int(in_channels)

        if self.backbone_name in {"resnet18", "resnet34"}:
            ctor = tv_models.resnet18 if self.backbone_name == "resnet18" else tv_models.resnet34
            net = ctor(weights=None)
            if self.in_channels != 3:
                net.conv1 = nn.Conv2d(
                    self.in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            feat_dim = int(net.fc.in_features)
            net.fc = nn.Identity()
            self.backbone = net
            self.classifier = nn.Linear(feat_dim, self.num_fonts)
            self._feat_dim = feat_dim
        elif self.backbone_name in {"light_cnn", "lightcnn", "small_cnn", "smallcnn"}:
            dim = int(light_cnn_dim)
            self.backbone = LightCNNBackbone(in_channels=self.in_channels, out_dim=dim)
            self.classifier = nn.Linear(dim, self.num_fonts)
            self._feat_dim = dim
            self.backbone_name = "light_cnn"
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Use one of: resnet18, resnet34, light_cnn."
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.classifier(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
