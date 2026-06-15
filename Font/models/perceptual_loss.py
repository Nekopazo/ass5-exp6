#!/usr/bin/env python3
"""Perceptual loss used by the FontDiffuser reference training code."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class VGG16FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            vgg16 = torchvision.models.vgg16(weights=weights)
        except AttributeError:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        hidden = image
        for block in (self.enc_1, self.enc_2, self.enc_3):
            hidden = block(hidden)
            features.append(hidden)
        return features


class ContentPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = VGG16FeatureExtractor()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
        return (image - self.mean.to(device=image.device, dtype=image.dtype)) / self.std.to(
            device=image.device,
            dtype=image.dtype,
        )

    def forward(self, generated_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        generated_features = self.vgg(self._normalize(generated_images.float()))
        target_features = self.vgg(self._normalize(target_images.float()))
        loss = torch.zeros((), device=generated_images.device, dtype=torch.float32)
        for generated_feature, target_feature in zip(generated_features, target_features):
            loss = loss + torch.mean((target_feature - generated_feature) ** 2)
        return loss / float(len(generated_features))
