#!/usr/bin/env python3
"""Glyph transforms used by DiffuFont.

The current pipeline intentionally avoids style augmentation. Both content and
style branches expect native 128x128 grayscale glyphs and only normalize them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch


def _to_gray_tensor01(img: Any) -> torch.Tensor:
    """Convert PIL/numpy/tensor input to grayscale float tensor in [0, 1]."""
    if isinstance(img, torch.Tensor):
        x = img.detach().clone().to(dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and int(x.size(0)) == 3:
            x = x.mean(dim=0, keepdim=True)
        elif x.dim() != 3 or int(x.size(0)) != 1:
            raise ValueError(f"unsupported tensor image shape: {tuple(x.shape)}")
        if float(x.max()) > 1.0 or float(x.min()) < 0.0:
            if float(x.min()) >= -1.0 and float(x.max()) <= 1.0:
                x = x.add(1.0).mul_(0.5)
            else:
                x = x.clamp_(0.0, 255.0).div_(255.0)
        return x.clamp_(0.0, 1.0)

    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    if isinstance(img, np.ndarray):
        arr = np.asarray(img)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] == 3:
                arr = arr.mean(axis=-1)
            elif arr.shape[-1] != 1:
                raise ValueError(f"unsupported ndarray image shape: {tuple(arr.shape)}")
            else:
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"unsupported ndarray image shape: {tuple(arr.shape)}")
        x = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0)
        if float(x.max()) > 1.0 or float(x.min()) < 0.0:
            x = x.clamp_(0.0, 255.0).div_(255.0)
        return x.clamp_(0.0, 1.0)

    raise TypeError(f"unsupported image type: {type(img)}")


class BaseGlyphTransform:
    """Validate grayscale glyph size and normalize to [-1, 1]."""

    def __init__(self, image_size: int = 128):
        self.image_size = int(image_size)

    def __call__(self, img: Any) -> torch.Tensor:
        x = _to_gray_tensor01(img)
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected glyph size {(self.image_size, self.image_size)}, "
                f"got {tuple(int(v) for v in x.shape[-2:])}"
            )
        return x.mul(2.0).sub(1.0)


def build_base_glyph_transform(image_size: int = 128) -> BaseGlyphTransform:
    return BaseGlyphTransform(image_size=int(image_size))


def build_style_reference_transform(
    image_size: int = 128,
    augment: bool | None = None,
    **_: Any,
) -> BaseGlyphTransform:
    """Compatibility wrapper.

    `augment` and all legacy augmentation kwargs are ignored on purpose.
    """

    _ = augment
    return build_base_glyph_transform(image_size=int(image_size))


__all__ = [
    "BaseGlyphTransform",
    "build_base_glyph_transform",
    "build_style_reference_transform",
]
