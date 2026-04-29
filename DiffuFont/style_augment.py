#!/usr/bin/env python3
"""Glyph transforms used by DiffuFont.

The current pipeline intentionally avoids style augmentation. Both content and
style branches expect native 128x128 RGB glyphs and only normalize them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch


def _to_rgb_tensor01(img: Any) -> torch.Tensor:
    """Convert PIL/numpy/tensor input to RGB float tensor in [0, 1]."""
    if isinstance(img, torch.Tensor):
        x = img.detach().clone().to(dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(3, -1, -1)
        elif x.dim() == 3 and int(x.size(0)) == 1:
            x = x.expand(3, -1, -1)
        elif x.dim() == 3 and int(x.size(0)) != 3 and int(x.size(-1)) == 3:
            x = x.permute(2, 0, 1).contiguous()
        elif x.dim() != 3 or int(x.size(0)) != 3:
            raise ValueError(f"unsupported tensor image shape: {tuple(x.shape)}")
        if float(x.max()) > 1.0 or float(x.min()) < 0.0:
            if float(x.min()) >= -1.0 and float(x.max()) <= 1.0:
                x = x.add(1.0).mul_(0.5)
            else:
                x = x.clamp_(0.0, 255.0).div_(255.0)
        return x.clamp_(0.0, 1.0)

    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    if isinstance(img, np.ndarray):
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3:
            raise ValueError(f"unsupported ndarray image shape: {tuple(arr.shape)}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] != 3:
            raise ValueError(f"unsupported ndarray image shape: {tuple(arr.shape)}")
        x = torch.from_numpy(arr.astype(np.float32, copy=False)).permute(2, 0, 1).contiguous()
        if float(x.max()) > 1.0 or float(x.min()) < 0.0:
            x = x.clamp_(0.0, 255.0).div_(255.0)
        return x.clamp_(0.0, 1.0)

    raise TypeError(f"unsupported image type: {type(img)}")


class BaseGlyphTransform:
    """Validate RGB glyph size and normalize to [-1, 1]."""

    def __init__(self, image_size: int = 128):
        self.image_size = int(image_size)

    def __call__(self, img: Any) -> torch.Tensor:
        x = _to_rgb_tensor01(img)
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected glyph size {(self.image_size, self.image_size)}, "
                f"got {tuple(int(v) for v in x.shape[-2:])}"
            )
        if x.dim() != 3 or int(x.size(0)) != 3:
            raise ValueError(f"expected RGB tensor shape (3,H,W), got {tuple(x.shape)}")
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
