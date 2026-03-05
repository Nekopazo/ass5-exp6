#!/usr/bin/env python3
"""Style-reference augmentation utilities for DiffuFont."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


class RandomMask:
    """Mask out one random rectangle area on a grayscale image tensor."""

    def __init__(self, area_ratio: Tuple[float, float] = (0.15, 0.3), fill: float = 1.0) -> None:
        amin = float(area_ratio[0])
        amax = float(area_ratio[1])
        if amin <= 0.0 or amax <= 0.0 or amax < amin:
            raise ValueError(f"invalid area_ratio={area_ratio}")
        self.area_ratio = (amin, amax)
        self.fill = _to_fill01(float(fill))

    def __call__(self, img, fill: float | None = None) -> torch.Tensor:
        x = _to_gray_tensor01(img)
        if x.dim() != 3 or int(x.size(0)) != 1:
            raise ValueError(f"RandomMask expects shape (1,H,W), got {tuple(x.shape)}")
        fill_v = self.fill if fill is None else _to_fill01(float(fill))

        _, h, w = x.shape
        if w <= 1 or h <= 1:
            return x

        ratio = _rand_uniform(self.area_ratio[0], self.area_ratio[1])
        area = max(1, int(round(w * h * ratio)))

        # Keep a near-square mask while allowing mild aspect variation.
        aspect = _rand_uniform(0.75, 1.33)
        mask_w = max(1, min(w, int(round(math.sqrt(area * aspect)))))
        mask_h = max(1, min(h, int(round(area / max(1, mask_w)))))

        max_x = max(0, w - mask_w)
        max_y = max(0, h - mask_h)
        x0 = int(torch.randint(0, max_x + 1, (1,)).item())
        y0 = int(torch.randint(0, max_y + 1, (1,)).item())
        x1 = min(w, x0 + mask_w)
        y1 = min(h, y0 + mask_h)

        out = x.clone()
        out[:, y0:y1, x0:x1] = fill_v
        return out


def _to_fill01(fill: float) -> float:
    if fill > 1.0:
        return max(0.0, min(1.0, fill / 255.0))
    return max(0.0, min(1.0, fill))


def _rand_uniform(low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float32).uniform_(float(low), float(high)).item())


def _to_gray_tensor01(img) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[2] in (1, 3):
            arr = arr.mean(axis=2) if arr.shape[2] == 3 else arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.mean(axis=0) if arr.shape[0] == 3 else arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Unsupported ndarray shape: {tuple(img.shape)}")
        if arr.dtype == np.uint8:
            t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(dtype=torch.float32)
            if t.numel() > 0 and float(t.max()) > 1.0:
                t = t / 255.0
        return t
    if isinstance(img, Image.Image):
        t = TF.pil_to_tensor(img)
    elif torch.is_tensor(img):
        t = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)

    if t.dim() != 3:
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")

    if t.shape[0] == 3:
        t = t.mean(dim=0, keepdim=True)
    elif t.shape[0] != 1:
        raise ValueError(f"Expected 1 or 3 channels, got {int(t.shape[0])}")

    if t.dtype == torch.uint8:
        t = t.to(dtype=torch.float32).div_(255.0)
    else:
        t = t.to(dtype=torch.float32)
        if t.numel() > 0 and float(t.max()) > 1.0:
            t = t / 255.0
    return t


def _to_gray_u8_numpy(img) -> np.ndarray | None:
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            arr = arr.mean(axis=2) if arr.shape[2] == 3 else arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.mean(axis=0) if arr.shape[0] == 3 else arr[0]
        else:
            return None
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr)
        x = arr.astype(np.float32, copy=False)
        vmax = float(x.max()) if x.size > 0 else 0.0
        vmin = float(x.min()) if x.size > 0 else 0.0
        if vmax <= 1.0 and vmin >= 0.0:
            x = x * 255.0
        elif vmax <= 1.0 and vmin >= -1.0:
            x = (x + 1.0) * 127.5
        return np.clip(np.rint(x), 0.0, 255.0).astype(np.uint8)
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.uint8)

    if not torch.is_tensor(img):
        return None

    t = img
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    if t.dim() != 3:
        return None

    if t.shape[0] == 3:
        t = t.mean(dim=0, keepdim=True)
    elif t.shape[0] != 1:
        return None

    t = t.squeeze(0)
    if t.device.type != "cpu":
        t = t.detach().cpu()

    if t.dtype == torch.uint8:
        arr = t.contiguous().numpy()
        return arr

    x = t.to(dtype=torch.float32).contiguous().numpy()
    vmax = float(x.max()) if x.size > 0 else 0.0
    vmin = float(x.min()) if x.size > 0 else 0.0
    if vmax <= 1.0 and vmin >= 0.0:
        x = x * 255.0
    elif vmax <= 1.0 and vmin >= -1.0:
        x = (x + 1.0) * 127.5
    x = np.clip(np.rint(x), 0.0, 255.0).astype(np.uint8)
    return x


def _resize(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    try:
        return TF.resize(t, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    except TypeError:
        return TF.resize(t, size=size, interpolation=InterpolationMode.BILINEAR)


def _resized_crop(
    t: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: tuple[int, int],
) -> torch.Tensor:
    try:
        return TF.resized_crop(
            t,
            top=top,
            left=left,
            height=height,
            width=width,
            size=size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    except TypeError:
        return TF.resized_crop(
            t,
            top=top,
            left=left,
            height=height,
            width=width,
            size=size,
            interpolation=InterpolationMode.BILINEAR,
        )


class BaseGlyphTransform:
    def __init__(self, image_size: int = 128) -> None:
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")

    def __call__(self, img) -> torch.Tensor:
        x_u8 = _to_gray_u8_numpy(img)
        if x_u8 is not None:
            x = cv2.resize(x_u8, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            out = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dtype=torch.float32)
            return out.div_(127.5).sub_(1.0)
        x = _to_gray_tensor01(img)
        x = _resize(x, (self.image_size, self.image_size))
        return x.mul(2.0).sub(1.0)


class StyleReferenceTransform:
    """Tensor augmentation path: crop -> mask -> affine -> resize."""

    def __init__(
        self,
        image_size: int = 128,
        pre_resize: int = 256,
        crop_scale_min: float = 0.6,
        crop_scale_max: float = 0.9,
        mask_prob: float = 0.5,
        mask_area_min: float = 0.15,
        mask_area_max: float = 0.3,
        affine_degrees: float = 5.0,
        affine_translate: float = 0.05,
        affine_scale_min: float = 1.0,
        affine_scale_max: float = 1.0,
    ) -> None:
        self.image_size = int(image_size)
        self.pre_resize = int(pre_resize)
        self.crop_scale_min = float(crop_scale_min)
        self.crop_scale_max = float(crop_scale_max)
        self.mask_prob = float(mask_prob)
        self.mask_area_min = float(mask_area_min)
        self.mask_area_max = float(mask_area_max)
        self.affine_degrees = float(affine_degrees)
        self.affine_translate = float(affine_translate)
        self.affine_scale_min = float(affine_scale_min)
        self.affine_scale_max = float(affine_scale_max)
        self.masker = RandomMask(area_ratio=(mask_area_min, mask_area_max), fill=1.0)

        if self.image_size <= 0 or self.pre_resize <= 0:
            raise ValueError(
                f"image_size/pre_resize must be > 0, got {self.image_size}/{self.pre_resize}"
            )
        if self.crop_scale_min <= 0.0 or self.crop_scale_max < self.crop_scale_min:
            raise ValueError(
                f"invalid crop scale range: ({self.crop_scale_min}, {self.crop_scale_max})"
            )
        if self.affine_scale_min <= 0.0 or self.affine_scale_max < self.affine_scale_min:
            raise ValueError(
                f"invalid affine scale range: ({self.affine_scale_min}, {self.affine_scale_max})"
            )
        if not (0.0 <= self.mask_prob <= 1.0):
            raise ValueError(f"mask_prob must be in [0,1], got {self.mask_prob}")

    def __call__(self, img) -> torch.Tensor:
        x_u8 = _to_gray_u8_numpy(img)
        if x_u8 is not None:
            return self._apply_cv2(x_u8)
        return self._apply_torch(img)

    def _apply_cv2(self, x_u8: np.ndarray) -> torch.Tensor:
        x = cv2.resize(
            x_u8,
            (self.pre_resize, self.pre_resize),
            interpolation=cv2.INTER_LINEAR,
        )

        scale = _rand_uniform(self.crop_scale_min, self.crop_scale_max)
        crop_size = int(round(self.pre_resize * math.sqrt(scale)))
        crop_size = max(1, min(self.pre_resize, crop_size))
        max_top = max(0, self.pre_resize - crop_size)
        max_left = max(0, self.pre_resize - crop_size)
        top = int(torch.randint(0, max_top + 1, (1,)).item())
        left = int(torch.randint(0, max_left + 1, (1,)).item())
        x = x[top : top + crop_size, left : left + crop_size]
        if crop_size != self.pre_resize:
            x = cv2.resize(x, (self.pre_resize, self.pre_resize), interpolation=cv2.INTER_LINEAR)

        if float(torch.rand((), dtype=torch.float32).item()) < self.mask_prob:
            ratio = _rand_uniform(self.mask_area_min, self.mask_area_max)
            area = max(1, int(round(float(self.pre_resize * self.pre_resize) * ratio)))
            aspect = _rand_uniform(0.75, 1.33)
            mask_w = max(1, min(self.pre_resize, int(round(math.sqrt(area * aspect)))))
            mask_h = max(1, min(self.pre_resize, int(round(area / max(1, mask_w)))))
            max_x = max(0, self.pre_resize - mask_w)
            max_y = max(0, self.pre_resize - mask_h)
            x0 = int(torch.randint(0, max_x + 1, (1,)).item())
            y0 = int(torch.randint(0, max_y + 1, (1,)).item())
            x1 = min(self.pre_resize, x0 + mask_w)
            y1 = min(self.pre_resize, y0 + mask_h)
            x[y0:y1, x0:x1] = 255

        angle = _rand_uniform(-self.affine_degrees, self.affine_degrees)
        trans_max = self.affine_translate * float(self.pre_resize)
        tx = float(round(_rand_uniform(-trans_max, trans_max)))
        ty = float(round(_rand_uniform(-trans_max, trans_max)))
        affine_scale = _rand_uniform(self.affine_scale_min, self.affine_scale_max)
        center = (0.5 * float(self.pre_resize), 0.5 * float(self.pre_resize))
        mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=affine_scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        x = cv2.warpAffine(
            x,
            mat,
            (self.pre_resize, self.pre_resize),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255.0,
        )

        x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        out = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dtype=torch.float32)
        return out.div_(127.5).sub_(1.0)

    def _apply_torch(self, img) -> torch.Tensor:
        x = _to_gray_tensor01(img)
        x = _resize(x, (self.pre_resize, self.pre_resize))

        scale = _rand_uniform(self.crop_scale_min, self.crop_scale_max)
        crop_size = int(round(self.pre_resize * math.sqrt(scale)))
        crop_size = max(1, min(self.pre_resize, crop_size))
        max_top = max(0, self.pre_resize - crop_size)
        max_left = max(0, self.pre_resize - crop_size)
        top = int(torch.randint(0, max_top + 1, (1,)).item())
        left = int(torch.randint(0, max_left + 1, (1,)).item())
        x = _resized_crop(
            x,
            top=top,
            left=left,
            height=crop_size,
            width=crop_size,
            size=(self.pre_resize, self.pre_resize),
        )

        if float(torch.rand((), dtype=torch.float32).item()) < self.mask_prob:
            # Doc order: crop -> mask -> affine -> resize.
            x = self.masker(x, fill=1.0)

        angle = _rand_uniform(-self.affine_degrees, self.affine_degrees)
        trans_max = self.affine_translate * float(self.pre_resize)
        tx = int(round(_rand_uniform(-trans_max, trans_max)))
        ty = int(round(_rand_uniform(-trans_max, trans_max)))
        affine_scale = _rand_uniform(self.affine_scale_min, self.affine_scale_max)
        x = TF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=affine_scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        )

        x = _resize(x, (self.image_size, self.image_size))
        return x.mul(2.0).sub(1.0)


def build_base_glyph_transform(image_size: int = 128):
    """Deterministic glyph transform used by content/target branches."""
    size = int(image_size)
    return BaseGlyphTransform(image_size=size)


def build_style_reference_transform(
    image_size: int = 128,
    augment: bool = True,
    pre_resize: int = 256,
    crop_scale_min: float = 0.6,
    crop_scale_max: float = 0.9,
    mask_prob: float = 0.5,
    mask_area_min: float = 0.15,
    mask_area_max: float = 0.3,
    affine_degrees: float = 5.0,
    affine_translate: float = 0.05,
    affine_scale_min: float = 1.0,
    affine_scale_max: float = 1.0,
):
    """Build style-image transform (doc pipeline: crop -> mask -> affine -> resize)."""
    out_size = int(image_size)
    if not bool(augment):
        return build_base_glyph_transform(image_size=out_size)

    canvas_size = int(pre_resize)
    crop_min = float(crop_scale_min)
    crop_max = float(crop_scale_max)
    if crop_min <= 0.0 or crop_max < crop_min:
        raise ValueError(f"invalid crop scale range: ({crop_min}, {crop_max})")

    mask_min = float(mask_area_min)
    mask_max = float(mask_area_max)
    if mask_min <= 0.0 or mask_max < mask_min:
        raise ValueError(f"invalid mask area range: ({mask_min}, {mask_max})")
    mask_p = float(mask_prob)
    if not (0.0 <= mask_p <= 1.0):
        raise ValueError(f"invalid mask_prob: {mask_p}")

    aff_scale_min = float(affine_scale_min)
    aff_scale_max = float(affine_scale_max)
    if aff_scale_min <= 0.0 or aff_scale_max < aff_scale_min:
        raise ValueError(f"invalid affine scale range: ({aff_scale_min}, {aff_scale_max})")

    return StyleReferenceTransform(
        image_size=out_size,
        pre_resize=canvas_size,
        crop_scale_min=crop_min,
        crop_scale_max=crop_max,
        mask_prob=mask_p,
        mask_area_min=mask_min,
        mask_area_max=mask_max,
        affine_degrees=float(affine_degrees),
        affine_translate=float(affine_translate),
        affine_scale_min=aff_scale_min,
        affine_scale_max=aff_scale_max,
    )


__all__ = [
    "BaseGlyphTransform",
    "RandomMask",
    "StyleReferenceTransform",
    "build_base_glyph_transform",
    "build_style_reference_transform",
]
