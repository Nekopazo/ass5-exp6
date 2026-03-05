#!/usr/bin/env python3
"""Export style-image augmentation visualization grids from TrainFont LMDB."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from style_augment import build_base_glyph_transform, build_style_reference_transform


def parse_chars(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def scan_lmdb_fonts_and_chars(env: lmdb.Environment) -> Dict[str, List[str]]:
    out: Dict[str, set[str]] = {}
    txn = env.begin(buffers=True)
    cursor = txn.cursor()
    for raw_key, _ in cursor:
        kb = bytes(raw_key) if isinstance(raw_key, memoryview) else raw_key
        if b"@" not in kb:
            continue
        try:
            key = kb.decode("utf-8")
        except UnicodeDecodeError:
            continue
        font, ch = key.split("@", 1)
        if not font or not ch:
            continue
        out.setdefault(font, set()).add(ch)
    return {k: sorted(v) for k, v in sorted(out.items(), key=lambda kv: kv[0])}


def load_font_char_images_u8(
    env: lmdb.Environment,
    font_name: str,
    chars: List[str],
) -> List[Tuple[str, np.ndarray]]:
    txn = env.begin(buffers=True)
    out: List[Tuple[str, np.ndarray]] = []
    for ch in chars:
        key = f"{font_name}@{ch}".encode("utf-8")
        value = txn.get(key)
        if value is None:
            continue
        enc = np.frombuffer(value, dtype=np.uint8)
        arr = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            img = Image.open(io.BytesIO(bytes(value))).convert("L")
            arr = np.asarray(img, dtype=np.uint8)
        out.append((ch, np.ascontiguousarray(arr)))
    return out


def build_grid(
    images_u8: List[Tuple[str, np.ndarray]],
    num_views: int,
    aug_transform,
    base_transform,
) -> Tuple[torch.Tensor, List[str]]:
    flat: List[torch.Tensor] = []
    row_chars: List[str] = []
    for ch, arr in images_u8:
        row = [base_transform(arr)]
        for _ in range(int(num_views)):
            row.append(aug_transform(arr))
        flat.extend(row)
        row_chars.append(ch)

    nrow = 1 + int(num_views)
    grid = make_grid(
        flat,
        nrow=nrow,
        padding=2,
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    return grid, row_chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Export style augmentation visualization from LMDB.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--font", type=str, default=None, help="Font name stem. Empty means auto-pick.")
    parser.add_argument("--chars", type=str, default=None, help="Comma-separated chars; defaults to random subset.")
    parser.add_argument("--num-chars", type=int, default=8)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--style-aug-canvas-size", type=int, default=256)
    parser.add_argument("--style-aug-crop-min", type=float, default=0.6)
    parser.add_argument("--style-aug-crop-max", type=float, default=0.9)
    parser.add_argument("--style-aug-mask-prob", type=float, default=0.5)
    parser.add_argument("--style-aug-mask-min", type=float, default=0.15)
    parser.add_argument("--style-aug-mask-max", type=float, default=0.3)
    parser.add_argument("--style-aug-affine-deg", type=float, default=5.0)
    parser.add_argument("--style-aug-translate", type=float, default=0.05)
    parser.add_argument("--style-aug-scale-min", type=float, default=1.0)
    parser.add_argument("--style-aug-scale-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/style_aug_preview.png"))
    args = parser.parse_args()

    rng = random.Random(int(args.seed))

    root = Path(args.data_root).resolve()
    lmdb_path = Path(args.train_lmdb)
    if not lmdb_path.is_absolute():
        lmdb_path = (root / lmdb_path).resolve()
    if not lmdb_path.exists():
        raise FileNotFoundError(f"Train LMDB not found: {lmdb_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        font_chars = scan_lmdb_fonts_and_chars(env)
        if not font_chars:
            raise RuntimeError(f"No '<font>@<char>' keys found in LMDB: {lmdb_path}")

        selected_font = str(args.font).strip() if args.font is not None else ""
        if not selected_font:
            selected_font = rng.choice(list(font_chars.keys()))
        if selected_font not in font_chars:
            raise KeyError(f"Font '{selected_font}' not found in LMDB. available={len(font_chars)} fonts")

        requested_chars = parse_chars(args.chars)
        available_chars = list(font_chars[selected_font])
        if requested_chars:
            chosen_chars = [ch for ch in requested_chars if ch in set(available_chars)]
            if not chosen_chars:
                raise KeyError(f"None of requested chars exist for font={selected_font}")
        else:
            k = max(1, min(int(args.num_chars), len(available_chars)))
            chosen_chars = rng.sample(available_chars, k=k)

        samples = load_font_char_images_u8(env, selected_font, chosen_chars)
        if not samples:
            raise RuntimeError(f"No images loaded for font={selected_font}")

        base_transform = build_base_glyph_transform(image_size=int(args.image_size))
        aug_transform = build_style_reference_transform(
            image_size=int(args.image_size),
            augment=True,
            pre_resize=int(args.style_aug_canvas_size),
            crop_scale_min=float(args.style_aug_crop_min),
            crop_scale_max=float(args.style_aug_crop_max),
            mask_prob=float(args.style_aug_mask_prob),
            mask_area_min=float(args.style_aug_mask_min),
            mask_area_max=float(args.style_aug_mask_max),
            affine_degrees=float(args.style_aug_affine_deg),
            affine_translate=float(args.style_aug_translate),
            affine_scale_min=float(args.style_aug_scale_min),
            affine_scale_max=float(args.style_aug_scale_max),
        )

        grid, row_chars = build_grid(
            images_u8=samples,
            num_views=int(args.num_views),
            aug_transform=aug_transform,
            base_transform=base_transform,
        )

        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, str(out_path))

        meta = {
            "lmdb": str(lmdb_path),
            "font": selected_font,
            "chars": row_chars,
            "columns": ["orig", *[f"aug_{i + 1}" for i in range(int(args.num_views))]],
            "seed": int(args.seed),
        }
        meta_path = out_path.with_suffix(out_path.suffix + ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[export_style_aug_visualization] saved grid: {out_path}")
        print(f"[export_style_aug_visualization] saved meta: {meta_path}")
        print(
            f"[export_style_aug_visualization] font={selected_font} rows={len(row_chars)} cols={1 + int(args.num_views)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
