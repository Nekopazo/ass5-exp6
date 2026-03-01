#!/usr/bin/env python3
"""Batch test: run part extraction on ALL TrainFonts using pre-generated images.

For each font directory under DataPreparation/Generated/TrainFonts/:
  - randomly pick 20 characters from available glyph images
  - call process_one_glyph() directly (in-process, no subprocess overhead)
  - save debug images (skeleton + sampling boxes)
  - collect statistics and report failures
"""

import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

TRAIN_FONTS_DIR = "DataPreparation/Generated/TrainFonts"
OUTPUT_DIR = "DataPreparation/PartBank_all_fonts_test"
MAX_CHARS = 20
SEED = 2026


def list_chars_in_dir(font_dir: Path) -> List[str]:
    """Extract available character list from glyph image filenames."""
    chars = []
    for p in sorted(font_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = p.stem
        if "@" in stem:
            ch = stem.rsplit("@", 1)[-1]
            if len(ch) == 1:
                chars.append(ch)
        elif len(stem) >= 1:
            ch = stem[-1]
            if len(ch) == 1:
                chars.append(ch)
    return chars


def find_image_for_char(font_dir: Path, ch: str) -> Optional[Path]:
    """Find the image file for a given character in the font directory."""
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    # Try Font@字.png pattern first
    for ext in exts:
        p = font_dir / f"{font_dir.name}@{ch}{ext}"
        if p.exists():
            return p
    # Fallback: scan
    for p in font_dir.iterdir():
        if not p.is_file() or p.suffix not in exts:
            continue
        stem = p.stem
        if "@" in stem:
            if stem.rsplit("@", 1)[-1] == ch:
                return p
        elif stem.endswith(ch):
            return p
    return None


def main():
    # Import the refactored part extractor directly (no subprocess)
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "scripts"))
    import build_glyph_parts_single_font_test as bld
    bld.require_cv2()
    import cv2
    from PIL import Image

    train_dir = project_root / TRAIN_FONTS_DIR
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found")
        sys.exit(1)

    out_root = (project_root / OUTPUT_DIR).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    font_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(font_dirs)} font directories in {train_dir}")
    print(f"Output: {out_root}")
    print(f"Each font: random {MAX_CHARS} chars, saving debug 2x2 images")
    print()

    rng = random.Random(SEED)
    failed: List[Tuple[str, str]] = []
    font_stats: List[Tuple[str, float, int, int]] = []  # (name, avg, min, max)
    t_start = time.time()

    for fi, fd in enumerate(font_dirs):
        font_name = fd.name
        all_chars = list_chars_in_dir(fd)
        if not all_chars:
            print(f"[{fi+1}/{len(font_dirs)}] SKIP (no images): {font_name}")
            continue

        sample = rng.sample(all_chars, min(MAX_CHARS, len(all_chars)))

        font_out = out_root / font_name
        font_out.mkdir(parents=True, exist_ok=True)

        part_counts: List[int] = []
        font_ok = True

        for ci, ch in enumerate(sample):
            try:
                img_path = find_image_for_char(fd, ch)
                if img_path is None:
                    continue
                gray = bld.load_glyph_gray(img_path, canvas_size=256)

                glyph_json, debug_img = bld.process_one_glyph(
                    gray=gray,
                    patch_size=40,
                    adaptive_block=35,
                    adaptive_c=8,
                    min_cc_area=30,
                    close_kernel=3,
                    noise_hole_max_area=120,
                    endpoint_cluster_r=6,
                    junction_cluster_r=8,
                    curvature_angle_thresh=0.45,
                    curvature_walk_delta=5,
                    curvature_cluster_r=8,
                    spur_max_len=4,
                    spur_rounds=2,
                    min_parts=4,
                    max_parts=12,
                    min_fg_ratio=0.08,
                )

                n_parts = glyph_json["meta"]["total_parts"]
                part_counts.append(n_parts)

                gid = f"{ci:03d}_U{ord(ch):04X}"
                glyph_dir_out = font_out / gid
                glyph_dir_out.mkdir(parents=True, exist_ok=True)

                # Save debug image
                Image.fromarray(
                    cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), mode="RGB"
                ).save(glyph_dir_out / f"{gid}_debug_2x2.png")

                # Save parts JSON (compact)
                glyph_json["glyph_id"] = gid
                glyph_json["char"] = ch
                glyph_json["char_code"] = f"U+{ord(ch):04X}"
                (glyph_dir_out / f"{gid}_parts.json").write_text(
                    json.dumps(glyph_json, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )

            except Exception as e:
                failed.append((font_name, f"char={ch}: {e}"))
                font_ok = False
                traceback.print_exc()

        if part_counts:
            avg = sum(part_counts) / len(part_counts)
            mn, mx = min(part_counts), max(part_counts)
            font_stats.append((font_name, avg, mn, mx))
            status = "OK" if font_ok else "PARTIAL"
            print(
                f"[{fi+1}/{len(font_dirs)}] {status}: {font_name}  "
                f"avg={avg:.1f} min={mn} max={mx}  ({len(part_counts)} chars)"
            )
        else:
            print(f"[{fi+1}/{len(font_dirs)}] EMPTY: {font_name}")

    elapsed = time.time() - t_start

    # Final report
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(font_dirs)} fonts, {elapsed:.1f}s elapsed")
    print(f"  Passed fonts: {len(font_stats)}")
    print(f"  Failed calls: {len(failed)}")

    if failed:
        print(f"\nFAILURES ({len(failed)}):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    if font_stats:
        all_avgs = [s[1] for s in font_stats]
        all_mins = [s[2] for s in font_stats]
        all_maxs = [s[3] for s in font_stats]
        print(f"\nGlobal statistics:")
        print(f"  Avg parts per char: [{min(all_avgs):.1f}, {max(all_avgs):.1f}]")
        print(f"  Global min parts: {min(all_mins)}")
        print(f"  Global max parts: {max(all_maxs)}")

    # Count debug images
    debug_count = sum(1 for _ in out_root.rglob("*_debug_2x2.png"))
    print(f"\nTotal debug images: {debug_count}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
