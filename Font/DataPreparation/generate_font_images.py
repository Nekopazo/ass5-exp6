#!/usr/bin/env python3
"""Render glyph images for configured fonts and characters.

Filename format:
    <FontStem>@<char>.png

Output layout:
    <out-dir>/<FontStem>/<FontStem>@<char>.png
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from PIL import Image, ImageDraw, ImageFont

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - optional dependency import guard
    TTFont = None


IMG_EXTS = {".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"}
_CMAP_CACHE: Dict[Path, Optional[Set[int]]] = {}


def load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fp:
        obj = json.load(fp)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [str(x) for x in obj]


def parse_indices(s: str | None) -> List[int] | None:
    if not s:
        return None
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out or None


def resolve_font_path(project_root: Path, font_dir: Path, item: str) -> Path:
    p = Path(item)
    if p.is_absolute() and p.exists():
        return p

    # 1) Prefer paths relative to project root, e.g. fonts/xxx.ttf
    p1 = (project_root / p).resolve()
    if p1.exists():
        return p1

    # 2) Fallback to font-dir + basename for older FontList formats.
    p2 = (font_dir / p.name).resolve()
    if p2.exists():
        return p2

    return p1


def load_font_cmap(font_path: Path) -> Optional[Set[int]]:
    cached = _CMAP_CACHE.get(font_path)
    if cached is not None or font_path in _CMAP_CACHE:
        return cached

    if TTFont is None:
        _CMAP_CACHE[font_path] = None
        return None

    try:
        tt = TTFont(str(font_path), lazy=True)
        cmap: Set[int] = set()
        for table in tt["cmap"].tables:
            cmap.update(int(cp) for cp in table.cmap.keys())
        tt.close()
        _CMAP_CACHE[font_path] = cmap
        return cmap
    except Exception:
        _CMAP_CACHE[font_path] = None
        return None


def char_supported(ch: str, font_path: Path) -> bool:
    cmap = load_font_cmap(font_path)
    if cmap is not None:
        return ord(ch) in cmap
    try:
        font = ImageFont.truetype(str(font_path), size=12)
        return font.getmask(ch).getbbox() is not None
    except Exception:
        return False


def draw_char(
    ch: str,
    font_path: Path,
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size=char_size)
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Center text by bbox and apply custom offsets.
    bbox = draw.textbbox((0, 0), ch, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2 - bbox[0] + x_offset
    y = (canvas_size - text_h) // 2 - bbox[1] + y_offset
    draw.text((x, y), ch, fill=(0, 0, 0), font=font)

    return img


def filter_fonts(all_fonts: List[str], font_indices: List[int] | None) -> List[str]:
    if not font_indices:
        return all_fonts
    selected: List[str] = []
    for idx in font_indices:
        if idx < 0 or idx >= len(all_fonts):
            raise ValueError(f"font index {idx} out of range [0, {len(all_fonts) - 1}]")
        selected.append(all_fonts[idx])
    return selected

def generate_images(
    chars: Iterable[str],
    font_items: Iterable[str],
    project_root: Path,
    font_dir: Path,
    out_dir: Path,
    char_size: int,
    canvas_size: int,
    x_offset: int,
    y_offset: int,
    num_workers: int,
    strict_coverage: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chars = list(chars)

    jobs = [
        (
            list(chars),
            str(font_item),
            str(project_root),
            str(font_dir),
            str(out_dir),
            int(char_size),
            int(canvas_size),
            int(x_offset),
            int(y_offset),
            bool(strict_coverage),
        )
        for font_item in font_items
    ]
    max_workers = max(1, int(num_workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_font_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            font_name, saved, unsupported_count = future.result()
            print(f"[generate] {font_name}: saved={saved} unsupported={unsupported_count}")


def _process_font_job(job: tuple) -> tuple[str, int, int]:
    (
        chars,
        font_item,
        project_root_raw,
        font_dir_raw,
        out_dir_raw,
        char_size,
        canvas_size,
        x_offset,
        y_offset,
        strict_coverage,
    ) = job
    project_root = Path(project_root_raw)
    font_dir = Path(font_dir_raw)
    out_dir = Path(out_dir_raw)
    font_path = resolve_font_path(project_root, font_dir, font_item)
    if not font_path.exists():
        raise FileNotFoundError(f"font not found: {font_item} -> {font_path}")
    if font_path.suffix not in IMG_EXTS:
        raise ValueError(f"invalid font suffix for: {font_item} -> {font_path.suffix}")

    font_name = font_path.stem
    font_out_dir = out_dir / font_name
    font_out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    unsupported_chars: List[str] = []
    for ch in chars:
        if len(ch) != 1:
            raise ValueError(f"invalid char entry '{ch}' in charset (expected single character)")
        if not char_supported(ch, font_path):
            unsupported_chars.append(ch)
            continue
        try:
            img = draw_char(
                ch=ch,
                font_path=font_path,
                char_size=char_size,
                canvas_size=canvas_size,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            save_path = font_out_dir / f"{font_name}@{ch}.png"
            img.save(save_path, dpi=(300, 300))
            saved += 1
        except Exception as exc:
            raise RuntimeError(
                f"failed to render char '{ch}' for font '{font_name}' ({font_path})"
            ) from exc

    if strict_coverage and unsupported_chars:
        sample = "".join(unsupported_chars[:20])
        raise RuntimeError(
            f"font '{font_name}' generation failed: "
            f"{len(unsupported_chars)} unsupported chars in charset. sample='{sample}'"
        )
    if saved == 0:
        raise RuntimeError(f"font '{font_name}' generation failed: no glyph images were saved")
    return font_name, saved, len(unsupported_chars)


def main() -> None:
    if TTFont is None:
        raise RuntimeError(
            "fontTools is required for strict glyph coverage checking. "
            "Please install it: pip install fonttools"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--char-list-json", type=Path, default=Path("CharacterData/CharList.json"))
    parser.add_argument("--font-list-json", type=Path, default=Path("DataPreparation/FontList.json"))
    parser.add_argument("--font-dir", type=Path, default=Path("DataPreparation/Font"))
    parser.add_argument("--font-indices", type=str, default=None, help="Comma-separated font indices from FontList")

    parser.add_argument("--char-size", type=int, default=120)
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("DataPreparation/Generated"))
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--allow-unsupported", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    char_list_json = (project_root / args.char_list_json).resolve()
    font_list_json = (project_root / args.font_list_json).resolve()
    font_dir = (project_root / args.font_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    chars = load_json_list(char_list_json)
    all_fonts = load_json_list(font_list_json)
    selected_fonts = filter_fonts(all_fonts, parse_indices(args.font_indices))

    generate_images(
        chars=chars,
        font_items=selected_fonts,
        project_root=project_root,
        font_dir=font_dir,
        out_dir=out_dir,
        char_size=args.char_size,
        canvas_size=args.canvas_size,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        num_workers=int(args.num_workers),
        strict_coverage=not bool(args.allow_unsupported),
    )


if __name__ == "__main__":
    main()
