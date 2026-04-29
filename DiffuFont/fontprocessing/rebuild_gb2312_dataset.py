#!/usr/bin/env python3
"""Rebuild DiffuFont glyph images and LMDBs from a GB2312 common font charset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Set

from fontTools.ttLib import TTFont


GB2312_MAPPING_URL = "https://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/CP936.TXT"
FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"}


def load_json_list(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [str(item) for item in obj]


def write_json_list(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(values), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text_chars(path: Path, chars: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chars) + "\n", encoding="utf-8")


def resolve_font_path(project_root: Path, font_dir: Path, item: str) -> Path:
    path = Path(item)
    if path.is_absolute() and path.exists():
        return path
    project_path = (project_root / path).resolve()
    if project_path.exists():
        return project_path
    return (font_dir / path.name).resolve()


def download_gb2312_mapping(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GB2312_MAPPING_URL, timeout=60) as response:
        payload = response.read()
    out_path.write_bytes(payload)


def parse_gb2312_hanzi(mapping_path: Path) -> List[str]:
    chars: List[str] = []
    for raw_line in mapping_path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        gb_hex = parts[0]
        if gb_hex.startswith("0x"):
            gb_hex = gb_hex[2:]
        if len(gb_hex) != 4:
            continue
        gb_code = int(gb_hex, 16)
        high = (gb_code >> 8) & 0xFF
        low = gb_code & 0xFF
        if not (0xB0 <= high <= 0xF7 and 0xA1 <= low <= 0xFE):
            continue

        unicode_hex = parts[1]
        if unicode_hex.startswith("0x"):
            unicode_hex = unicode_hex[2:]
        codepoint = int(unicode_hex, 16)
        if 0x4E00 <= codepoint <= 0x9FFF:
            chars.append(chr(codepoint))

    seen: set[str] = set()
    unique_chars: List[str] = []
    for ch in chars:
        if ch in seen:
            continue
        seen.add(ch)
        unique_chars.append(ch)
    if len(unique_chars) < 6000:
        raise RuntimeError(f"Downloaded GB2312 hanzi list is unexpectedly small: {len(unique_chars)}")
    return unique_chars


def load_font_cmap(font_path: Path) -> Set[int]:
    font = TTFont(str(font_path), lazy=True, fontNumber=0)
    try:
        cmap: Set[int] = set()
        for table in font["cmap"].tables:
            cmap.update(int(codepoint) for codepoint in table.cmap.keys())
        return cmap
    finally:
        font.close()


def build_common_charset(chars: List[str], font_paths: List[Path]) -> tuple[List[str], list[dict]]:
    common_codepoints: Optional[Set[int]] = {ord(ch) for ch in chars}
    coverage: list[dict] = []
    candidate_codepoints = set(common_codepoints)
    for font_path in font_paths:
        cmap = load_font_cmap(font_path)
        supported = candidate_codepoints & cmap
        missing = candidate_codepoints - cmap
        coverage.append(
            {
                "font": font_path.stem,
                "path": str(font_path),
                "supported_gb2312": len(supported),
                "missing_gb2312": len(missing),
                "missing_sample": "".join(chr(cp) for cp in sorted(missing)[:50]),
            }
        )
        common_codepoints &= cmap

    common_chars = [ch for ch in chars if ord(ch) in common_codepoints]
    if len(common_chars) <= 1:
        raise RuntimeError("Common GB2312 charset across selected fonts is empty or too small.")
    return common_chars, coverage


def require_fonts(project_root: Path, font_items: List[str], font_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for item in font_items:
        font_path = resolve_font_path(project_root, font_dir, item)
        if not font_path.exists():
            raise FileNotFoundError(f"font not found: {item} -> {font_path}")
        if font_path.suffix not in FONT_SUFFIXES:
            raise ValueError(f"invalid font suffix: {item} -> {font_path.suffix}")
        paths.append(font_path)
    return paths


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def run_command(args: list[str], *, cwd: Path) -> None:
    print("[run]", " ".join(args), flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GB2312, choose common glyphs across all fonts, and rebuild RGB PNG/LMDB data."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--font-list-json", type=Path, default=Path("DataPreparation/FontList.json"))
    parser.add_argument("--content-font-list-json", type=Path, default=Path("DataPreparation/ContentFontList.json"))
    parser.add_argument("--font-dir", type=Path, default=Path("DataPreparation/Font"))
    parser.add_argument("--char-size", type=int, default=120)
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--lmdb-map-size", type=int, default=2**40)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-lmdb", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    font_dir = (project_root / args.font_dir).resolve()
    font_list_path = (project_root / args.font_list_json).resolve()
    content_font_list_path = (project_root / args.content_font_list_json).resolve()

    style_font_items = load_json_list(font_list_path)
    content_font_items = load_json_list(content_font_list_path)
    all_font_paths = require_fonts(project_root, content_font_items + style_font_items, font_dir)

    charset_dir = project_root / "fontprocessing" / "charsets"
    mapping_path = charset_dir / "GB2312.TXT"
    gb2312_path = charset_dir / "gb2312_hanzi.txt"
    common_path = charset_dir / "gb2312_common_all_fonts.txt"
    report_path = charset_dir / "gb2312_common_report.json"

    print(f"[gb2312] downloading {GB2312_MAPPING_URL}", flush=True)
    download_gb2312_mapping(mapping_path)
    gb2312_chars = parse_gb2312_hanzi(mapping_path)
    write_text_chars(gb2312_path, gb2312_chars)
    print(f"[gb2312] hanzi={len(gb2312_chars)}", flush=True)

    common_chars, coverage = build_common_charset(gb2312_chars, all_font_paths)
    write_text_chars(common_path, common_chars)
    write_json_list(project_root / "CharacterData" / "CharList.json", common_chars)
    report_path.write_text(
        json.dumps(
            {
                "gb2312_mapping_url": GB2312_MAPPING_URL,
                "gb2312_hanzi_count": len(gb2312_chars),
                "common_count": len(common_chars),
                "font_count": len(all_font_paths),
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[charset] common_count={len(common_chars)}", flush=True)

    generated_root = project_root / "DataPreparation" / "Generated"
    content_generated = generated_root / "ContentFont"
    train_generated = generated_root / "TrainFonts"
    lmdb_root = project_root / "DataPreparation" / "LMDB"
    content_lmdb = lmdb_root / "ContentFont.lmdb"
    train_lmdb = lmdb_root / "TrainFont.lmdb"

    print("[cleanup] deleting old generated image trees and LMDBs", flush=True)
    for path in (content_generated, train_generated, content_lmdb, train_lmdb):
        remove_path(path)

    if not args.skip_render:
        common_render_args = [
            sys.executable,
            "DataPreparation/generate_font_images.py",
            "--project-root",
            str(project_root),
            "--char-list-json",
            "CharacterData/CharList.json",
            "--char-size",
            str(int(args.char_size)),
            "--canvas-size",
            str(int(args.canvas_size)),
            "--x-offset",
            str(int(args.x_offset)),
            "--y-offset",
            str(int(args.y_offset)),
            "--num-workers",
            str(int(args.num_workers)),
        ]
        run_command(
            common_render_args
            + [
                "--font-list-json",
                "DataPreparation/ContentFontList.json",
                "--out-dir",
                "DataPreparation/Generated/ContentFont",
            ],
            cwd=project_root,
        )
        run_command(
            common_render_args
            + [
                "--font-list-json",
                "DataPreparation/FontList.json",
                "--out-dir",
                "DataPreparation/Generated/TrainFonts",
            ],
            cwd=project_root,
        )

    if not args.skip_lmdb:
        run_command(
            [
                sys.executable,
                "DataPreparation/images_to_lmdb.py",
                "--project-root",
                str(project_root),
                "--img-roots",
                "DataPreparation/Generated/ContentFont",
                "--lmdb-path",
                "DataPreparation/LMDB/ContentFont.lmdb",
                "--map-size",
                str(int(args.lmdb_map_size)),
                "--overwrite",
            ],
            cwd=project_root,
        )
        run_command(
            [
                sys.executable,
                "DataPreparation/images_to_lmdb.py",
                "--project-root",
                str(project_root),
                "--img-roots",
                "DataPreparation/Generated/TrainFonts",
                "--lmdb-path",
                "DataPreparation/LMDB/TrainFont.lmdb",
                "--map-size",
                str(int(args.lmdb_map_size)),
                "--overwrite",
            ],
            cwd=project_root,
        )

    print("[done] rebuilt GB2312 common RGB dataset", flush=True)


if __name__ == "__main__":
    main()
