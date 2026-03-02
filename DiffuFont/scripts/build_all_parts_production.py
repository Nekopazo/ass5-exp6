#!/usr/bin/env python3
"""Production script: extract style-aware part patches for ALL fonts × ALL chars.

Output structure (nested: font → char):
    DataPreparation/PartBank/<font>/<Uxxxx>/part_NNN_UXXXX.png  (40×40 grayscale)
    DataPreparation/PartBank/manifest.json
    DataPreparation/LMDB/PartBank.lmdb

Multi-process: uses ProcessPoolExecutor (bypasses GIL, true CPU parallelism
for cv2/numpy/skimage C extensions).

Usage:
    python scripts/build_all_parts_production.py                  # auto workers
    python scripts/build_all_parts_production.py --workers 6
    python scripts/build_all_parts_production.py --debug          # + debug imgs
    python scripts/build_all_parts_production.py --skip-lmdb      # PNGs only
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 256
PATCH_SIZE = 40
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
#  Helpers  (all top-level for pickle)
# ---------------------------------------------------------------------------

def char_to_unicode_tag(ch: str) -> str:
    return f"U{ord(ch):04X}"


def list_chars_in_dir(font_dir: str) -> List[Tuple[str, str]]:
    results = []
    for fname in os.listdir(font_dir):
        if not fname.lower().endswith(".png"):
            continue
        stem = fname[:-4]
        ch = stem.split("@", 1)[1] if "@" in stem else stem
        if ch:
            results.append((ch, os.path.join(font_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


def save_part_png(gray_patch: np.ndarray, out_path: str) -> None:
    """Save a 40×40 grayscale patch as PNG (black stroke, white background)."""
    arr = gray_patch.astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG", compress_level=1)


# ---------------------------------------------------------------------------
#  Per-font worker  (runs in child process – fully self-contained)
# ---------------------------------------------------------------------------

def _process_font_worker(args_tuple: tuple) -> Dict[str, Any]:
    """Process ALL chars of one font.  Runs in a child process.

    args_tuple = (font_dir, out_bank_root, patch_size, min_fg_ratio, save_debug, scripts_dir)
    """
    font_dir_str, out_bank_root_str, patch_size, min_fg_ratio, save_debug, scripts_dir = args_tuple

    # -- late imports in child --
    import sys as _sys
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    from build_glyph_parts_single_font_test import process_one_glyph  # noqa

    font_name = os.path.basename(font_dir_str)
    chars = list_chars_in_dir(font_dir_str)
    empty = {"font": font_name, "status": "skip", "reason": "no chars",
             "n_chars": 0, "n_fail": 0, "n_parts": 0,
             "avg_parts": 0, "min_parts": 0, "max_parts": 0,
             "_manifest_parts": []}
    if not chars:
        return empty

    manifest_parts: List[Dict[str, Any]] = []
    total_parts = 0
    n_ok = 0
    n_fail = 0
    part_counts: List[int] = []

    for idx, (ch, img_path_str) in enumerate(chars):
        try:
            pil_img = Image.open(img_path_str).convert("L")
            gray = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

            result, debug_img = process_one_glyph(
                gray,
                patch_size=patch_size,
                min_fg_ratio=float(min_fg_ratio),
            )

            utag = char_to_unicode_tag(ch)
            all_parts = result.get("all_parts", [])
            n_parts = len(all_parts)
            part_counts.append(n_parts)

            # Nested dir: PartBank/<font>/<Uxxxx>/
            char_out_dir = os.path.join(out_bank_root_str, font_name, utag)

            for pi, part_info in enumerate(all_parts):
                fname = f"part_{pi:03d}_{utag}.png"
                part_path = os.path.join(char_out_dir, fname)
                patch_gray = part_info["patch_gray"]
                gray_arr = np.array(patch_gray, dtype=np.uint8)
                save_part_png(gray_arr, part_path)

                lmdb_key = f"DataPreparation/PartBank/{font_name}/{utag}/{fname}"
                manifest_parts.append({
                    "path": f"DataPreparation/PartBank/{font_name}/{utag}/{fname}",
                    "lmdb_key": lmdb_key,
                    "char": ch,
                    "char_code": utag,
                    "type": part_info.get("type", ""),
                    "center": part_info.get("center", []),
                    "fg_ratio": part_info.get("fg_ratio", 0),
                    "width": part_info.get("width", 0),
                })
                total_parts += 1

            if save_debug and debug_img is not None:
                dbg_path = os.path.join(char_out_dir, f"_debug_{utag}.png")
                Image.fromarray(debug_img[..., ::-1]).save(dbg_path)

            n_ok += 1
        except Exception:
            n_fail += 1
            if n_fail <= 3:
                traceback.print_exc()

    avg = round(float(np.mean(part_counts)), 2) if part_counts else 0
    mn = int(np.min(part_counts)) if part_counts else 0
    mx = int(np.max(part_counts)) if part_counts else 0

    return {
        "font": font_name,
        "status": "ok" if n_fail == 0 else "partial",
        "n_chars": n_ok,
        "n_fail": n_fail,
        "n_parts": total_parts,
        "avg_parts": avg,
        "min_parts": mn,
        "max_parts": mx,
        "_manifest_parts": manifest_parts,
    }


# ---------------------------------------------------------------------------
#  LMDB packing
# ---------------------------------------------------------------------------

def build_lmdb_from_manifest(
    manifest: Dict[str, Any],
    project_root: Path,
    lmdb_path: Path,
    map_size_gb: int = 8,
) -> Dict[str, int]:
    import lmdb as _lmdb

    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = _lmdb.open(str(lmdb_path), map_size=map_size_gb * (1 << 30),
                      subdir=True, lock=True, readahead=False, meminit=False)
    txn = env.begin(write=True)

    written = 0
    missing = 0
    total_bytes = 0

    for font_name, info in manifest.get("fonts", {}).items():
        for row in info.get("parts", []):
            key = row.get("lmdb_key", "")
            rel_path = row.get("path", "")
            if not key or not rel_path:
                missing += 1
                continue
            full_path = project_root / rel_path
            if not full_path.exists():
                missing += 1
                continue
            b = full_path.read_bytes()
            txn.put(key.encode("utf-8"), b, overwrite=True)
            written += 1
            total_bytes += len(b)
            if written % 10000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"  [lmdb] written {written} ...", flush=True)

    txn.commit()
    env.sync()
    env.close()
    return {"written": written, "missing": missing, "total_bytes": total_bytes}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build PartBank for ALL fonts (multi-process).")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--train-fonts-dir", type=Path,
                    default=Path("DataPreparation/Generated/TrainFonts"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("DataPreparation/PartBank"))
    ap.add_argument("--lmdb-dir", type=Path,
                    default=Path("DataPreparation/LMDB"))
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    ap.add_argument("--min-fg-ratio", type=float, default=0.05,
                    help="Minimum foreground ratio of a patch (lower helps ultra-thin glyphs).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Process pool size. 0 = auto (nproc).")
    ap.add_argument("--debug", action="store_true",
                    help="Save debug 2x2 images per glyph.")
    ap.add_argument("--skip-lmdb", action="store_true",
                    help="Only PNGs + manifest, skip LMDB.")
    ap.add_argument("--lmdb-map-size-gb", type=int, default=8)
    ap.add_argument("--fonts", type=str, default="",
                    help="Comma-separated font names (default = all).")
    args = ap.parse_args()

    root = args.project_root.resolve()
    train_root = (root / args.train_fonts_dir).resolve()
    out_bank = (root / args.output_dir).resolve()
    lmdb_dir = (root / args.lmdb_dir).resolve()
    scripts_dir = str(Path(__file__).resolve().parent)

    if not train_root.is_dir():
        print(f"ERROR: TrainFonts not found: {train_root}", file=sys.stderr)
        sys.exit(1)

    font_dirs = sorted(
        [p for p in train_root.iterdir() if p.is_dir() and any(p.glob("*.png"))],
        key=lambda p: p.name,
    )
    if args.fonts:
        keep = {f.strip() for f in args.fonts.split(",") if f.strip()}
        font_dirs = [d for d in font_dirs if d.name in keep]

    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() or 2)
    workers = min(workers, len(font_dirs), 16)

    print(f"Fonts:   {len(font_dirs)}")
    print(f"Output:  {out_bank}")
    print(f"LMDB:    {lmdb_dir / 'PartBank.lmdb'}")
    print(f"Patch:   {args.patch_size}")
    print(f"Workers: {workers}  (ProcessPoolExecutor)")
    print(f"Debug:   {args.debug}")
    print(flush=True)

    out_bank.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(fd), str(out_bank), args.patch_size, args.min_fg_ratio, args.debug, scripts_dir)
        for fd in font_dirs
    ]

    # ------- Multi-process --------
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    manifest_fonts: Dict[str, Dict[str, Any]] = {}
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_font_worker, t): os.path.basename(t[0]) for t in tasks}
        for future in as_completed(futures):
            fname = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"font": fname, "status": "error", "reason": str(e),
                       "n_chars": 0, "n_fail": 0, "n_parts": 0,
                       "avg_parts": 0, "min_parts": 0, "max_parts": 0,
                       "_manifest_parts": []}
                traceback.print_exc()

            m_parts = res.pop("_manifest_parts", [])
            results.append(res)
            if m_parts:
                manifest_fonts[res["font"]] = {"parts": m_parts}

            done += 1
            elapsed_so_far = time.time() - t0
            eta = (elapsed_so_far / done) * (len(font_dirs) - done) if done else 0
            print(f"  [{done}/{len(font_dirs)}] {res['status']}: {res['font']}  "
                  f"chars={res.get('n_chars',0)} parts={res.get('n_parts',0)} "
                  f"avg={res.get('avg_parts',0)}  "
                  f"ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - t0

    # ------- Manifest --------
    manifest = {"fonts": manifest_fonts}
    manifest_path = out_bank / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nManifest: {manifest_path}  ({len(manifest_fonts)} fonts)")

    # ------- LMDB --------
    if not args.skip_lmdb:
        print("\nPacking LMDB ...")
        lmdb_path = lmdb_dir / "PartBank.lmdb"
        st = build_lmdb_from_manifest(manifest, root, lmdb_path, args.lmdb_map_size_gb)
        print(f"  LMDB: written={st['written']} missing={st['missing']} "
              f"size={st['total_bytes']/1e6:.1f}MB")

    # ------- Summary --------
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_partial = sum(1 for r in results if r["status"] == "partial")
    n_err = sum(1 for r in results if r["status"] in ("error", "skip"))
    total_chars = sum(r.get("n_chars", 0) for r in results)
    total_parts = sum(r.get("n_parts", 0) for r in results)
    all_avgs = [r["avg_parts"] for r in results if r.get("avg_parts", 0) > 0]
    all_mins = [r["min_parts"] for r in results if r.get("min_parts", 0) > 0]
    all_maxs = [r["max_parts"] for r in results if r.get("max_parts", 0) > 0]

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Fonts: {n_ok} ok / {n_partial} partial / {n_err} error")
    print(f"  Total chars: {total_chars}")
    print(f"  Total parts: {total_parts}")
    if all_avgs:
        print(f"  Avg parts/char: [{min(all_avgs):.1f}, {max(all_avgs):.1f}]")
    if all_mins:
        print(f"  Global min parts: {min(all_mins)}")
    if all_maxs:
        print(f"  Global max parts: {max(all_maxs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
