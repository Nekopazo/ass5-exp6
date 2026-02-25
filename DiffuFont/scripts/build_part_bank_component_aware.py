#!/usr/bin/env python3
"""Build a component-aware PartBank with better structural coverage and diversity.

Compared with keypoint-only sampling, this script:
1) Extracts candidates from connected components (centroid/extrema/distance peaks).
2) Scores candidates by structure quality (edge, ink, component coverage).
3) Selects final parts with balanced constraints (char coverage + spatial coverage + descriptor diversity).

Output manifest is compatible with existing PartBank consumers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_EXTS = {".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"}


@dataclass
class Candidate:
    char: str
    x: int
    y: int
    component_id: int
    component_area: int
    score: float
    descriptor: np.ndarray
    patch: np.ndarray
    ink_ratio: float
    edge_ratio: float
    comp_coverage: float
    quadrant: int


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def parse_indices(raw: str | None) -> List[int] | None:
    if not raw:
        return None
    out: List[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    return out or None


def load_charset(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Charset JSON must be a list: {path}")
    chars = [str(x) for x in obj if isinstance(x, str) and len(x) == 1]
    if not chars:
        raise ValueError(f"No valid chars in charset: {path}")
    return chars


def collect_fonts(project_root: Path, fonts_dir: Path, font_list_json: Path | None, indices: Sequence[int] | None) -> List[Path]:
    if font_list_json is not None and font_list_json.exists():
        listed = json.loads(font_list_json.read_text(encoding="utf-8"))
        all_fonts = [(project_root / str(x)).resolve() for x in listed]
    else:
        all_fonts = sorted([p for p in fonts_dir.rglob("*") if p.suffix in FONT_EXTS])

    if indices:
        picked: List[Path] = []
        for i in indices:
            if i < 0 or i >= len(all_fonts):
                raise IndexError(f"font index {i} out of range [0,{len(all_fonts)-1}]")
            picked.append(all_fonts[i])
        return picked
    return all_fonts


def char_supported(ch: str, font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), size=12)
        return font.getmask(ch).getbbox() is not None
    except Exception:
        return False


def render_char(ch: str, font_path: Path, canvas_size: int, char_size: int, x_offset: int, y_offset: int) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size=char_size)
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), ch, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (canvas_size - tw) // 2 - bbox[0] + x_offset
    y = (canvas_size - th) // 2 - bbox[1] + y_offset
    draw.text((x, y), ch, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


def extract_patch(img: np.ndarray, cx: int, cy: int, patch_size: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    r = patch_size // 2
    if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
        return None
    patch = img[cy - r : cy + r, cx - r : cx + r]
    if patch.shape != (patch_size, patch_size):
        return None
    return patch


def patch_edge_ratio(patch: np.ndarray) -> float:
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > 20.0).mean())


def patch_descriptor(patch: np.ndarray) -> np.ndarray:
    small = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    bins = 8
    hist = np.zeros((bins,), dtype=np.float32)
    bin_ids = np.floor((ang % (2.0 * np.pi)) / (2.0 * np.pi) * bins).astype(np.int32)
    for b in range(bins):
        hist[b] = float(mag[bin_ids == b].sum())

    feat = np.concatenate([small.reshape(-1), hist], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(feat)) + 1e-8
    return feat / norm


def topk_distance_peaks(mask: np.ndarray, k: int, min_dist: int) -> List[Tuple[int, int]]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return []

    ys, xs = np.where(dist > 0)
    if ys.size == 0:
        return []

    order = np.argsort(-dist[ys, xs])
    picked: List[Tuple[int, int]] = []
    min_d2 = float(min_dist * min_dist)

    for oi in order:
        y = int(ys[oi])
        x = int(xs[oi])
        ok = True
        for px, py in picked:
            dx = float(x - px)
            dy = float(y - py)
            if dx * dx + dy * dy < min_d2:
                ok = False
                break
        if ok:
            picked.append((x, y))
            if len(picked) >= k:
                break
    return picked


def component_points(comp_mask: np.ndarray, comp_bbox: Tuple[int, int, int, int], extra_peaks: int, min_peak_dist: int) -> List[Tuple[int, int]]:
    x0, y0, w, h = comp_bbox
    ys, xs = np.where(comp_mask > 0)
    if ys.size == 0:
        return []

    pts: List[Tuple[int, int]] = []

    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    pts.append((cx, cy))

    # Component bbox center as a stable fallback.
    pts.append((x0 + w // 2, y0 + h // 2))

    left_i = int(np.argmin(xs))
    right_i = int(np.argmax(xs))
    top_i = int(np.argmin(ys))
    bottom_i = int(np.argmax(ys))
    pts.extend(
        [
            (int(xs[left_i]), int(ys[left_i])),
            (int(xs[right_i]), int(ys[right_i])),
            (int(xs[top_i]), int(ys[top_i])),
            (int(xs[bottom_i]), int(ys[bottom_i])),
        ]
    )

    peaks = topk_distance_peaks(comp_mask.astype(np.uint8), k=max(0, int(extra_peaks)), min_dist=max(1, int(min_peak_dist)))
    pts.extend(peaks)

    # Deduplicate while preserving order.
    out: List[Tuple[int, int]] = []
    seen = set()
    for x, y in pts:
        key = (int(x), int(y))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def dedupe_candidates(cands: List[Candidate], cos_threshold: float, anchor_limit: int) -> List[Candidate]:
    if not cands:
        return []

    # Location dedupe first.
    loc_seen = set()
    loc_out: List[Candidate] = []
    for c in cands:
        key = (c.char, int(c.x), int(c.y))
        if key in loc_seen:
            continue
        loc_seen.add(key)
        loc_out.append(c)

    if cos_threshold <= 0:
        return loc_out

    anchor_limit = max(1, int(anchor_limit))
    out: List[Candidate] = []
    anchors: List[np.ndarray] = []

    for c in sorted(loc_out, key=lambda x: x.score, reverse=True):
        desc = c.descriptor
        if anchors:
            sims = [cosine_sim(desc, a) for a in anchors]
            if max(sims) >= cos_threshold:
                continue
        out.append(c)
        if len(anchors) < anchor_limit:
            anchors.append(desc)
    return out


def quadrant_of(x: int, y: int, canvas_size: int) -> int:
    cx = canvas_size // 2
    cy = canvas_size // 2
    if x < cx and y < cy:
        return 0
    if x >= cx and y < cy:
        return 1
    if x < cx and y >= cy:
        return 2
    return 3


def select_balanced(cands: List[Candidate], k: int, min_char_coverage: int, ensure_quadrants: bool, seed: int) -> List[Candidate]:
    if not cands or k <= 0:
        return []

    ranked = sorted(cands, key=lambda x: x.score, reverse=True)
    if len(ranked) <= k:
        return ranked

    rng = np.random.default_rng(seed)
    chosen: List[Candidate] = []
    chosen_ids = set()

    def add_one(idx: int) -> bool:
        if idx in chosen_ids:
            return False
        chosen_ids.add(idx)
        chosen.append(ranked[idx])
        return True

    # 1) Character coverage pass.
    char_best: Dict[str, int] = {}
    for i, c in enumerate(ranked):
        if c.char not in char_best:
            char_best[c.char] = i
    for _, i in sorted(char_best.items(), key=lambda kv: ranked[kv[1]].score, reverse=True):
        add_one(i)
        if len(chosen) >= min(k, max(1, min_char_coverage)):
            break

    # 2) Spatial coverage pass.
    if ensure_quadrants and len(chosen) < k:
        for q in range(4):
            for i, c in enumerate(ranked):
                if c.quadrant == q and add_one(i):
                    break
                if len(chosen) >= k:
                    break

    # 3) Diversity pass (farthest-point style over descriptor space).
    desc = np.stack([c.descriptor for c in ranked], axis=0)
    while len(chosen) < k:
        best_i = -1
        best_val = -1.0
        for i, c in enumerate(ranked):
            if i in chosen_ids:
                continue
            if not chosen:
                val = c.score
            else:
                dmin = 1.0
                for cc in chosen:
                    sim = cosine_sim(c.descriptor, cc.descriptor)
                    dmin = min(dmin, 1.0 - sim)
                # Blend quality and diversity.
                val = 0.65 * float(c.score) + 0.35 * float(dmin)
            # tiny jitter for stable tie-breaking
            val += float(rng.uniform(0.0, 1e-6))
            if val > best_val:
                best_val = val
                best_i = i
        if best_i < 0:
            break
        add_one(best_i)

    chosen.sort(key=lambda x: x.score, reverse=True)
    return chosen[:k]


def build_candidates_for_char(
    img: np.ndarray,
    ch: str,
    canvas_size: int,
    patch_size: int,
    bin_threshold: int,
    min_component_area: int,
    max_component_area_ratio: float,
    min_ink_ratio: float,
    max_ink_ratio: float,
    min_edge_ratio: float,
    min_comp_coverage: float,
    peaks_per_component: int,
    peak_min_dist: int,
) -> Tuple[List[Candidate], int]:
    binary = (img < int(bin_threshold)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h, w = img.shape[:2]
    max_comp_area = int(float(max_component_area_ratio) * float(h * w))
    out: List[Candidate] = []
    valid_components = 0

    for cid in range(1, n_labels):
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        chh = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        if area > max_comp_area:
            continue

        comp_mask = (labels == cid).astype(np.uint8)
        pts = component_points(comp_mask, (x, y, cw, chh), extra_peaks=peaks_per_component, min_peak_dist=peak_min_dist)
        if not pts:
            continue
        valid_components += 1

        for px, py in pts:
            patch = extract_patch(img, px, py, patch_size)
            if patch is None:
                continue

            ink_ratio = float((patch < 245).mean())
            if ink_ratio < min_ink_ratio or ink_ratio > max_ink_ratio:
                continue

            edge_ratio = patch_edge_ratio(patch)
            if edge_ratio < min_edge_ratio:
                continue

            r = patch_size // 2
            local_comp = comp_mask[py - r : py + r, px - r : px + r]
            if local_comp.shape != (patch_size, patch_size):
                continue
            comp_cov = float(local_comp.sum() / (area + 1e-8))
            if comp_cov < min_comp_coverage:
                continue

            # Score favors: high edge detail, compact component focus, and informative ink occupancy.
            occ = 1.0 - abs(ink_ratio - 0.28) / 0.28
            occ = max(0.0, occ)
            area_term = min(1.0, float(area) / max(1.0, 0.02 * (h * w)))
            score = 0.45 * edge_ratio + 0.30 * comp_cov + 0.15 * occ + 0.10 * area_term

            out.append(
                Candidate(
                    char=ch,
                    x=int(px),
                    y=int(py),
                    component_id=int(cid),
                    component_area=int(area),
                    score=float(score),
                    descriptor=patch_descriptor(patch),
                    patch=patch,
                    ink_ratio=float(ink_ratio),
                    edge_ratio=float(edge_ratio),
                    comp_coverage=float(comp_cov),
                    quadrant=quadrant_of(int(px), int(py), canvas_size),
                )
            )

    return out, valid_components


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--fonts-dir", type=Path, default=Path("fonts"))
    p.add_argument("--font-list-json", type=Path, default=Path("DataPreparation/FontList.json"))
    p.add_argument("--font-indices", type=str, default=None)
    p.add_argument("--max-fonts", type=int, default=0)

    p.add_argument("--charset-json", type=Path, default=Path("CharacterData/ReferenceCharList.json"))
    p.add_argument("--max-chars", type=int, default=0)

    p.add_argument("--output-dir", type=Path, default=Path("DataPreparation/PartBank_component_aware"))
    p.add_argument("--parts-per-font", type=int, default=64)
    p.add_argument("--max-candidates", type=int, default=9000)

    p.add_argument("--canvas-size", type=int, default=256)
    p.add_argument("--char-size", type=int, default=224)
    p.add_argument("--x-offset", type=int, default=0)
    p.add_argument("--y-offset", type=int, default=0)

    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--bin-threshold", type=int, default=245)
    p.add_argument("--min-component-area", type=int, default=24)
    p.add_argument("--max-component-area-ratio", type=float, default=0.35)
    p.add_argument("--peaks-per-component", type=int, default=3)
    p.add_argument("--peak-min-dist", type=int, default=8)

    p.add_argument("--min-ink-ratio", type=float, default=0.02)
    p.add_argument("--max-ink-ratio", type=float, default=0.72)
    p.add_argument("--min-edge-ratio", type=float, default=0.04)
    p.add_argument("--min-comp-coverage", type=float, default=0.08)

    p.add_argument("--sim-dedupe-cos-threshold", type=float, default=0.993)
    p.add_argument("--sim-dedupe-anchor-limit", type=int, default=1800)

    p.add_argument("--min-char-coverage", type=int, default=12)
    p.add_argument("--ensure-quadrants", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    if args.patch_size % 2 != 0:
        raise ValueError("--patch-size must be even")
    if args.parts_per_font <= 0:
        raise ValueError("--parts-per-font must be > 0")

    root = args.project_root.resolve()
    fonts_dir = (root / args.fonts_dir).resolve()
    font_list_json = (root / args.font_list_json).resolve() if args.font_list_json else None
    charset_json = (root / args.charset_json).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = parse_indices(args.font_indices)
    fonts = collect_fonts(root, fonts_dir, font_list_json, indices)
    if args.max_fonts > 0:
        fonts = fonts[: args.max_fonts]

    chars = load_charset(charset_json)
    if args.max_chars > 0:
        chars = chars[: args.max_chars]

    manifest: Dict[str, Dict] = {
        "meta": {
            "selection_method": "component_aware_balanced_diversity",
            "parts_per_font": int(args.parts_per_font),
            "patch_size": int(args.patch_size),
            "charset_size": len(chars),
            "canvas_size": int(args.canvas_size),
            "char_size": int(args.char_size),
            "min_component_area": int(args.min_component_area),
            "max_component_area_ratio": float(args.max_component_area_ratio),
            "peaks_per_component": int(args.peaks_per_component),
            "peak_min_dist": int(args.peak_min_dist),
            "min_char_coverage": int(args.min_char_coverage),
            "ensure_quadrants": bool(args.ensure_quadrants),
            "sim_dedupe_cos_threshold": float(args.sim_dedupe_cos_threshold),
        },
        "fonts": {},
    }

    for fi, font_path in enumerate(fonts):
        if not font_path.exists():
            print(f"[skip] missing font: {font_path}")
            continue

        font_name = font_path.stem
        print(f"[{fi + 1}/{len(fonts)}] {font_name}")

        cands: List[Candidate] = []
        chars_used = 0
        comps_used = 0

        for ch in chars:
            if not char_supported(ch, font_path):
                continue
            try:
                img = render_char(
                    ch,
                    font_path,
                    canvas_size=args.canvas_size,
                    char_size=args.char_size,
                    x_offset=args.x_offset,
                    y_offset=args.y_offset,
                )
            except Exception:
                continue

            cc, valid_components = build_candidates_for_char(
                img=img,
                ch=ch,
                canvas_size=args.canvas_size,
                patch_size=args.patch_size,
                bin_threshold=args.bin_threshold,
                min_component_area=args.min_component_area,
                max_component_area_ratio=args.max_component_area_ratio,
                min_ink_ratio=args.min_ink_ratio,
                max_ink_ratio=args.max_ink_ratio,
                min_edge_ratio=args.min_edge_ratio,
                min_comp_coverage=args.min_comp_coverage,
                peaks_per_component=args.peaks_per_component,
                peak_min_dist=args.peak_min_dist,
            )
            if cc:
                chars_used += 1
                comps_used += int(valid_components)
                cands.extend(cc)

            if len(cands) > args.max_candidates * 2:
                cands = sorted(cands, key=lambda x: x.score, reverse=True)[: args.max_candidates]

        if not cands:
            print("  [skip] no candidates")
            continue

        cands = sorted(cands, key=lambda x: x.score, reverse=True)[: args.max_candidates]
        cands = dedupe_candidates(
            cands,
            cos_threshold=float(args.sim_dedupe_cos_threshold),
            anchor_limit=int(args.sim_dedupe_anchor_limit),
        )

        picked = select_balanced(
            cands,
            k=int(args.parts_per_font),
            min_char_coverage=int(args.min_char_coverage),
            ensure_quadrants=bool(args.ensure_quadrants),
            seed=int(args.random_seed),
        )

        font_out = out_dir / font_name
        font_out.mkdir(parents=True, exist_ok=True)

        rows: List[Dict] = []
        for i, cand in enumerate(picked):
            img_rgb = np.repeat(cand.patch[..., None], 3, axis=2)
            rel_name = f"part_{i:03d}_U{ord(cand.char):04X}.png"
            out_path = font_out / rel_name
            Image.fromarray(img_rgb).save(out_path)
            rows.append(
                {
                    "path": rel_or_abs(out_path, root),
                    "char": cand.char,
                    "char_code": f"U+{ord(cand.char):04X}",
                    "x": int(cand.x),
                    "y": int(cand.y),
                    "response": float(cand.score),
                    "ink_ratio": float(cand.ink_ratio),
                    "edge_ratio": float(cand.edge_ratio),
                    "component_id": int(cand.component_id),
                    "component_area": int(cand.component_area),
                    "component_coverage": float(cand.comp_coverage),
                    "quadrant": int(cand.quadrant),
                    "selection_score": float(cand.score),
                }
            )

        manifest["fonts"][font_name] = {
            "font_path": rel_or_abs(font_path, root),
            "num_candidates": int(len(cands)),
            "num_parts": int(len(rows)),
            "num_chars_with_candidates": int(chars_used),
            "num_valid_components": int(comps_used),
            "parts": rows,
        }
        print(f"  saved {len(rows)} parts from {len(cands)} candidates")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
