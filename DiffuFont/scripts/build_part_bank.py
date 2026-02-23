#!/usr/bin/env python3
"""Build a paper-aligned few-part bank from font files.

Pipeline:
1. Render glyph images for each font and charset.
2. Detect keypoints/descriptors (paper setting: SIFT).
3. Keep meaningful patches (in-bounds, non-blank, non-overfilled, edge-rich).
4. Select K representative parts per font via descriptor-space K-medoids.
5. Save part patches and a manifest for downstream training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"}


@dataclass
class Candidate:
    char: str
    x: int
    y: int
    response: float
    descriptor: np.ndarray
    patch: np.ndarray
    ink_ratio: float
    edge_ratio: float


def load_charset(path: Path) -> List[str]:
    chars = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chars, list):
        raise ValueError(f"Charset file must be a list: {path}")
    return [str(c) for c in chars if isinstance(c, str) and len(c) == 1]


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def collect_fonts(project_root: Path, fonts_dir: Path, font_list_json: Path | None, font_indices: Sequence[int] | None) -> List[Path]:
    if font_list_json is not None and font_list_json.exists():
        font_list = json.loads(font_list_json.read_text(encoding="utf-8"))
        all_fonts = [(project_root / x).resolve() for x in font_list]
        if font_indices:
            selected = []
            for i in font_indices:
                if i < 0 or i >= len(all_fonts):
                    raise IndexError(f"font index {i} out of range [0,{len(all_fonts)-1}]")
                selected.append(all_fonts[i])
            return selected
        return all_fonts

    all_fonts = sorted([p for p in fonts_dir.rglob("*") if p.suffix in IMG_EXTS])
    if font_indices:
        selected = []
        for i in font_indices:
            if i < 0 or i >= len(all_fonts):
                raise IndexError(f"font index {i} out of range [0,{len(all_fonts)-1}]")
            selected.append(all_fonts[i])
        return selected
    return all_fonts


def build_detector(mode: str):
    mode = mode.lower()
    if mode == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV build does not support SIFT. Install opencv-contrib-python or use --detector orb")
        return cv2.SIFT_create(nfeatures=800)
    if mode == "auto":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=800)
        print("[warn] SIFT unavailable, fallback to ORB (not paper-aligned)")
        return cv2.ORB_create(nfeatures=1200)
    if mode == "orb":
        return cv2.ORB_create(nfeatures=1200)
    raise ValueError(f"Unsupported detector: {mode}")


def char_supported(ch: str, font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), size=12)
        return font.getmask(ch).getbbox() is not None
    except Exception:
        return False


def render_char(
    ch: str,
    font_path: Path,
    canvas_size: int,
    char_size: int,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size=char_size)
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), ch, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2 - bbox[0] + x_offset
    y = (canvas_size - text_h) // 2 - bbox[1] + y_offset
    draw.text((x, y), ch, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


def descriptor_to_unit(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    norm = float(np.linalg.norm(d)) + 1e-8
    return d / norm


def patch_edge_ratio(patch: np.ndarray) -> float:
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > 20.0).mean())


def extract_candidates_for_char(
    img: np.ndarray,
    ch: str,
    detector,
    patch_size: int,
    min_ink_ratio: float,
    max_ink_ratio: float,
    min_edge_ratio: float,
) -> List[Candidate]:
    keypoints, desc = detector.detectAndCompute(img, None)
    if desc is None or len(keypoints) == 0:
        return []

    h, w = img.shape[:2]
    r = patch_size // 2
    out: List[Candidate] = []

    for kp, d in zip(keypoints, desc):
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x - r < 0 or y - r < 0 or x + r > w or y + r > h:
            continue

        patch = img[y - r : y + r, x - r : x + r]
        if patch.shape != (patch_size, patch_size):
            continue

        ink_ratio = float((patch < 245).mean())
        if ink_ratio < min_ink_ratio or ink_ratio > max_ink_ratio:
            continue

        edge_ratio = patch_edge_ratio(patch)
        if edge_ratio < min_edge_ratio:
            continue

        out.append(
            Candidate(
                char=ch,
                x=x,
                y=y,
                response=float(kp.response),
                descriptor=descriptor_to_unit(d),
                patch=patch,
                ink_ratio=ink_ratio,
                edge_ratio=edge_ratio,
            )
        )
    return out


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


def pairwise_l2(desc: np.ndarray) -> np.ndarray:
    # desc is expected to be row-normalized; this is stable and fast.
    sq = np.sum(desc * desc, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (desc @ desc.T)
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2 + 1e-12)
    return dist.astype(np.float32, copy=False)


def dedupe_keep_order(seq: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def dedupe_candidates_by_location(cands: List[Candidate]) -> tuple[List[Candidate], int]:
    """Remove exact duplicates by (char, x, y), keeping highest-response entries first."""
    seen = set()
    out: List[Candidate] = []
    removed = 0
    for cand in cands:
        key = (cand.char, int(cand.x), int(cand.y))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(cand)
    return out, removed


def dedupe_candidates_by_similarity(
    cands: List[Candidate],
    cos_threshold: float,
    anchor_limit: int,
) -> tuple[List[Candidate], int]:
    """Greedy descriptor-level dedupe.

    Notes:
    - descriptors are already unit-normalized, so cosine similarity is fast and stable.
    - set cos_threshold<=0 to disable.
    """
    if not cands or cos_threshold <= 0:
        return cands, 0

    anchor_limit = max(1, int(anchor_limit))
    dim = int(cands[0].descriptor.shape[0])
    anchors = np.empty((min(anchor_limit, len(cands)), dim), dtype=np.float32)
    anchor_count = 0

    out: List[Candidate] = []
    removed = 0
    for cand in cands:
        desc = cand.descriptor.astype(np.float32, copy=False)
        if anchor_count > 0:
            sim = anchors[:anchor_count] @ desc
            if float(sim.max()) >= cos_threshold:
                removed += 1
                continue

        out.append(cand)
        if anchor_count < anchors.shape[0]:
            anchors[anchor_count] = desc
            anchor_count += 1

    return out, removed


def init_medoids_farthest(dist: np.ndarray, k: int, seed: int) -> List[int]:
    n = dist.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    medoids = [first]
    while len(medoids) < k:
        d = dist[:, medoids].min(axis=1)
        d[medoids] = -1.0
        medoids.append(int(np.argmax(d)))
    return medoids


def fill_medoids(medoids: List[int], dist: np.ndarray, k: int, seed: int) -> List[int]:
    n = dist.shape[0]
    medoids = dedupe_keep_order(medoids)
    rng = np.random.default_rng(seed)
    while len(medoids) < k:
        if not medoids:
            cand = int(rng.integers(0, n))
        else:
            d = dist[:, medoids].min(axis=1)
            d[medoids] = -1.0
            cand = int(np.argmax(d))
        if cand not in medoids:
            medoids.append(cand)
    return medoids[:k]


def kmedoids_voronoi(dist: np.ndarray, k: int, max_iter: int, seed: int) -> tuple[List[int], float]:
    n = dist.shape[0]
    if n <= k:
        medoids = list(range(n))
        cost = float(dist[:, medoids].min(axis=1).sum())
        return medoids, cost

    medoids = init_medoids_farthest(dist, k, seed)
    for it in range(max_iter):
        assign = np.argmin(dist[:, medoids], axis=1)
        new_medoids: List[int] = []
        for ci in range(k):
            members = np.where(assign == ci)[0]
            if len(members) == 0:
                continue
            sub = dist[np.ix_(members, members)]
            idx = int(np.argmin(sub.sum(axis=1)))
            new_medoids.append(int(members[idx]))

        new_medoids = fill_medoids(new_medoids, dist, k, seed + it + 1)
        if new_medoids == medoids:
            break
        medoids = new_medoids

    cost = float(dist[:, medoids].min(axis=1).sum())
    return medoids, cost


def pick_representative_indices_kmedoids(
    cands: List[Candidate],
    k: int,
    pool_size: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> List[int]:
    if not cands:
        return []

    by_resp = list(np.argsort([-c.response for c in cands]))
    if len(cands) <= k:
        return by_resp

    pool_size = max(k, int(pool_size))
    pool_idx = by_resp[: min(pool_size, len(cands))]
    desc = np.stack([cands[i].descriptor for i in pool_idx], axis=0).astype(np.float32)

    dist = pairwise_l2(desc)

    best_medoids: List[int] | None = None
    best_cost = float("inf")
    for t in range(max(1, n_init)):
        medoids, cost = kmedoids_voronoi(dist, min(k, len(pool_idx)), max_iter=max_iter, seed=seed + t * 997)
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids

    assert best_medoids is not None
    chosen = [pool_idx[i] for i in best_medoids]

    chosen = dedupe_keep_order(chosen)
    if len(chosen) < k:
        for i in by_resp:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= k:
                break

    # Stable order by confidence for easier visual inspection.
    chosen.sort(key=lambda i: cands[i].response, reverse=True)
    return chosen[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--fonts-dir", type=Path, default=Path("fonts"))
    parser.add_argument("--font-list-json", type=Path, default=Path("DataPreparation/FontList.json"))
    parser.add_argument("--font-indices", type=str, default=None, help="Comma-separated font indices from FontList")
    parser.add_argument("--max-fonts", type=int, default=0)

    parser.add_argument("--charset-json", type=Path, default=Path("CharacterData/ReferenceCharList.json"))
    parser.add_argument("--max-chars", type=int, default=0)

    parser.add_argument("--output-dir", type=Path, default=Path("DataPreparation/PartBank"))
    parser.add_argument("--parts-per-font", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--min-ink-ratio", type=float, default=0.02)
    parser.add_argument("--max-ink-ratio", type=float, default=0.85)
    parser.add_argument("--min-edge-ratio", type=float, default=0.03)
    parser.add_argument("--max-candidates", type=int, default=6000)
    parser.add_argument(
        "--location-dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dedupe candidates by exact (char,x,y) before representative selection.",
    )
    parser.add_argument(
        "--sim-dedupe-cos-threshold",
        type=float,
        default=0.995,
        help="Cosine threshold for descriptor-level dedupe. <=0 disables.",
    )
    parser.add_argument(
        "--sim-dedupe-anchor-limit",
        type=int,
        default=1500,
        help="Number of high-response anchors used for similarity dedupe.",
    )

    parser.add_argument("--medoid-pool-size", type=int, default=1200)
    parser.add_argument("--kmedoids-n-init", type=int, default=4)
    parser.add_argument("--kmedoids-max-iter", type=int, default=25)

    parser.add_argument("--detector", type=str, default="sift", choices=["sift", "auto", "orb"])
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--char-size", type=int, default=224)
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    if args.patch_size % 2 != 0:
        raise ValueError("patch-size must be even, e.g. 32 or 64")

    root = args.project_root.resolve()
    fonts_dir = (root / args.fonts_dir).resolve()
    font_list_json = (root / args.font_list_json).resolve() if args.font_list_json else None
    charset_json = (root / args.charset_json).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    font_indices = parse_indices(args.font_indices)
    fonts = collect_fonts(root, fonts_dir, font_list_json, font_indices)
    if args.max_fonts > 0:
        fonts = fonts[: args.max_fonts]

    chars = load_charset(charset_json)
    if args.max_chars > 0:
        chars = chars[: args.max_chars]

    detector = build_detector(args.detector)

    manifest: Dict[str, Dict] = {
        "meta": {
            "parts_per_font": int(args.parts_per_font),
            "patch_size": int(args.patch_size),
            "detector": str(args.detector),
            "charset_size": len(chars),
            "selection_method": "k_medoids",
            "medoid_pool_size": int(args.medoid_pool_size),
            "kmedoids_n_init": int(args.kmedoids_n_init),
            "kmedoids_max_iter": int(args.kmedoids_max_iter),
            "min_ink_ratio": float(args.min_ink_ratio),
            "max_ink_ratio": float(args.max_ink_ratio),
            "min_edge_ratio": float(args.min_edge_ratio),
            "canvas_size": int(args.canvas_size),
            "char_size": int(args.char_size),
        },
        "fonts": {},
    }

    for fi, font_path in enumerate(fonts):
        if not font_path.exists():
            print(f"[skip] missing font file: {font_path}")
            continue

        font_name = font_path.stem
        print(f"[{fi+1}/{len(fonts)}] {font_name}")
        cands: List[Candidate] = []

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

            cands.extend(
                extract_candidates_for_char(
                    img=img,
                    ch=ch,
                    detector=detector,
                    patch_size=args.patch_size,
                    min_ink_ratio=args.min_ink_ratio,
                    max_ink_ratio=args.max_ink_ratio,
                    min_edge_ratio=args.min_edge_ratio,
                )
            )

            if len(cands) > args.max_candidates * 2:
                cands = sorted(cands, key=lambda x: x.response, reverse=True)[: args.max_candidates]

        if not cands:
            print("  [skip] no candidates")
            continue

        cands = sorted(cands, key=lambda x: x.response, reverse=True)[: args.max_candidates]
        loc_removed = 0
        sim_removed = 0
        if args.location_dedupe:
            cands, loc_removed = dedupe_candidates_by_location(cands)
        cands, sim_removed = dedupe_candidates_by_similarity(
            cands,
            cos_threshold=float(args.sim_dedupe_cos_threshold),
            anchor_limit=int(args.sim_dedupe_anchor_limit),
        )
        if loc_removed > 0 or sim_removed > 0:
            print(
                f"  dedupe: location_removed={loc_removed} "
                f"similarity_removed={sim_removed} remain={len(cands)}"
            )

        picked_idx = pick_representative_indices_kmedoids(
            cands,
            k=args.parts_per_font,
            pool_size=args.medoid_pool_size,
            n_init=args.kmedoids_n_init,
            max_iter=args.kmedoids_max_iter,
            seed=args.random_seed,
        )
        picked = [cands[i] for i in picked_idx]

        font_out = output_dir / font_name
        font_out.mkdir(parents=True, exist_ok=True)
        parts_meta: List[Dict] = []

        for i, cand in enumerate(picked):
            img_rgb = np.repeat(cand.patch[..., None], 3, axis=2)
            rel_name = f"part_{i:03d}_U{ord(cand.char):04X}.png"
            out_path = font_out / rel_name
            Image.fromarray(img_rgb).save(out_path)

            parts_meta.append(
                {
                    "path": rel_or_abs(out_path, root),
                    "char": cand.char,
                    "char_code": f"U+{ord(cand.char):04X}",
                    "x": int(cand.x),
                    "y": int(cand.y),
                    "response": float(cand.response),
                    "ink_ratio": float(cand.ink_ratio),
                    "edge_ratio": float(cand.edge_ratio),
                }
            )

        manifest["fonts"][font_name] = {
            "font_path": rel_or_abs(font_path, root),
            "num_candidates": len(cands),
            "num_parts": len(parts_meta),
            "parts": parts_meta,
        }
        print(f"  saved {len(parts_meta)} parts from {len(cands)} candidates")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
