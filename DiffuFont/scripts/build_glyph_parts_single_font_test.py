#!/usr/bin/env python3
"""Generate per-glyph style-aware part patches for one test font.

Refactored pipeline focusing on **style features** (stroke width, endpoint shape,
curvature) while excluding global layout and absolute positions.

Pipeline per 256x256 glyph image:
1) Binarize (Otsu + adaptive, fill only small noise holes, preserve structural holes)
2) Distance transform -> stroke width map
3) Skeletonize on properly stroked mask
4) Classify keypoints: endpoints (degree=1), junctions (crossing>=3)
5) Detect high-curvature skeleton points via local angle change
6) Adaptive count: farthest-point sampling -> [MIN_PARTS, MAX_PARTS]
7) Quality filter: reject low-foreground or overly-clipped patches
8) Extract patch images (grayscale mask crops) + metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None

try:
    from skimage.morphology import thin as _skimage_thin
except ImportError:
    _skimage_thin = None


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 256
MIN_PARTS = 4
MAX_PARTS = 12


def require_cv2() -> None:
    if cv2 is not None:
        return
    raise RuntimeError(
        "OpenCV import failed. Please install a usable OpenCV build "
        "(e.g. opencv-python-headless) or system libGL. "
        f"Original error: {_CV2_IMPORT_ERROR}"
    )


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------

def _neighbors8(y: int, x: int) -> List[Tuple[int, int]]:
    return [
        (y - 1, x),
        (y - 1, x + 1),
        (y, x + 1),
        (y + 1, x + 1),
        (y + 1, x),
        (y + 1, x - 1),
        (y, x - 1),
        (y - 1, x - 1),
    ]


# ---------------------------------------------------------------------------
#  Skeletonization
# ---------------------------------------------------------------------------

def thinning_zhang_suen(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8).copy()
    h, w = img.shape
    changed = True
    while changed:
        changed = False
        for pass_id in (0, 1):
            to_remove: List[Tuple[int, int]] = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] != 1:
                        continue
                    nbr = [img[yy, xx] for yy, xx in _neighbors8(y, x)]
                    n = int(sum(nbr))
                    if n < 2 or n > 6:
                        continue
                    s = 0
                    for i in range(8):
                        if nbr[i] == 0 and nbr[(i + 1) % 8] == 1:
                            s += 1
                    if s != 1:
                        continue
                    p2, p4, p6, p8 = nbr[0], nbr[2], nbr[4], nbr[6]
                    if pass_id == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0
    return img


def skeletonize(mask: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        sk = cv2.ximgproc.thinning((mask > 0).astype(np.uint8) * 255)
        return (sk > 0).astype(np.uint8)
    if _skimage_thin is not None:
        return _skimage_thin((mask > 0)).astype(np.uint8)
    return thinning_zhang_suen(mask)


# ---------------------------------------------------------------------------
#  Character / image loading helpers
# ---------------------------------------------------------------------------

def load_chars(path: Path, max_chars: int) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"char json must be a list: {path}")
    chars: List[str] = []
    seen: Set[str] = set()
    for x in obj:
        if not isinstance(x, str) or len(x) != 1:
            continue
        if x in seen:
            continue
        seen.add(x)
        chars.append(x)
        if len(chars) >= max_chars:
            break
    if not chars:
        raise ValueError("no valid chars loaded")
    return chars


def parse_char_from_stem(stem: str) -> Optional[str]:
    if "@" in stem:
        ch = stem.rsplit("@", 1)[-1]
        if len(ch) == 1:
            return ch
    if len(stem) >= 1:
        ch = stem[-1]
        if len(ch) == 1:
            return ch
    return None


def collect_glyph_images_from_dir(
    glyph_dir: Path,
    max_chars: int,
    explicit_chars: str,
) -> List[Tuple[str, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    files = [p for p in sorted(glyph_dir.iterdir()) if p.is_file() and p.suffix in exts]
    if not files:
        raise RuntimeError(f"no glyph images found in: {glyph_dir}")

    wanted: Optional[Set[str]] = None
    if explicit_chars:
        wanted = set(explicit_chars)

    out: List[Tuple[str, Path]] = []
    seen: Set[str] = set()
    for p in files:
        ch = parse_char_from_stem(p.stem)
        if ch is None:
            continue
        if wanted is not None and ch not in wanted:
            continue
        if ch in seen:
            continue
        seen.add(ch)
        out.append((ch, p))
        if len(out) >= int(max_chars):
            break

    if not out:
        raise RuntimeError(
            "no usable glyph image matched. expected filename pattern like 'Font@X.png' "
            "or ending with char"
        )
    return out


def load_glyph_gray(path: Path, canvas_size: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.shape != (canvas_size, canvas_size):
        arr = cv2.resize(arr, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    return arr


def render_char_gray(ch: str, font_path: Path, canvas: int, margin: int) -> np.ndarray:
    font_size = canvas - 2 * margin
    font = ImageFont.truetype(str(font_path), size=max(8, int(font_size)))
    img = Image.new("L", (canvas, canvas), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - tw) // 2 - bbox[0]
    y = (canvas - th) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


def ensure_black_foreground(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    b = 8
    border = np.concatenate([
        gray[:b, :].ravel(), gray[h - b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, w - b:].ravel(),
    ])
    if float(border.mean()) < 127.0:
        return 255 - gray
    return gray


# ---------------------------------------------------------------------------
#  Binarization -- FIXED: preserve structural holes
# ---------------------------------------------------------------------------

def _hole_components(mask: np.ndarray):
    """Return (bg_mask, labels, stats, border_label_set) for enclosed-hole analysis."""
    m = (mask > 0).astype(np.uint8)
    bg = (m == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    h, w = m.shape
    border_labels: Set[int] = set()
    if n > 0:
        border_labels = set(
            labels[0, :].tolist()
            + labels[h - 1, :].tolist()
            + labels[:, 0].tolist()
            + labels[:, w - 1].tolist()
        )
    return bg, labels, stats, border_labels


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill only small enclosed holes (noise).  Structural holes are preserved."""
    m = (mask > 0).astype(np.uint8).copy()
    if max_hole_area <= 0:
        return m
    _, labels, stats, border_labels = _hole_components(m)
    n = int(stats.shape[0])
    for i in range(1, n):
        if i in border_labels:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= int(max_hole_area):
            m[labels == i] = 1
    return m


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 1
    return out


def preprocess_binary(
    gray: np.ndarray,
    adaptive_block: int = 35,
    adaptive_c: int = 8,
    min_area: int = 30,
    close_kernel: int = 3,
    noise_hole_max_area: int = 120,
) -> np.ndarray:
    """Binarize glyph image, preserving large structural holes.

    Unlike the old version which called fill_all_holes, this only fills
    tiny noise holes (<= noise_hole_max_area pixels).  This keeps stroke
    topology intact for proper skeletonization.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = ensure_black_foreground(gray)

    # --- Dual threshold: Otsu global + adaptive local, combined via OR --------
    _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block = int(adaptive_block)
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    bw_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, int(adaptive_c),
    )

    # Union: take foreground from either method for robustness
    m = ((bw_otsu > 0) | (bw_adapt > 0)).astype(np.uint8)

    # --- Morphological close to bridge tiny gaps ---
    ck = int(max(0, close_kernel))
    if ck > 1:
        if ck % 2 == 0:
            ck += 1
        k = np.ones((ck, ck), dtype=np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # --- Fill only small noise holes; preserve structural holes ----
    m = fill_small_holes(m, max_hole_area=int(noise_hole_max_area))
    m = remove_small_components(m, min_area=min_area)
    return m


# ---------------------------------------------------------------------------
#  Skeleton analysis
# ---------------------------------------------------------------------------

def compute_crossing_numbers(skel: np.ndarray) -> np.ndarray:
    """Crossing number per skeleton pixel (1=endpoint, 2=regular, >=3=junction)."""
    h, w = skel.shape
    out = np.zeros((h, w), dtype=np.int16)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        nbr = []
        for yy, xx in _neighbors8(y, x):
            if 0 <= yy < h and 0 <= xx < w and skel[yy, xx] > 0:
                nbr.append(1)
            else:
                nbr.append(0)
        transitions = 0
        for i in range(8):
            if nbr[i] == 0 and nbr[(i + 1) % 8] == 1:
                transitions += 1
        out[y, x] = int(transitions)
    return out


def _skeleton_neighbors_xy(skel: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
    h, w = skel.shape
    out: List[Tuple[int, int]] = []
    for yy, xx in _neighbors8(y, x):
        if 0 <= yy < h and 0 <= xx < w and skel[yy, xx] > 0:
            out.append((xx, yy))
    return out


# ---------------------------------------------------------------------------
#  Spur pruning
# ---------------------------------------------------------------------------

def prune_short_spurs(skel: np.ndarray, max_spur_len: int, rounds: int) -> np.ndarray:
    """Remove short endpoint->junction branches to suppress spur noise."""
    out = (skel > 0).astype(np.uint8).copy()
    if max_spur_len <= 0 or rounds <= 0:
        return out

    for _ in range(int(rounds)):
        cn = compute_crossing_numbers(out)
        ys, xs = np.where(out > 0)
        removed_any = False

        for x0, y0 in zip(xs.tolist(), ys.tolist()):
            if int(cn[y0, x0]) != 1:
                continue
            path = [(x0, y0)]
            prev: Optional[Tuple[int, int]] = None
            curr = (x0, y0)
            end_cn = 1

            for _step in range(int(max_spur_len) + 2):
                nbrs = _skeleton_neighbors_xy(out, curr[0], curr[1])
                if prev is not None:
                    nbrs = [p for p in nbrs if p != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, curr = curr, nxt
                end_cn = int(cn[curr[1], curr[0]])
                if end_cn != 2:
                    break

            if end_cn >= 3 and len(path) - 1 <= int(max_spur_len):
                for px, py in path[:-1]:
                    out[py, px] = 0
                removed_any = True

        if not removed_any:
            break
    return out


def prune_short_spurs_safe(
    skel: np.ndarray,
    max_spur_len: int,
    rounds: int,
    keep_ratio: float = 0.40,
    min_keep_pixels: int = 24,
) -> np.ndarray:
    raw = (skel > 0).astype(np.uint8)
    pruned = prune_short_spurs(raw, max_spur_len=max_spur_len, rounds=rounds)
    raw_n = int(raw.sum())
    pruned_n = int(pruned.sum())
    if raw_n <= 0:
        return pruned
    if pruned_n < int(max(min_keep_pixels, keep_ratio * raw_n)):
        return raw
    return pruned


# ---------------------------------------------------------------------------
#  Keypoint detection -- simplified & robust
# ---------------------------------------------------------------------------

def _cluster_and_pick(
    points: Sequence[Tuple[int, int]],
    radius: int,
    score_map: np.ndarray,
) -> List[Tuple[int, int]]:
    """Cluster nearby points and pick the one with highest score per cluster."""
    if not points:
        return []
    r2 = float(radius * radius)
    pts = list(points)
    used = [False] * len(pts)
    reps: List[Tuple[int, int]] = []

    for i, p in enumerate(pts):
        if used[i]:
            continue
        stack = [i]
        used[i] = True
        comp: List[int] = []
        while stack:
            j = stack.pop()
            comp.append(j)
            xj, yj = pts[j]
            for k in range(len(pts)):
                if used[k]:
                    continue
                xk, yk = pts[k]
                if float((xj - xk) ** 2 + (yj - yk) ** 2) <= r2:
                    used[k] = True
                    stack.append(k)
        comp_pts = [pts[j] for j in comp]
        best = max(comp_pts, key=lambda q: float(score_map[q[1], q[0]]))
        reps.append(best)

    return reps


def detect_skeleton_keypoints(
    skel: np.ndarray,
    crossing: np.ndarray,
    dist_map: np.ndarray,
    endpoint_cluster_r: int = 6,
    junction_cluster_r: int = 8,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect endpoints (crossing==1) and junctions (crossing>=3), then cluster."""
    ys, xs = np.where(skel > 0)

    raw_ep = [(int(x), int(y)) for x, y in zip(xs, ys) if int(crossing[y, x]) == 1]
    raw_jn = [(int(x), int(y)) for x, y in zip(xs, ys) if int(crossing[y, x]) >= 3]

    endpoints = _cluster_and_pick(raw_ep, radius=endpoint_cluster_r, score_map=dist_map)
    junctions = _cluster_and_pick(raw_jn, radius=junction_cluster_r, score_map=dist_map)

    return endpoints, junctions


def detect_curvature_points(
    skel: np.ndarray,
    crossing: np.ndarray,
    dist_map: np.ndarray,
    angle_thresh: float = 0.45,
    walk_delta: int = 5,
    cluster_radius: int = 8,
) -> List[Tuple[int, int]]:
    """Detect high-curvature points by walking along skeleton chains.

    For each regular skeleton pixel (crossing==2), compute the angle between
    the direction from (delta steps back) and the direction to (delta steps
    forward).  If the curvature exceeds angle_thresh radians, record it.
    """
    h, w = skel.shape
    d = max(1, int(walk_delta))

    def _walk(start_x, start_y, avoid_x, avoid_y, steps):
        """Walk along skeleton from (start_x,start_y) avoiding (avoid_x,avoid_y)."""
        cx, cy = start_x, start_y
        px, py = avoid_x, avoid_y
        for _ in range(steps):
            nbrs = _skeleton_neighbors_xy(skel, cx, cy)
            nbrs = [(nx, ny) for nx, ny in nbrs if not (nx == px and ny == py)]
            if not nbrs:
                return cx, cy
            px, py = cx, cy
            cx, cy = nbrs[0]
        return cx, cy

    raw: List[Tuple[int, int, float]] = []
    ys, xs = np.where(skel > 0)

    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        cn = int(crossing[y0, x0])
        if cn != 2:
            continue
        nbrs = _skeleton_neighbors_xy(skel, x0, y0)
        if len(nbrs) != 2:
            continue
        n1, n2 = nbrs

        # Walk d steps in each direction
        fx1, fy1 = _walk(x0, y0, n2[0], n2[1], d)
        fx2, fy2 = _walk(x0, y0, n1[0], n1[1], d)

        v1 = np.array([fx1 - x0, fy1 - y0], dtype=np.float32)
        v2 = np.array([fx2 - x0, fy2 - y0], dtype=np.float32)
        n1_len = float(np.linalg.norm(v1))
        n2_len = float(np.linalg.norm(v2))
        if n1_len < 1e-6 or n2_len < 1e-6:
            continue
        v1 /= n1_len
        v2 /= n2_len
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        # angle between the two direction vectors: pi = straight, 0 = fold-back
        angle = float(np.arccos(dot))
        # curvature = deviation from straight line
        curvature = np.pi - angle
        if curvature > float(angle_thresh):
            raw.append((x0, y0, curvature))

    if not raw:
        return []

    # Non-maximum suppression via clustering
    score_map = np.zeros((h, w), dtype=np.float32)
    for x, y, k in raw:
        if 0 <= y < h and 0 <= x < w:
            score_map[y, x] = max(score_map[y, x], k)

    pts = [(x, y) for x, y, _ in raw]
    return _cluster_and_pick(pts, radius=int(cluster_radius), score_map=score_map)


# ---------------------------------------------------------------------------
#  Adaptive keypoint selection -- farthest-point sampling
# ---------------------------------------------------------------------------

def _fps_select_indices(
    candidates: List[Tuple[int, int]],
    n: int,
) -> List[int]:
    """Farthest-point sampling returning indices."""
    if len(candidates) <= n:
        return list(range(len(candidates)))

    pts = np.array(candidates, dtype=np.float32)
    selected: List[int] = [0]
    min_dists = np.full(len(pts), np.inf, dtype=np.float32)

    for _ in range(n - 1):
        last = pts[selected[-1]]
        dists = np.sum((pts - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        for s in selected:
            min_dists[s] = -1.0
        nxt = int(np.argmax(min_dists))
        selected.append(nxt)

    return selected


def _add_midstroke_samples(
    skel: np.ndarray,
    existing: Set[Tuple[int, int]],
    need: int,
    dist_map: np.ndarray,
    min_spacing: int = 12,
) -> List[Tuple[int, int]]:
    """Sample additional points along skeleton, far from existing keypoints."""
    if need <= 0:
        return []

    ys, xs = np.where(skel > 0)
    skel_pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if not skel_pts:
        return []

    # Score: prefer thicker strokes + far from existing keypoints
    sp2 = float(min_spacing ** 2)
    scores: List[float] = []
    for x, y in skel_pts:
        min_d2 = float("inf")
        for ex, ey in existing:
            dd = float((x - ex) ** 2 + (y - ey) ** 2)
            if dd < min_d2:
                min_d2 = dd
        scores.append(float(dist_map[y, x]) + 0.05 * min_d2 ** 0.5)

    ranked = sorted(zip(scores, skel_pts), key=lambda t: t[0], reverse=True)
    out: List[Tuple[int, int]] = []
    combined = set(existing)
    for _, (x, y) in ranked:
        if len(out) >= need:
            break
        too_close = False
        for cx, cy in combined:
            if float((x - cx) ** 2 + (y - cy) ** 2) < sp2:
                too_close = True
                break
        if too_close:
            continue
        out.append((x, y))
        combined.add((x, y))

    return out


def adaptive_select_keypoints(
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
    curvature_pts: List[Tuple[int, int]],
    skel: np.ndarray,
    dist_map: np.ndarray,
    min_parts: int = MIN_PARTS,
    max_parts: int = MAX_PARTS,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Select keypoints adaptively within [min_parts, max_parts] range.

    Returns (selected_points, point_types).
    """
    all_pts: List[Tuple[int, int]] = []
    all_types: List[str] = []

    for p in endpoints:
        all_pts.append(p)
        all_types.append("endpoint")
    for p in junctions:
        all_pts.append(p)
        all_types.append("junction")
    for p in curvature_pts:
        # Skip curvature points too close to endpoints/junctions
        too_close = False
        for ep in endpoints + junctions:
            if float((p[0] - ep[0]) ** 2 + (p[1] - ep[1]) ** 2) < 64.0:
                too_close = True
                break
        if not too_close:
            all_pts.append(p)
            all_types.append("curvature")

    total = len(all_pts)

    if total > max_parts:
        # Too many -> farthest-point sampling
        selected_idx = _fps_select_indices(all_pts, max_parts)
        all_pts = [all_pts[i] for i in selected_idx]
        all_types = [all_types[i] for i in selected_idx]
    elif total < min_parts:
        # Too few -> add mid-stroke samples
        existing = set(all_pts)
        extra = _add_midstroke_samples(
            skel, existing, need=min_parts - total, dist_map=dist_map
        )
        for p in extra:
            all_pts.append(p)
            all_types.append("midstroke")

    return all_pts, all_types


# ---------------------------------------------------------------------------
#  Part patch extraction with quality filtering
# ---------------------------------------------------------------------------

@dataclass
class Part:
    ptype: str
    x: int
    y: int
    width: float
    degree: Optional[int]
    patch_mask: np.ndarray
    patch_distance: np.ndarray
    patch_bbox: Tuple[int, int, int, int]
    clipped: bool
    fg_ratio: float = 0.0


def extract_patch(
    arr: np.ndarray, x: int, y: int, size: int, fill: float
) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    r = size // 2
    h, w = arr.shape
    x0, y0, x1, y1 = x - r, y - r, x + r, y + r
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    clipped = (cx0 != x0) or (cy0 != y0) or (cx1 != x1) or (cy1 != y1)

    out = np.full((size, size), fill, dtype=arr.dtype)
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    if cx1 > cx0 and cy1 > cy0:
        out[sy0:sy1, sx0:sx1] = arr[cy0:cy1, cx0:cx1]
    return out, (int(cx0), int(cy0), int(cx1), int(cy1)), clipped


def build_parts(
    keypoints: List[Tuple[int, int]],
    types: List[str],
    crossing: np.ndarray,
    mask: np.ndarray,
    dist_map: np.ndarray,
    patch_size: int,
    min_fg_ratio: float = 0.08,
) -> List[Part]:
    """Build Part objects with quality filtering.

    Patches with foreground ratio below min_fg_ratio are discarded.
    """
    parts: List[Part] = []
    area = float(patch_size * patch_size)

    for (x, y), ptype in zip(keypoints, types):
        pm, bbox, clipped = extract_patch(mask, x, y, patch_size, fill=0)
        pd, _, _ = extract_patch(dist_map, x, y, patch_size, fill=0.0)

        fg = float(np.sum(pm > 0))
        fg_ratio = fg / area if area > 0 else 0.0

        if fg_ratio < min_fg_ratio:
            continue

        deg = int(crossing[y, x]) if ptype == "junction" else None

        parts.append(Part(
            ptype=ptype,
            x=int(x),
            y=int(y),
            width=float(2.0 * float(dist_map[y, x])),
            degree=deg,
            patch_mask=pm.astype(np.uint8),
            patch_distance=pd.astype(np.float32),
            patch_bbox=bbox,
            clipped=bool(clipped),
            fg_ratio=round(fg_ratio, 4),
        ))

    return parts


# ---------------------------------------------------------------------------
#  Debug visualization
# ---------------------------------------------------------------------------

def to_color(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


_TYPE_COLOR = {
    "endpoint": (0, 0, 255),     # red
    "junction": (0, 255, 0),     # green
    "curvature": (255, 0, 0),    # blue
    "midstroke": (255, 255, 0),  # cyan
}


def draw_panel_mask_contour(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = to_color(gray)
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    return vis


def draw_panel_mask_skeleton(mask: np.ndarray, skel: np.ndarray) -> np.ndarray:
    base = np.where(mask > 0, 225, 25).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    sk_vis = cv2.dilate(
        (skel > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    vis[sk_vis > 0] = (0, 165, 255)
    return vis


def draw_panel_keypoints(
    gray: np.ndarray,
    skel: np.ndarray,
    parts: List[Part],
) -> np.ndarray:
    vis = to_color(gray)
    vis[skel > 0] = (180, 220, 255)
    for i, p in enumerate(parts):
        c = _TYPE_COLOR.get(p.ptype, (255, 255, 255))
        cv2.circle(vis, (p.x, p.y), 4, c, -1)
        label = f"{p.ptype[0].upper()}{i}"
        cv2.putText(vis, label, (p.x + 5, p.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    return vis


def draw_panel_patches(
    gray: np.ndarray,
    skel: np.ndarray,
    parts: List[Part],
    patch_size: int,
) -> np.ndarray:
    vis = to_color(gray)
    vis[skel > 0] = (220, 220, 220)
    r = patch_size // 2
    for i, p in enumerate(parts):
        c = _TYPE_COLOR.get(p.ptype, (255, 255, 255))
        x0, y0 = max(0, p.x - r), max(0, p.y - r)
        x1, y1 = min(IMG_SIZE - 1, p.x + r), min(IMG_SIZE - 1, p.y + r)
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 1)
        label = f"{p.ptype[0].upper()}{i} w={p.width:.1f}"
        cv2.putText(vis, label, (x0, max(10, y0 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1, cv2.LINE_AA)
    return vis


def make_debug_2x2(
    gray: np.ndarray,
    mask: np.ndarray,
    skel: np.ndarray,
    parts: List[Part],
    patch_size: int,
) -> np.ndarray:
    p1 = draw_panel_mask_contour(gray, mask)
    p2 = draw_panel_mask_skeleton(mask, skel)
    p3 = draw_panel_keypoints(gray, skel, parts)
    p4 = draw_panel_patches(gray, skel, parts, patch_size)

    cv2.putText(p1, "gray + contour (holes preserved)", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p2, "mask + skeleton", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p3, "keypoints (E=red J=grn C=blue M=cyan)", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p4, "part patches", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    top = np.concatenate([p1, p2], axis=1)
    bot = np.concatenate([p3, p4], axis=1)
    return np.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
#  Serialization
# ---------------------------------------------------------------------------

def serialize_part(part: Part) -> Dict:
    # Export a clean grayscale patch contract: black stroke (0), white background (255).
    patch_gray = np.where(part.patch_mask > 0, 0, 255).astype(np.uint8)
    row = {
        "type": part.ptype,
        "center": [int(part.x), int(part.y)],
        "patch_bbox": list(part.patch_bbox),
        "clipped": bool(part.clipped),
        "width": round(float(part.width), 4),
        "fg_ratio": part.fg_ratio,
        "patch_gray": patch_gray.tolist(),
        "patch_distance": np.round(part.patch_distance.astype(np.float32), 4).tolist(),
    }
    if part.degree is not None:
        row["degree"] = int(part.degree)
    return row


# ---------------------------------------------------------------------------
#  Main per-glyph processing
# ---------------------------------------------------------------------------

def process_one_glyph(
    gray: np.ndarray,
    patch_size: int = 40,
    adaptive_block: int = 35,
    adaptive_c: int = 8,
    min_cc_area: int = 30,
    close_kernel: int = 3,
    noise_hole_max_area: int = 120,
    endpoint_cluster_r: int = 6,
    junction_cluster_r: int = 8,
    curvature_angle_thresh: float = 0.45,
    curvature_walk_delta: int = 5,
    curvature_cluster_r: int = 8,
    spur_max_len: int = 4,
    spur_rounds: int = 2,
    min_parts: int = MIN_PARTS,
    max_parts: int = MAX_PARTS,
    min_fg_ratio: float = 0.05,
) -> Tuple[Dict, np.ndarray]:
    """Process one glyph: binarize -> skeleton -> keypoints -> patches."""

    # 1) Binarize (preserve structural holes)
    mask = preprocess_binary(
        gray,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        min_area=min_cc_area,
        close_kernel=close_kernel,
        noise_hole_max_area=noise_hole_max_area,
    )

    # 2) Distance transform
    dist_map = cv2.distanceTransform(
        (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
    )

    # 3) Skeletonize + prune spurs
    skel = skeletonize(mask)
    skel = prune_short_spurs_safe(
        skel, max_spur_len=spur_max_len, rounds=spur_rounds
    )

    # 4) Crossing numbers for keypoint classification
    crossing = compute_crossing_numbers(skel)

    # 5) Detect structural keypoints
    endpoints, junctions = detect_skeleton_keypoints(
        skel, crossing, dist_map,
        endpoint_cluster_r=endpoint_cluster_r,
        junction_cluster_r=junction_cluster_r,
    )

    # 6) Detect curvature keypoints
    curvature_pts = detect_curvature_points(
        skel, crossing, dist_map,
        angle_thresh=curvature_angle_thresh,
        walk_delta=curvature_walk_delta,
        cluster_radius=curvature_cluster_r,
    )

    # 7) Adaptive selection within [min_parts, max_parts]
    sel_pts, sel_types = adaptive_select_keypoints(
        endpoints, junctions, curvature_pts,
        skel, dist_map,
        min_parts=min_parts,
        max_parts=max_parts,
    )

    # 8) Extract patches with quality filtering
    parts = build_parts(
        sel_pts, sel_types,
        crossing, mask, dist_map,
        patch_size=patch_size,
        min_fg_ratio=min_fg_ratio,
    )

    # Build output dict
    parts_by_type: Dict[str, List[Dict]] = {}
    for p in parts:
        parts_by_type.setdefault(p.ptype, []).append(serialize_part(p))

    out = {
        "endpoints": parts_by_type.get("endpoint", []),
        "corners": parts_by_type.get("curvature", []),
        "junctions": parts_by_type.get("junction", []),
        "midstrokes": parts_by_type.get("midstroke", []),
        "all_parts": [serialize_part(p) for p in parts],
        "meta": {
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "patch_size": int(patch_size),
            "total_parts": len(parts),
            "parts_by_type": {k: len(v) for k, v in parts_by_type.items()},
            "raw_endpoint_count": len(endpoints),
            "raw_junction_count": len(junctions),
            "raw_curvature_count": len(curvature_pts),
            "params": {
                "adaptive_block": int(adaptive_block),
                "adaptive_c": int(adaptive_c),
                "min_cc_area": int(min_cc_area),
                "close_kernel": int(close_kernel),
                "noise_hole_max_area": int(noise_hole_max_area),
                "endpoint_cluster_r": int(endpoint_cluster_r),
                "junction_cluster_r": int(junction_cluster_r),
                "curvature_angle_thresh": float(curvature_angle_thresh),
                "curvature_walk_delta": int(curvature_walk_delta),
                "curvature_cluster_r": int(curvature_cluster_r),
                "spur_max_len": int(spur_max_len),
                "spur_rounds": int(spur_rounds),
                "min_parts": int(min_parts),
                "max_parts": int(max_parts),
                "min_fg_ratio": float(min_fg_ratio),
            },
        },
    }

    debug_img = make_debug_2x2(gray, mask, skel, parts, patch_size)
    return out, debug_img


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    require_cv2()
    ap = argparse.ArgumentParser(
        description="Generate style-aware part patches for one font (refactored)."
    )
    ap.add_argument("--project-root", type=Path, default=Path("."))

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--font-path", type=Path, default=None,
                     help="Path to a TTF/OTF font file")
    src.add_argument("--glyph-dir", type=Path, default=None,
                     help="Directory of pre-generated glyph images")

    ap.add_argument("--char-json", type=Path,
                    default=Path("CharacterData/ReferenceCharList.json"))
    ap.add_argument("--max-chars", type=int, default=20)
    ap.add_argument("--chars", type=str, default="",
                    help="Optional explicit chars string, overrides --char-json")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("DataPreparation/PartBank_single_font_test"))

    ap.add_argument("--canvas-size", type=int, default=256)
    ap.add_argument("--render-margin", type=int, default=24)
    ap.add_argument("--patch-size", type=int, default=40)

    # Binarization
    ap.add_argument("--adaptive-block", type=int, default=35)
    ap.add_argument("--adaptive-c", type=int, default=8)
    ap.add_argument("--min-cc-area", type=int, default=30)
    ap.add_argument("--close-kernel", type=int, default=3)
    ap.add_argument("--noise-hole-max-area", type=int, default=120,
                    help="Max area of noise holes to fill (structural holes preserved)")

    # Keypoint detection
    ap.add_argument("--endpoint-cluster-r", type=int, default=6)
    ap.add_argument("--junction-cluster-r", type=int, default=8)
    ap.add_argument("--curvature-angle-thresh", type=float, default=0.45)
    ap.add_argument("--curvature-walk-delta", type=int, default=5)
    ap.add_argument("--curvature-cluster-r", type=int, default=8)

    # Skeleton
    ap.add_argument("--spur-max-len", type=int, default=4)
    ap.add_argument("--spur-rounds", type=int, default=2)

    # Adaptive count
    ap.add_argument("--min-parts", type=int, default=MIN_PARTS)
    ap.add_argument("--max-parts", type=int, default=MAX_PARTS)
    ap.add_argument("--min-fg-ratio", type=float, default=0.05,
                    help="Minimum foreground ratio in patch (0-1)")

    args = ap.parse_args()

    if args.canvas_size != 256:
        raise ValueError("This test script currently expects --canvas-size=256")
    if args.patch_size <= 0 or args.patch_size % 2 != 0:
        raise ValueError("--patch-size must be positive and even")

    root = args.project_root.resolve()
    out_root = (root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    font_path: Optional[Path] = None
    glyph_pairs: List[Tuple[str, Path]] = []
    chars: List[str] = []
    source_mode: str

    if args.glyph_dir is not None:
        glyph_dir = (root / args.glyph_dir).resolve()
        if not glyph_dir.exists():
            raise FileNotFoundError(f"glyph dir not found: {glyph_dir}")
        glyph_pairs = collect_glyph_images_from_dir(
            glyph_dir=glyph_dir,
            max_chars=int(args.max_chars),
            explicit_chars=str(args.chars),
        )
        chars = [ch for ch, _ in glyph_pairs]
        source_mode = "glyph_dir"
        font_name = glyph_dir.name
    else:
        font_path = (root / args.font_path).resolve()
        if not font_path.exists():
            raise FileNotFoundError(f"font not found: {font_path}")
        if args.chars:
            seen: Set[str] = set()
            for ch in args.chars:
                if ch not in seen:
                    seen.add(ch)
                    chars.append(ch)
                if len(chars) >= int(args.max_chars):
                    break
        else:
            char_json = (root / args.char_json).resolve()
            chars = load_chars(char_json, max_chars=int(args.max_chars))
        source_mode = "font_render"
        font_name = font_path.stem

    font_out = out_root / font_name
    font_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "meta": {
            "source_mode": source_mode,
            "font_path": str(font_path) if font_path else None,
            "glyph_dir": str((root / args.glyph_dir).resolve()) if args.glyph_dir else None,
            "font_name": font_name,
            "num_chars": len(chars),
            "chars": chars,
            "pipeline": "binarize(preserve_holes)->distance->skeleton->keypoints->adaptive_select->quality_filter->patch",
        },
        "glyphs": {},
    }

    total_parts = 0
    char_part_counts: List[int] = []

    for idx, ch in enumerate(chars):
        if source_mode == "glyph_dir":
            _, img_path = glyph_pairs[idx]
            gray = load_glyph_gray(img_path, canvas_size=int(args.canvas_size))
        else:
            gray = render_char_gray(
                ch, font_path=font_path,
                canvas=int(args.canvas_size),
                margin=int(args.render_margin),
            )

        glyph_json, debug_img = process_one_glyph(
            gray=gray,
            patch_size=int(args.patch_size),
            adaptive_block=int(args.adaptive_block),
            adaptive_c=int(args.adaptive_c),
            min_cc_area=int(args.min_cc_area),
            close_kernel=int(args.close_kernel),
            noise_hole_max_area=int(args.noise_hole_max_area),
            endpoint_cluster_r=int(args.endpoint_cluster_r),
            junction_cluster_r=int(args.junction_cluster_r),
            curvature_angle_thresh=float(args.curvature_angle_thresh),
            curvature_walk_delta=int(args.curvature_walk_delta),
            curvature_cluster_r=int(args.curvature_cluster_r),
            spur_max_len=int(args.spur_max_len),
            spur_rounds=int(args.spur_rounds),
            min_parts=int(args.min_parts),
            max_parts=int(args.max_parts),
            min_fg_ratio=float(args.min_fg_ratio),
        )

        n_parts = glyph_json["meta"]["total_parts"]
        total_parts += n_parts
        char_part_counts.append(n_parts)

        gid = f"{idx:03d}_U{ord(ch):04X}"
        glyph_json["glyph_id"] = gid
        glyph_json["char"] = ch
        glyph_json["char_code"] = f"U+{ord(ch):04X}"

        glyph_dir_out = font_out / gid
        glyph_dir_out.mkdir(parents=True, exist_ok=True)

        Image.fromarray(gray, mode="L").save(glyph_dir_out / f"{gid}.png")
        Image.fromarray(
            cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), mode="RGB"
        ).save(glyph_dir_out / f"{gid}_debug_2x2.png")
        (glyph_dir_out / f"{gid}_parts.json").write_text(
            json.dumps(glyph_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest["glyphs"][gid] = {
            "char": ch,
            "char_code": f"U+{ord(ch):04X}",
            "source_image": str(glyph_pairs[idx][1]) if source_mode == "glyph_dir" else None,
            "parts_json": str((glyph_dir_out / f"{gid}_parts.json").relative_to(out_root)),
            "glyph_image": str((glyph_dir_out / f"{gid}.png").relative_to(out_root)),
            "debug_image": str((glyph_dir_out / f"{gid}_debug_2x2.png").relative_to(out_root)),
            "counts": glyph_json["meta"]["parts_by_type"],
            "total_parts": n_parts,
        }

        type_str = " ".join(
            f"{k[0].upper()}={v}" for k, v in glyph_json["meta"]["parts_by_type"].items() if v > 0
        )
        print(f"[{idx + 1}/{len(chars)}] {ch}  total={n_parts}  {type_str}")

    manifest_path = font_out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary stats
    if char_part_counts:
        avg = sum(char_part_counts) / len(char_part_counts)
        mn, mx = min(char_part_counts), max(char_part_counts)
        print(f"\nSummary: {len(chars)} chars, {total_parts} total parts")
        print(f"Parts per char: avg={avg:.1f}, min={mn}, max={mx}")
        print(f"Target range: [{args.min_parts}, {args.max_parts}]")

    print(f"Output: {font_out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
