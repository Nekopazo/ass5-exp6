#!/usr/bin/env python3
"""Build PartBank from pre-generated glyph images with strict geometry constraints.

Framework (reconstructed):
1) No geometric pre-normalization (use original glyph image directly).
2) Keypoints anchored on foreground skeleton only.
3) Center constraints + center-connected-component isolation.
4) Hole rejection for hollow structures.
5) Quota selection by keypoint type + cosine dedupe.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
FEATURE_TYPES: Tuple[str, ...] = ("terminal", "junction")


@dataclass
class Candidate:
    char: str
    x: int
    y: int
    feature_type: str
    component_id: int
    component_area: int
    score: float
    patch: np.ndarray
    descriptor: np.ndarray
    ink_ratio: float
    edge_ratio: float
    center_comp_ratio: float
    hole_ratio: float
    width_std: float
    width_grad: float
    skeleton_span: float


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def load_reference_chars(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"reference char json must be a list: {path}")
    out: List[str] = []
    seen = set()
    for x in obj:
        if not isinstance(x, str) or len(x) != 1:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    if not out:
        raise ValueError(f"no valid chars in reference char json: {path}")
    return out


def parse_type_ratios(raw: str) -> Dict[str, float]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--type-ratios must have 2 values: terminal,junction")
    vals = [max(0.0, float(x)) for x in parts]
    s = float(sum(vals))
    if s <= 0.0:
        raise ValueError("--type-ratios sum must be > 0")
    vals = [v / s for v in vals]
    return {k: v for k, v in zip(FEATURE_TYPES, vals)}


def parse_char_from_stem(stem: str) -> str | None:
    if "@" not in stem:
        return None
    ch = stem.rsplit("@", 1)[-1]
    return ch if len(ch) == 1 else None


def build_glyph_index(font_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(font_dir.glob("*")):
        if p.suffix not in IMG_EXTS:
            continue
        ch = parse_char_from_stem(p.stem)
        if ch is None:
            continue
        if ch not in out:
            out[ch] = p
    return out


def to_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _neighbors8(y: int, x: int) -> List[Tuple[int, int]]:
    return [
        (y - 1, x), (y - 1, x + 1), (y, x + 1), (y + 1, x + 1),
        (y + 1, x), (y + 1, x - 1), (y, x - 1), (y - 1, x - 1),
    ]


def thinning_zhang_suen(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8).copy()
    h, w = img.shape
    changed = True
    while changed:
        changed = False
        for pass_id in (0, 1):
            rem: List[Tuple[int, int]] = []
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
                    rem.append((y, x))
            if rem:
                changed = True
                for y, x in rem:
                    img[y, x] = 0
    return img


def skeletonize(mask: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        sk = cv2.ximgproc.thinning((mask > 0).astype(np.uint8) * 255)
        return (sk > 0).astype(np.uint8)
    return thinning_zhang_suen(mask)


def nms_points(points: List[Tuple[int, int, float]], min_dist: int, max_k: int) -> List[Tuple[int, int, float]]:
    if not points or max_k <= 0:
        return []
    md2 = float(min_dist * min_dist)
    out: List[Tuple[int, int, float]] = []
    for x, y, s in sorted(points, key=lambda t: t[2], reverse=True):
        ok = True
        for px, py, _ in out:
            dx = float(x - px)
            dy = float(y - py)
            if dx * dx + dy * dy < md2:
                ok = False
                break
        if ok:
            out.append((x, y, s))
            if len(out) >= max_k:
                break
    return out


def detect_skeleton_keypoints(
    fg: np.ndarray,
    skel: np.ndarray,
    dist_fg: np.ndarray,
    point_min_dist: int,
    max_points_per_type: int,
) -> List[Tuple[int, int, str, float, float, float]]:
    """Return points: (x,y,type,base_strength,width_std,width_grad)."""
    ys, xs = np.where(skel > 0)
    if ys.size == 0:
        return []

    per: Dict[str, List[Tuple[int, int, float, float, float]]] = {k: [] for k in FEATURE_TYPES}

    h, w = skel.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        nbr: List[Tuple[int, int]] = []
        for yy, xx in _neighbors8(y, x):
            if 0 <= yy < h and 0 <= xx < w and skel[yy, xx] > 0:
                nbr.append((yy, xx))
        deg = len(nbr)
        base = float(max(1e-6, dist_fg[y, x]))

        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        x0 = max(0, x - 2)
        x1 = min(w, x + 3)
        local = dist_fg[y0:y1, x0:x1]
        width_std = float(local.std()) if local.size > 0 else 0.0

        if deg <= 1:
            per["terminal"].append((x, y, base, width_std, 0.0))
            continue
        if deg >= 3:
            # Width gradient around junction can express stroke fusion style.
            nbr_w = [float(dist_fg[yy, xx]) for yy, xx in nbr] if nbr else [float(dist_fg[y, x])]
            wgrad = float(np.std(np.array(nbr_w, dtype=np.float32)))
            per["junction"].append((x, y, base + 0.2 * deg + 0.1 * wgrad, width_std, wgrad))

    out: List[Tuple[int, int, str, float, float, float]] = []
    for t in FEATURE_TYPES:
        pts = [(x, y, s) for x, y, s, _, _ in per[t]]
        keep = nms_points(pts, min_dist=int(point_min_dist), max_k=int(max_points_per_type))
        keep_set = {(x, y) for x, y, _ in keep}
        for x, y, s, wstd, wgrad in per[t]:
            if (x, y) in keep_set:
                out.append((x, y, t, s, wstd, wgrad))
    return out


def extract_patch(gray: np.ndarray, fg: np.ndarray, cx: int, cy: int, patch_size: int) -> Tuple[np.ndarray, np.ndarray] | None:
    h, w = gray.shape
    r = patch_size // 2
    if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
        return None
    p_gray = gray[cy - r : cy + r, cx - r : cx + r]
    p_fg = fg[cy - r : cy + r, cx - r : cx + r]
    if p_gray.shape != (patch_size, patch_size):
        return None
    return p_gray, p_fg


def center_window_ink_ratio(p_fg: np.ndarray, k: int) -> float:
    h, w = p_fg.shape
    cy, cx = h // 2, w // 2
    r = int(max(1, k // 2))
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    win = p_fg[y0:y1, x0:x1]
    return float(win.mean()) if win.size > 0 else 0.0


def center_dist_to_fg(p_fg: np.ndarray) -> float:
    h, w = p_fg.shape
    cy, cx = h // 2, w // 2
    inv = (p_fg == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return float(dist[cy, cx])


def isolate_center_component(
    p_gray: np.ndarray,
    p_fg: np.ndarray,
    min_fragment_keep_ratio: float,
    min_center_comp_ratio: float,
    max_center_comp_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, float] | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(p_fg.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None

    h, w = p_fg.shape
    cy, cx = h // 2, w // 2
    center_label = int(labels[cy, cx])
    if center_label == 0:
        # More stable than nearest-pixel fallback: vote labels in center window.
        r = 5  # 11x11
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        win = labels[y0:y1, x0:x1]
        fg_win = win[win > 0]
        if fg_win.size > 0:
            uniq, cnt = np.unique(fg_win, return_counts=True)
            center_label = int(uniq[int(np.argmax(cnt))])
        else:
            ys, xs = np.where(p_fg > 0)
            if ys.size <= 0:
                return None
            d2 = (ys - cy) * (ys - cy) + (xs - cx) * (xs - cx)
            i = int(np.argmin(d2))
            center_label = int(labels[int(ys[i]), int(xs[i])])
        if center_label == 0:
            return None

    mask_center = (labels == center_label)
    fg_sum = int(p_fg.sum())
    center_sum = int(mask_center.sum())
    if fg_sum <= 0 or center_sum <= 0:
        return None

    outside = fg_sum - center_sum
    if float(outside) / float(fg_sum) > float(1.0 - min_fragment_keep_ratio):
        return None

    comp_ratio = float(center_sum) / float(h * w)
    if comp_ratio < float(min_center_comp_ratio) or comp_ratio > float(max_center_comp_ratio):
        return None

    out_fg = np.zeros_like(p_fg, dtype=np.uint8)
    out_fg[mask_center] = 1
    out_gray = np.full_like(p_gray, 255)
    out_gray[mask_center] = p_gray[mask_center]
    return out_gray, out_fg, comp_ratio


def hole_ratio(mask_fg: np.ndarray) -> float:
    bg = (mask_fg == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if n <= 1:
        return 0.0

    h, w = bg.shape
    border_labels = set(labels[0, :].tolist() + labels[h - 1, :].tolist() + labels[:, 0].tolist() + labels[:, w - 1].tolist())
    hole = 0
    for i in range(1, n):
        if i in border_labels:
            continue
        hole += int(stats[i, cv2.CC_STAT_AREA])
    return float(hole) / float(max(1, h * w))


def patch_edge_ratio(patch_gray: np.ndarray) -> float:
    gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > 20.0).mean())


def tight_bbox_fill(mask_fg: np.ndarray) -> float:
    ys, xs = np.where(mask_fg > 0)
    if ys.size <= 0:
        return 0.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    area = max(1, (y1 - y0) * (x1 - x0))
    return float(mask_fg[y0:y1, x0:x1].sum()) / float(area)


def skeleton_length_ratio(mask_fg: np.ndarray) -> float:
    sk = skeletonize(mask_fg)
    return float(sk.sum()) / float(max(1, mask_fg.size))


def skeleton_span_ratio(mask_fg: np.ndarray) -> float:
    sk = skeletonize(mask_fg)
    ys, xs = np.where(sk > 0)
    if ys.size <= 1:
        return 0.0
    h, w = mask_fg.shape
    span_x = float(xs.max() - xs.min()) / float(max(1, w - 1))
    span_y = float(ys.max() - ys.min()) / float(max(1, h - 1))
    return float(max(span_x, span_y))


def patch_descriptor_hog_like(patch_gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(patch_gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    bins = 8
    hist = np.zeros((bins,), dtype=np.float32)
    bid = np.floor((ang % (2.0 * np.pi)) / (2.0 * np.pi) * bins).astype(np.int32)
    for b in range(bins):
        hist[b] = float(mag[bid == b].sum())

    feat = np.concatenate([small.reshape(-1), hist], axis=0).astype(np.float32)
    return feat / (float(np.linalg.norm(feat)) + 1e-8)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def dedupe_by_cosine(cands: List[Candidate], threshold: float, anchor_limit: int) -> List[Candidate]:
    if not cands:
        return []
    out: List[Candidate] = []
    anchors: List[np.ndarray] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        if anchors:
            sims = [cosine_sim(c.descriptor, a) for a in anchors]
            if max(sims) >= float(threshold):
                continue
        out.append(c)
        if len(anchors) < max(1, int(anchor_limit)):
            anchors.append(c.descriptor)
    return out


def dedupe_by_cosine_per_type(cands: List[Candidate], threshold: float, anchor_limit: int) -> List[Candidate]:
    out: List[Candidate] = []
    by_type: Dict[str, List[Candidate]] = {t: [] for t in FEATURE_TYPES}
    for c in cands:
        by_type.setdefault(c.feature_type, []).append(c)
    for t in FEATURE_TYPES:
        out.extend(dedupe_by_cosine(by_type.get(t, []), threshold=float(threshold), anchor_limit=int(anchor_limit)))
    return out


def enforce_max_parts_per_char(cands: List[Candidate], max_per_char: int) -> List[Candidate]:
    if max_per_char <= 0:
        return cands
    out: List[Candidate] = []
    cnt: Dict[str, int] = {}
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        k = c.char
        v = cnt.get(k, 0)
        if v >= int(max_per_char):
            continue
        cnt[k] = v + 1
        out.append(c)
    return out


def split_quota_counts(k: int, ratios: Dict[str, float], avail: Dict[str, int]) -> Dict[str, int]:
    cnt = {t: 0 for t in FEATURE_TYPES}
    raw = {t: float(k) * float(ratios.get(t, 0.0)) for t in FEATURE_TYPES}
    for t in FEATURE_TYPES:
        cnt[t] = min(int(np.floor(raw[t])), int(avail.get(t, 0)))

    remain = int(k - sum(cnt.values()))
    if remain > 0:
        order = sorted(FEATURE_TYPES, key=lambda t: (raw[t] - np.floor(raw[t])), reverse=True)
        for t in order:
            if remain <= 0:
                break
            room = int(avail.get(t, 0)) - cnt[t]
            take = min(remain, max(0, room))
            cnt[t] += take
            remain -= take
    if remain > 0:
        order = sorted(FEATURE_TYPES, key=lambda t: avail.get(t, 0), reverse=True)
        for t in order:
            if remain <= 0:
                break
            room = int(avail.get(t, 0)) - cnt[t]
            if room <= 0:
                continue
            take = min(remain, room)
            cnt[t] += take
            remain -= take
    return cnt


def pick_diverse(pool: Sequence[Candidate], k: int, already: Sequence[Candidate], seed: int) -> List[Candidate]:
    if k <= 0 or not pool:
        return []
    ranked = sorted(pool, key=lambda x: x.score, reverse=True)
    rng = np.random.default_rng(seed)

    chosen: List[Candidate] = []
    chosen_ids = set()

    def dmin(c: Candidate, refs: Sequence[Candidate]) -> float:
        if not refs:
            return 1.0
        v = 1.0
        for r in refs:
            v = min(v, 1.0 - cosine_sim(c.descriptor, r.descriptor))
        return v

    while len(chosen) < k:
        best_i = -1
        best_v = -1.0
        refs = list(already) + chosen
        for i, c in enumerate(ranked):
            if i in chosen_ids:
                continue
            v = 0.7 * float(c.score) + 0.3 * float(dmin(c, refs)) + float(rng.uniform(0, 1e-6))
            if v > best_v:
                best_v = v
                best_i = i
        if best_i < 0:
            break
        chosen_ids.add(best_i)
        chosen.append(ranked[best_i])
    return chosen


def select_with_quota(cands: List[Candidate], k: int, ratios: Dict[str, float], seed: int) -> List[Candidate]:
    if not cands or k <= 0:
        return []
    if len(cands) <= k:
        return sorted(cands, key=lambda x: x.score, reverse=True)

    by_type: Dict[str, List[Candidate]] = {t: [] for t in FEATURE_TYPES}
    for c in cands:
        by_type[c.feature_type].append(c)
    avail = {t: len(by_type[t]) for t in FEATURE_TYPES}
    quota = split_quota_counts(k, ratios, avail)

    chosen: List[Candidate] = []
    for i, t in enumerate(FEATURE_TYPES):
        chosen.extend(pick_diverse(by_type[t], quota[t], chosen, seed + 101 * (i + 1)))

    if len(chosen) < k:
        remain = [c for c in cands if c not in chosen]
        chosen.extend(pick_diverse(remain, k - len(chosen), chosen, seed + 7777))

    return sorted(chosen, key=lambda x: x.score, reverse=True)[:k]


def build_candidates_for_glyph(
    gray: np.ndarray,
    fg: np.ndarray,
    ch: str,
    patch_size: int,
    point_min_dist: int,
    max_points_per_type: int,
    center_window_size: int,
    min_center_ink: float,
    max_center_dist: float,
    min_fragment_keep_ratio: float,
    min_center_comp_ratio: float,
    max_center_comp_ratio: float,
    max_hole_ratio: float,
    min_ink_ratio: float,
    max_ink_ratio: float,
    min_edge_ratio: float,
    max_tight_bbox_fill: float,
    max_skeleton_len_ratio: float,
    max_skeleton_span_terminal: float,
    max_skeleton_span_junction: float,
    junction_max_center_comp_ratio: float,
    junction_max_hole_ratio: float,
    junction_min_edge_ratio: float,
    junction_max_tight_bbox_fill: float,
    junction_max_skeleton_len_ratio: float,
) -> List[Candidate]:
    dist_fg = cv2.distanceTransform((fg > 0).astype(np.uint8), cv2.DIST_L2, 3)
    skel = skeletonize(fg)
    keypoints = detect_skeleton_keypoints(
        fg=fg,
        skel=skel,
        dist_fg=dist_fg,
        point_min_dist=int(point_min_dist),
        max_points_per_type=int(max_points_per_type),
    )
    if not keypoints:
        return []

    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)

    out: List[Candidate] = []
    for x, y, t, base_strength, width_std, width_grad in keypoints:
        comp_id = int(labels[y, x]) if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0
        comp_area = int(stats[comp_id, cv2.CC_STAT_AREA]) if 0 < comp_id < ncc else 0

        ret = extract_patch(gray, fg, x, y, patch_size)
        if ret is None:
            continue
        p_gray, p_fg = ret

        if center_window_ink_ratio(p_fg, int(center_window_size)) < float(min_center_ink):
            continue
        if center_dist_to_fg(p_fg) > float(max_center_dist):
            continue

        local_max_center_comp_ratio = (
            float(junction_max_center_comp_ratio) if t == "junction" else float(max_center_comp_ratio)
        )
        local_max_hole_ratio = float(junction_max_hole_ratio) if t == "junction" else float(max_hole_ratio)
        local_min_edge_ratio = float(junction_min_edge_ratio) if t == "junction" else float(min_edge_ratio)
        local_max_tight_bbox_fill = (
            float(junction_max_tight_bbox_fill) if t == "junction" else float(max_tight_bbox_fill)
        )
        local_max_skeleton_len_ratio = (
            float(junction_max_skeleton_len_ratio) if t == "junction" else float(max_skeleton_len_ratio)
        )

        iso = isolate_center_component(
            p_gray,
            p_fg,
            min_fragment_keep_ratio=float(min_fragment_keep_ratio),
            min_center_comp_ratio=float(min_center_comp_ratio),
            max_center_comp_ratio=float(local_max_center_comp_ratio),
        )
        if iso is None:
            continue
        p_gray_iso, p_fg_iso, comp_ratio = iso

        hratio = hole_ratio(p_fg_iso)
        if hratio > float(local_max_hole_ratio):
            continue

        ink = float(p_fg_iso.mean())
        if ink < float(min_ink_ratio) or ink > float(max_ink_ratio):
            continue

        edge = patch_edge_ratio(p_gray_iso)
        if edge < float(local_min_edge_ratio):
            continue

        tfill = tight_bbox_fill(p_fg_iso)
        if tfill > float(local_max_tight_bbox_fill):
            continue

        sklen = skeleton_length_ratio(p_fg_iso)
        if sklen > float(local_max_skeleton_len_ratio):
            continue

        skspan = skeleton_span_ratio(p_fg_iso)
        span_limit = {
            "terminal": float(max_skeleton_span_terminal),
            "junction": float(max_skeleton_span_junction),
        }[t]
        if skspan > span_limit:
            continue

        # Score: simple and pragmatic after hard constraints.
        type_bonus = {"terminal": 1.06, "junction": 1.10}[t]
        score = (
            0.34 * edge
            + 0.18 * float(base_strength)
            + 0.12 * float(width_std)
            + 0.12 * float(width_grad)
            + 0.14 * float(type_bonus)
            - 0.08 * max(0.0, ink - 0.38)
            - 0.08 * max(0.0, tfill - 0.72)
            - 0.08 * max(0.0, sklen - 0.18)
        )

        out.append(
            Candidate(
                char=ch,
                x=int(x),
                y=int(y),
                feature_type=str(t),
                component_id=int(comp_id),
                component_area=int(comp_area),
                score=float(score),
                patch=p_gray_iso,
                descriptor=patch_descriptor_hog_like(p_gray_iso),
                ink_ratio=float(ink),
                edge_ratio=float(edge),
                center_comp_ratio=float(comp_ratio),
                hole_ratio=float(hratio),
                width_std=float(width_std),
                width_grad=float(width_grad),
                skeleton_span=float(skspan),
            )
        )
    return out


def _process_font_chunk(
    font_dir_str: str,
    ref_chars_chunk: List[str],
    cfg: Dict[str, Any],
) -> Tuple[str, List[Candidate], int, int, bool, float, float]:
    """Worker entry for one (font, char-chunk)."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    font_dir = Path(font_dir_str)
    font_name = font_dir.name
    glyph_index = build_glyph_index(font_dir)
    if not glyph_index:
        return font_name, [], 0, 0, False, float(cfg["min_ink_ratio"]), float(cfg["max_ink_ratio"])

    cands: List[Candidate] = []
    chars_found = 0
    chars_used = 0
    cap = int(cfg["max_candidates"])

    for ch in ref_chars_chunk:
        ip = glyph_index.get(ch)
        if ip is None:
            continue
        chars_found += 1

        try:
            gray = to_gray(ip)
        except Exception:
            continue

        fg = (gray < int(cfg["binary_threshold"])).astype(np.uint8)
        cc = build_candidates_for_glyph(
            gray=gray,
            fg=fg,
            ch=ch,
            patch_size=int(cfg["patch_size"]),
            point_min_dist=int(cfg["point_min_dist"]),
            max_points_per_type=int(cfg["max_points_per_type"]),
            center_window_size=int(cfg["center_window_size"]),
            min_center_ink=float(cfg["min_center_ink"]),
            max_center_dist=float(cfg["max_center_dist"]),
            min_fragment_keep_ratio=float(cfg["min_fragment_keep_ratio"]),
            min_center_comp_ratio=float(cfg["min_center_comp_ratio"]),
            max_center_comp_ratio=float(cfg["max_center_comp_ratio"]),
            max_hole_ratio=float(cfg["max_hole_ratio"]),
            min_ink_ratio=float(cfg["min_ink_ratio"]),
            max_ink_ratio=float(cfg["max_ink_ratio"]),
            min_edge_ratio=float(cfg["min_edge_ratio"]),
            max_tight_bbox_fill=float(cfg["max_tight_bbox_fill"]),
            max_skeleton_len_ratio=float(cfg["max_skeleton_len_ratio"]),
            max_skeleton_span_terminal=float(cfg["max_skeleton_span_terminal"]),
            max_skeleton_span_junction=float(cfg["max_skeleton_span_junction"]),
            junction_max_center_comp_ratio=float(cfg["junction_max_center_comp_ratio"]),
            junction_max_hole_ratio=float(cfg["junction_max_hole_ratio"]),
            junction_min_edge_ratio=float(cfg["junction_min_edge_ratio"]),
            junction_max_tight_bbox_fill=float(cfg["junction_max_tight_bbox_fill"]),
            junction_max_skeleton_len_ratio=float(cfg["junction_max_skeleton_len_ratio"]),
        )
        if cc:
            chars_used += 1
            cands.extend(cc)

        if len(cands) > cap * 2:
            cands = sorted(cands, key=lambda x: x.score, reverse=True)[:cap]

    if len(cands) > cap:
        cands = sorted(cands, key=lambda x: x.score, reverse=True)[:cap]
    return font_name, cands, chars_found, chars_used, True, float(cfg["min_ink_ratio"]), float(cfg["max_ink_ratio"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--glyph-root", type=Path, default=Path("DataPreparation/Generated/TrainFonts"))
    p.add_argument("--font-list-json", type=Path, default=Path("DataPreparation/FontList.json"))
    p.add_argument("--max-fonts", type=int, default=0)
    p.add_argument("--reference-char-json", type=Path, default=Path("CharacterData/ReferenceCharList.json"))
    p.add_argument("--max-reference-chars", type=int, default=0)

    p.add_argument("--output-dir", type=Path, default=Path("DataPreparation/PartBank_component_aware"))
    p.add_argument("--parts-per-font", type=int, default=32)
    p.add_argument("--max-candidates", type=int, default=12000)

    p.add_argument("--binary-threshold", type=int, default=245)

    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--point-min-dist", type=int, default=7)
    p.add_argument("--max-points-per-type", type=int, default=20)

    p.add_argument("--center-window-size", type=int, default=11)
    p.add_argument("--min-center-ink", type=float, default=0.15)
    p.add_argument("--max-center-dist", type=float, default=1.5)

    p.add_argument("--min-fragment-keep-ratio", type=float, default=0.90,
                   help="center component must keep at least this fraction of fg mass")
    p.add_argument("--min-center-comp-ratio", type=float, default=0.02)
    p.add_argument("--max-center-comp-ratio", type=float, default=0.35)
    p.add_argument("--max-hole-ratio", type=float, default=0.08)

    p.add_argument("--min-ink-ratio", type=float, default=0.02)
    p.add_argument("--max-ink-ratio", type=float, default=0.50)
    p.add_argument("--min-edge-ratio", type=float, default=0.04)
    p.add_argument("--max-tight-bbox-fill", type=float, default=0.82)
    p.add_argument("--max-skeleton-len-ratio", type=float, default=0.28)
    p.add_argument("--max-skeleton-span-terminal", type=float, default=0.84)
    p.add_argument("--max-skeleton-span-junction", type=float, default=0.93)
    p.add_argument("--junction-max-center-comp-ratio", type=float, default=0.50)
    p.add_argument("--junction-max-hole-ratio", type=float, default=0.12)
    p.add_argument("--junction-min-edge-ratio", type=float, default=0.02)
    p.add_argument("--junction-max-tight-bbox-fill", type=float, default=0.92)
    p.add_argument("--junction-max-skeleton-len-ratio", type=float, default=0.40)

    p.add_argument("--type-ratios", type=str, default="0.6,0.4",
                   help="terminal,junction")
    p.add_argument("--dedupe-cos-threshold", type=float, default=0.85,
                   help="Global cosine dedupe threshold.")
    p.add_argument("--dedupe-cos-threshold-per-type", type=float, default=0.82,
                   help="Class-wise cosine dedupe threshold (stricter).")
    p.add_argument("--dedupe-anchor-limit", type=int, default=4000)
    p.add_argument("--max-parts-per-char", type=int, default=3)
    p.add_argument("--executor", type=str, default="process", choices=["process", "thread"])
    p.add_argument("--char-chunk-size", type=int, default=25)
    p.add_argument("--workers", "--num-threads", dest="workers", type=int, default=48)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    if args.patch_size % 2 != 0:
        raise ValueError("--patch-size must be even")

    ratios = parse_type_ratios(args.type_ratios)

    root = args.project_root.resolve()
    glyph_root = (root / args.glyph_root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_char_path = (root / args.reference_char_json).resolve()
    ref_chars = load_reference_chars(ref_char_path)
    if args.max_reference_chars > 0:
        ref_chars = ref_chars[: int(args.max_reference_chars)]
    print(f"[config] reference chars: {len(ref_chars)} from {ref_char_path}")
    print(f"[config] type ratios(terminal,junction): {ratios}")
    print(
        f"[config] executor={args.executor} workers={int(args.workers)} "
        f"char_chunk_size={int(args.char_chunk_size)}"
    )

    allowed_fonts: set[str] | None = None
    font_list_path = (root / args.font_list_json).resolve()
    if font_list_path.exists():
        font_list = json.loads(font_list_path.read_text(encoding="utf-8"))
        allowed_fonts = {Path(str(x)).stem for x in font_list if isinstance(x, str)}

    font_dirs = [d for d in sorted(glyph_root.iterdir()) if d.is_dir()]
    if allowed_fonts is not None:
        font_dirs = [d for d in font_dirs if d.name in allowed_fonts]
    if args.max_fonts > 0:
        font_dirs = font_dirs[: int(args.max_fonts)]
    if not font_dirs:
        raise RuntimeError(f"No font directories found in: {glyph_root}")

    manifest: Dict[str, Dict] = {
        "meta": {
            "selection_method": "skeleton_center_connected_quota",
            "glyph_root": rel_or_abs(glyph_root, root),
            "reference_char_json": rel_or_abs(ref_char_path, root),
            "reference_char_count": len(ref_chars),
            "parts_per_font": int(args.parts_per_font),
            "patch_size": int(args.patch_size),
            "type_ratios": {k: float(v) for k, v in ratios.items()},
            "dedupe_cos_threshold": float(args.dedupe_cos_threshold),
            "dedupe_cos_threshold_per_type": float(args.dedupe_cos_threshold_per_type),
            "max_parts_per_char": int(args.max_parts_per_char),
            "ink_mode": "absolute",
        },
        "fonts": {},
    }

    worker_cfg: Dict[str, Any] = {
        "binary_threshold": int(args.binary_threshold),
        "patch_size": int(args.patch_size),
        "point_min_dist": int(args.point_min_dist),
        "max_points_per_type": int(args.max_points_per_type),
        "center_window_size": int(args.center_window_size),
        "min_center_ink": float(args.min_center_ink),
        "max_center_dist": float(args.max_center_dist),
        "min_fragment_keep_ratio": float(args.min_fragment_keep_ratio),
        "min_center_comp_ratio": float(args.min_center_comp_ratio),
        "max_center_comp_ratio": float(args.max_center_comp_ratio),
        "max_hole_ratio": float(args.max_hole_ratio),
        "min_ink_ratio": float(args.min_ink_ratio),
        "max_ink_ratio": float(args.max_ink_ratio),
        "min_edge_ratio": float(args.min_edge_ratio),
        "max_tight_bbox_fill": float(args.max_tight_bbox_fill),
        "max_skeleton_len_ratio": float(args.max_skeleton_len_ratio),
        "max_skeleton_span_terminal": float(args.max_skeleton_span_terminal),
        "max_skeleton_span_junction": float(args.max_skeleton_span_junction),
        "junction_max_center_comp_ratio": float(args.junction_max_center_comp_ratio),
        "junction_max_hole_ratio": float(args.junction_max_hole_ratio),
        "junction_min_edge_ratio": float(args.junction_min_edge_ratio),
        "junction_max_tight_bbox_fill": float(args.junction_max_tight_bbox_fill),
        "junction_max_skeleton_len_ratio": float(args.junction_max_skeleton_len_ratio),
        "max_candidates": int(args.max_candidates),
    }

    font_states: Dict[str, Dict[str, Any]] = {
        d.name: {
            "cands": [],
            "chars_found": 0,
            "chars_used": 0,
            "has_images": False,
            "pending": 0,
            "flushed": False,
            "ink_min_ratio": float(worker_cfg["min_ink_ratio"]),
            "ink_max_ratio": float(worker_cfg["max_ink_ratio"]),
            "ink_sample_count": 0,
        }
        for d in font_dirs
    }
    font_index = {d.name: i for i, d in enumerate(font_dirs)}
    font_dir_by_name = {d.name: d for d in font_dirs}

    chunk_size = max(1, int(args.char_chunk_size))
    jobs: List[Tuple[str, List[str], Dict[str, Any]]] = []

    for d in font_dirs:
        lo, hi, n = float(worker_cfg["min_ink_ratio"]), float(worker_cfg["max_ink_ratio"]), 0
        font_states[d.name]["ink_min_ratio"] = float(lo)
        font_states[d.name]["ink_max_ratio"] = float(hi)
        font_states[d.name]["ink_sample_count"] = int(n)
        cfg_for_font = dict(worker_cfg)
        cfg_for_font["min_ink_ratio"] = float(lo)
        cfg_for_font["max_ink_ratio"] = float(hi)
        for i in range(0, len(ref_chars), chunk_size):
            jobs.append((str(d), ref_chars[i : i + chunk_size], cfg_for_font))
            font_states[d.name]["pending"] += 1

    def _flush_one_font(font_name: str) -> None:
        st = font_states[font_name]
        if st["flushed"]:
            return
        fi = int(font_index[font_name])
        prefix = f"[{fi+1}/{len(font_dirs)}] {font_name}"
        d = font_dir_by_name[font_name]

        if not st["has_images"]:
            print(f"{prefix} [skip] no images")
            st["flushed"] = True
            return

        cands: List[Candidate] = list(st["cands"])
        chars_found = int(st["chars_found"])
        chars_used = int(st["chars_used"])
        ink_min_ratio = float(st["ink_min_ratio"])
        ink_max_ratio = float(st["ink_max_ratio"])
        ink_sample_count = int(st["ink_sample_count"])
        if not cands:
            print(f"{prefix} [skip] no candidates (ref_found={chars_found}/{len(ref_chars)})")
            st["flushed"] = True
            return

        cands = sorted(cands, key=lambda x: x.score, reverse=True)[: int(args.max_candidates)]
        picked = select_with_quota(cands, int(args.parts_per_font), ratios, int(args.random_seed))
        picked = dedupe_by_cosine_per_type(
            picked,
            threshold=float(args.dedupe_cos_threshold_per_type),
            anchor_limit=int(args.dedupe_anchor_limit),
        )
        picked = dedupe_by_cosine(picked, float(args.dedupe_cos_threshold), int(args.dedupe_anchor_limit))
        picked = enforce_max_parts_per_char(picked, int(args.max_parts_per_char))
        if len(picked) < int(args.parts_per_font):
            remain = [c for c in cands if c not in picked]
            remain = dedupe_by_cosine_per_type(
                remain,
                threshold=float(args.dedupe_cos_threshold_per_type),
                anchor_limit=int(args.dedupe_anchor_limit),
            )
            remain = dedupe_by_cosine(remain, float(args.dedupe_cos_threshold), int(args.dedupe_anchor_limit))
            remain = enforce_max_parts_per_char(remain, int(args.max_parts_per_char))
            need = int(args.parts_per_font) - len(picked)
            picked.extend(remain[:need])
        picked = sorted(picked, key=lambda x: x.score, reverse=True)[: int(args.parts_per_font)]

        type_counts = {t: 0 for t in FEATURE_TYPES}
        for c in picked:
            type_counts[c.feature_type] = type_counts.get(c.feature_type, 0) + 1

        font_out = out_dir / font_name
        font_out.mkdir(parents=True, exist_ok=True)
        rows: List[Dict] = []
        for i, c in enumerate(picked):
            rel_name = f"part_{i:03d}_U{ord(c.char):04X}.png"
            out_path = font_out / rel_name
            Image.fromarray(c.patch, mode="L").save(out_path)
            rows.append(
                {
                    "path": rel_or_abs(out_path, root),
                    "char": c.char,
                    "char_code": f"U+{ord(c.char):04X}",
                    "x": int(c.x),
                    "y": int(c.y),
                    "response": float(c.score),
                    "feature_type": str(c.feature_type),
                    "ink_ratio": float(c.ink_ratio),
                    "edge_ratio": float(c.edge_ratio),
                    "center_comp_ratio": float(c.center_comp_ratio),
                    "hole_ratio": float(c.hole_ratio),
                    "width_std": float(c.width_std),
                    "width_grad": float(c.width_grad),
                    "skeleton_span": float(c.skeleton_span),
                    "component_id": int(c.component_id),
                    "component_area": int(c.component_area),
                }
            )

        manifest["fonts"][font_name] = {
            "font_path": rel_or_abs(d, root),
            "num_candidates": int(len(cands)),
            "num_parts": int(len(rows)),
            "type_counts": type_counts,
            "num_reference_chars_total": int(len(ref_chars)),
            "num_reference_chars_found": int(chars_found),
            "num_chars_with_candidates": int(chars_used),
            "ink_ratio_bounds": [float(ink_min_ratio), float(ink_max_ratio)],
            "ink_ratio_sample_count": int(ink_sample_count),
            "parts": rows,
        }
        # Flush manifest incrementally after each completed font.
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        st["flushed"] = True
        print(
            f"{prefix} ref_found={chars_found}/{len(ref_chars)} "
            f"candidates={len(cands)} saved={len(rows)} "
            f"ink=[{ink_min_ratio:.4f},{ink_max_ratio:.4f}] type_counts={type_counts} [flushed]"
        )

    max_workers = max(1, int(args.workers))
    ex_cls = ProcessPoolExecutor if str(args.executor).lower() == "process" else ThreadPoolExecutor
    with ex_cls(max_workers=max_workers) as ex:
        futs = [ex.submit(_process_font_chunk, a, b, c) for a, b, c in jobs]
        for fut in as_completed(futs):
            font_name, cands_chunk, found_chunk, used_chunk, has_images, ink_lo, ink_hi = fut.result()
            st = font_states[font_name]
            st["has_images"] = st["has_images"] or bool(has_images)
            st["chars_found"] += int(found_chunk)
            st["chars_used"] += int(used_chunk)
            st["ink_min_ratio"] = float(ink_lo)
            st["ink_max_ratio"] = float(ink_hi)
            if cands_chunk:
                st["cands"].extend(cands_chunk)
                if len(st["cands"]) > int(args.max_candidates) * 2:
                    st["cands"] = sorted(st["cands"], key=lambda x: x.score, reverse=True)[: int(args.max_candidates)]
            st["pending"] -= 1
            if st["pending"] == 0:
                _flush_one_font(font_name)

    # Safety: flush any stragglers (should already be flushed).
    for d in font_dirs:
        _flush_one_font(d.name)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
