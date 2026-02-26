#!/usr/bin/env python3
"""Build a structure-balanced reference char list from an existing list.

Structure categories are inferred from ContentFont glyph geometry:
- left_right
- up_down
- enclosing
- single
- other
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def load_json_list(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list: {path}")
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
        raise ValueError(f"No valid chars in list: {path}")
    return out


def parse_ratios(raw: str) -> Dict[str, float]:
    names = ["left_right", "up_down", "enclosing", "single", "other"]
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 5:
        raise ValueError("--ratios must have 5 comma-separated values: left_right,up_down,enclosing,single,other")
    vals = [max(0.0, v) for v in vals]
    s = sum(vals)
    if s <= 0:
        raise ValueError("ratio sum must be > 0")
    vals = [v / s for v in vals]
    return {k: v for k, v in zip(names, vals)}


def glyph_path(content_dir: Path, ch: str) -> Path:
    return content_dir / f"ContentFont@{ch}.png"


def binarize(gray: np.ndarray, thr: int) -> np.ndarray:
    return (gray < int(thr)).astype(np.uint8)


def hole_ratio(fg: np.ndarray) -> float:
    bg = (fg == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if n <= 1:
        return 0.0
    h, w = bg.shape
    border = set(labels[0, :].tolist() + labels[h - 1, :].tolist() + labels[:, 0].tolist() + labels[:, w - 1].tolist())
    holes = 0
    for i in range(1, n):
        if i in border:
            continue
        holes += int(stats[i, cv2.CC_STAT_AREA])
    return float(holes) / float(max(1, h * w))


def best_split_score(fg: np.ndarray, axis: int) -> Tuple[float, float, float]:
    # Returns (best_valley, mass_a, mass_b). Lower valley means cleaner split.
    h, w = fg.shape
    if axis == 0:  # vertical split for left-right
        proj = fg.sum(axis=0).astype(np.float32)
        total = float(proj.sum()) + 1e-8
        lo = int(0.28 * w)
        hi = int(0.72 * w)
        if hi <= lo:
            lo, hi = 1, w - 1
        best = (1e9, 0.0, 0.0)
        win = 2
        for s in range(lo, hi):
            valley = float(proj[max(0, s - win) : min(w, s + win + 1)].mean())
            ma = float(proj[:s].sum()) / total
            mb = float(proj[s:].sum()) / total
            if min(ma, mb) < 0.18:
                continue
            # prefer balanced halves
            val = valley + 0.35 * abs(ma - mb) * float(proj.mean() + 1e-8)
            if val < best[0]:
                best = (val, ma, mb)
        return best

    proj = fg.sum(axis=1).astype(np.float32)  # horizontal split for up-down
    total = float(proj.sum()) + 1e-8
    lo = int(0.28 * h)
    hi = int(0.72 * h)
    if hi <= lo:
        lo, hi = 1, h - 1
    best = (1e9, 0.0, 0.0)
    win = 2
    for s in range(lo, hi):
        valley = float(proj[max(0, s - win) : min(h, s + win + 1)].mean())
        ma = float(proj[:s].sum()) / total
        mb = float(proj[s:].sum()) / total
        if min(ma, mb) < 0.18:
            continue
        val = valley + 0.35 * abs(ma - mb) * float(proj.mean() + 1e-8)
        if val < best[0]:
            best = (val, ma, mb)
    return best


def classify_structure(gray: np.ndarray, thr: int, hole_thr: float, valley_rel_thr: float) -> str:
    fg = binarize(gray, thr)
    if int(fg.sum()) <= 0:
        return "other"

    # enclosing first
    hratio = hole_ratio(fg)
    if hratio >= float(hole_thr):
        return "enclosing"

    ncc, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comp_n = max(0, ncc - 1)

    vbest, vma, vmb = best_split_score(fg, axis=0)  # LR
    hbest, hma, hmb = best_split_score(fg, axis=1)  # UD

    # normalize by mean projection magnitude to get relative valley
    px = fg.sum(axis=0).astype(np.float32)
    py = fg.sum(axis=1).astype(np.float32)
    xmean = float(px.mean()) + 1e-8
    ymean = float(py.mean()) + 1e-8
    vrel = float(vbest / xmean) if vbest < 1e8 else 1e9
    hrel = float(hbest / ymean) if hbest < 1e8 else 1e9

    lr_ok = vrel <= float(valley_rel_thr)
    ud_ok = hrel <= float(valley_rel_thr)

    if lr_ok and (not ud_ok or vrel <= hrel * 0.95):
        return "left_right"
    if ud_ok and (not lr_ok or hrel < vrel * 0.95):
        return "up_down"

    if comp_n <= 1:
        return "single"
    return "other"


def split_counts(total: int, ratios: Dict[str, float], avail: Dict[str, int]) -> Dict[str, int]:
    cats = list(ratios.keys())
    raw = {k: float(total) * float(ratios[k]) for k in cats}
    cnt = {k: min(int(np.floor(raw[k])), int(avail.get(k, 0))) for k in cats}
    rem = int(total - sum(cnt.values()))

    if rem > 0:
        order = sorted(cats, key=lambda k: (raw[k] - np.floor(raw[k])), reverse=True)
        for k in order:
            if rem <= 0:
                break
            room = int(avail.get(k, 0)) - cnt[k]
            if room <= 0:
                continue
            take = min(rem, room)
            cnt[k] += take
            rem -= take

    if rem > 0:
        order = sorted(cats, key=lambda k: avail.get(k, 0), reverse=True)
        for k in order:
            if rem <= 0:
                break
            room = int(avail.get(k, 0)) - cnt[k]
            if room <= 0:
                continue
            take = min(rem, room)
            cnt[k] += take
            rem -= take
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--in-reference-json", type=Path, default=Path("CharacterData/ReferenceCharList.json"))
    ap.add_argument("--content-glyph-dir", type=Path, default=Path("DataPreparation/Generated/ContentFont/ContentFont"))
    ap.add_argument("--out-reference-json", type=Path, default=Path("CharacterData/ReferenceCharList_200_balanced.json"))
    ap.add_argument("--out-report-json", type=Path, default=Path("CharacterData/reference_200_balanced_report.json"))
    ap.add_argument("--target-count", type=int, default=200)
    ap.add_argument("--binary-threshold", type=int, default=245)
    ap.add_argument("--hole-threshold", type=float, default=0.06)
    ap.add_argument("--valley-rel-threshold", type=float, default=0.78)
    ap.add_argument("--ratios", type=str, default="0.34,0.26,0.14,0.16,0.10",
                    help="left_right,up_down,enclosing,single,other")
    args = ap.parse_args()

    root = args.project_root.resolve()
    in_ref = (root / args.in_reference_json).resolve()
    content_dir = (root / args.content_glyph_dir).resolve()
    out_ref = (root / args.out_reference_json).resolve()
    out_report = (root / args.out_report_json).resolve()

    chars = load_json_list(in_ref)
    ratios = parse_ratios(args.ratios)

    groups: Dict[str, List[str]] = {k: [] for k in ratios.keys()}
    missing: List[str] = []

    for ch in chars:
        gp = glyph_path(content_dir, ch)
        if not gp.exists():
            missing.append(ch)
            groups["other"].append(ch)
            continue
        try:
            gray = np.asarray(Image.open(gp).convert("L"), dtype=np.uint8)
        except Exception:
            groups["other"].append(ch)
            continue
        cls = classify_structure(
            gray,
            thr=int(args.binary_threshold),
            hole_thr=float(args.hole_threshold),
            valley_rel_thr=float(args.valley_rel_threshold),
        )
        groups[cls].append(ch)

    avail = {k: len(v) for k, v in groups.items()}
    target = max(1, int(args.target_count))
    counts = split_counts(target, ratios, avail)

    selected: List[str] = []
    for k in ratios.keys():
        selected.extend(groups[k][: counts[k]])

    if len(selected) < target:
        used = set(selected)
        for ch in chars:
            if ch in used:
                continue
            selected.append(ch)
            used.add(ch)
            if len(selected) >= target:
                break

    selected = selected[:target]
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    out_ref.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # actual distribution in selected
    sel_set = set(selected)
    actual = {k: sum(1 for x in groups[k] if x in sel_set) for k in ratios.keys()}

    report = {
        "input_reference": str(in_ref),
        "output_reference": str(out_ref),
        "target_count": int(target),
        "selected_count": int(len(selected)),
        "missing_content_glyph_count": int(len(missing)),
        "ratios_target": ratios,
        "groups_available": avail,
        "groups_selected": actual,
        "counts_target_by_ratio": counts,
        "params": {
            "binary_threshold": int(args.binary_threshold),
            "hole_threshold": float(args.hole_threshold),
            "valley_rel_threshold": float(args.valley_rel_threshold),
        },
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
