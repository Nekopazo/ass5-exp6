#!/usr/bin/env python3
"""Offline component-overlap analyzer.

This tool is intentionally offline-only and does not modify training/runtime logic.
It estimates how much component overlap exists between target chars and sampled
style chars.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


def load_json_list(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected list json: {path}")
    out = [x for x in obj if isinstance(x, str) and len(x) == 1]
    if not out:
        raise ValueError(f"Empty/invalid char list: {path}")
    return out


def load_decomposition(path: Path) -> Dict[str, Set[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict json: {path}")
    out: Dict[str, Set[str]] = {}
    for ch, raw in obj.items():
        if not isinstance(ch, str) or len(ch) != 1:
            continue
        tokens: List[str] = []
        if isinstance(raw, str) and raw:
            tokens.append(raw)
        elif isinstance(raw, list):
            tokens.extend([x for x in raw if isinstance(x, str) and x])
        if not tokens:
            continue
        parts: Set[str] = set()
        for t in tokens:
            parts.add(t)
            if len(t) > 1:
                parts.update([c for c in t if c.strip()])
        out[ch] = parts
    if not out:
        raise ValueError(f"No valid decomposition entries: {path}")
    return out


def overlap(a: str, b: str, dec: Dict[str, Set[str]]) -> int:
    pa = dec.get(a)
    pb = dec.get(b)
    if not pa or not pb:
        return 0
    return len(pa.intersection(pb))


def quantile(vals_sorted: List[int], q: float) -> float:
    if not vals_sorted:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    if len(vals_sorted) == 1:
        return float(vals_sorted[0])
    pos = q * (len(vals_sorted) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals_sorted) - 1)
    w = pos - lo
    return float(vals_sorted[lo] * (1.0 - w) + vals_sorted[hi] * w)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--char-list", type=str, default="CharacterData/CharList.json")
    p.add_argument("--reference-char-list", type=str, default="CharacterData/ReferenceCharList.json")
    p.add_argument("--decomposition-json", type=str, default="CharacterData/decomposition.json")
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--style-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "topk_overlap"],
        help="random: random style chars; topk_overlap: choose highest overlap style chars.",
    )
    p.add_argument("--out-json", type=str, default="checkpoints/overlap_stats/component_overlap_report.json")
    args = p.parse_args()

    root = args.project_root.resolve()
    char_list_path = (root / args.char_list).resolve()
    ref_list_path = (root / args.reference_char_list).resolve()
    dec_path = (root / args.decomposition_json).resolve()

    chars = load_json_list(char_list_path)
    refs = load_json_list(ref_list_path)
    dec = load_decomposition(dec_path)

    rng = random.Random(int(args.seed))
    n = max(1, int(args.samples))
    k = max(1, int(args.style_k))

    vals: List[int] = []
    for _ in range(n):
        ch = rng.choice(chars)
        pool = [x for x in refs if x != ch]
        if not pool:
            pool = refs[:]
        if not pool:
            continue
        if str(args.mode) == "topk_overlap":
            sorted_pool = sorted(pool, key=lambda x: overlap(ch, x, dec), reverse=True)
            picked = sorted_pool[:k]
        else:
            if len(pool) >= k:
                picked = rng.sample(pool, k)
            else:
                picked = pool[:]
                while len(picked) < k:
                    picked.append(rng.choice(pool))
        for sc in picked:
            vals.append(int(overlap(ch, sc, dec)))

    if not vals:
        raise RuntimeError("No overlap pairs were collected.")

    vals_sorted = sorted(vals)
    hist = Counter(vals)
    total = len(vals)
    pos = sum(1 for v in vals if v > 0)
    report = {
        "samples": n,
        "style_k": k,
        "mode": str(args.mode),
        "total_pairs": total,
        "positive_pairs": pos,
        "positive_rate": float(pos / total),
        "mean_overlap": float(sum(vals) / total),
        "median_overlap": quantile(vals_sorted, 0.5),
        "p90_overlap": quantile(vals_sorted, 0.9),
        "p95_overlap": quantile(vals_sorted, 0.95),
        "max_overlap": int(vals_sorted[-1]),
        "overlap_hist": {str(k): int(v) for k, v in sorted(hist.items(), key=lambda x: x[0])},
        "paths": {
            "char_list": str(char_list_path),
            "reference_char_list": str(ref_list_path),
            "decomposition_json": str(dec_path),
        },
    }

    out_path = (root / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze_component_overlap] saved: {out_path}")
    print(
        f"[analyze_component_overlap] mode={report['mode']} "
        f"positive_rate={report['positive_rate']:.4f} mean={report['mean_overlap']:.4f}"
    )


if __name__ == "__main__":
    main()

