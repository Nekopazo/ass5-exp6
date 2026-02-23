#!/usr/bin/env python3
"""Analyze and visualize component-overlap statistics for style sampling."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from dataset import FontImageDataset


def resolve_path(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def print_summary(report: Dict) -> None:
    print("=== Component Overlap Summary ===")
    print(f"enabled: {report.get('enabled')}")
    print(f"eval_samples: {report.get('num_eval_samples')}")
    print(f"style_k: {report.get('style_k')}")
    print(f"total_pairs: {report.get('total_pairs')}")
    print(f"positive_pairs: {report.get('positive_pairs')}")
    print(f"positive_rate: {report.get('positive_rate', 0.0):.4f}")
    print(f"mean_overlap: {report.get('mean_overlap', 0.0):.4f}")
    print(f"median_overlap: {report.get('median_overlap', 0.0):.4f}")
    print(f"p90_overlap: {report.get('p90_overlap', 0.0):.4f}")
    print(f"p95_overlap: {report.get('p95_overlap', 0.0):.4f}")
    print(f"max_overlap: {report.get('max_overlap', 0)}")

    hist = report.get("overlap_hist", {})
    if hist:
        print("histogram:", ", ".join([f"{k}:{v}" for k, v in hist.items()]))

    top_fonts: List[Dict] = report.get("top_fonts_by_mean_overlap", [])[:5]
    if top_fonts:
        print("top fonts by mean overlap:")
        for row in top_fonts:
            print(
                f"  {row['font']}: mean={row['mean_overlap']:.3f}, "
                f"pos_rate={row['positive_rate']:.3f}, pairs={row['pairs']}"
            )

    top_chars: List[Dict] = report.get("top_chars_by_mean_overlap", [])[:8]
    if top_chars:
        print("top chars by mean overlap:")
        for row in top_chars:
            print(
                f"  {row['char']}: mean={row['mean_overlap']:.3f}, "
                f"pos_rate={row['positive_rate']:.3f}, pairs={row['pairs']}"
            )


def save_top_pairs_csv(top_pairs: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["target_char", "style_char", "count"])
        writer.writeheader()
        for row in top_pairs:
            writer.writerow(row)


def save_hist_plot(report: Dict, out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable, skip plot: {e}")
        return

    hist = report.get("overlap_hist", {})
    if not hist:
        print("[warn] empty histogram, skip plot")
        return

    x = sorted((int(k) for k in hist.keys()))
    y = [int(hist[str(k)]) for k in x]

    font_rows: List[Dict] = report.get("top_fonts_by_mean_overlap", [])[:10]
    font_names = [r["font"] for r in font_rows]
    font_vals = [float(r["mean_overlap"]) for r in font_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([str(v) for v in x], y)
    axes[0].set_title("Component Overlap Histogram")
    axes[0].set_xlabel("overlap size")
    axes[0].set_ylabel("pair count")

    if font_rows:
        axes[1].barh(font_names[::-1], font_vals[::-1])
        axes[1].set_title("Top Fonts by Mean Overlap")
        axes[1].set_xlabel("mean overlap")
    else:
        axes[1].set_title("Top Fonts by Mean Overlap")
        axes[1].text(0.5, 0.5, "no font rows", ha="center", va="center")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-name", type=str, default=None)
    parser.add_argument("--font-mode", type=str, default="random", choices=["fixed", "random"])
    parser.add_argument("--max-fonts", type=int, default=0)

    parser.add_argument("--style-k", type=int, default=3)
    parser.add_argument("--include-target-in-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--component-guided-style", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decomposition-json", type=Path, default=Path("CharacterData/decomposition.json"))

    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-char-k", type=int, default=20)
    parser.add_argument("--top-pair-k", type=int, default=80)

    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/overlap_stats"))
    parser.add_argument("--json-name", type=str, default="component_overlap_report.json")
    parser.add_argument("--pairs-csv-name", type=str, default="component_overlap_top_pairs.csv")
    parser.add_argument("--plot-name", type=str, default="component_overlap_hist.png")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FontImageDataset(
        project_root=root,
        font_index=args.font_index,
        font_name=args.font_name,
        font_mode=args.font_mode,
        max_fonts=args.max_fonts,
        num_style_refs=args.style_k,
        include_target_in_style=args.include_target_in_style,
        component_guided_style=args.component_guided_style,
        decomposition_json=args.decomposition_json,
        transform=None,
    )
    report = dataset.component_overlap_stats(
        num_samples=args.samples,
        random_seed=args.seed,
        top_char_k=args.top_char_k,
        top_pair_k=args.top_pair_k,
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["config"] = {
        "project_root": str(root),
        "font_index": args.font_index,
        "font_name": args.font_name,
        "font_mode": args.font_mode,
        "max_fonts": args.max_fonts,
        "style_k": args.style_k,
        "include_target_in_style": args.include_target_in_style,
        "component_guided_style": args.component_guided_style,
        "decomposition_json": str(args.decomposition_json),
        "samples": args.samples,
        "seed": args.seed,
    }

    print_summary(report)

    json_path = out_dir / args.json_name
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved json: {json_path}")

    top_pairs: List[Dict] = report.get("top_positive_pairs", [])
    csv_path = out_dir / args.pairs_csv_name
    save_top_pairs_csv(top_pairs, csv_path)
    print(f"saved top pairs csv: {csv_path}")

    if not args.skip_plot:
        plot_path = out_dir / args.plot_name
        save_hist_plot(report, plot_path)
        if plot_path.exists():
            print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
