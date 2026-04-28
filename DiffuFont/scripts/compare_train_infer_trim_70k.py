#!/usr/bin/env python3
"""Compare 70k checkpoints trained/inferred with mean vs drop-min-max ref fusion."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import FontImageDataset
from inference import load_trainer, tensor_to_pil
from models.source_part_ref_dit import ContentStyleCrossAttention
from style_augment import build_base_glyph_transform


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(raw_device)


def mean_refs(context_per_ref: torch.Tensor) -> torch.Tensor:
    return context_per_ref.mean(dim=1)


def drop_minmax_refs(context_per_ref: torch.Tensor) -> torch.Tensor:
    ref_count = int(context_per_ref.size(1))
    if ref_count <= 2:
        return context_per_ref.mean(dim=1)
    return (
        context_per_ref.sum(dim=1)
        - context_per_ref.max(dim=1).values
        - context_per_ref.min(dim=1).values
    ) / float(ref_count - 2)


AGGREGATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": mean_refs,
    "drop_minmax": drop_minmax_refs,
}


def set_aggregation(name: str) -> None:
    ContentStyleCrossAttention._trimmed_mean_refs = staticmethod(AGGREGATIONS[name])


def choose_unique_font_samples(dataset: FontImageDataset, count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    font_names = list(dataset.font_names)
    rng.shuffle(font_names)
    sample_indices: list[int] = []
    for font_name in font_names:
        candidates = list(dataset.sample_indices_by_font[font_name])
        if not candidates:
            continue
        sample_indices.append(int(rng.choice(candidates)))
        if len(sample_indices) >= int(count):
            break
    return sample_indices


def metrics(generation: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    gen = generation.detach().float().cpu()
    tgt = target.detach().float().cpu()
    diff = gen - tgt
    return {
        "l1": float(diff.abs().mean().item()),
        "mse": float(diff.pow(2).mean().item()),
        "cosine": float(F.cosine_similarity(gen.reshape(1, -1), tgt.reshape(1, -1), dim=1).item()),
    }


def stats(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


@torch.no_grad()
def generate_one(
    trainer,
    sample: dict[str, Any],
    *,
    seed: int,
    inference_steps: int,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return trainer.sample(
        sample["content"].unsqueeze(0),
        content_index=torch.tensor([0], dtype=torch.long),
        style_img=sample["style_img"].unsqueeze(0),
        style_index=torch.tensor([0], dtype=torch.long),
        style_ref_mask=sample["style_ref_mask"].unsqueeze(0),
        num_inference_steps=int(inference_steps),
        use_ema=True,
    ).squeeze(0).cpu()


def build_grid(samples: list[dict[str, Any]], rows: list[dict[str, Any]], *, cell_size: int) -> Image.Image:
    combo_keys = [
        "train_mean__infer_mean",
        "train_mean__infer_drop_minmax",
        "train_drop_minmax__infer_mean",
        "train_drop_minmax__infer_drop_minmax",
    ]
    label_w = 220
    cols = ["content", "target", "style1", *combo_keys]
    header_h = 50
    width = label_w + len(cols) * cell_size
    height = header_h + len(samples) * cell_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for col_idx, label in enumerate(cols):
        draw.text((label_w + col_idx * cell_size + 3, 8), label.replace("__", "\n"), fill=(0, 0, 0), font=font)
    by_sample_combo = {(row["sample_pos"], row["combo"]): row for row in rows}
    for sample_pos, sample in enumerate(samples):
        y = header_h + sample_pos * cell_size
        draw.text(
            (6, y + 6),
            f"{sample['char']} {sample['font'][:20]}\n{''.join(sample['style_chars'])}",
            fill=(0, 0, 0),
            font=font,
        )
        tensors = [
            sample["content"],
            sample["target"],
            sample["style_img"][0],
            *[by_sample_combo[(sample_pos, key)]["generation"] for key in combo_keys],
        ]
        for col_idx, tensor in enumerate(tensors):
            canvas.paste(tensor_to_pil(tensor, size=cell_size).convert("RGB"), (label_w + col_idx * cell_size, y))
    return canvas


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_rows = []
    for row in rows:
        metric_rows.append({key: value for key, value in row.items() if not isinstance(value, torch.Tensor)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mean-checkpoint", type=Path, required=True)
    parser.add_argument("--train-drop-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "compare_train_infer_trim_70k")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--style-ref-count", type=int, default=6)
    parser.add_argument("--font-split", type=str, default="test")
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    parser.add_argument("--cell-size", type=int, default=88)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_device(args.device)

    set_aggregation("mean")
    probe_trainer = load_trainer(args.train_mean_checkpoint, device)
    transform = build_base_glyph_transform(image_size=int(probe_trainer.model.image_size))
    del probe_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = FontImageDataset(
        project_root=args.data_root,
        style_ref_count=int(args.style_ref_count),
        style_ref_count_min=int(args.style_ref_count),
        style_ref_count_max=int(args.style_ref_count),
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=int(args.font_split_seed),
        font_train_ratio=float(args.font_train_ratio),
        transform=transform,
        style_transform=transform,
    )
    sample_indices = choose_unique_font_samples(dataset, int(args.num_samples), int(args.seed))
    samples = [dataset[idx] for idx in sample_indices]

    ckpts = [
        ("train_mean", args.train_mean_checkpoint),
        ("train_drop_minmax", args.train_drop_checkpoint),
    ]
    rows: list[dict[str, Any]] = []
    for train_label, ckpt_path in ckpts:
        set_aggregation("mean")
        trainer = load_trainer(ckpt_path, device)
        for infer_agg in ("mean", "drop_minmax"):
            set_aggregation(infer_agg)
            for sample_pos, sample in enumerate(samples):
                generation_seed = int(args.seed) * 1000 + sample_pos
                generation = generate_one(
                    trainer,
                    sample,
                    seed=generation_seed,
                    inference_steps=int(args.inference_steps),
                )
                row_metrics = metrics(generation, sample["target"])
                rows.append(
                    {
                        "combo": f"{train_label}__infer_{infer_agg}",
                        "train_label": train_label,
                        "infer_aggregation": infer_agg,
                        "checkpoint": str(ckpt_path),
                        "sample_pos": int(sample_pos),
                        "sample_index": int(sample_indices[sample_pos]),
                        "font": str(sample["font"]),
                        "target_char": str(sample["char"]),
                        "style_chars": "".join(str(ch) for ch in sample["style_chars"]),
                        **row_metrics,
                        "generation": generation,
                    }
                )
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "metrics.tsv", rows)
    build_grid(samples, rows, cell_size=int(args.cell_size)).save(output_dir / "comparison_grid.png")

    combos = sorted({row["combo"] for row in rows})
    summary = {}
    for combo in combos:
        combo_rows = [row for row in rows if row["combo"] == combo]
        summary[combo] = {
            key: stats([float(row[key]) for row in combo_rows])
            for key in ("l1", "mse", "cosine")
        }

    # Pairwise win counts by L1 for useful direct comparisons.
    win_counts: dict[str, dict[str, int]] = {}
    by_sample = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_pos"]), {})[row["combo"]] = row
    for a in combos:
        win_counts[a] = {}
        for b in combos:
            if a == b:
                continue
            wins = sum(1 for sample_map in by_sample.values() if float(sample_map[a]["l1"]) < float(sample_map[b]["l1"]))
            win_counts[a][b] = int(wins)

    report = {
        "train_mean_checkpoint": str(args.train_mean_checkpoint),
        "train_drop_checkpoint": str(args.train_drop_checkpoint),
        "num_samples": len(samples),
        "inference_steps": int(args.inference_steps),
        "style_ref_count": int(args.style_ref_count),
        "summary": summary,
        "l1_win_counts": win_counts,
        "rows": [
            {key: value for key, value in row.items() if not isinstance(value, torch.Tensor)}
            for row in rows
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "num_samples": len(samples),
            "summary": summary,
            "l1_win_counts": win_counts,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
