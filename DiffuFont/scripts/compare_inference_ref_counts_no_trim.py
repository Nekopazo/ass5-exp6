#!/usr/bin/env python3
"""Compare inference with 6 vs 12 style refs while disabling ref trimming."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

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


def disable_ref_trimming() -> None:
    ContentStyleCrossAttention._trimmed_mean_refs = staticmethod(lambda context_per_ref: context_per_ref.mean(dim=1))


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


def image_metrics(generation: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    gen = generation.detach().float().cpu()
    tgt = target.detach().float().cpu()
    diff = gen - tgt
    return {
        "l1": float(diff.abs().mean().item()),
        "mse": float(diff.pow(2).mean().item()),
        "cosine": float(F.cosine_similarity(gen.reshape(1, -1), tgt.reshape(1, -1), dim=1).item()),
    }


def compare_generation(gen6: torch.Tensor, gen12: torch.Tensor) -> dict[str, float]:
    a = gen6.detach().float().cpu()
    b = gen12.detach().float().cpu()
    diff = a - b
    return {
        "gen6_vs_gen12_l1": float(diff.abs().mean().item()),
        "gen6_vs_gen12_mse": float(diff.pow(2).mean().item()),
        "gen6_vs_gen12_cosine": float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).item()),
    }


@torch.no_grad()
def run_sample(
    trainer,
    sample: dict[str, Any],
    *,
    ref_count: int,
    seed: int,
    inference_steps: int,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    style_img = sample["style_img"][:ref_count].unsqueeze(0)
    style_ref_mask = sample["style_ref_mask"][:ref_count].unsqueeze(0)
    return trainer.sample(
        sample["content"].unsqueeze(0),
        content_index=torch.tensor([0], dtype=torch.long),
        style_img=style_img,
        style_index=torch.tensor([0], dtype=torch.long),
        style_ref_mask=style_ref_mask,
        num_inference_steps=int(inference_steps),
        use_ema=True,
    ).squeeze(0).cpu()


def build_grid(rows: list[dict[str, Any]], *, cell_size: int) -> Image.Image:
    label_w = 220
    cols = ["content", "target", "ref1", "ref6", "ref12", "gen6", "gen12", "absdiff"]
    header_h = 44
    width = label_w + len(cols) * cell_size
    height = header_h + len(rows) * cell_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for col_idx, label in enumerate(cols):
        draw.text((label_w + col_idx * cell_size + 4, 12), label, fill=(0, 0, 0), font=font)
    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * cell_size
        draw.text(
            (6, y + 8),
            f"{row['target_char']} {row['font'][:20]}\n{row['style_chars_12']}",
            fill=(0, 0, 0),
            font=font,
        )
        tensors = [
            row["content"],
            row["target"],
            row["ref1"],
            row["ref6"],
            row["ref12"],
            row["gen6"],
            row["gen12"],
            row["absdiff"],
        ]
        for col_idx, tensor in enumerate(tensors):
            img = tensor_to_pil(tensor, size=cell_size).convert("RGB")
            canvas.paste(img, (label_w + col_idx * cell_size, y))
    return canvas


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    metric_rows = []
    for row in rows:
        metric_rows.append({key: value for key, value in row.items() if not isinstance(value, torch.Tensor)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)


def stats(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "compare_ref_counts_no_trim")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--font-split", type=str, default="test")
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    parser.add_argument("--cell-size", type=int, default=96)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    disable_ref_trimming()

    device = resolve_device(args.device)
    trainer = load_trainer(args.checkpoint, device)
    transform = build_base_glyph_transform(image_size=int(trainer.model.image_size))
    dataset = FontImageDataset(
        project_root=args.data_root,
        style_ref_count=12,
        style_ref_count_min=12,
        style_ref_count_max=12,
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=int(args.font_split_seed),
        font_train_ratio=float(args.font_train_ratio),
        transform=transform,
        style_transform=transform,
    )
    sample_indices = choose_unique_font_samples(dataset, int(args.num_samples), int(args.seed))
    rows: list[dict[str, Any]] = []
    for row_idx, sample_index in enumerate(sample_indices):
        sample = dataset[int(sample_index)]
        generation_seed = int(args.seed) * 1000 + row_idx
        gen6 = run_sample(
            trainer,
            sample,
            ref_count=6,
            seed=generation_seed,
            inference_steps=int(args.inference_steps),
        )
        gen12 = run_sample(
            trainer,
            sample,
            ref_count=12,
            seed=generation_seed,
            inference_steps=int(args.inference_steps),
        )
        target = sample["target"].cpu()
        metrics6 = {f"gen6_{key}": value for key, value in image_metrics(gen6, target).items()}
        metrics12 = {f"gen12_{key}": value for key, value in image_metrics(gen12, target).items()}
        compare = compare_generation(gen6, gen12)
        absdiff = (gen6 - gen12).abs().mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        rows.append(
            {
                "sample_index": int(sample_index),
                "font": str(sample["font"]),
                "target_char": str(sample["char"]),
                "style_chars_6": "".join(str(ch) for ch in sample["style_chars"][:6]),
                "style_chars_12": "".join(str(ch) for ch in sample["style_chars"][:12]),
                **metrics6,
                **metrics12,
                **compare,
                "content": sample["content"].cpu(),
                "target": target,
                "ref1": sample["style_img"][0].cpu(),
                "ref6": sample["style_img"][5].cpu(),
                "ref12": sample["style_img"][11].cpu(),
                "gen6": gen6,
                "gen12": gen12,
                "absdiff": absdiff,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "metrics.tsv", rows)
    grid = build_grid(rows, cell_size=int(args.cell_size))
    grid.save(output_dir / "comparison_grid.png")
    metric_keys = [
        "gen6_l1",
        "gen12_l1",
        "gen6_mse",
        "gen12_mse",
        "gen6_cosine",
        "gen12_cosine",
        "gen6_vs_gen12_l1",
        "gen6_vs_gen12_mse",
        "gen6_vs_gen12_cosine",
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "num_samples": len(rows),
        "font_split": str(args.font_split),
        "inference_steps": int(args.inference_steps),
        "aggregation": "plain_mean_no_trim",
        "summary": {key: stats([float(row[key]) for row in rows]) for key in metric_keys},
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
            "num_samples": len(rows),
            "summary": report["summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
