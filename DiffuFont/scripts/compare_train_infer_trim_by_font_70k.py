#!/usr/bin/env python3
"""Compare 70k trim/no-trim checkpoints by font split and per-font samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import FontImageDataset
from inference import load_trainer
from models.source_part_ref_dit import ContentStyleCrossAttention
from style_augment import build_base_glyph_transform


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            best_index = 0
            best_free_bytes = -1
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free_bytes, _ = torch.cuda.mem_get_info()
                if int(free_bytes) > best_free_bytes:
                    best_index = int(index)
                    best_free_bytes = int(free_bytes)
            return torch.device(f"cuda:{best_index}")
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


def stats(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def image_metrics(generation: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    gen = generation.detach().float().cpu()
    tgt = target.detach().float().cpu()
    diff = gen - tgt
    return {
        "l1": float(diff.abs().mean().item()),
        "mse": float(diff.pow(2).mean().item()),
        "cosine": float(F.cosine_similarity(gen.reshape(1, -1), tgt.reshape(1, -1), dim=1).item()),
    }


def choose_font_names(dataset: FontImageDataset, max_fonts: int, seed: int) -> list[str]:
    font_names = list(dataset.font_names)
    rng = random.Random(seed)
    rng.shuffle(font_names)
    if max_fonts > 0:
        font_names = font_names[: int(max_fonts)]
    return font_names


def choose_samples_for_font(
    dataset: FontImageDataset,
    font_name: str,
    samples_per_font: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    candidates = list(dataset.sample_indices_by_font[font_name])
    rng.shuffle(candidates)
    if len(candidates) >= samples_per_font:
        return [int(idx) for idx in candidates[:samples_per_font]]
    return [int(rng.choice(candidates)) for _ in range(samples_per_font)]


def build_split_samples(
    dataset: FontImageDataset,
    *,
    split: str,
    samples_per_font: int,
    max_fonts: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for font_pos, font_name in enumerate(choose_font_names(dataset, max_fonts=max_fonts, seed=seed)):
        indices = choose_samples_for_font(
            dataset,
            font_name,
            samples_per_font=samples_per_font,
            seed=seed * 1009 + font_pos,
        )
        for sample_pos, sample_index in enumerate(indices):
            sample = dataset[int(sample_index)]
            records.append(
                {
                    "split": split,
                    "font_pos": int(font_pos),
                    "sample_pos_in_font": int(sample_pos),
                    "sample_index": int(sample_index),
                    "sample": sample,
                }
            )
    return records


@torch.no_grad()
def generate_batch(trainer, samples: list[dict[str, Any]], *, seed: int, inference_steps: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    batch_size = len(samples)
    content = torch.stack([sample["content"] for sample in samples], dim=0)
    style_img = torch.stack([sample["style_img"] for sample in samples], dim=0)
    style_ref_mask = torch.stack([sample["style_ref_mask"] for sample in samples], dim=0)
    return trainer.sample(
        content,
        content_index=torch.arange(batch_size, dtype=torch.long),
        style_img=style_img,
        style_index=torch.arange(batch_size, dtype=torch.long),
        style_ref_mask=style_ref_mask,
        num_inference_steps=int(inference_steps),
        use_ema=True,
    ).cpu()


def iter_chunks(records: list[dict[str, Any]], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


METRIC_FIELDS = [
    "combo",
    "train_label",
    "infer_aggregation",
    "checkpoint",
    "split",
    "font",
    "font_pos",
    "sample_pos_in_font",
    "sample_index",
    "target_char",
    "style_chars",
    "l1",
    "mse",
    "cosine",
]


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in group_keys), []).append(row)
    output: dict[str, Any] = {}
    for key, group_rows in sorted(grouped.items()):
        name = "__".join(f"{group_keys[idx]}={value}" for idx, value in enumerate(key))
        output[name] = {
            metric: stats([float(row[metric]) for row in group_rows])
            for metric in ("l1", "mse", "cosine")
        }
    return output


def win_counts(rows: list[dict[str, Any]], *, group_key: str) -> dict[str, Any]:
    combos = sorted({str(row["combo"]) for row in rows})
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row[group_key]), []).append(row)

    output: dict[str, Any] = {}
    for group_name, group in sorted(grouped_rows.items()):
        by_sample: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for row in group:
            sample_key = (str(row["split"]), int(row["sample_index"]))
            by_sample.setdefault(sample_key, {})[str(row["combo"])] = row
        group_counts: dict[str, dict[str, int]] = {}
        for a in combos:
            group_counts[a] = {}
            for b in combos:
                if a == b:
                    continue
                wins = 0
                valid = 0
                for sample_map in by_sample.values():
                    if a not in sample_map or b not in sample_map:
                        continue
                    valid += 1
                    if float(sample_map[a]["l1"]) < float(sample_map[b]["l1"]):
                        wins += 1
                group_counts[a][b] = int(wins)
                group_counts[a][f"{b}__valid"] = int(valid)
        output[group_name] = group_counts
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-trained-checkpoint", type=Path, required=True)
    parser.add_argument("--mean-trained-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "compare_train_infer_trim_by_font_70k")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-font", type=int, default=30)
    parser.add_argument("--max-train-fonts", type=int, default=0)
    parser.add_argument("--max-test-fonts", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--style-ref-count", type=int, default=6)
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_device(args.device)
    set_aggregation("mean")
    probe_trainer = load_trainer(args.mean_trained_checkpoint, device)
    transform = build_base_glyph_transform(image_size=int(probe_trainer.model.image_size))
    del probe_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    split_records: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    for split, max_fonts in (("train", int(args.max_train_fonts)), ("test", int(args.max_test_fonts))):
        dataset = FontImageDataset(
            project_root=args.data_root,
            style_ref_count=int(args.style_ref_count),
            style_ref_count_min=int(args.style_ref_count),
            style_ref_count_max=int(args.style_ref_count),
            random_seed=int(args.seed),
            font_split=split,
            font_split_seed=int(args.font_split_seed),
            font_train_ratio=float(args.font_train_ratio),
            transform=transform,
            style_transform=transform,
        )
        records = build_split_samples(
            dataset,
            split=split,
            samples_per_font=int(args.samples_per_font),
            max_fonts=max_fonts,
            seed=int(args.seed) + (0 if split == "train" else 100000),
        )
        split_records.extend(records)
        split_meta[split] = {
            "available_fonts": int(len(dataset.font_names)),
            "used_fonts": int(len({record["sample"]["font"] for record in records})),
            "samples": int(len(records)),
            "max_fonts": int(max_fonts),
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.tsv"

    ckpts = [
        ("trained_drop_minmax", args.drop_trained_checkpoint),
        ("trained_mean", args.mean_trained_checkpoint),
    ]
    metric_rows: list[dict[str, Any]] = []
    total_jobs = len(ckpts) * 2 * len(split_records)
    completed = 0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, delimiter="\t")
        writer.writeheader()
        handle.flush()
        for train_label, ckpt_path in ckpts:
            set_aggregation("mean")
            trainer = load_trainer(ckpt_path, device)
            for infer_agg in ("mean", "drop_minmax"):
                set_aggregation(infer_agg)
                for batch_start, record_batch in iter_chunks(split_records, int(args.eval_batch_size)):
                    sample_batch = [record["sample"] for record in record_batch]
                    generations = generate_batch(
                        trainer,
                        sample_batch,
                        seed=int(args.seed) * 1000003 + batch_start,
                        inference_steps=int(args.inference_steps),
                    )
                    for offset, record in enumerate(record_batch):
                        sample = record["sample"]
                        row = {
                            "combo": f"{train_label}__infer_{infer_agg}",
                            "train_label": train_label,
                            "infer_aggregation": infer_agg,
                            "checkpoint": str(ckpt_path),
                            "split": str(record["split"]),
                            "font": str(sample["font"]),
                            "font_pos": int(record["font_pos"]),
                            "sample_pos_in_font": int(record["sample_pos_in_font"]),
                            "sample_index": int(record["sample_index"]),
                            "target_char": str(sample["char"]),
                            "style_chars": "".join(str(ch) for ch in sample["style_chars"]),
                        }
                        row.update(image_metrics(generations[offset], sample["target"]))
                        metric_rows.append(row)
                        writer.writerow(row)
                        completed += 1
                    if completed == len(record_batch) or completed % 100 <= len(record_batch) or completed == total_jobs:
                        last_row = metric_rows[-1]
                        print(
                            f"[progress] {completed}/{total_jobs} "
                            f"({completed / max(total_jobs, 1):.1%}) "
                            f"{last_row['combo']} split={last_row['split']} font={last_row['font']}",
                            flush=True,
                        )
                    handle.flush()
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report = {
        "drop_trained_checkpoint": str(args.drop_trained_checkpoint),
        "mean_trained_checkpoint": str(args.mean_trained_checkpoint),
        "samples_per_font": int(args.samples_per_font),
        "eval_batch_size": int(args.eval_batch_size),
        "inference_steps": int(args.inference_steps),
        "style_ref_count": int(args.style_ref_count),
        "split_meta": split_meta,
        "summary_by_split_combo": summarize(metric_rows, ("split", "combo")),
        "summary_by_combo": summarize(metric_rows, ("combo",)),
        "summary_by_split_font_combo": summarize(metric_rows, ("split", "font", "combo")),
        "l1_win_counts_by_split": win_counts(metric_rows, group_key="split"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "split_meta": split_meta,
            "summary_by_split_combo": report["summary_by_split_combo"],
            "l1_win_counts_by_split": report["l1_win_counts_by_split"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
