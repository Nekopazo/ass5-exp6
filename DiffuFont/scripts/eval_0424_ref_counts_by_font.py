#!/usr/bin/env python3
"""Evaluate one training run at multiple checkpoints and style-ref counts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

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


METRIC_FIELDS = [
    "group",
    "checkpoint_label",
    "checkpoint",
    "ref_count",
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


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best_index = 0
        best_free_bytes = -1
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
            if int(free_bytes) > best_free_bytes:
                best_index = int(index)
                best_free_bytes = int(free_bytes)
        return torch.device(f"cuda:{best_index}")
    return torch.device(raw_device)


def force_plain_ref_mean() -> None:
    ContentStyleCrossAttention._trimmed_mean_refs = staticmethod(lambda context_per_ref: context_per_ref.mean(dim=1))


def image_metrics(generation: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
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
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def choose_font_names(dataset: FontImageDataset, max_fonts: int, seed: int) -> list[str]:
    font_names = list(dataset.font_names)
    rng = random.Random(seed)
    rng.shuffle(font_names)
    if max_fonts > 0:
        return font_names[: int(max_fonts)]
    return font_names


def choose_samples_for_font(dataset: FontImageDataset, font_name: str, samples_per_font: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    candidates = list(dataset.sample_indices_by_font[font_name])
    rng.shuffle(candidates)
    if len(candidates) >= samples_per_font:
        return [int(idx) for idx in candidates[:samples_per_font]]
    return [int(rng.choice(candidates)) for _ in range(samples_per_font)]


def choose_style_indices(
    dataset: FontImageDataset,
    *,
    font_name: str,
    target_char_id: int,
    max_ref_count: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    candidates = [
        int(idx)
        for idx in dataset.valid_indices_by_font[font_name]
        if int(idx) != int(target_char_id)
    ]
    if not candidates:
        raise RuntimeError(f"Font '{font_name}' has no style candidates.")
    rng.shuffle(candidates)
    if len(candidates) >= max_ref_count:
        return candidates[:max_ref_count]
    output = list(candidates)
    while len(output) < max_ref_count:
        output.append(int(rng.choice(candidates)))
    return output


def build_records(
    dataset: FontImageDataset,
    *,
    split: str,
    samples_per_font: int,
    max_fonts: int,
    max_ref_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    font_names = choose_font_names(dataset, max_fonts=max_fonts, seed=seed)
    for font_pos, font_name in enumerate(font_names):
        sample_indices = choose_samples_for_font(
            dataset,
            font_name,
            samples_per_font=samples_per_font,
            seed=seed * 1009 + font_pos,
        )
        for sample_pos, sample_index in enumerate(sample_indices):
            target_font, target_char_id = dataset.samples[int(sample_index)]
            if target_font != font_name:
                raise RuntimeError(f"sample/font mismatch: {target_font} vs {font_name}")
            style_indices = choose_style_indices(
                dataset,
                font_name=font_name,
                target_char_id=int(target_char_id),
                max_ref_count=max_ref_count,
                seed=seed * 1_000_003 + int(sample_index),
            )
            records.append(
                {
                    "split": split,
                    "font": font_name,
                    "font_pos": int(font_pos),
                    "sample_pos_in_font": int(sample_pos),
                    "sample_index": int(sample_index),
                    "target_char_id": int(target_char_id),
                    "style_indices": style_indices,
                }
            )
    return records


def load_record_sample(dataset: FontImageDataset, record: dict[str, Any], ref_count: int) -> dict[str, Any]:
    sample = dataset[int(record["sample_index"])]
    style_img, style_ref_mask, style_chars = dataset.load_style_refs_by_indices(
        str(record["font"]),
        [int(idx) for idx in record["style_indices"][: int(ref_count)]],
    )
    sample["style_img"] = style_img
    sample["style_ref_mask"] = style_ref_mask
    sample["style_chars"] = style_chars
    return sample


def iter_chunks(records: list[dict[str, Any]], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


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


def win_counts_by_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(row["group"]) for row in rows})
    output: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        by_sample: dict[int, dict[str, dict[str, Any]]] = {}
        for row in split_rows:
            by_sample.setdefault(int(row["sample_index"]), {})[str(row["group"])] = row
        split_counts: dict[str, dict[str, int]] = {}
        for a in groups:
            split_counts[a] = {}
            for b in groups:
                if a == b:
                    continue
                valid = 0
                wins = 0
                for sample_map in by_sample.values():
                    if a not in sample_map or b not in sample_map:
                        continue
                    valid += 1
                    if float(sample_map[a]["l1"]) < float(sample_map[b]["l1"]):
                        wins += 1
                split_counts[a][b] = int(wins)
                split_counts[a][f"{b}__valid"] = int(valid)
        output[split] = split_counts
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-label", type=str, action="append", required=True)
    parser.add_argument("--ref-counts", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-font", type=int, default=30)
    parser.add_argument("--max-train-fonts", type=int, default=0)
    parser.add_argument("--max-test-fonts", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    args = parser.parse_args()

    if len(args.checkpoint) != len(args.checkpoint_label):
        raise ValueError("--checkpoint and --checkpoint-label must have the same length.")
    ref_counts = [int(x) for x in args.ref_counts]
    max_ref_count = max(ref_counts)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    force_plain_ref_mean()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.tsv"

    device = resolve_device(args.device)
    probe_trainer = load_trainer(args.checkpoint[0], device)
    transform = build_base_glyph_transform(image_size=int(probe_trainer.model.image_size))
    del probe_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    datasets: dict[str, FontImageDataset] = {}
    all_records: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    for split, max_fonts in (("train", int(args.max_train_fonts)), ("test", int(args.max_test_fonts))):
        dataset = FontImageDataset(
            project_root=args.data_root,
            style_ref_count=max_ref_count,
            style_ref_count_min=max_ref_count,
            style_ref_count_max=max_ref_count,
            random_seed=int(args.seed),
            font_split=split,
            font_split_seed=int(args.font_split_seed),
            font_train_ratio=float(args.font_train_ratio),
            transform=transform,
            style_transform=transform,
            load_style_refs=False,
        )
        records = build_records(
            dataset,
            split=split,
            samples_per_font=int(args.samples_per_font),
            max_fonts=int(max_fonts),
            max_ref_count=max_ref_count,
            seed=int(args.seed) + (0 if split == "train" else 100000),
        )
        datasets[split] = dataset
        all_records.extend(records)
        split_meta[split] = {
            "available_fonts": int(len(dataset.font_names)),
            "used_fonts": int(len({record["font"] for record in records})),
            "samples": int(len(records)),
            "max_fonts": int(max_fonts),
        }

    total_jobs = len(args.checkpoint) * len(ref_counts) * len(all_records)
    completed = 0
    rows: list[dict[str, Any]] = []
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, delimiter="\t")
        writer.writeheader()
        handle.flush()
        for ckpt_label, ckpt_path in zip(args.checkpoint_label, args.checkpoint):
            force_plain_ref_mean()
            trainer = load_trainer(ckpt_path, device)
            for ref_count in ref_counts:
                group = f"{ckpt_label}__ref{ref_count}"
                for batch_start, record_batch in iter_chunks(all_records, int(args.eval_batch_size)):
                    sample_batch = [
                        load_record_sample(datasets[str(record["split"])], record, int(ref_count))
                        for record in record_batch
                    ]
                    generations = generate_batch(
                        trainer,
                        sample_batch,
                        seed=int(args.seed) * 1_000_003 + batch_start + int(ref_count) * 10_000,
                        inference_steps=int(args.inference_steps),
                    )
                    for offset, record in enumerate(record_batch):
                        sample = sample_batch[offset]
                        row = {
                            "group": group,
                            "checkpoint_label": ckpt_label,
                            "checkpoint": str(ckpt_path),
                            "ref_count": int(ref_count),
                            "split": str(record["split"]),
                            "font": str(record["font"]),
                            "font_pos": int(record["font_pos"]),
                            "sample_pos_in_font": int(record["sample_pos_in_font"]),
                            "sample_index": int(record["sample_index"]),
                            "target_char": str(sample["char"]),
                            "style_chars": "".join(str(ch) for ch in sample["style_chars"]),
                        }
                        row.update(image_metrics(generations[offset], sample["target"]))
                        rows.append(row)
                        writer.writerow(row)
                        completed += 1
                    if completed == len(record_batch) or completed % 100 <= len(record_batch) or completed == total_jobs:
                        last = rows[-1]
                        print(
                            f"[progress] {completed}/{total_jobs} "
                            f"({completed / max(total_jobs, 1):.1%}) "
                            f"{last['group']} split={last['split']} font={last['font']}",
                            flush=True,
                        )
                    handle.flush()
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "checkpoints": [
            {"label": label, "path": str(path)}
            for label, path in zip(args.checkpoint_label, args.checkpoint)
        ],
        "ref_counts": ref_counts,
        "samples_per_font": int(args.samples_per_font),
        "eval_batch_size": int(args.eval_batch_size),
        "inference_steps": int(args.inference_steps),
        "aggregation": "plain_mean_no_drop_minmax",
        "split_meta": split_meta,
        "summary_by_split_group": summarize(rows, ("split", "group")),
        "summary_by_group": summarize(rows, ("group",)),
        "summary_by_split_font_group": summarize(rows, ("split", "font", "group")),
        "l1_win_counts_by_split": win_counts_by_split(rows),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "split_meta": split_meta,
            "summary_by_split_group": report["summary_by_split_group"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
