#!/usr/bin/env python3
"""Compare normal mean vs drop-min-max fusion over multiple style-ref groups."""

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
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(raw_device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SourcePartRefDiT, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("stage") != "xpred":
        raise RuntimeError(f"Not an x-pred checkpoint: {checkpoint_path}")
    model = SourcePartRefDiT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def tensor_stats(values: torch.Tensor | list[float]) -> dict[str, float]:
    x = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "q1": float(torch.quantile(x, 0.25).item()),
        "q3": float(torch.quantile(x, 0.75).item()),
    }


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


def sample_ref_groups(
    dataset: FontImageDataset,
    *,
    font_name: str,
    target_char_id: int,
    ref_count: int,
    group_count: int,
    rng: random.Random,
) -> list[list[int]]:
    candidates = dataset.list_style_candidate_indices(font_name, excluded_indices=[int(target_char_id)])
    if len(candidates) < int(ref_count):
        raise RuntimeError(
            f"Font {font_name!r} only has {len(candidates)} style candidates, need {ref_count}."
        )
    groups = []
    for _ in range(int(group_count)):
        groups.append(rng.sample(candidates, k=int(ref_count)))
    return groups


@torch.inference_mode()
def compute_style_context_per_group(
    model: SourcePartRefDiT,
    *,
    content: torch.Tensor,
    style_img: torch.Tensor,
    style_ref_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    content = content.to(device)
    style_img = style_img.to(device)
    style_ref_mask = style_ref_mask.to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        content_tokens = model.encode_content_tokens(content)
        style_token_bank, token_valid_mask = model.encode_style_token_bank(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
        )
        content_query = model.precompute_content_query(content_tokens)
        style_key, style_value = model.precompute_style_kv(
            style_token_bank,
            token_valid_mask=token_valid_mask,
        )

        batch, refs, heads, key_len, head_dim = style_key.shape
        query_len = int(content_query.size(2))
        expanded_query = (
            content_query.unsqueeze(1)
            .expand(batch, refs, heads, query_len, head_dim)
            .reshape(batch * refs, heads, query_len, head_dim)
        )
        flat_key = style_key.reshape(batch * refs, heads, key_len, head_dim)
        flat_value = style_value.reshape(batch * refs, heads, key_len, head_dim)
        style_context_per_ref, _ = model.content_style_attn.attn.attend_projected(
            expanded_query,
            flat_key,
            flat_value,
            need_weights=False,
        )
    return style_context_per_ref.view(batch, refs, query_len, model.encoder_hidden_dim).float().cpu()


def offdiag_cosine_summary(vectors: torch.Tensor) -> dict[str, float]:
    if vectors.size(0) <= 1:
        return tensor_stats([1.0])
    flat = vectors.reshape(vectors.size(0), -1).float()
    normed = F.normalize(flat, dim=1, eps=1e-8)
    matrix = normed @ normed.T
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool)
    return tensor_stats(matrix[mask])


def compare_groups(
    model: SourcePartRefDiT,
    dataset: FontImageDataset,
    sample_indices: list[int],
    *,
    ref_count: int,
    groups_per_sample: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed) + 1009)
    group_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for sample_pos, sample_index in enumerate(sample_indices):
        base = dataset[int(sample_index)]
        font_name = str(base["font"])
        target_char = str(base["char"])
        target_char_id = int(base["char_id"])
        ref_groups = sample_ref_groups(
            dataset,
            font_name=font_name,
            target_char_id=target_char_id,
            ref_count=int(ref_count),
            group_count=int(groups_per_sample),
            rng=rng,
        )

        style_imgs = []
        style_masks = []
        style_chars_by_group = []
        for indices in ref_groups:
            style_img, style_mask, style_chars = dataset.load_style_refs_by_indices(font_name, indices)
            style_imgs.append(style_img)
            style_masks.append(style_mask)
            style_chars_by_group.append(style_chars)

        content = base["content"].unsqueeze(0).expand(len(style_imgs), -1, -1, -1).contiguous()
        style_img_batch = torch.stack(style_imgs, dim=0)
        style_mask_batch = torch.stack(style_masks, dim=0)
        per_ref_context = compute_style_context_per_group(
            model,
            content=content,
            style_img=style_img_batch,
            style_ref_mask=style_mask_batch,
            device=device,
        )
        normal_context = per_ref_context.mean(dim=1)
        trimmed_context = model.content_style_attn._trimmed_mean_refs(per_ref_context)

        normal_flat = normal_context.reshape(normal_context.size(0), -1)
        trimmed_flat = trimmed_context.reshape(trimmed_context.size(0), -1)
        pair_cos = F.cosine_similarity(normal_flat, trimmed_flat, dim=1)
        diff = (trimmed_context - normal_context).float()
        normal_abs = normal_context.abs().mean(dim=(1, 2)).clamp_min(1e-8)

        for group_idx in range(len(ref_groups)):
            group_rows.append(
                {
                    "sample_pos": sample_pos,
                    "sample_index": int(sample_index),
                    "font": font_name,
                    "target_char": target_char,
                    "group_idx": group_idx,
                    "style_chars": "".join(style_chars_by_group[group_idx]),
                    "normal_vs_trimmed_cosine": float(pair_cos[group_idx].item()),
                    "l1": float(diff[group_idx].abs().mean().item()),
                    "rmse": float(diff[group_idx].pow(2).mean().sqrt().item()),
                    "max_abs": float(diff[group_idx].abs().max().item()),
                    "relative_l1_to_normal_abs_mean": float((diff[group_idx].abs().mean() / normal_abs[group_idx]).item()),
                }
            )

        sample_rows.append(
            {
                "sample_pos": sample_pos,
                "sample_index": int(sample_index),
                "font": font_name,
                "target_char": target_char,
                "groups": int(len(ref_groups)),
                "normal_vs_trimmed_cosine": tensor_stats(pair_cos),
                "l1": tensor_stats([row["l1"] for row in group_rows if row["sample_pos"] == sample_pos]),
                "rmse": tensor_stats([row["rmse"] for row in group_rows if row["sample_pos"] == sample_pos]),
                "max_abs": tensor_stats([row["max_abs"] for row in group_rows if row["sample_pos"] == sample_pos]),
                "relative_l1_to_normal_abs_mean": tensor_stats(
                    [row["relative_l1_to_normal_abs_mean"] for row in group_rows if row["sample_pos"] == sample_pos]
                ),
                "normal_cross_group_cosine": offdiag_cosine_summary(normal_context),
                "trimmed_cross_group_cosine": offdiag_cosine_summary(trimmed_context),
            }
        )

    return group_rows, sample_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for stat_key, stat_value in value.items():
                    flat[f"{key}_{stat_key}"] = stat_value
            else:
                flat[key] = value
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "compare_trimmed_style_groups")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--groups-per-sample", type=int, default=8)
    parser.add_argument("--style-ref-count", type=int, default=6)
    parser.add_argument("--font-split", type=str, default="train")
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    parser.add_argument("--max-fonts", type=int, default=0)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    transform = build_base_glyph_transform(image_size=int(model.image_size))
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
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
    group_rows, sample_rows = compare_groups(
        model,
        dataset,
        sample_indices,
        ref_count=int(args.style_ref_count),
        groups_per_sample=int(args.groups_per_sample),
        seed=int(args.seed),
        device=device,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "group_comparison.tsv", group_rows)
    write_tsv(output_dir / "sample_summary.tsv", sample_rows)

    all_cos = [row["normal_vs_trimmed_cosine"] for row in group_rows]
    all_l1 = [row["l1"] for row in group_rows]
    all_rel = [row["relative_l1_to_normal_abs_mean"] for row in group_rows]
    normal_group_cos = []
    trimmed_group_cos = []
    for row in sample_rows:
        normal_group_cos.append(row["normal_cross_group_cosine"]["mean"])
        trimmed_group_cos.append(row["trimmed_cross_group_cosine"]["mean"])
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "num_samples": len(sample_indices),
        "groups_per_sample": int(args.groups_per_sample),
        "style_ref_count": int(args.style_ref_count),
        "sample_indices": sample_indices,
        "overall": {
            "normal_vs_trimmed_cosine": tensor_stats(all_cos),
            "l1": tensor_stats(all_l1),
            "relative_l1_to_normal_abs_mean": tensor_stats(all_rel),
            "normal_cross_group_cosine_mean_by_sample": tensor_stats(normal_group_cos),
            "trimmed_cross_group_cosine_mean_by_sample": tensor_stats(trimmed_group_cos),
            "trimmed_minus_normal_cross_group_cosine": tensor_stats(
                [t - n for t, n in zip(trimmed_group_cos, normal_group_cos)]
            ),
        },
        "samples": sample_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "checkpoint_step": report["checkpoint_step"],
            "num_samples": report["num_samples"],
            "groups_per_sample": report["groups_per_sample"],
            "overall": report["overall"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
