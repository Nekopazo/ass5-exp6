#!/usr/bin/env python3
"""Probe six-reference cross-attention value/context similarity for x-pred checkpoints."""

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
    model_config = dict(checkpoint.get("model_config", {}))
    if not model_config:
        raise RuntimeError(f"Checkpoint is missing model_config: {checkpoint_path}")
    model = SourcePartRefDiT(**model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def build_dataset(args: argparse.Namespace, image_size: int) -> FontImageDataset:
    glyph_transform = build_base_glyph_transform(image_size=int(image_size))
    return FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=int(args.style_ref_count),
        style_ref_count_min=int(args.style_ref_count),
        style_ref_count_max=int(args.style_ref_count),
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=int(args.font_split_seed),
        font_train_ratio=float(args.font_train_ratio),
        transform=glyph_transform,
        style_transform=glyph_transform,
    )


def choose_sample_indices(dataset: FontImageDataset, count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    sample_count = min(int(count), len(dataset))
    if sample_count <= 0:
        raise RuntimeError("No samples available.")
    return rng.sample(range(len(dataset)), k=sample_count)


def choose_unique_font_sample_indices(dataset: FontImageDataset, count: int, seed: int) -> list[int]:
    rng = random.Random(int(seed))
    font_names = list(dataset.font_names)
    rng.shuffle(font_names)
    selected_fonts = font_names[: min(int(count), len(font_names))]
    sample_indices = []
    for font_name in selected_fonts:
        candidates = list(dataset.sample_indices_by_font[font_name])
        if not candidates:
            continue
        sample_indices.append(int(rng.choice(candidates)))
    return sample_indices


def apply_train_like_shared_style_refs(
    dataset: FontImageDataset,
    samples: list[dict[str, Any]],
    *,
    ref_count: int,
    seed: int,
) -> list[str]:
    excluded_by_font: dict[str, list[int]] = {}
    for sample in samples:
        excluded_by_font.setdefault(str(sample["font"]), []).append(int(sample["char_id"]))

    shared_candidates: set[int] | None = None
    for font_name, excluded_indices in excluded_by_font.items():
        candidates = set(dataset.list_style_candidate_indices(font_name, excluded_indices=excluded_indices))
        shared_candidates = candidates if shared_candidates is None else shared_candidates & candidates
    if not shared_candidates:
        raise RuntimeError("Selected samples have no shared non-overlapping style references.")

    ordered_candidates = sorted(int(idx) for idx in shared_candidates)
    ref_count = min(max(1, int(ref_count)), len(ordered_candidates))
    rng = random.Random(int(seed))
    shared_style_indices = rng.sample(ordered_candidates, k=ref_count)

    style_chars_by_font: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]] = {}
    for sample in samples:
        font_name = str(sample["font"])
        cached = style_chars_by_font.get(font_name)
        if cached is None:
            cached = dataset.load_style_refs_by_indices(font_name, shared_style_indices)
            style_chars_by_font[font_name] = cached
        style_img, style_ref_mask, style_chars = cached
        sample["style_img"] = style_img
        sample["style_ref_mask"] = style_ref_mask
        sample["style_chars"] = style_chars
        sample["style_ref_count_min"] = ref_count
        sample["style_ref_count_max"] = ref_count
    return [dataset.char_list[idx] for idx in shared_style_indices]


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "q1": float(torch.quantile(flat, 0.25).item()),
        "q3": float(torch.quantile(flat, 0.75).item()),
    }


def pairwise_cosine(vectors: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(vectors.detach().float(), dim=-1, eps=1e-8)
    return normalized @ normalized.transpose(-1, -2)


def offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    n = int(matrix.size(-1))
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def iqr_outlier_flags(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().cpu()
    q1 = torch.quantile(values, 0.25)
    q3 = torch.quantile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flags = (values < lower) | (values > upper)
    return {
        "q1": float(q1.item()),
        "q3": float(q3.item()),
        "iqr": float(iqr.item()),
        "lower_fence": float(lower.item()),
        "upper_fence": float(upper.item()),
        "count": int(flags.sum().item()),
        "fraction": float(flags.float().mean().item()),
    }


def coordinate_ref_stats(values: torch.Tensor, *, top_k: int = 50) -> dict[str, Any]:
    """Summarize per-coordinate stats across refs.

    Input shape is [B, R, T, D]. For every [B, T, D] coordinate, compute stats
    over the R reference values only.
    """
    values = values.detach().float().cpu()
    mean = values.mean(dim=1)
    median = values.median(dim=1).values
    std = values.std(dim=1, unbiased=False)
    min_values = values.min(dim=1).values
    max_values = values.max(dim=1).values
    ranges = max_values - min_values

    q1 = torch.quantile(values, 0.25, dim=1)
    q3 = torch.quantile(values, 0.75, dim=1)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (values < lower.unsqueeze(1)) | (values > upper.unsqueeze(1))
    outlier_count = outlier_mask.sum(dim=1)

    flat_range = ranges.reshape(-1)
    flat_std = std.reshape(-1)
    flat_outlier_count = outlier_count.reshape(-1)
    coordinate_count = int(flat_range.numel())
    ref_count = int(values.size(1))
    token_count = int(values.size(2))
    channel_count = int(values.size(3))

    top_count = min(int(top_k), coordinate_count)
    top_values, top_indices = torch.topk(flat_range, k=top_count)
    top_rows = []
    for rank, (range_value, flat_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        sample_idx = flat_idx // (token_count * channel_count)
        rem = flat_idx % (token_count * channel_count)
        token_idx = rem // channel_count
        channel_idx = rem % channel_count
        ref_values = values[sample_idx, :, token_idx, channel_idx]
        top_rows.append(
            {
                "rank": int(rank),
                "sample_pos": int(sample_idx),
                "token_idx": int(token_idx),
                "channel_idx": int(channel_idx),
                "mean": float(mean[sample_idx, token_idx, channel_idx].item()),
                "median": float(median[sample_idx, token_idx, channel_idx].item()),
                "std": float(std[sample_idx, token_idx, channel_idx].item()),
                "min": float(min_values[sample_idx, token_idx, channel_idx].item()),
                "max": float(max_values[sample_idx, token_idx, channel_idx].item()),
                "range": float(range_value),
                "iqr": float(iqr[sample_idx, token_idx, channel_idx].item()),
                "outlier_count": int(outlier_count[sample_idx, token_idx, channel_idx].item()),
                "ref_values": [float(value) for value in ref_values.tolist()],
            }
        )

    return {
        "shape": {
            "samples": int(values.size(0)),
            "refs": ref_count,
            "tokens": token_count,
            "channels": channel_count,
            "coordinates": coordinate_count,
        },
        "mean_over_refs_distribution": tensor_stats(mean),
        "median_over_refs_distribution": tensor_stats(median),
        "std_over_refs_distribution": tensor_stats(std),
        "min_over_refs_distribution": tensor_stats(min_values),
        "max_over_refs_distribution": tensor_stats(max_values),
        "range_over_refs_distribution": tensor_stats(ranges),
        "iqr_over_refs_distribution": tensor_stats(iqr),
        "coordinate_outlier_count_distribution": tensor_stats(outlier_count.float()),
        "coordinates_with_any_iqr_outlier": {
            "count": int((flat_outlier_count > 0).sum().item()),
            "fraction": float((flat_outlier_count > 0).float().mean().item()),
        },
        "scalar_values_flagged_by_iqr": {
            "count": int(outlier_mask.sum().item()),
            "fraction": float(outlier_mask.float().mean().item()),
        },
        "top_coordinates_by_range": top_rows,
    }


@torch.inference_mode()
def probe_batch(
    model: SourcePartRefDiT,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    content = torch.stack([sample["content"] for sample in samples], dim=0).to(device)
    style_img = torch.stack([sample["style_img"] for sample in samples], dim=0).to(device)
    style_ref_mask = torch.stack([sample["style_ref_mask"] for sample in samples], dim=0).to(device)

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

    attn_logits = torch.matmul(expanded_query, flat_key.transpose(-2, -1)) * model.content_style_attn.attn.scale
    attn_weights = torch.softmax(attn_logits.float(), dim=-1)
    context_heads = torch.matmul(attn_weights.to(dtype=flat_value.dtype), flat_value)
    context_tokens = (
        context_heads.transpose(1, 2)
        .contiguous()
        .view(batch * refs, query_len, model.encoder_hidden_dim)
    )
    context_tokens = model.content_style_attn.attn.out_proj(context_tokens)
    context_tokens = context_tokens.view(batch, refs, query_len, model.encoder_hidden_dim)

    value_tokens = (
        style_value.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, refs, key_len, model.encoder_hidden_dim)
    )
    value_vectors = value_tokens.mean(dim=2)
    context_vectors = context_tokens.mean(dim=2)
    token_value_vectors = (
        style_token_bank.mean(dim=2)
    )
    same_position_mass = attn_weights.diagonal(dim1=-2, dim2=-1)
    same_position_mass = same_position_mass.view(batch, refs, heads, query_len).mean(dim=(2, 3))
    entropy = -(attn_weights.clamp_min(1e-12) * attn_weights.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy.view(batch, refs, heads, query_len).mean(dim=(2, 3))

    return {
        "value_vectors": value_vectors,
        "context_vectors": context_vectors,
        "value_tokens": value_tokens,
        "context_tokens": context_tokens,
        "token_value_vectors": token_value_vectors,
        "same_position_mass": same_position_mass,
        "attention_entropy": entropy,
    }


def summarize_probe(
    probe: dict[str, torch.Tensor],
    samples: list[dict[str, Any]],
    sample_indices: list[int],
    checkpoint_path: Path,
) -> dict[str, Any]:
    value_vectors = probe["value_vectors"].cpu()
    context_vectors = probe["context_vectors"].cpu()
    value_tokens = probe["value_tokens"].cpu()
    context_tokens = probe["context_tokens"].cpu()
    token_value_vectors = probe["token_value_vectors"].cpu()
    same_position_mass = probe["same_position_mass"].cpu()
    attention_entropy = probe["attention_entropy"].cpu()

    per_sample = []
    per_ref_rows = []
    value_offdiag_all = []
    context_offdiag_all = []
    token_offdiag_all = []
    for sample_pos, sample in enumerate(samples):
        value_cos = pairwise_cosine(value_vectors[sample_pos]).cpu()
        context_cos = pairwise_cosine(context_vectors[sample_pos]).cpu()
        token_cos = pairwise_cosine(token_value_vectors[sample_pos]).cpu()
        value_offdiag = offdiag_values(value_cos)
        context_offdiag = offdiag_values(context_cos)
        token_offdiag = offdiag_values(token_cos)
        value_offdiag_all.append(value_offdiag)
        context_offdiag_all.append(context_offdiag)
        token_offdiag_all.append(token_offdiag)
        per_sample.append(
            {
                "sample_index": int(sample_indices[sample_pos]),
                "font": str(sample["font"]),
                "target_char": str(sample["char"]),
                "style_chars": [str(ch) for ch in sample["style_chars"]],
                "value_cosine_offdiag": tensor_stats(value_offdiag),
                "context_cosine_offdiag": tensor_stats(context_offdiag),
                "raw_style_token_cosine_offdiag": tensor_stats(token_offdiag),
                "same_position_mass": tensor_stats(same_position_mass[sample_pos]),
                "attention_entropy": tensor_stats(attention_entropy[sample_pos]),
                "value_cosine_matrix": value_cos.tolist(),
                "context_cosine_matrix": context_cos.tolist(),
            }
        )
        for ref_idx, style_char in enumerate(sample["style_chars"]):
            row = {
                "sample_pos": sample_pos,
                "sample_index": int(sample_indices[sample_pos]),
                "font": str(sample["font"]),
                "target_char": str(sample["char"]),
                "ref_idx": ref_idx,
                "style_char": str(style_char),
            }
            for prefix, vectors in (
                ("value", value_vectors),
                ("context", context_vectors),
                ("raw_style_token", token_value_vectors),
            ):
                stats = tensor_stats(vectors[sample_pos, ref_idx])
                row.update({f"{prefix}_{key}": val for key, val in stats.items()})
            row["same_position_mass"] = float(same_position_mass[sample_pos, ref_idx].item())
            row["attention_entropy"] = float(attention_entropy[sample_pos, ref_idx].item())
            per_ref_rows.append(row)

    value_offdiag_all_t = torch.cat(value_offdiag_all)
    context_offdiag_all_t = torch.cat(context_offdiag_all)
    token_offdiag_all_t = torch.cat(token_offdiag_all)
    per_ref_context_means = torch.tensor([row["context_mean"] for row in per_ref_rows])
    per_ref_value_means = torch.tensor([row["value_mean"] for row in per_ref_rows])

    return {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(samples),
        "num_refs_per_sample": int(value_vectors.size(1)),
        "vector_dim": int(value_vectors.size(2)),
        "similarity_summary": {
            "value_cosine_offdiag_all_pairs": tensor_stats(value_offdiag_all_t),
            "context_cosine_offdiag_all_pairs": tensor_stats(context_offdiag_all_t),
            "raw_style_token_cosine_offdiag_all_pairs": tensor_stats(token_offdiag_all_t),
        },
        "per_ref_dimension_mean_summary": {
            "value_mean_across_all_refs": tensor_stats(per_ref_value_means),
            "context_mean_across_all_refs": tensor_stats(per_ref_context_means),
            "value_mean_iqr_outliers": iqr_outlier_flags(per_ref_value_means),
            "context_mean_iqr_outliers": iqr_outlier_flags(per_ref_context_means),
        },
        "attention_summary": {
            "same_position_mass_all_refs": tensor_stats(same_position_mass),
            "attention_entropy_all_refs": tensor_stats(attention_entropy),
        },
        "per_token_channel_ref_stats": {
            "value_tokens": coordinate_ref_stats(value_tokens),
            "context_tokens": coordinate_ref_stats(context_tokens),
        },
        "per_sample": per_sample,
        "per_ref_rows": per_ref_rows,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = report["per_ref_rows"]
    if rows:
        with (output_dir / "per_ref_stats.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    sample_rows = []
    for sample in report["per_sample"]:
        row = {
            "sample_index": sample["sample_index"],
            "font": sample["font"],
            "target_char": sample["target_char"],
            "style_chars": ",".join(sample["style_chars"]),
        }
        for prefix in ("value_cosine_offdiag", "context_cosine_offdiag", "raw_style_token_cosine_offdiag"):
            row.update({f"{prefix}_{key}": val for key, val in sample[prefix].items()})
        sample_rows.append(row)
    if sample_rows:
        with (output_dir / "per_sample_similarity.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(sample_rows)
    coord_stats = report.get("per_token_channel_ref_stats", {})
    for name, stats in coord_stats.items():
        rows = stats.get("top_coordinates_by_range", [])
        if not rows:
            continue
        flat_rows = []
        for row in rows:
            flat = dict(row)
            flat["ref_values"] = ",".join(f"{value:.8g}" for value in flat["ref_values"])
            flat_rows.append(flat)
        with (output_dir / f"{name}_top_coordinates_by_range.tsv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "cross_attn_six_ref_stats")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unique-fonts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--style-ref-count", type=int, default=6)
    parser.add_argument("--train-like-shared-refs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--font-split", type=str, default="test")
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    parser.add_argument("--max-fonts", type=int, default=0)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = build_dataset(args, image_size=int(model.image_size))
    if bool(args.unique_fonts):
        sample_indices = choose_unique_font_sample_indices(dataset, int(args.num_samples), int(args.seed))
    else:
        sample_indices = choose_sample_indices(dataset, int(args.num_samples), int(args.seed))
    samples = [dataset[index] for index in sample_indices]
    shared_style_chars = None
    if bool(args.train_like_shared_refs):
        shared_style_chars = apply_train_like_shared_style_refs(
            dataset,
            samples,
            ref_count=int(args.style_ref_count),
            seed=int(args.seed),
        )
    probe_chunks: list[dict[str, torch.Tensor]] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        chunk_probe = probe_batch(model, chunk, device)
        probe_chunks.append({key: value.detach().cpu() for key, value in chunk_probe.items()})
    probe = {
        key: torch.cat([chunk[key] for chunk in probe_chunks], dim=0)
        for key in probe_chunks[0]
    }
    report = summarize_probe(probe, samples, sample_indices, args.checkpoint)
    report["checkpoint_step"] = int(checkpoint.get("step", -1))
    report["font_split"] = str(args.font_split)
    report["seed"] = int(args.seed)
    report["train_like_shared_refs"] = bool(args.train_like_shared_refs)
    report["shared_style_chars"] = shared_style_chars
    write_outputs(report, args.output_dir)

    sim = report["similarity_summary"]
    means = report["per_ref_dimension_mean_summary"]
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "checkpoint_step": report["checkpoint_step"],
        "num_samples": report["num_samples"],
        "num_refs_per_sample": report["num_refs_per_sample"],
        "value_cosine_offdiag_all_pairs": sim["value_cosine_offdiag_all_pairs"],
        "context_cosine_offdiag_all_pairs": sim["context_cosine_offdiag_all_pairs"],
        "context_mean_iqr_outliers": means["context_mean_iqr_outliers"],
        "value_mean_iqr_outliers": means["value_mean_iqr_outliers"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
