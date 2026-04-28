#!/usr/bin/env python3
"""Compare aggregated style context against original content and style tokens."""

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


def tensor_stats(values: list[float] | torch.Tensor) -> dict[str, float]:
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


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SourcePartRefDiT, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("stage") != "xpred":
        raise RuntimeError(f"Not an x-pred checkpoint: {checkpoint_path}")
    model = SourcePartRefDiT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


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


def flattened_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.reshape(a.size(0), -1).float(), b.reshape(b.size(0), -1).float(), dim=1)


def token_cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=1)


@torch.inference_mode()
def compute_batch(
    model: SourcePartRefDiT,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    content = torch.stack([sample["content"] for sample in samples], dim=0).to(device)
    style_img = torch.stack([sample["style_img"] for sample in samples], dim=0).to(device)
    style_ref_mask = torch.stack([sample["style_ref_mask"] for sample in samples], dim=0).to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
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
        per_ref_context, _ = model.content_style_attn.attn.attend_projected(
            expanded_query,
            flat_key,
            flat_value,
            need_weights=False,
        )
        per_ref_context = per_ref_context.view(batch, refs, query_len, model.encoder_hidden_dim)
        normal_context = per_ref_context.mean(dim=1)
        trimmed_context = model.content_style_attn._trimmed_mean_refs(per_ref_context)
        raw_style_mean = style_token_bank.mean(dim=1)

    return {
        "content_tokens": content_tokens.float().cpu(),
        "raw_style_mean": raw_style_mean.float().cpu(),
        "normal_context": normal_context.float().cpu(),
        "trimmed_context": trimmed_context.float().cpu(),
    }


def build_rows(outputs: dict[str, torch.Tensor], samples: list[dict[str, Any]], sample_indices: list[int]) -> list[dict[str, Any]]:
    content = outputs["content_tokens"]
    style = outputs["raw_style_mean"]
    normal = outputs["normal_context"]
    trimmed = outputs["trimmed_context"]
    rows = []
    for idx, sample in enumerate(samples):
        rows.append(
            {
                "sample_index": int(sample_indices[idx]),
                "font": str(sample["font"]),
                "target_char": str(sample["char"]),
                "style_chars": "".join(str(ch) for ch in sample["style_chars"]),
                "normal_context_vs_content_flat_cos": float(flattened_cosine(normal[idx:idx + 1], content[idx:idx + 1]).item()),
                "normal_context_vs_content_token_cos_mean": float(token_cosine_mean(normal[idx:idx + 1], content[idx:idx + 1]).item()),
                "normal_context_vs_raw_style_mean_flat_cos": float(flattened_cosine(normal[idx:idx + 1], style[idx:idx + 1]).item()),
                "normal_context_vs_raw_style_mean_token_cos_mean": float(token_cosine_mean(normal[idx:idx + 1], style[idx:idx + 1]).item()),
                "trimmed_context_vs_content_flat_cos": float(flattened_cosine(trimmed[idx:idx + 1], content[idx:idx + 1]).item()),
                "trimmed_context_vs_content_token_cos_mean": float(token_cosine_mean(trimmed[idx:idx + 1], content[idx:idx + 1]).item()),
                "trimmed_context_vs_raw_style_mean_flat_cos": float(flattened_cosine(trimmed[idx:idx + 1], style[idx:idx + 1]).item()),
                "trimmed_context_vs_raw_style_mean_token_cos_mean": float(token_cosine_mean(trimmed[idx:idx + 1], style[idx:idx + 1]).item()),
                "normal_vs_trimmed_flat_cos": float(flattened_cosine(normal[idx:idx + 1], trimmed[idx:idx + 1]).item()),
                "normal_vs_trimmed_l1": float((normal[idx] - trimmed[idx]).abs().mean().item()),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "context_vs_content_style")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
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
    samples = [dataset[index] for index in sample_indices]
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(samples), batch_size):
        chunk_samples = samples[start:start + batch_size]
        chunk_indices = sample_indices[start:start + batch_size]
        outputs = compute_batch(model, chunk_samples, device)
        rows.extend(build_rows(outputs, chunk_samples, chunk_indices))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "per_sample_similarity.tsv", rows)
    keys = [key for key in rows[0] if key not in {"sample_index", "font", "target_char", "style_chars"}]
    overall = {key: tensor_stats([row[key] for row in rows]) for key in keys}
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "num_samples": len(rows),
        "style_ref_count": int(args.style_ref_count),
        "overall": overall,
        "rows": rows,
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
            "overall": overall,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
