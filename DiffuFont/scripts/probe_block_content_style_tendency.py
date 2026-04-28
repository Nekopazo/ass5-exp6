#!/usr/bin/env python3
"""Probe per-block content/style tendency in the DiT conditioning path."""

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


def build_batch(dataset: FontImageDataset, sample_indices: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = [dataset[int(idx)] for idx in sample_indices]
    batch = {
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "content_index": torch.arange(len(samples), dtype=torch.long),
        "style_img": torch.stack([sample["style_img"] for sample in samples], dim=0),
        "style_ref_mask": torch.stack([sample["style_ref_mask"] for sample in samples], dim=0),
        "style_index": torch.arange(len(samples), dtype=torch.long),
    }
    metadata = [
        {
            "sample_index": int(sample_indices[idx]),
            "font": str(sample["font"]),
            "target_char": str(sample["char"]),
            "style_chars": "".join(str(ch) for ch in sample["style_chars"]),
        }
        for idx, sample in enumerate(samples)
    ]
    return batch, metadata


@torch.inference_mode()
def build_conditioning_tokens(
    model: SourcePartRefDiT,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    content = batch["content"].to(device)
    style_img = batch["style_img"].to(device)
    style_ref_mask = batch["style_ref_mask"].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        content_tokens = model.encode_content_tokens(content)
        style_token_bank, token_valid_mask = model.encode_style_token_bank(
            style_img=style_img,
            style_ref_mask=style_ref_mask,
        )
        conditioning_tokens = model.build_conditioning_tokens(
            content_tokens,
            style_token_bank,
            token_valid_mask=token_valid_mask,
        )
    return conditioning_tokens.float()


@torch.inference_mode()
def probe_blocks(
    model: SourcePartRefDiT,
    conditioning_tokens: torch.Tensor,
    *,
    timesteps: list[float],
    device: torch.device,
) -> list[dict[str, Any]]:
    conditioning_tokens = conditioning_tokens.to(device)
    content_tokens, style_tokens = conditioning_tokens.split(model.encoder_hidden_dim, dim=-1)
    rows: list[dict[str, Any]] = []
    eps = 1e-8

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        for block_idx, block in enumerate(model.backbone.blocks, start=1):
            if not block.use_content_injection:
                continue
            content_hidden = block.content_condition_to_hidden(block.content_condition_norm(content_tokens))
            style_hidden = block.style_condition_to_hidden(block.style_condition_norm(style_tokens))

            content_rms = content_hidden.float().pow(2).mean(dim=-1).sqrt()
            style_rms = style_hidden.float().pow(2).mean(dim=-1).sqrt()

            for timestep in timesteps:
                t = torch.full(
                    (conditioning_tokens.size(0),),
                    float(timestep),
                    device=device,
                    dtype=torch.float32,
                )
                time_cond = model.backbone.build_time_cond(t, dtype=conditioning_tokens.dtype)
                time_hidden = block.time_to_hidden(time_cond).unsqueeze(1).expand_as(content_hidden)

                mod_full = block.joint_mod(torch.nn.functional.silu(time_hidden + content_hidden + style_hidden))
                mod_no_content = block.joint_mod(torch.nn.functional.silu(time_hidden + style_hidden))
                mod_no_style = block.joint_mod(torch.nn.functional.silu(time_hidden + content_hidden))
                content_delta = (mod_full - mod_no_content).float()
                style_delta = (mod_full - mod_no_style).float()
                content_delta_rms = content_delta.pow(2).mean(dim=-1).sqrt()
                style_delta_rms = style_delta.pow(2).mean(dim=-1).sqrt()

                rows.append(
                    {
                        "block": int(block_idx),
                        "timestep": float(timestep),
                        "content_hidden_rms": tensor_stats(content_rms.cpu()),
                        "style_hidden_rms": tensor_stats(style_rms.cpu()),
                        "content_hidden_over_style_hidden": tensor_stats(
                            (content_rms / style_rms.clamp_min(eps)).cpu()
                        ),
                        "content_delta_rms": tensor_stats(content_delta_rms.cpu()),
                        "style_delta_rms": tensor_stats(style_delta_rms.cpu()),
                        "content_delta_over_style_delta": tensor_stats(
                            (content_delta_rms / style_delta_rms.clamp_min(eps)).cpu()
                        ),
                        "tendency": (
                            "content"
                            if float(content_delta_rms.mean().item()) > float(style_delta_rms.mean().item())
                            else "style"
                        ),
                    }
                )
    return rows


def static_weight_rows(model: SourcePartRefDiT) -> list[dict[str, Any]]:
    rows = []
    for block_idx, block in enumerate(model.backbone.blocks, start=1):
        if not block.use_content_injection:
            continue
        content_weight_norm = float(block.content_condition_to_hidden.weight.detach().float().norm().item())
        style_weight_norm = float(block.style_condition_to_hidden.weight.detach().float().norm().item())
        rows.append(
            {
                "block": int(block_idx),
                "content_weight_norm": content_weight_norm,
                "style_weight_norm": style_weight_norm,
                "content_over_style_weight_norm": content_weight_norm / max(style_weight_norm, 1e-8),
            }
        )
    return rows


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return flat_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = flatten_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(flat_rows)


def summarize_by_block(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(int(row["block"]), []).append(row)
    summary = []
    for block, block_rows in sorted(by_block.items()):
        ratio_values = [row["content_delta_over_style_delta"]["mean"] for row in block_rows]
        content_delta = [row["content_delta_rms"]["mean"] for row in block_rows]
        style_delta = [row["style_delta_rms"]["mean"] for row in block_rows]
        summary.append(
            {
                "block": int(block),
                "content_delta_rms_mean_over_timesteps": float(np.mean(content_delta)),
                "style_delta_rms_mean_over_timesteps": float(np.mean(style_delta)),
                "content_delta_over_style_delta_mean": float(np.mean(ratio_values)),
                "tendency": "content" if float(np.mean(ratio_values)) > 1.0 else "style",
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "block_content_style_tendency")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--style-ref-count", type=int, default=6)
    parser.add_argument("--font-split", type=str, default="train")
    parser.add_argument("--font-split-seed", type=int, default=42)
    parser.add_argument("--font-train-ratio", type=float, default=0.95)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--timesteps", type=str, default="0.1,0.5,0.9")
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
    batch, metadata = build_batch(dataset, sample_indices)
    conditioning_tokens = build_conditioning_tokens(model, batch, device)
    timesteps = [float(item.strip()) for item in str(args.timesteps).split(",") if item.strip()]
    rows = probe_blocks(model, conditioning_tokens, timesteps=timesteps, device=device)
    block_summary = summarize_by_block(rows)
    weight_rows = static_weight_rows(model)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "block_tendency_by_timestep.tsv", rows)
    write_tsv(output_dir / "block_tendency_summary.tsv", block_summary)
    write_tsv(output_dir / "block_static_weight_norms.tsv", weight_rows)
    (output_dir / "sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "num_samples": len(sample_indices),
        "timesteps": timesteps,
        "block_summary": block_summary,
        "static_weight_norms": weight_rows,
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
            "block_summary": block_summary,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
