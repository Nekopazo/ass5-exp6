#!/usr/bin/env python3
"""Profile peak CUDA memory by major pixel-flow training stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import FontImageDataset, UniqueFontBatchSampler
from models.model import FlowTrainer
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_batch_ref_count(samples) -> int:
    min_refs = max(int(sample.get("style_ref_count_min", sample["style_img"].size(0))) for sample in samples)
    max_refs = min(int(sample.get("style_ref_count_max", sample["style_img"].size(0))) for sample in samples)
    available_refs = min(int(sample["style_img"].size(0)) for sample in samples)
    max_refs = min(max_refs, available_refs)
    if max_refs < min_refs:
        raise RuntimeError(f"Invalid style ref bounds in batch: min_refs={min_refs} max_refs={max_refs}")
    return random.randint(min_refs, max_refs) if min_refs < max_refs else max_refs


def collate_fn(samples):
    ref_count = _resolve_batch_ref_count(samples)
    return {
        "font": [sample["font"] for sample in samples],
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "style_img": torch.stack([sample["style_img"][:ref_count] for sample in samples], dim=0),
        "style_ref_mask": torch.stack([sample["style_ref_mask"][:ref_count] for sample in samples], dim=0),
    }


def to_gb(value: int) -> float:
    return round(value / float(1024**3), 4)


def stage_record(device: torch.device, name: str, fn, records: list[dict]) -> object:
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    records.append(
        {
            "stage": name,
            "allocated_before_gb": to_gb(before),
            "allocated_after_gb": to_gb(after),
            "allocated_delta_gb": to_gb(after - before),
            "peak_delta_gb": to_gb(max(0, peak - before)),
        }
    )
    return result


def build_model_from_args(args: argparse.Namespace) -> SourcePartRefDiT:
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise RuntimeError("Flow checkpoint is missing 'model_config'.")
        model = SourcePartRefDiT(**model_config)
        state_dict = checkpoint.get("model_state")
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=True)
        return model
    return SourcePartRefDiT(
        in_channels=1,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        encoder_patch_size=int(args.encoder_patch_size),
        encoder_hidden_dim=int(args.encoder_hidden_dim),
        encoder_depth=int(args.encoder_depth),
        encoder_heads=int(args.encoder_heads),
        dit_hidden_dim=int(args.dit_hidden_dim),
        dit_depth=int(args.dit_depth),
        dit_heads=int(args.dit_heads),
        dit_mlp_ratio=float(args.dit_mlp_ratio),
        content_fusion_start=int(args.content_fusion_start),
        content_fusion_end=int(args.content_fusion_end),
        style_fusion_start=int(args.style_fusion_start),
        style_fusion_end=int(args.style_fusion_end),
        detailer_base_channels=int(args.detailer_base_channels),
        detailer_max_channels=int(args.detailer_max_channels),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--style-ref-count", type=int, default=0)
    parser.add_argument("--style-ref-count-min", type=int, default=6)
    parser.add_argument("--style-ref-count-max", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--font-split-seed", type=int, default=None)
    parser.add_argument("--font-train-ratio", type=float, default=0.9)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis" / "memory_profile.json")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--encoder-patch-size", type=int, default=8)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--dit-hidden-dim", type=int, default=512)
    parser.add_argument("--dit-depth", type=int, default=12)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--content-fusion-start", type=int, default=0)
    parser.add_argument("--content-fusion-end", type=int, default=8)
    parser.add_argument("--style-fusion-start", type=int, default=6)
    parser.add_argument("--style-fusion-end", type=int, default=12)
    parser.add_argument("--detailer-base-channels", type=int, default=32)
    parser.add_argument("--detailer-max-channels", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("profile_memory.py requires a CUDA device.")
    font_split_seed = int(args.seed) if args.font_split_seed is None else int(args.font_split_seed)
    style_ref_count = None if int(args.style_ref_count) <= 0 else int(args.style_ref_count)
    glyph_transform = build_base_glyph_transform(image_size=int(args.image_size))
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=style_ref_count,
        style_ref_count_min=int(args.style_ref_count_min),
        style_ref_count_max=int(args.style_ref_count_max),
        include_positive_style=False,
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=font_split_seed,
        font_train_ratio=float(args.font_train_ratio),
        transform=glyph_transform,
        style_transform=glyph_transform,
    )
    batch_sampler = UniqueFontBatchSampler(
        dataset,
        batch_size=int(args.batch),
        seed=int(args.seed),
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    batch = next(iter(dataloader))

    model = build_model_from_args(args)
    trainer = FlowTrainer(
        model,
        device,
        lr=1e-4,
        total_steps=1,
        lambda_flow=1.0,
        flow_sample_steps=24,
        ema_decay=float(args.ema_decay),
        log_every_steps=1,
        save_every_steps=None,
        grad_clip_norm=float(args.grad_clip_norm),
    )
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    target = batch["target"].to(device)
    content = batch["content"].to(device)
    style = batch["style_img"].to(device)
    style_ref_mask = batch["style_ref_mask"].to(device)

    records: list[dict] = []
    torch.cuda.empty_cache()
    with trainer._autocast_context():
        x1 = target
        x0 = stage_record(device, "noise_sample", lambda: torch.randn_like(x1), records)
        timesteps = stage_record(device, "time_sample", lambda: torch.rand(x1.size(0), device=device), records)
        t_view = timesteps.view(-1, 1, 1, 1).to(dtype=x1.dtype)
        xt = stage_record(device, "flow_interpolate", lambda: (1.0 - t_view) * x0 + t_view * x1, records)
        target_flow = stage_record(device, "flow_target", lambda: x1 - x0, records)
        content_features = stage_record(device, "content_encode_features", lambda: trainer.model.encode_content_features(content), records)
        content_tokens = stage_record(device, "content_project", lambda: trainer.model.content_proj(content_features), records)
        style_pack = stage_record(
            device,
            "style_encode",
            lambda: trainer.model.encode_style(
                style_img=style,
                style_ref_mask=style_ref_mask,
                return_contrastive=False,
                detach_style_encoder=(not trainer.style_grad_enabled),
            ),
            records,
        )
        style_tokens = style_pack["style_tokens"]
        style_global = style_pack["style_global"]
        style_token_mask = style_pack["style_token_mask"]
        if style_tokens is None or style_token_mask is None or style_global is None:
            raise RuntimeError("profile_memory requires style tokens.")
        pred_flow = stage_record(
            device,
            "dip_forward",
            lambda: trainer.model.predict_flow(
                xt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_tokens,
                style_global=style_global,
                style_token_mask=style_token_mask,
            ),
            records,
        )
        loss = stage_record(device, "flow_loss", lambda: F.mse_loss(pred_flow, target_flow), records)

    stage_record(device, "backward", lambda: loss.backward(), records)
    if trainer.grad_clip_norm is not None:
        stage_record(device, "grad_clip", lambda: trainer._apply_grad_clip(), records)
    trainer.optimizer.step()

    result = {
        "config": {
            "device": str(device),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "batch": int(args.batch),
            "style_ref_count": None if style_ref_count is None else int(style_ref_count),
            "style_ref_count_min": int(args.style_ref_count_min),
            "style_ref_count_max": int(args.style_ref_count_max),
            "max_fonts": int(args.max_fonts),
            "num_workers": int(args.num_workers),
            "grad_clip_norm": float(args.grad_clip_norm),
            "ema_decay": float(args.ema_decay),
        },
        "largest_peak_stage": max(records, key=lambda row: row["peak_delta_gb"]),
        "largest_alloc_delta_stage": max(records, key=lambda row: row["allocated_delta_gb"]),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
