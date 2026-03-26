#!/usr/bin/env python3
"""Profile peak CUDA memory by major training stage for one flow step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

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


def collate_fn(samples):
    return {
        "font": [sample["font"] for sample in samples],
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "style_img": torch.stack([sample["style_img"] for sample in samples], dim=0),
        "style_ref_mask": torch.stack([sample["style_ref_mask"] for sample in samples], dim=0),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--style-ref-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--font-split-seed", type=int, default=None)
    parser.add_argument("--font-train-ratio", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "analysis" / "memory_profile.json")
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device)
    font_split_seed = int(args.seed) if args.font_split_seed is None else int(args.font_split_seed)
    glyph_transform = build_base_glyph_transform(image_size=128)
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=int(args.style_ref_count),
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

    model = SourcePartRefDiT(
        in_channels=1,
        image_size=128,
        latent_channels=10,
        latent_size=16,
        encoder_patch_size=8,
        encoder_hidden_dim=512,
        encoder_depth=4,
        encoder_heads=8,
        local_style_tokens_per_ref=24,
        style_mid_tokens_per_ref=12,
        style_residual_tokens=8,
        dit_hidden_dim=512,
        dit_depth=16,
        dit_heads=8,
        dit_mlp_ratio=4.0,
        content_cross_attn_layers=8,
        style_cross_attn_every_n_layers=1,
    )
    model.load_vae_checkpoint(args.vae_checkpoint)
    trainer = FlowTrainer(
        model,
        device,
        lr=1e-4,
        total_steps=1,
        lambda_flow=1.0,
        freeze_vae=True,
        flow_sample_steps=24,
        log_every_steps=1,
        save_every_steps=None,
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
        z1, _, _ = stage_record(device, "vae_encode", lambda: trainer._encode_latent(target), records)
        z0 = stage_record(device, "noise_sample", lambda: torch.randn_like(z1), records)
        timesteps = stage_record(device, "time_sample", lambda: torch.rand(z1.size(0), device=device), records)
        t_view = timesteps.view(-1, 1, 1, 1).to(dtype=z1.dtype)
        zt = stage_record(device, "flow_interpolate", lambda: (1.0 - t_view) * z0 + t_view * z1, records)
        target_flow = stage_record(device, "flow_target", lambda: z1 - z0, records)
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
            "dit_backbone",
            lambda: trainer.model.predict_flow(
                zt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_tokens,
                style_global=style_global,
                style_token_mask=style_token_mask,
            ),
            records,
        )
        loss = stage_record(device, "flow_loss", lambda: torch.nn.functional.mse_loss(pred_flow, target_flow), records)

    stage_record(device, "backward", lambda: loss.backward(), records)
    trainer.optimizer.step()

    result = {
        "config": {
            "device": str(device),
            "batch": int(args.batch),
            "style_ref_count": int(args.style_ref_count),
            "max_fonts": int(args.max_fonts),
            "num_workers": int(args.num_workers),
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
