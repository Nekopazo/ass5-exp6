#!/usr/bin/env python3
"""Profile peak CUDA memory by major training stage for one diffusion step."""

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
from models.model import DiffusionTrainer, glyph_perceptual_loss, info_nce_loss
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
        "style_img_pos": torch.stack([sample["style_img_pos"] for sample in samples], dim=0),
        "style_ref_mask_pos": torch.stack([sample["style_ref_mask_pos"] for sample in samples], dim=0),
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
        include_positive_style=True,
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
        latent_channels=4,
        latent_size=16,
        encoder_patch_size=8,
        encoder_hidden_dim=512,
        encoder_depth=4,
        encoder_heads=8,
        dit_hidden_dim=512,
        dit_depth=12,
        dit_heads=8,
        dit_mlp_ratio=4.0,
    )
    model.load_vae_checkpoint(args.vae_checkpoint)
    trainer = DiffusionTrainer(
        model,
        device,
        lr=1e-4,
        timesteps=1000,
        total_steps=1,
        lambda_diff=1.0,
        lambda_rec=0.5,
        lambda_perc=0.1,
        lambda_kl=0.0,
        lambda_ctr=0.02,
        contrastive_temperature=0.1,
        contrastive_warmup_steps=5000,
        freeze_vae=True,
        log_every_steps=1,
        save_every_steps=None,
    )
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    target = batch["target"].to(device)
    content = batch["content"].to(device)
    style = batch["style_img"].to(device)
    style_pos = batch["style_img_pos"].to(device)
    style_ref_mask = batch["style_ref_mask"].to(device)
    style_ref_mask_pos = batch["style_ref_mask_pos"].to(device)

    records: list[dict] = []
    torch.cuda.empty_cache()
    with trainer._autocast_context():
        z0, _, _ = stage_record(device, "vae_encode", lambda: trainer._encode_latent(target), records)
        timesteps = torch.randint(0, trainer.scheduler.timesteps, (z0.size(0),), device=device)
        zt, noise = stage_record(device, "noise_add", lambda: trainer.scheduler.add_noise(z0, timesteps), records)
        content_tokens = stage_record(device, "content_encode", lambda: trainer.model.encode_content(content), records)
        style_tokens, style_global, style_token_mask, anchor_style_embed = stage_record(
            device,
            "style_encode_anchor",
            lambda: trainer.model.encode_style(style_img=style, style_ref_mask=style_ref_mask),
            records,
        )
        noise_pred = stage_record(
            device,
            "dit_backbone",
            lambda: trainer.model.predict_noise(
                zt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_tokens,
                style_global=style_global,
                style_token_mask=style_token_mask,
            ),
            records,
        )
        loss_diff = stage_record(device, "diff_loss", lambda: torch.nn.functional.mse_loss(noise_pred, noise), records)
        z0_pred = stage_record(
            device,
            "predict_x0",
            lambda: trainer.scheduler.predict_start_from_noise(zt, timesteps, noise_pred).clamp(-4.0, 4.0),
            records,
        )
        recon = stage_record(device, "vae_decode", lambda: trainer.model.decode_from_latent(z0_pred), records)
        loss_rec = stage_record(device, "recon_l1", lambda: torch.nn.functional.l1_loss(recon, target), records)
        loss_perc = stage_record(device, "recon_perc", lambda: glyph_perceptual_loss(recon, target), records)
        _, _, _, positive_style_embed = stage_record(
            device,
            "style_encode_positive",
            lambda: trainer.model.encode_style(style_img=style_pos, style_ref_mask=style_ref_mask_pos),
            records,
        )
        loss_ctr = stage_record(
            device,
            "contrastive_loss",
            lambda: info_nce_loss(anchor_style_embed, positive_style_embed, trainer.contrastive_temperature),
            records,
        )
        loss = loss_diff + 0.5 * loss_rec + 0.1 * loss_perc + (trainer._current_contrastive_weight() * loss_ctr)

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
