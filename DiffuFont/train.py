#!/usr/bin/env python3
"""Training entry for the pixel-space content+style glyph DiT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from dataset import FontImageDataset, GroupedCharFontBatchSampler, UniqueFontBatchSampler
from models.model import FlowTrainer
from models.sdpa_attention import describe_torch_sdpa_backends, enable_torch_sdpa_backends
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + int(worker_id))
    np.random.seed(worker_seed + int(worker_id))


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(raw_device)


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    return {
        "font": [sample["font"] for sample in samples],
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "style_img": torch.stack([sample["style_img"] for sample in samples], dim=0),
    }


def build_dataloader(
    dataset: FontImageDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampling_mode: str,
    grouped_char_count: int = 8,
    grouped_fonts_per_char: int = 0,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    sampling_mode = str(sampling_mode).strip().lower()
    if sampling_mode not in {"shuffle", "sequential", "unique_font", "grouped_char_font"}:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}")
    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if int(num_workers) > 0 else None,
    }
    if sampling_mode == "unique_font":
        if sampler is not None:
            raise ValueError("sampler cannot be combined with unique-font batch sampling")
        batch_sampler = UniqueFontBatchSampler(
            dataset,
            batch_size=int(batch_size),
            seed=int(seed),
            drop_last=False,
        )
        return DataLoader(batch_sampler=batch_sampler, **dataloader_kwargs)
    if sampling_mode == "grouped_char_font":
        if sampler is not None:
            raise ValueError("sampler cannot be combined with grouped-char-font batch sampling")
        batch_sampler = GroupedCharFontBatchSampler(
            dataset,
            batch_size=int(batch_size),
            seed=int(seed),
            drop_last=False,
            grouped_char_count=int(grouped_char_count),
            grouped_fonts_per_char=int(grouped_fonts_per_char),
        )
        return DataLoader(batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        batch_size=int(batch_size),
        shuffle=(sampling_mode == "shuffle") and sampler is None,
        sampler=sampler,
        **dataloader_kwargs,
    )


def slice_batch(batch: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        if isinstance(value, list):
            output[key] = value[:count]
        elif torch.is_tensor(value):
            output[key] = value[:count]
    return output


def concat_batches(*batches: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    keys = batches[0].keys()
    for key in keys:
        values = [batch[key] for batch in batches if key in batch]
        if not values:
            continue
        first = values[0]
        if isinstance(first, list):
            merged[key] = [item for value in values for item in value]
        elif torch.is_tensor(first):
            merged[key] = torch.cat(values, dim=0)
    return merged


def build_sample_batch(
    train_dataset: FontImageDataset,
    val_dataset: FontImageDataset | None,
    *,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    seen_count = min(4, len(train_dataset.font_names))
    seen_loader = build_dataloader(
        train_dataset,
        batch_size=max(1, seen_count),
        num_workers=0,
        device=device,
        seed=seed,
        sampling_mode="unique_font",
    )
    seen_batch = slice_batch(next(iter(seen_loader)), seen_count)
    if val_dataset is None or len(val_dataset.font_names) == 0:
        return seen_batch

    unseen_count = min(4, len(val_dataset.font_names))
    unseen_loader = build_dataloader(
        val_dataset,
        batch_size=max(1, unseen_count),
        num_workers=0,
        device=device,
        seed=seed + 1000,
        sampling_mode="unique_font",
    )
    unseen_batch = slice_batch(next(iter(unseen_loader)), unseen_count)
    return concat_batches(seen_batch, unseen_batch)


def build_model(args: argparse.Namespace) -> SourcePartRefDiT:
    return SourcePartRefDiT(
        in_channels=1,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        encoder_hidden_dim=int(args.encoder_hidden_dim),
        encoder_depth=int(args.encoder_depth),
        encoder_heads=int(args.encoder_heads),
        encoder_mlp_ratio=float(args.encoder_mlp_ratio),
        style_feature_dim=int(args.style_feature_dim),
        style_memory_k=int(args.style_memory_k),
        patch_hidden_dim=int(args.patch_hidden_dim),
        patch_depth=int(args.patch_depth),
        patch_heads=int(args.patch_heads),
        patch_mlp_ratio=float(args.patch_mlp_ratio),
        pixel_hidden_dim=int(args.pixel_hidden_dim),
        pit_depth=int(args.pit_depth),
        pit_heads=int(args.pit_heads),
        pit_mlp_ratio=float(args.pit_mlp_ratio),
        style_fusion_start=int(args.style_fusion_start),
        style_fusion_end=int(args.style_fusion_end),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--font-split-seed", type=int, default=None)
    parser.add_argument("--font-train-ratio", type=float, default=0.9)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--style-ref-count", type=int, default=8)
    parser.add_argument(
        "--train-sampling",
        type=str,
        default="shuffle",
        choices=["shuffle", "unique_font", "grouped_char_font", "sequential"],
    )
    parser.add_argument("--grouped-char-count", type=int, default=8)
    parser.add_argument("--grouped-fonts-per-char", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--style-feature-dim", type=int, default=256)
    parser.add_argument("--style-memory-k", type=int, default=4)
    parser.add_argument("--patch-hidden-dim", type=int, default=512)
    parser.add_argument("--patch-depth", type=int, default=12)
    parser.add_argument("--patch-heads", type=int, default=8)
    parser.add_argument("--patch-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--pixel-hidden-dim", type=int, default=32)
    parser.add_argument("--pit-depth", type=int, default=2)
    parser.add_argument("--pit-heads", type=int, default=8)
    parser.add_argument("--pit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--style-fusion-start", type=int, default=4)
    parser.add_argument("--style-fusion-end", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--sample-every-steps", type=int, default=0)
    parser.add_argument("--flow-lambda-rf", type=float, default=1.0)
    parser.add_argument("--flow-sample-steps", type=int, default=20)
    parser.add_argument("--flow-sampler", type=str, default="flow_dpm", choices=["flow_dpm", "euler", "heun"])
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
    )
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] device={device} seed={int(args.seed)}")
    print(f"[train] attention_backend={describe_torch_sdpa_backends()}")

    font_split_seed = int(args.seed) if args.font_split_seed is None else int(args.font_split_seed)
    glyph_transform = build_base_glyph_transform(image_size=int(args.image_size))

    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=int(args.style_ref_count),
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=font_split_seed,
        font_train_ratio=float(args.font_train_ratio),
        transform=glyph_transform,
        style_transform=glyph_transform,
    )
    val_dataset = None
    if str(args.font_split) == "train":
        val_dataset = FontImageDataset(
            project_root=args.data_root,
            max_fonts=int(args.max_fonts),
            style_ref_count=int(args.style_ref_count),
            random_seed=int(args.seed),
            font_split="test",
            font_split_seed=font_split_seed,
            font_train_ratio=float(args.font_train_ratio),
            transform=glyph_transform,
            style_transform=glyph_transform,
        )

    dataloader = build_dataloader(
        dataset,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        device=device,
        seed=int(args.seed),
        sampling_mode=str(args.train_sampling),
        grouped_char_count=int(args.grouped_char_count),
        grouped_fonts_per_char=int(args.grouped_fonts_per_char),
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=int(args.batch),
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            sampling_mode="sequential",
        )

    total_steps = int(args.total_steps)
    if total_steps <= 0:
        total_steps = max(1, len(dataloader) * int(args.epochs))
    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))
    resolved_lr_decay_start_step = max(int(args.lr_warmup_steps), int(math.ceil(total_steps * 0.8)))

    model = build_model(args)

    save_every_steps = int(args.save_every_steps)
    if save_every_steps <= 0:
        save_every_steps = 5000
    sample_every_steps = int(args.sample_every_steps)
    if sample_every_steps <= 0:
        sample_every_steps = 300

    trainer = FlowTrainer(
        model,
        device,
        lr=float(args.lr),
        total_steps=total_steps,
        lambda_rf=float(args.flow_lambda_rf),
        flow_sample_steps=int(args.flow_sample_steps),
        flow_sampler=str(args.flow_sampler),
        timestep_sampling=str(args.timestep_sampling),
        log_every_steps=int(args.log_every_steps),
        save_every_steps=save_every_steps,
        val_every_steps=int(args.val_every_steps),
        val_max_batches=int(args.val_max_batches),
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_min_ratio=float(args.lr_min_ratio),
        weight_decay=float(args.weight_decay),
    )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume}")

    trainer.sample_batch = build_sample_batch(dataset, val_dataset, device=device, seed=int(args.seed))
    trainer.sample_every_steps = sample_every_steps if sample_every_steps > 0 else None
    trainer.sample_dir = args.save_dir / "samples"

    run_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_config["resolved_device"] = str(device)
    run_config["resolved_font_split_seed"] = int(font_split_seed)
    run_config["resolved_epochs"] = int(resolved_epochs)
    run_config["computed_total_steps"] = int(total_steps)
    run_config["lr_schedule"] = "warmup_hold_then_final20pct_cosine"
    run_config["lr_tail_decay_portion"] = 0.2
    run_config["resolved_lr_decay_start_step"] = int(resolved_lr_decay_start_step)
    run_config["train_fonts"] = int(len(dataset.font_names))
    run_config["train_samples"] = int(len(dataset.samples))
    run_config["val_fonts"] = 0 if val_dataset is None else int(len(val_dataset.font_names))
    run_config["val_samples"] = 0 if val_dataset is None else int(len(val_dataset.samples))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[train] save_dir={args.save_dir}")
    print(f"[train] total_steps={total_steps} epochs={resolved_epochs}")
    print(
        "[train] lr_schedule=warmup_hold_then_final20pct_cosine "
        f"warmup_steps={int(args.lr_warmup_steps)} decay_start_step={resolved_lr_decay_start_step} "
        f"lr_min_ratio={float(args.lr_min_ratio)}"
    )

    trainer.fit(
        dataloader,
        epochs=resolved_epochs,
        save_dir=args.save_dir,
        val_dataloader=val_dataloader,
    )


if __name__ == "__main__":
    main()
