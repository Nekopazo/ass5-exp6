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

from dataset import FontImageDataset, UniqueFontBatchSampler
from models.model import FlowTrainer, StylePretrainTrainer
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
    batch = {
        "font": [sample["font"] for sample in samples],
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "style_img": torch.stack([sample["style_img"] for sample in samples], dim=0),
        "style_ref_mask": torch.stack([sample["style_ref_mask"] for sample in samples], dim=0),
    }
    if "style_img_pos" in samples[0]:
        batch["style_img_pos"] = torch.stack([sample["style_img_pos"] for sample in samples], dim=0)
        batch["style_ref_mask_pos"] = torch.stack([sample["style_ref_mask_pos"] for sample in samples], dim=0)
    return batch


def build_dataloader(
    dataset: FontImageDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    use_unique_font_batches: bool,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if int(num_workers) > 0 else None,
    }
    if use_unique_font_batches:
        if sampler is not None:
            raise ValueError("sampler cannot be combined with unique-font batch sampling")
        batch_sampler = UniqueFontBatchSampler(
            dataset,
            batch_size=int(batch_size),
            seed=int(seed),
            drop_last=False,
        )
        return DataLoader(batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        batch_size=int(batch_size),
        shuffle=bool(shuffle) and sampler is None,
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
        use_unique_font_batches=True,
        shuffle=False,
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
        use_unique_font_batches=True,
        shuffle=False,
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
        style_global_dim=int(args.style_global_dim),
        patch_hidden_dim=int(args.patch_hidden_dim),
        patch_depth=int(args.patch_depth),
        patch_heads=int(args.patch_heads),
        patch_mlp_ratio=float(args.patch_mlp_ratio),
        pixel_hidden_dim=int(args.pixel_hidden_dim),
        pit_depth=int(args.pit_depth),
        pit_heads=int(args.pit_heads),
        pit_mlp_ratio=float(args.pit_mlp_ratio),
        style_fusion_start=int(args.style_fusion_start),
        use_style_tokens=bool(args.use_style_tokens),
        contrastive_proj_dim=int(args.contrastive_proj_dim),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="flow", choices=["style", "flow"])
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--style-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--freeze-style-global",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only used in flow stage. If true, freeze the global style encoder loaded from style pretraining.",
    )

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
    parser.add_argument("--image-size", type=int, default=128)

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--style-global-dim", type=int, default=256)
    parser.add_argument("--patch-hidden-dim", type=int, default=512)
    parser.add_argument("--patch-depth", type=int, default=12)
    parser.add_argument("--patch-heads", type=int, default=8)
    parser.add_argument("--patch-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--pixel-hidden-dim", type=int, default=32)
    parser.add_argument("--pit-depth", type=int, default=2)
    parser.add_argument("--pit-heads", type=int, default=8)
    parser.add_argument("--pit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--style-fusion-start", type=int, default=8)
    parser.add_argument(
        "--use-style-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, disable the local style-token branch and keep only content + global style conditioning.",
    )
    parser.add_argument("--contrastive-proj-dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--sample-every-steps", type=int, default=0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--flow-lambda-rf", type=float, default=1.0)
    parser.add_argument("--flow-lambda-img-l1", type=float, default=0.0)
    parser.add_argument("--flow-lambda-img-perc", type=float, default=0.0)
    parser.add_argument("--flow-sample-steps", type=int, default=20)
    parser.add_argument("--flow-sampler", type=str, default="flow_dpm", choices=["flow_dpm", "euler", "heun"])
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
    )
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] stage={args.stage} device={device} seed={int(args.seed)}")
    print(f"[train] attention_backend={describe_torch_sdpa_backends()}")

    font_split_seed = int(args.seed) if args.font_split_seed is None else int(args.font_split_seed)
    include_positive_style = args.stage == "style"
    glyph_transform = build_base_glyph_transform(image_size=int(args.image_size))

    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=int(args.style_ref_count),
        include_positive_style=include_positive_style,
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
            include_positive_style=include_positive_style,
            random_seed=int(args.seed),
            font_split="test",
            font_split_seed=font_split_seed,
            font_train_ratio=float(args.font_train_ratio),
            transform=glyph_transform,
            style_transform=glyph_transform,
        )

    use_unique_font_batches = args.stage == "style"
    dataloader = build_dataloader(
        dataset,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        device=device,
        seed=int(args.seed),
        use_unique_font_batches=use_unique_font_batches,
        shuffle=True,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=int(args.batch),
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            use_unique_font_batches=use_unique_font_batches,
            shuffle=False,
        )

    total_steps = int(args.total_steps)
    if total_steps <= 0:
        if args.stage == "style":
            total_steps = 5000
        else:
            total_steps = max(1, len(dataloader) * int(args.epochs))
    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))

    model = build_model(args)
    if args.style_checkpoint is not None and args.resume is None and args.stage == "flow":
        model.load_style_checkpoint(args.style_checkpoint)
        print(f"[train] loaded style checkpoint: {args.style_checkpoint}")
    if args.stage == "flow" and bool(args.freeze_style_global) and args.resume is None and args.style_checkpoint is None:
        raise ValueError("Flow stage cannot freeze global style without --style-checkpoint unless resuming.")

    save_every_steps = int(args.save_every_steps)
    if save_every_steps <= 0:
        save_every_steps = 1000 if args.stage == "style" else 5000
    sample_every_steps = int(args.sample_every_steps)
    if sample_every_steps <= 0:
        sample_every_steps = 0 if args.stage == "style" else 300

    if args.stage == "style":
        trainer = StylePretrainTrainer(
            model,
            device,
            lr=float(args.lr),
            total_steps=total_steps,
            contrastive_temperature=float(args.contrastive_temperature),
            log_every_steps=int(args.log_every_steps),
            save_every_steps=save_every_steps,
            val_every_steps=int(args.val_every_steps),
            val_max_batches=int(args.val_max_batches),
            lr_warmup_steps=int(args.lr_warmup_steps),
            lr_min_ratio=float(args.lr_min_ratio),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip_norm),
        )
    else:
        trainer = FlowTrainer(
            model,
            device,
            lr=float(args.lr),
            total_steps=total_steps,
            lambda_rf=float(args.flow_lambda_rf),
            lambda_img_l1=float(args.flow_lambda_img_l1),
            lambda_img_perc=float(args.flow_lambda_img_perc),
            freeze_style_global=bool(args.freeze_style_global),
            flow_sample_steps=int(args.flow_sample_steps),
            flow_sampler=str(args.flow_sampler),
            timestep_sampling=str(args.timestep_sampling),
            ema_decay=float(args.ema_decay),
            log_every_steps=int(args.log_every_steps),
            save_every_steps=save_every_steps,
            val_every_steps=int(args.val_every_steps),
            val_max_batches=int(args.val_max_batches),
            lr_warmup_steps=int(args.lr_warmup_steps),
            lr_min_ratio=float(args.lr_min_ratio),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip_norm),
        )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume}")

    trainer.sample_batch = (
        build_sample_batch(dataset, val_dataset, device=device, seed=int(args.seed))
        if args.stage == "flow"
        else None
    )
    trainer.sample_every_steps = sample_every_steps if sample_every_steps > 0 else None
    trainer.sample_dir = args.save_dir / "samples"

    run_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_config["resolved_device"] = str(device)
    run_config["resolved_font_split_seed"] = int(font_split_seed)
    run_config["resolved_epochs"] = int(resolved_epochs)
    run_config["computed_total_steps"] = int(total_steps)
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

    trainer.fit(
        dataloader,
        epochs=resolved_epochs,
        save_dir=args.save_dir,
        val_dataloader=val_dataloader,
    )


if __name__ == "__main__":
    main()
