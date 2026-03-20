#!/usr/bin/env python3
"""Training entry for the document-defined content+style latent flow path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FontImageDataset, UniqueFontBatchSampler
from models.model import FlowTrainer, StylePretrainTrainer, VAETrainer
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
) -> DataLoader:
    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker if int(num_workers) > 0 else None,
    }
    if use_unique_font_batches:
        batch_sampler = UniqueFontBatchSampler(
            dataset,
            batch_size=int(batch_size),
            seed=int(seed),
            drop_last=False,
        )
        return DataLoader(batch_sampler=batch_sampler, **dataloader_kwargs)
    return DataLoader(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
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
    content_cross_attn_layers = int(args.content_cross_attn_layers)
    if content_cross_attn_layers <= 0:
        content_cross_attn_layers = math.ceil(int(args.dit_depth) * 2 / 3)
    return SourcePartRefDiT(
        in_channels=1,
        image_size=int(args.image_size),
        latent_channels=int(args.latent_channels),
        latent_size=int(args.latent_size),
        encoder_patch_size=int(args.encoder_patch_size),
        encoder_hidden_dim=int(args.encoder_hidden_dim),
        encoder_depth=int(args.encoder_depth),
        encoder_heads=int(args.encoder_heads),
        dit_hidden_dim=int(args.dit_hidden_dim),
        dit_depth=int(args.dit_depth),
        dit_heads=int(args.dit_heads),
        dit_mlp_ratio=float(args.dit_mlp_ratio),
        local_style_tokens_per_ref=int(args.local_style_tokens_per_ref),
        content_cross_attn_layers=content_cross_attn_layers,
        style_cross_attn_every_n_layers=int(args.style_cross_attn_every_n_layers),
        contrastive_proj_dim=int(args.contrastive_proj_dim),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="flow", choices=["vae", "style", "flow"])
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--style-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--train-vae-jointly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only used in flow stage. If false, a pretrained VAE checkpoint is required.",
    )
    parser.add_argument(
        "--train-style-jointly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only used in flow stage. If false, a pretrained style checkpoint is required.",
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

    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--encoder-patch-size", type=int, default=8)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--dit-hidden-dim", type=int, default=512)
    parser.add_argument("--dit-depth", type=int, default=12)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--local-style-tokens-per-ref", type=int, default=24)
    parser.add_argument("--content-cross-attn-layers", type=int, default=8)
    parser.add_argument("--style-cross-attn-every-n-layers", type=int, default=1)
    parser.add_argument("--contrastive-proj-dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-min-scale", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--sample-every-steps", type=int, default=0)

    parser.add_argument("--vae-lambda-rec", type=float, default=1.0)
    parser.add_argument("--vae-lambda-perc", type=float, default=0.1)
    parser.add_argument("--vae-lambda-kl", type=float, default=1e-4)

    parser.add_argument("--flow-lambda", type=float, default=1.0)
    parser.add_argument("--flow-lambda-img-l1", type=float, default=0.2)
    parser.add_argument("--flow-lambda-img-perc", type=float, default=0.02)
    parser.add_argument("--flow-sample-steps", type=int, default=24)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--style-lr-scale", type=float, default=1.0)
    parser.add_argument("--style-lr-warmup-steps", type=int, default=5000)
    parser.add_argument("--style-unfreeze-step", type=int, default=5000)
    parser.add_argument("--style-unfreeze-last-encoder-blocks", type=int, default=1)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] stage={args.stage} device={device} seed={int(args.seed)}")
    print(f"[train] attention_backend={describe_torch_sdpa_backends()}")
    font_split_seed = int(args.seed) if args.font_split_seed is None else int(args.font_split_seed)
    log_every_steps = max(1, int(args.log_every_steps))
    val_every_steps = max(1, int(args.val_every_steps))
    save_every_steps = int(args.save_every_steps)
    if save_every_steps <= 0:
        save_every_steps = 1000 if args.stage in {"vae", "style"} else 5000
    sample_every_steps = int(args.sample_every_steps)
    if sample_every_steps <= 0:
        sample_every_steps = 300 if args.stage == "flow" else 0

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
    train_use_unique_font_batches = args.stage == "style"
    val_use_unique_font_batches = args.stage == "style"
    dataloader = build_dataloader(
        dataset,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        device=device,
        seed=int(args.seed),
        use_unique_font_batches=train_use_unique_font_batches,
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
            use_unique_font_batches=val_use_unique_font_batches,
            shuffle=False,
        )
    if train_use_unique_font_batches:
        print(f"[train] using unique-font batch sampler for contrastive training, batch={int(args.batch)}")
    if val_use_unique_font_batches and val_dataset is not None:
        print(
            "[train] using unique-font batch sampler for contrastive validation, "
            f"effective_batch<={min(int(args.batch), len(val_dataset.font_names))}"
        )

    total_steps = int(args.total_steps)
    if total_steps <= 0:
        if args.stage == "vae":
            total_steps = 20_000
        elif args.stage == "style":
            total_steps = 5000
        else:
            total_steps = max(1, len(dataloader) * int(args.epochs))
    resolved_content_cross_attn_layers = int(args.content_cross_attn_layers)
    if resolved_content_cross_attn_layers <= 0:
        resolved_content_cross_attn_layers = math.ceil(int(args.dit_depth) * 2 / 3)
    style_unfreeze_step = int(args.style_unfreeze_step)
    if args.stage == "flow" and style_unfreeze_step < 0:
        style_unfreeze_step = 5000
    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))

    model = build_model(args)

    if args.stage == "flow" and args.resume is None and not bool(args.train_vae_jointly) and args.vae_checkpoint is None:
        raise ValueError("Flow stage requires --vae-checkpoint unless --train-vae-jointly is enabled.")
    if args.stage == "flow" and args.resume is None and not bool(args.train_style_jointly) and args.style_checkpoint is None:
        raise ValueError("Flow stage requires --style-checkpoint unless --train-style-jointly is enabled.")
    if args.vae_checkpoint is not None:
        model.load_vae_checkpoint(args.vae_checkpoint)
        print(f"[train] loaded VAE checkpoint: {args.vae_checkpoint}")
    if args.style_checkpoint is not None and args.resume is None:
        model.load_style_checkpoint(args.style_checkpoint)
        print(f"[train] loaded style checkpoint: {args.style_checkpoint}")

    if args.stage == "vae":
        trainer = VAETrainer(
            model,
            device,
            lr=float(args.lr),
            total_steps=total_steps,
            lambda_rec=float(args.vae_lambda_rec),
            lambda_perc=float(args.vae_lambda_perc),
            lambda_kl=float(args.vae_lambda_kl),
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=int(args.val_max_batches),
        )
    elif args.stage == "style":
        trainer = StylePretrainTrainer(
            model,
            device,
            lr=float(args.lr),
            total_steps=total_steps,
            contrastive_temperature=float(args.contrastive_temperature),
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=int(args.val_max_batches),
        )
    else:
        trainer = FlowTrainer(
            model,
            device,
            lr=float(args.lr),
            total_steps=total_steps,
            lambda_flow=float(args.flow_lambda),
            lambda_img_l1=float(args.flow_lambda_img_l1),
            lambda_img_perc=float(args.flow_lambda_img_perc),
            style_lr_scale=float(args.style_lr_scale),
            style_lr_warmup_steps=int(args.style_lr_warmup_steps),
            style_unfreeze_step=style_unfreeze_step,
            style_unfreeze_last_encoder_blocks=int(args.style_unfreeze_last_encoder_blocks),
            freeze_vae=not bool(args.train_vae_jointly),
            freeze_style=not bool(args.train_style_jointly),
            flow_sample_steps=int(args.flow_sample_steps),
            lr_warmup_steps=int(args.lr_warmup_steps),
            lr_min_scale=float(args.lr_min_scale),
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=int(args.val_max_batches),
        )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume}")

    first_batch = next(iter(dataloader))
    trainer.sample_batch = (
        build_sample_batch(dataset, val_dataset, device=device, seed=int(args.seed))
        if args.stage == "flow"
        else first_batch
    )
    trainer.sample_every_steps = sample_every_steps if sample_every_steps > 0 else None
    trainer.sample_dir = args.save_dir / "samples"

    run_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_config["resolved_device"] = str(device)
    run_config["resolved_font_split_seed"] = int(font_split_seed)
    run_config["resolved_log_every_steps"] = int(log_every_steps)
    run_config["resolved_val_every_steps"] = int(val_every_steps)
    run_config["resolved_save_every_steps"] = int(save_every_steps)
    run_config["resolved_sample_every_steps"] = int(sample_every_steps)
    run_config["resolved_epochs"] = int(resolved_epochs)
    run_config["resolved_content_cross_attn_layers"] = int(resolved_content_cross_attn_layers)
    run_config["resolved_style_unfreeze_step"] = int(style_unfreeze_step)
    run_config["train_fonts"] = int(len(dataset.font_names))
    run_config["train_samples"] = int(len(dataset.samples))
    run_config["val_fonts"] = 0 if val_dataset is None else int(len(val_dataset.font_names))
    run_config["val_samples"] = 0 if val_dataset is None else int(len(val_dataset.samples))
    run_config["computed_total_steps"] = int(total_steps)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer.fit(dataloader, epochs=resolved_epochs, save_dir=args.save_dir, val_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
