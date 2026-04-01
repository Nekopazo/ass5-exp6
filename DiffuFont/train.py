#!/usr/bin/env python3
"""Training entry for the pixel-space DiP glyph generation path."""

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

from dataset import CartesianFontCharBatchSampler, FontImageDataset, UniqueFontBatchSampler
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


def _resolve_batch_ref_count(samples) -> int:
    min_refs = max(int(sample.get("style_ref_count_min", sample["style_img"].size(0))) for sample in samples)
    max_refs = min(int(sample.get("style_ref_count_max", sample["style_img"].size(0))) for sample in samples)
    available_refs = min(int(sample["style_img"].size(0)) for sample in samples)
    max_refs = min(max_refs, available_refs)
    if max_refs < min_refs:
        raise RuntimeError(f"Invalid style ref bounds in batch: min_refs={min_refs} max_refs={max_refs}")
    return random.randint(min_refs, max_refs) if min_refs < max_refs else max_refs


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    ref_count = _resolve_batch_ref_count(samples)
    batch = {
        "font": [sample["font"] for sample in samples],
        "font_id": torch.tensor([sample["font_id"] for sample in samples], dtype=torch.long),
        "char": [sample["char"] for sample in samples],
        "char_id": torch.tensor([sample["char_id"] for sample in samples], dtype=torch.long),
        "content": torch.stack([sample["content"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "style_img": torch.stack([sample["style_img"][:ref_count] for sample in samples], dim=0),
        "style_ref_mask": torch.stack([sample["style_ref_mask"][:ref_count] for sample in samples], dim=0),
    }
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
    sampling_mode: str,
    cartesian_fonts_per_batch: int,
    cartesian_chars_per_batch: int,
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
    if str(sampling_mode) == "cartesian_font_char":
        batch_sampler = CartesianFontCharBatchSampler(
            dataset,
            fonts_per_batch=int(cartesian_fonts_per_batch),
            chars_per_batch=int(cartesian_chars_per_batch),
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
            if key in {"style_img", "style_ref_mask"}:
                ref_count = min(int(value.size(1)) for value in values)
                values = [value[:, :ref_count] for value in values]
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
        sampling_mode="shuffle",
        cartesian_fonts_per_batch=1,
        cartesian_chars_per_batch=1,
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
        sampling_mode="shuffle",
        cartesian_fonts_per_batch=1,
        cartesian_chars_per_batch=1,
    )
    unseen_batch = slice_batch(next(iter(unseen_loader)), unseen_count)
    return concat_batches(seen_batch, unseen_batch)


def build_model(args: argparse.Namespace) -> SourcePartRefDiT:
    return SourcePartRefDiT(
        in_channels=1,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        encoder_hidden_dim=int(args.encoder_hidden_dim),
        dit_hidden_dim=int(args.dit_hidden_dim),
        dit_depth=int(args.dit_depth),
        dit_heads=int(args.dit_heads),
        content_cross_attn_heads=None if args.content_cross_attn_heads is None else int(args.content_cross_attn_heads),
        dit_mlp_ratio=float(args.dit_mlp_ratio),
        content_cross_attn_layers=args.content_cross_attn_layers,
        style_modulation_layers=args.style_modulation_layers,
        detailer_base_channels=int(args.detailer_base_channels),
        detailer_max_channels=int(args.detailer_max_channels),
    )


def parse_layer_indices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("layer list must be a comma-separated list like 1,2,3,4,5,6")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid layer list: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--font-split", type=str, required=True, choices=["train", "test", "all"])
    parser.add_argument("--font-split-seed", type=int, required=True)
    parser.add_argument("--font-train-ratio", type=float, required=True)
    parser.add_argument("--max-fonts", type=int, required=True)
    parser.add_argument("--style-ref-count", type=int, required=True)
    parser.add_argument("--style-ref-count-min", type=int, required=True)
    parser.add_argument("--style-ref-count-max", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)

    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--encoder-hidden-dim", type=int, required=True)
    parser.add_argument("--dit-hidden-dim", type=int, required=True)
    parser.add_argument("--dit-depth", type=int, required=True)
    parser.add_argument("--dit-heads", type=int, required=True)
    parser.add_argument("--content-cross-attn-heads", type=int, default=None)
    parser.add_argument("--dit-mlp-ratio", type=float, required=True)
    parser.add_argument("--content-cross-attn-layers", type=parse_layer_indices, required=True)
    parser.add_argument("--style-modulation-layers", type=parse_layer_indices, required=True)
    parser.add_argument("--detailer-base-channels", type=int, required=True)
    parser.add_argument("--detailer-max-channels", type=int, required=True)
    parser.add_argument("--train-sampling", type=str, required=True, choices=["shuffle", "cartesian_font_char"])
    parser.add_argument("--cartesian-fonts-per-batch", type=int, required=True)
    parser.add_argument("--cartesian-chars-per-batch", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=-1)
    parser.add_argument("--lr-min-scale", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--log-every-steps", type=int, required=True)
    parser.add_argument("--val-every-steps", type=int, required=True)
    parser.add_argument("--val-max-batches", type=int, required=True)
    parser.add_argument("--save-every-steps", type=int, required=True)
    parser.add_argument("--sample-every-steps", type=int, required=True)
    parser.add_argument("--grad-clip-norm", type=float, required=True)
    parser.add_argument("--grad-clip-min-norm", type=float, default=None)

    parser.add_argument("--flow-lambda", type=float, required=True)
    use_cnn_perceptor_group = parser.add_mutually_exclusive_group(required=True)
    use_cnn_perceptor_group.add_argument("--use-cnn-perceptor", dest="use_cnn_perceptor", action="store_true")
    use_cnn_perceptor_group.add_argument("--no-use-cnn-perceptor", dest="use_cnn_perceptor", action="store_false")
    parser.add_argument("--perceptor-checkpoint", type=Path)
    parser.add_argument("--perceptual-loss-lambda", type=float, required=True)
    parser.add_argument("--style-loss-lambda", type=float, required=True)
    parser.add_argument("--aux-loss-t-logistic-steepness", type=float, default=8.0)
    parser.add_argument("--perceptual-loss-t-midpoint", type=float, default=0.35)
    parser.add_argument("--style-loss-t-midpoint", type=float, default=0.45)
    parser.add_argument("--flow-sample-steps", type=int, required=True)
    parser.add_argument("--ema-decay", type=float, required=True)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] mode=pixel_flow device={device} seed={int(args.seed)}")
    print(f"[train] attention_backend={describe_torch_sdpa_backends()}")

    font_split_seed = int(args.font_split_seed)
    log_every_steps = int(args.log_every_steps)
    val_every_steps = int(args.val_every_steps)
    save_every_steps = int(args.save_every_steps)
    sample_every_steps = int(args.sample_every_steps)
    resolved_save_every_steps = None if save_every_steps == 0 else save_every_steps
    resolved_sample_every_steps = None if sample_every_steps == 0 else sample_every_steps
    total_steps = int(args.total_steps)
    if log_every_steps <= 0 or val_every_steps <= 0:
        raise ValueError("log_every_steps and val_every_steps must be > 0.")
    if save_every_steps < 0 or sample_every_steps < 0:
        raise ValueError("save_every_steps and sample_every_steps must be >= 0.")
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")
    if int(args.lr_warmup_steps) < 0:
        raise ValueError("lr_warmup_steps must be >= 0.")
    if float(args.lr_min_scale) < 0.0 or float(args.lr_min_scale) > 1.0:
        raise ValueError("lr_min_scale must be in [0, 1].")

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
    val_dataset = None
    if str(args.font_split) == "train":
        val_dataset = FontImageDataset(
            project_root=args.data_root,
            max_fonts=int(args.max_fonts),
            style_ref_count=style_ref_count,
            style_ref_count_min=int(args.style_ref_count_min),
            style_ref_count_max=int(args.style_ref_count_max),
            include_positive_style=False,
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
        use_unique_font_batches=False,
        shuffle=(str(args.train_sampling) == "shuffle"),
        sampling_mode=str(args.train_sampling),
        cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
        cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=int(args.batch),
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            use_unique_font_batches=False,
            shuffle=False,
            sampling_mode="shuffle",
            cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
            cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
        )
    if str(args.train_sampling) == "cartesian_font_char":
        print(
            "[train] using cartesian font-char sampler "
            f"fonts_per_batch={int(args.cartesian_fonts_per_batch)} "
            f"chars_per_batch={int(args.cartesian_chars_per_batch)} "
            f"effective_batch={int(args.cartesian_fonts_per_batch) * int(args.cartesian_chars_per_batch)} "
            "pad_missing_with_random_samples=1"
        )

    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))

    effective_perceptor_checkpoint = args.perceptor_checkpoint
    effective_perceptual_loss_lambda = float(args.perceptual_loss_lambda)
    effective_style_loss_lambda = float(args.style_loss_lambda)
    if not bool(args.use_cnn_perceptor):
        if args.perceptor_checkpoint is not None or effective_perceptual_loss_lambda > 0.0 or effective_style_loss_lambda > 0.0:
            print("[train] use_cnn_perceptor=0, ignoring perceptor checkpoint and auxiliary loss weights")
        effective_perceptor_checkpoint = None
        effective_perceptual_loss_lambda = 0.0
        effective_style_loss_lambda = 0.0

    model = build_model(args)
    trainer = FlowTrainer(
        model,
        device,
        lr=float(args.lr),
        total_steps=total_steps,
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_decay_start_step=None if int(args.lr_decay_start_step) < 0 else int(args.lr_decay_start_step),
        lr_min_scale=float(args.lr_min_scale),
        lambda_flow=float(args.flow_lambda),
        use_cnn_perceptor=bool(args.use_cnn_perceptor),
        perceptor_checkpoint=effective_perceptor_checkpoint,
        perceptual_loss_lambda=effective_perceptual_loss_lambda,
        style_loss_lambda=effective_style_loss_lambda,
        aux_loss_t_logistic_steepness=float(args.aux_loss_t_logistic_steepness),
        perceptual_loss_t_midpoint=float(args.perceptual_loss_t_midpoint),
        style_loss_t_midpoint=float(args.style_loss_t_midpoint),
        flow_sample_steps=int(args.flow_sample_steps),
        ema_decay=float(args.ema_decay),
        log_every_steps=log_every_steps,
        save_every_steps=resolved_save_every_steps,
        val_every_steps=val_every_steps,
        val_max_batches=int(args.val_max_batches),
        grad_clip_norm=float(args.grad_clip_norm),
        grad_clip_min_norm=None if args.grad_clip_min_norm is None else float(args.grad_clip_min_norm),
    )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume}")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.use_cnn_perceptor) and trainer.perceptor_report is not None:
        (args.save_dir / "perceptor_qualification_snapshot.json").write_text(
            json.dumps(trainer.perceptor_report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ready_flag = int(bool(trainer.perceptor_report.get("can_integrate_directly")))
        print(f"[train] loaded perceptor_report ready_for_flow={ready_flag}")
        if not ready_flag:
            print(f"[train] warning: perceptor report suggests not integrating directly yet: {args.perceptor_checkpoint}")
    elif bool(args.use_cnn_perceptor) and args.perceptor_checkpoint is not None:
        print(f"[train] loaded perceptor checkpoint without qualification report: {args.perceptor_checkpoint}")

    trainer.sample_batch = build_sample_batch(dataset, val_dataset, device=device, seed=int(args.seed))
    trainer.sample_every_steps = resolved_sample_every_steps
    trainer.sample_dir = args.save_dir / "samples"

    run_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_config["resolved_device"] = str(device)
    run_config["resolved_font_split_seed"] = int(font_split_seed)
    run_config["resolved_log_every_steps"] = int(log_every_steps)
    run_config["resolved_val_every_steps"] = int(val_every_steps)
    run_config["resolved_save_every_steps"] = None if resolved_save_every_steps is None else int(resolved_save_every_steps)
    run_config["resolved_sample_every_steps"] = None if resolved_sample_every_steps is None else int(resolved_sample_every_steps)
    run_config["resolved_epochs"] = int(resolved_epochs)
    run_config["train_fonts"] = int(len(dataset.font_names))
    run_config["train_samples"] = int(len(dataset.samples))
    run_config["val_fonts"] = 0 if val_dataset is None else int(len(val_dataset.font_names))
    run_config["val_samples"] = 0 if val_dataset is None else int(len(val_dataset.samples))
    run_config["computed_total_steps"] = int(total_steps)
    run_config["model_type"] = "pixel_dip"
    run_config["use_cnn_perceptor"] = int(bool(args.use_cnn_perceptor))
    run_config["perceptor_checkpoint"] = None if effective_perceptor_checkpoint is None else str(effective_perceptor_checkpoint)
    run_config["perceptual_loss_lambda"] = float(effective_perceptual_loss_lambda)
    run_config["style_loss_lambda"] = float(effective_style_loss_lambda)
    run_config["aux_loss_t_logistic_steepness"] = float(args.aux_loss_t_logistic_steepness)
    run_config["perceptual_loss_t_midpoint"] = float(args.perceptual_loss_t_midpoint)
    run_config["style_loss_t_midpoint"] = float(args.style_loss_t_midpoint)
    run_config["lr_warmup_steps"] = int(args.lr_warmup_steps)
    run_config["lr_decay_start_step"] = None if int(args.lr_decay_start_step) < 0 else int(args.lr_decay_start_step)
    run_config["lr_min_scale"] = float(args.lr_min_scale)
    run_config["grad_clip_min_norm"] = None if args.grad_clip_min_norm is None else float(args.grad_clip_min_norm)

    (args.save_dir / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer.fit(dataloader, epochs=resolved_epochs, save_dir=args.save_dir, val_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
