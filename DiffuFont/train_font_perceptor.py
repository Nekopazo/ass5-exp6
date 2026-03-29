#!/usr/bin/env python3
"""Pretrain the custom grayscale font perceptor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CartesianFontCharBatchSampler, FontImageDataset
from models.font_perceptor import FontPerceptor
from models.model import FontPerceptorTrainer
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


def build_dataloader(
    dataset: FontImageDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampling_mode: str,
    cartesian_fonts_per_batch: int,
    cartesian_chars_per_batch: int,
    shuffle: bool,
) -> DataLoader:
    common_kwargs = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "worker_init_fn": seed_worker if int(num_workers) > 0 else None,
    }
    if str(sampling_mode) == "cartesian_font_char":
        batch_sampler = CartesianFontCharBatchSampler(
            dataset,
            fonts_per_batch=int(cartesian_fonts_per_batch),
            chars_per_batch=int(cartesian_chars_per_batch),
            seed=int(seed),
            drop_last=False,
        )
        return DataLoader(batch_sampler=batch_sampler, **common_kwargs)
    return DataLoader(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        **common_kwargs,
    )


def parse_feature_stage_names(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _load_last_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def _stamp_checkpoint_with_qualification(path: Path, report: dict[str, Any]) -> None:
    if not path.is_file():
        return
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["qualification"] = report
    torch.save(checkpoint, path)


def build_qualification_report(
    *,
    save_dir: Path,
    min_char_acc: float,
    min_style_margin: float,
) -> dict[str, Any]:
    best_metrics_path = save_dir / "best_val_metrics.json"
    val_log_path = save_dir / "val_step_metrics.jsonl"
    best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8")) if best_metrics_path.is_file() else None
    latest_val_metrics = _load_last_jsonl_row(val_log_path)

    reasons: list[str] = []
    qualified = False
    if best_metrics is None:
        reasons.append("missing_best_val_metrics")
    else:
        char_acc = float(best_metrics.get("char_acc", 0.0))
        style_margin = float(best_metrics.get("style_cos_margin", 0.0))
        pos_pairs = float(best_metrics.get("style_pos_pairs", 0.0))
        neg_pairs = float(best_metrics.get("style_neg_pairs", 0.0))
        if char_acc < float(min_char_acc):
            reasons.append("char_acc_below_threshold")
        if style_margin < float(min_style_margin):
            reasons.append("style_margin_below_threshold")
        if pos_pairs <= 0.0 or neg_pairs <= 0.0:
            reasons.append("insufficient_style_pairs")
        qualified = len(reasons) == 0

    best_checkpoint = save_dir / "best.pt"
    last_checkpoint = save_dir / "last.pt"
    report = {
        "stage": "font_perceptor_pretrain",
        "qualified": bool(qualified),
        "can_integrate_directly": bool(qualified and best_checkpoint.is_file()),
        "criteria": {
            "min_char_acc": float(min_char_acc),
            "min_style_margin": float(min_style_margin),
        },
        "reason_codes": reasons,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint.is_file() else None,
        "best_val_metrics": best_metrics,
        "latest_val_metrics": latest_val_metrics,
    }
    (save_dir / "qualification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _stamp_checkpoint_with_qualification(best_checkpoint, report)
    _stamp_checkpoint_with_qualification(last_checkpoint, report)
    return report


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
    parser.add_argument("--image-size", type=int, required=True)

    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--style-proj-dim", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--feature-stages", type=str, required=True)

    parser.add_argument("--train-sampling", type=str, required=True, choices=["shuffle", "cartesian_font_char"])
    parser.add_argument("--cartesian-fonts-per-batch", type=int, required=True)
    parser.add_argument("--cartesian-chars-per-batch", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--log-every-steps", type=int, required=True)
    parser.add_argument("--val-every-steps", type=int, required=True)
    parser.add_argument("--val-max-batches", type=int, required=True)
    parser.add_argument("--save-every-steps", type=int, required=True)
    parser.add_argument("--grad-clip-norm", type=float, required=True)

    parser.add_argument("--style-supcon-lambda", type=float, required=True)
    parser.add_argument("--style-temperature", type=float, required=True)
    parser.add_argument("--qualify-min-char-acc", type=float, required=True)
    parser.add_argument("--qualify-min-style-margin", type=float, required=True)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    print(f"[font_perceptor_train] device={device} seed={int(args.seed)}")

    font_split_seed = int(args.font_split_seed)
    log_every_steps = int(args.log_every_steps)
    val_every_steps = int(args.val_every_steps)
    save_every_steps = int(args.save_every_steps)
    resolved_save_every_steps = None if save_every_steps == 0 else save_every_steps
    total_steps = int(args.total_steps)
    if log_every_steps <= 0 or val_every_steps <= 0:
        raise ValueError("log_every_steps and val_every_steps must be > 0.")
    if save_every_steps < 0:
        raise ValueError("save_every_steps must be >= 0.")
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")

    glyph_transform = build_base_glyph_transform(image_size=int(args.image_size))
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=font_split_seed,
        font_train_ratio=float(args.font_train_ratio),
        transform=glyph_transform,
        style_transform=glyph_transform,
        load_style_refs=False,
    )
    val_dataset = None
    if str(args.font_split) == "train":
        val_dataset = FontImageDataset(
            project_root=args.data_root,
            max_fonts=int(args.max_fonts),
            random_seed=int(args.seed),
            font_split="test",
            font_split_seed=font_split_seed,
            font_train_ratio=float(args.font_train_ratio),
            transform=glyph_transform,
            style_transform=glyph_transform,
            load_style_refs=False,
        )

    dataloader = build_dataloader(
        dataset,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        device=device,
        seed=int(args.seed),
        sampling_mode=str(args.train_sampling),
        cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
        cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
        shuffle=(str(args.train_sampling) == "shuffle"),
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=int(args.batch),
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            sampling_mode=str(args.train_sampling),
            cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
            cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
            shuffle=False,
        )

    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))

    model = FontPerceptor(
        in_channels=1,
        base_channels=int(args.base_channels),
        proj_dim=int(args.style_proj_dim),
        num_chars=len(dataset.char_list),
        dropout=float(args.dropout),
        feature_stage_names=parse_feature_stage_names(args.feature_stages),
    )
    trainer = FontPerceptorTrainer(
        model,
        device,
        lr=float(args.lr),
        total_steps=total_steps,
        style_supcon_lambda=float(args.style_supcon_lambda),
        style_temperature=float(args.style_temperature),
        qualify_min_char_acc=float(args.qualify_min_char_acc),
        qualify_min_style_margin=float(args.qualify_min_style_margin),
        log_every_steps=log_every_steps,
        save_every_steps=resolved_save_every_steps,
        val_every_steps=val_every_steps,
        val_max_batches=int(args.val_max_batches),
        grad_clip_norm=float(args.grad_clip_norm),
    )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[font_perceptor_train] resumed from {args.resume}")

    run_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run_config["resolved_device"] = str(device)
    run_config["resolved_font_split_seed"] = int(font_split_seed)
    run_config["resolved_epochs"] = int(resolved_epochs)
    run_config["resolved_log_every_steps"] = int(log_every_steps)
    run_config["resolved_val_every_steps"] = int(val_every_steps)
    run_config["resolved_save_every_steps"] = None if resolved_save_every_steps is None else int(resolved_save_every_steps)
    run_config["computed_total_steps"] = int(total_steps)
    run_config["train_fonts"] = int(len(dataset.font_names))
    run_config["train_samples"] = int(len(dataset.samples))
    run_config["val_fonts"] = 0 if val_dataset is None else int(len(val_dataset.font_names))
    run_config["val_samples"] = 0 if val_dataset is None else int(len(val_dataset.samples))
    run_config["num_chars"] = int(len(dataset.char_list))
    run_config["model_type"] = "font_perceptor"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer.fit(dataloader, epochs=resolved_epochs, save_dir=args.save_dir, val_dataloader=val_dataloader)
    trainer.save(args.save_dir / "last.pt")
    report = build_qualification_report(
        save_dir=args.save_dir,
        min_char_acc=float(args.qualify_min_char_acc),
        min_style_margin=float(args.qualify_min_style_margin),
    )
    print(
        "[font_perceptor_train] "
        f"qualified={int(bool(report['qualified']))} "
        f"can_integrate_directly={int(bool(report['can_integrate_directly']))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
