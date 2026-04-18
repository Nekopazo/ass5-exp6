#!/usr/bin/env python3
"""Training entry for the DiT x-pred glyph generation path."""

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
from models.model import XPredTrainer
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


def _pack_unique_content(samples) -> tuple[torch.Tensor, torch.Tensor]:
    unique_contents = []
    content_index = []
    content_slot_by_char_id: dict[int, int] = {}
    for sample in samples:
        char_id = int(sample["char_id"])
        slot = content_slot_by_char_id.get(char_id)
        if slot is None:
            slot = len(unique_contents)
            content_slot_by_char_id[char_id] = slot
            unique_contents.append(sample["content"])
        content_index.append(slot)
    return torch.stack(unique_contents, dim=0), torch.tensor(content_index, dtype=torch.long)


def _compact_unique_indices(sliced_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if sliced_index.numel() == 0:
        return sliced_index, sliced_index
    unique_positions = []
    remapped_indices = []
    position_remap: dict[int, int] = {}
    for old_position in sliced_index.tolist():
        old_position = int(old_position)
        new_position = position_remap.get(old_position)
        if new_position is None:
            new_position = len(unique_positions)
            position_remap[old_position] = new_position
            unique_positions.append(old_position)
        remapped_indices.append(new_position)
    return torch.tensor(unique_positions, dtype=torch.long), torch.tensor(remapped_indices, dtype=torch.long)


class XPredBatchCollator:
    def __init__(self, dataset: FontImageDataset) -> None:
        self.dataset = dataset

    def _build_excluded_style_indices(self, samples) -> dict[str, list[int]]:
        excluded_by_font: dict[str, list[int]] = {}
        for sample in samples:
            font_name = str(sample["font"])
            excluded_by_font.setdefault(font_name, []).append(int(sample["char_id"]))
        return excluded_by_font

    def _sample_shared_style_indices(self, excluded_by_font: dict[str, list[int]]) -> list[int]:
        shared_candidates: set[int] | None = None
        for font_name, excluded_indices in excluded_by_font.items():
            candidates = set(self.dataset.list_style_candidate_indices(font_name, excluded_indices=excluded_indices))
            shared_candidates = candidates if shared_candidates is None else shared_candidates & candidates
        if not shared_candidates:
            raise RuntimeError("Batch has no shared non-overlapping style references available across fonts.")
        ordered_candidates = sorted(int(idx) for idx in shared_candidates)
        max_refs = min(int(self.dataset.style_ref_count_max), len(ordered_candidates))
        if max_refs < 1:
            raise RuntimeError("Batch has no shared non-overlapping style references available across fonts.")
        min_refs = min(int(self.dataset.style_ref_count_min), max_refs)
        ref_count = random.randint(min_refs, max_refs) if min_refs < max_refs else max_refs
        return random.sample(ordered_candidates, k=ref_count)

    def _pack_unique_style(
        self,
        samples,
        *,
        shared_style_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        unique_style_imgs = []
        unique_style_masks = []
        style_index = []
        style_fonts = []
        style_char_ids = []
        style_slot_by_font: dict[str, int] = {}
        for sample in samples:
            font_name = str(sample["font"])
            slot = style_slot_by_font.get(font_name)
            if slot is None:
                slot = len(unique_style_imgs)
                style_slot_by_font[font_name] = slot
                style_img, style_ref_mask, _ = self.dataset.load_style_refs_by_indices(
                    font_name,
                    shared_style_indices,
                )
                unique_style_imgs.append(style_img)
                unique_style_masks.append(style_ref_mask)
                style_fonts.append(font_name)
                style_char_ids.append(int(sample["char_id"]))
            style_index.append(slot)
        return (
            torch.stack(unique_style_imgs, dim=0),
            torch.stack(unique_style_masks, dim=0),
            torch.tensor(style_index, dtype=torch.long),
            style_fonts,
            torch.tensor(style_char_ids, dtype=torch.long),
        )

    def __call__(self, samples) -> Dict[str, torch.Tensor]:
        excluded_by_font = self._build_excluded_style_indices(samples)
        shared_style_indices = self._sample_shared_style_indices(excluded_by_font)
        content, content_index = _pack_unique_content(samples)
        style_img, style_ref_mask, style_index, style_font, style_char_id = self._pack_unique_style(
            samples,
            shared_style_indices=shared_style_indices,
        )
        return {
            "font": [sample["font"] for sample in samples],
            "font_id": torch.tensor([sample["font_id"] for sample in samples], dtype=torch.long),
            "char": [sample["char"] for sample in samples],
            "char_id": torch.tensor([sample["char_id"] for sample in samples], dtype=torch.long),
            "content": content,
            "content_index": content_index,
            "target": torch.stack([sample["target"] for sample in samples], dim=0),
            "style_img": style_img,
            "style_ref_mask": style_ref_mask,
            "style_index": style_index,
            "style_font": style_font,
            "style_char_id": style_char_id,
        }


class StyleEvalBatchCollator:
    def __call__(self, samples) -> Dict[str, torch.Tensor]:
        content, content_index = _pack_unique_content(samples)
        return {
            "font": [sample["font"] for sample in samples],
            "font_id": torch.tensor([sample["font_id"] for sample in samples], dtype=torch.long),
            "char": [sample["char"] for sample in samples],
            "char_id": torch.tensor([sample["char_id"] for sample in samples], dtype=torch.long),
            "content": content,
            "content_index": content_index,
            "target": torch.stack([sample["target"] for sample in samples], dim=0),
            "style_img": torch.stack([sample["style_img"] for sample in samples], dim=0),
            "style_ref_mask": torch.stack([sample["style_ref_mask"] for sample in samples], dim=0),
            "style_index": torch.arange(len(samples), dtype=torch.long),
        }


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
        "collate_fn": XPredBatchCollator(dataset),
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


def build_style_eval_dataloader(
    dataset: FontImageDataset,
    *,
    num_workers: int,
    device: torch.device,
    seed: int,
    cartesian_fonts_per_batch: int,
    cartesian_chars_per_batch: int,
) -> DataLoader:
    batch_sampler = CartesianFontCharBatchSampler(
        dataset,
        fonts_per_batch=int(cartesian_fonts_per_batch),
        chars_per_batch=int(cartesian_chars_per_batch),
        seed=int(seed),
        drop_last=False,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=StyleEvalBatchCollator(),
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
    )


def slice_batch(batch: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        if key in {"content", "content_index", "style_img", "style_ref_mask", "style_index", "style_font", "style_char_id"}:
            continue
        if isinstance(value, list):
            output[key] = value[:count]
        elif torch.is_tensor(value):
            output[key] = value[:count]
    content_positions, content_index = _compact_unique_indices(batch["content_index"][:count])
    style_positions, style_index = _compact_unique_indices(batch["style_index"][:count])
    output["content"] = batch["content"][content_positions]
    output["content_index"] = content_index
    output["style_img"] = batch["style_img"][style_positions]
    output["style_ref_mask"] = batch["style_ref_mask"][style_positions]
    output["style_index"] = style_index
    output["style_font"] = [batch["style_font"][idx] for idx in style_positions.tolist()]
    output["style_char_id"] = batch["style_char_id"][style_positions]
    return output


def concat_batches(*batches: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    keys = batches[0].keys()
    for key in keys:
        if key in {"content", "content_index", "style_img", "style_ref_mask", "style_index", "style_font", "style_char_id"}:
            continue
        values = [batch[key] for batch in batches if key in batch]
        if not values:
            continue
        first = values[0]
        if isinstance(first, list):
            merged[key] = [item for value in values for item in value]
        elif torch.is_tensor(first):
            merged[key] = torch.cat(values, dim=0)
    merged["content"] = torch.cat([batch["content"] for batch in batches], dim=0)
    content_indices = []
    content_offset = 0
    for batch in batches:
        content_indices.append(batch["content_index"] + content_offset)
        content_offset += int(batch["content"].size(0))
    merged["content_index"] = torch.cat(content_indices, dim=0)

    ref_count = min(int(batch["style_img"].size(1)) for batch in batches)
    merged["style_img"] = torch.cat([batch["style_img"][:, :ref_count] for batch in batches], dim=0)
    merged["style_ref_mask"] = torch.cat([batch["style_ref_mask"][:, :ref_count] for batch in batches], dim=0)
    style_indices = []
    style_offset = 0
    for batch in batches:
        style_indices.append(batch["style_index"] + style_offset)
        style_offset += int(batch["style_img"].size(0))
    merged["style_index"] = torch.cat(style_indices, dim=0)
    merged["style_font"] = [font_name for batch in batches for font_name in batch["style_font"]]
    merged["style_char_id"] = torch.cat([batch["style_char_id"] for batch in batches], dim=0)
    return merged


def apply_fixed_style_refs(
    batch: Dict[str, torch.Tensor],
    train_dataset: FontImageDataset,
    val_dataset: FontImageDataset | None,
    *,
    seen_count: int,
) -> Dict[str, torch.Tensor]:
    ref_count = int(batch["style_img"].size(1))
    shared_candidates: set[int] | None = None
    for idx, (font_name, char_id) in enumerate(zip(batch["style_font"], batch["style_char_id"].tolist(), strict=True)):
        dataset = train_dataset if idx < seen_count else val_dataset
        if dataset is None:
            raise RuntimeError("val dataset is required for unseen sample style references")
        candidates = set(dataset.list_style_candidate_indices(font_name, excluded_indices=[int(char_id)]))
        shared_candidates = candidates if shared_candidates is None else shared_candidates & candidates
    if shared_candidates is None or len(shared_candidates) < ref_count:
        raise RuntimeError(
            f"Sample batch needs {ref_count} shared fixed style refs, only found {0 if shared_candidates is None else len(shared_candidates)}."
        )
    shared_style_indices = sorted(int(idx) for idx in shared_candidates)[:ref_count]
    fixed_style_imgs = []
    fixed_style_masks = []
    for idx, font_name in enumerate(batch["style_font"]):
        dataset = train_dataset if idx < seen_count else val_dataset
        if dataset is None:
            raise RuntimeError("val dataset is required for unseen sample style references")
        style_img, style_ref_mask, _ = dataset.load_style_refs_by_indices(
            font_name,
            shared_style_indices,
        )
        fixed_style_imgs.append(style_img)
        fixed_style_masks.append(style_ref_mask)
    batch["style_img"] = torch.stack(fixed_style_imgs, dim=0)
    batch["style_ref_mask"] = torch.stack(fixed_style_masks, dim=0)
    return batch


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
    sample_batch = concat_batches(seen_batch, unseen_batch)
    return apply_fixed_style_refs(
        sample_batch,
        train_dataset,
        val_dataset,
        seen_count=seen_count,
    )


def build_model(args: argparse.Namespace) -> SourcePartRefDiT:
    return SourcePartRefDiT(
        in_channels=1,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        encoder_hidden_dim=int(args.encoder_hidden_dim),
        dit_hidden_dim=int(args.dit_hidden_dim),
        dit_depth=int(args.dit_depth),
        dit_heads=int(args.dit_heads),
        dit_mlp_ratio=float(args.dit_mlp_ratio),
        content_injection_layers=None,
        content_style_fusion_heads=4,
    )


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
    parser.add_argument("--dit-mlp-ratio", type=float, required=True)
    parser.add_argument("--train-sampling", type=str, required=True, choices=["shuffle", "cartesian_font_char"])
    parser.add_argument("--cartesian-fonts-per-batch", type=int, required=True)
    parser.add_argument("--cartesian-chars-per-batch", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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

    parser.add_argument("--v-loss-lambda", type=float, required=True)
    parser.add_argument("--sample-steps", type=int, required=True)
    parser.add_argument("--ema-decay", type=float, required=True)
    parser.add_argument("--ema-start-step", type=int, default=-1)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] mode=dit_xpred device={device} seed={int(args.seed)}")
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
    if float(args.weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
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
        load_style_refs=False,
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
            load_style_refs=True,
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
        val_dataloader = build_style_eval_dataloader(
            val_dataset,
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
            cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
        )
    if str(args.train_sampling) == "cartesian_font_char":
        print(
            "[train] using cartesian font-char sampler "
            f"fonts_per_batch={int(args.cartesian_fonts_per_batch)} "
            f"chars_per_batch={int(args.cartesian_chars_per_batch)} "
            f"effective_batch={int(args.cartesian_fonts_per_batch) * int(args.cartesian_chars_per_batch)} "
            "carry_over_partial_batches=1 duplicate_padding=0"
        )

    resolved_epochs = max(1, int(args.epochs))
    if len(dataloader) > 0:
        resolved_epochs = max(resolved_epochs, math.ceil(total_steps / len(dataloader)))

    model = build_model(args)
    trainer = XPredTrainer(
        model,
        device,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        total_steps=total_steps,
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_decay_start_step=None if int(args.lr_decay_start_step) < 0 else int(args.lr_decay_start_step),
        lr_min_scale=float(args.lr_min_scale),
        v_loss_lambda=float(args.v_loss_lambda),
        sample_steps=int(args.sample_steps),
        ema_decay=float(args.ema_decay),
        ema_start_step=None if int(args.ema_start_step) < 0 else int(args.ema_start_step),
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
    run_config["model_type"] = "dit_xpred"
    run_config["lr_warmup_steps"] = int(args.lr_warmup_steps)
    run_config["lr_decay_start_step"] = None if int(args.lr_decay_start_step) < 0 else int(args.lr_decay_start_step)
    run_config["lr_min_scale"] = float(args.lr_min_scale)
    run_config["weight_decay"] = float(args.weight_decay)
    run_config["ffn_activation"] = "swiglu"
    run_config["norm_variant"] = "rms"
    run_config["ode_solver"] = "euler"
    run_config["content_injection_layers"] = list(range(1, int(args.dit_depth) + 1))
    run_config["content_style_fusion_heads"] = 4
    run_config["grad_clip_min_norm"] = None if args.grad_clip_min_norm is None else float(args.grad_clip_min_norm)

    (args.save_dir / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer.fit(dataloader, epochs=resolved_epochs, save_dir=args.save_dir, val_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
