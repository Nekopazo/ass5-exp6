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
from torch.utils.data import DataLoader, Sampler

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


def configure_torch_cuda_performance() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


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

    def _sample_shared_style_indices(
        self,
        excluded_by_font: dict[str, list[int]],
        *,
        avoid_indices: list[int] | None = None,
    ) -> list[int]:
        shared_candidates: set[int] | None = None
        for font_name, excluded_indices in excluded_by_font.items():
            candidates = set(self.dataset.list_style_candidate_indices(font_name, excluded_indices=excluded_indices))
            shared_candidates = candidates if shared_candidates is None else shared_candidates & candidates
        if not shared_candidates:
            raise RuntimeError("Batch has no shared non-overlapping style references available across fonts.")
        ordered_candidates = sorted(int(idx) for idx in shared_candidates)
        if avoid_indices is not None:
            avoid_set = {int(idx) for idx in avoid_indices}
            preferred_candidates = [idx for idx in ordered_candidates if idx not in avoid_set]
        else:
            preferred_candidates = ordered_candidates
        max_refs = min(int(self.dataset.style_ref_count_max), len(ordered_candidates))
        if max_refs < 1:
            raise RuntimeError("Batch has no shared non-overlapping style references available across fonts.")
        min_refs = min(int(self.dataset.style_ref_count_min), max_refs)
        ref_count = random.randint(min_refs, max_refs) if min_refs < max_refs else max_refs
        candidate_pool = preferred_candidates if len(preferred_candidates) >= ref_count else ordered_candidates
        return random.sample(candidate_pool, k=ref_count)

    def _pack_unique_style(
        self,
        samples,
        *,
        shared_style_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        unique_style_imgs = []
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
                style_img, _ = self.dataset.load_style_refs_by_indices(
                    font_name,
                    shared_style_indices,
                )
                unique_style_imgs.append(style_img)
                style_fonts.append(font_name)
                style_char_ids.append(int(sample["char_id"]))
            style_index.append(slot)
        return (
            torch.stack(unique_style_imgs, dim=0),
            torch.tensor(style_index, dtype=torch.long),
            style_fonts,
            torch.tensor(style_char_ids, dtype=torch.long),
        )

    def __call__(self, samples) -> Dict[str, torch.Tensor]:
        excluded_by_font = self._build_excluded_style_indices(samples)
        shared_style_indices = self._sample_shared_style_indices(excluded_by_font)
        content, content_index = _pack_unique_content(samples)
        style_img, style_index, style_font, style_char_id = self._pack_unique_style(
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
            "style_index": style_index,
            "style_font": style_font,
            "style_char_id": style_char_id,
        }


class StyleEvalBatchCollator:
    def __init__(self, dataset: FontImageDataset) -> None:
        self.dataset = dataset

    def _build_excluded_style_indices(self, samples) -> dict[str, list[int]]:
        excluded_by_font: dict[str, list[int]] = {}
        for sample in samples:
            font_name = str(sample["font"])
            excluded_by_font.setdefault(font_name, []).append(int(sample["char_id"]))
        return excluded_by_font

    def _select_shared_style_indices(self, excluded_by_font: dict[str, list[int]]) -> list[int]:
        shared_candidates: set[int] | None = None
        for font_name, excluded_indices in excluded_by_font.items():
            candidates = set(self.dataset.list_style_candidate_indices(font_name, excluded_indices=excluded_indices))
            shared_candidates = candidates if shared_candidates is None else shared_candidates & candidates
        if not shared_candidates:
            raise RuntimeError("Validation batch has no shared non-overlapping style references available across fonts.")
        ordered_candidates = sorted(int(idx) for idx in shared_candidates)
        ref_count = min(int(self.dataset.style_ref_count_max), len(ordered_candidates))
        if ref_count < 1:
            raise RuntimeError("Validation batch has no available style references.")
        return ordered_candidates[:ref_count]

    def _pack_unique_style(
        self,
        samples,
        *,
        shared_style_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        unique_style_imgs = []
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
                style_img, _ = self.dataset.load_style_refs_by_indices(
                    font_name,
                    shared_style_indices,
                )
                unique_style_imgs.append(style_img)
                style_fonts.append(font_name)
                style_char_ids.append(int(sample["char_id"]))
            style_index.append(slot)
        return (
            torch.stack(unique_style_imgs, dim=0),
            torch.tensor(style_index, dtype=torch.long),
            style_fonts,
            torch.tensor(style_char_ids, dtype=torch.long),
        )

    def __call__(self, samples) -> Dict[str, torch.Tensor]:
        excluded_by_font = self._build_excluded_style_indices(samples)
        shared_style_indices = self._select_shared_style_indices(excluded_by_font)
        content, content_index = _pack_unique_content(samples)
        style_img, style_index, style_font, style_char_id = self._pack_unique_style(
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
            "style_index": style_index,
            "style_font": style_font,
            "style_char_id": style_char_id,
        }


class FixedFontCharBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: FontImageDataset,
        *,
        font_names: list[str],
        char_indices: list[int],
        fonts_per_batch: int,
        chars_per_batch: int,
    ) -> None:
        self.dataset = dataset
        self.font_names = [str(name) for name in font_names]
        self.char_indices = [int(idx) for idx in char_indices]
        self.fonts_per_batch = max(1, int(fonts_per_batch))
        self.chars_per_batch = max(1, int(chars_per_batch))
        if not self.font_names:
            raise ValueError("FixedFontCharBatchSampler requires at least one font.")
        if not self.char_indices:
            raise ValueError("FixedFontCharBatchSampler requires at least one char.")

    def __len__(self) -> int:
        return len(self._chunk_items(self.font_names, self.fonts_per_batch)) * len(
            self._chunk_items(self.char_indices, self.chars_per_batch)
        )

    def __iter__(self):
        for font_group in self._chunk_items(self.font_names, self.fonts_per_batch):
            for char_group in self._chunk_items(self.char_indices, self.chars_per_batch):
                batch: list[int] = []
                for font_name in font_group:
                    lookup = self.dataset.sample_index_by_font_char[font_name]
                    for char_index in char_group:
                        sample_index = lookup.get(int(char_index))
                        if sample_index is not None:
                            batch.append(int(sample_index))
                if batch:
                    yield batch

    @staticmethod
    def _chunk_items(items, chunk_size: int):
        return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


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
    fixed_font_count: int = 16,
    fixed_char_count: int = 16,
) -> DataLoader:
    _ = seed
    fixed_font_names = list(dataset.font_names)[: max(1, int(fixed_font_count))]
    fixed_char_indices = list(dataset.split_char_indices)[: max(1, int(fixed_char_count))]
    batch_sampler = FixedFontCharBatchSampler(
        dataset,
        font_names=fixed_font_names,
        char_indices=fixed_char_indices,
        fonts_per_batch=int(cartesian_fonts_per_batch),
        chars_per_batch=int(cartesian_chars_per_batch),
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=StyleEvalBatchCollator(dataset),
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
    )


def slice_batch(batch: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    output = {}
    for key, value in batch.items():
        if key in {
            "content",
            "content_index",
            "style_img",
            "style_index",
            "style_font",
            "style_char_id",
        }:
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
    output["style_index"] = style_index
    output["style_font"] = [batch["style_font"][idx] for idx in style_positions.tolist()]
    output["style_char_id"] = batch["style_char_id"][style_positions]
    return output


def concat_batches(*batches: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    keys = batches[0].keys()
    for key in keys:
        if key in {
            "content",
            "content_index",
            "style_img",
            "style_index",
            "style_font",
            "style_char_id",
        }:
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
    style_indices = []
    style_offset = 0
    for batch in batches:
        style_indices.append(batch["style_index"] + style_offset)
        style_offset += int(batch["style_img"].size(0))
    merged["style_index"] = torch.cat(style_indices, dim=0)
    merged["style_font"] = [font_name for batch in batches for font_name in batch["style_font"]]
    merged["style_char_id"] = torch.cat([batch["style_char_id"] for batch in batches], dim=0)
    return merged


def build_sample_batch(
    train_dataset: FontImageDataset,
    val_dataset: FontImageDataset | None,
    *,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    _ = device
    _ = seed

    def fixed_diagonal_batch(dataset: FontImageDataset, count: int) -> Dict[str, torch.Tensor]:
        fixed_font_names = list(dataset.font_names)[: max(1, int(count))]
        fixed_char_indices = list(dataset.split_char_indices)[: max(1, int(count))]
        pair_count = min(len(fixed_font_names), len(fixed_char_indices), max(1, int(count)))
        if pair_count <= 0:
            raise RuntimeError("dataset has no fixed font/char pair for sampling")
        samples = [
            dataset[
                dataset.sample_index_by_font_char[fixed_font_names[idx]][int(fixed_char_indices[idx])]
            ]
            for idx in range(pair_count)
        ]
        return StyleEvalBatchCollator(dataset)(samples)

    seen_count = min(8, len(train_dataset.font_names), len(train_dataset.split_char_indices))
    if seen_count <= 0:
        raise RuntimeError("train dataset has no fixed font/char pair for sampling")
    seen_batch = fixed_diagonal_batch(train_dataset, seen_count)
    if val_dataset is None or len(val_dataset.font_names) == 0:
        return seen_batch

    unseen_count = min(8, len(val_dataset.font_names), len(val_dataset.split_char_indices))
    if unseen_count <= 0:
        return seen_batch
    unseen_batch = fixed_diagonal_batch(val_dataset, unseen_count)
    return concat_batches(seen_batch, unseen_batch)


def build_model(args: argparse.Namespace) -> SourcePartRefDiT:
    return SourcePartRefDiT(
        in_channels=3,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        patch_embed_bottleneck_dim=int(args.patch_embed_bottleneck_dim),
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
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--max-fonts", type=int, required=True)
    parser.add_argument("--style-ref-count", type=int, required=True)
    parser.add_argument("--style-ref-count-min", type=int, required=True)
    parser.add_argument("--style-ref-count-max", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)

    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patch-embed-bottleneck-dim", type=int, default=0)
    parser.add_argument("--encoder-hidden-dim", type=int, required=True)
    parser.add_argument("--dit-hidden-dim", type=int, required=True)
    parser.add_argument("--dit-depth", type=int, required=True)
    parser.add_argument("--dit-heads", type=int, required=True)
    parser.add_argument("--dit-mlp-ratio", type=float, required=True)
    parser.add_argument("--train-sampling", type=str, required=True, choices=["shuffle", "cartesian_font_char"])
    parser.add_argument("--cartesian-fonts-per-batch", type=int, required=True)
    parser.add_argument("--cartesian-chars-per-batch", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-min-scale", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--log-every-steps", type=int, required=True)
    parser.add_argument("--val-every-steps", type=int, required=True)
    parser.add_argument("--save-every-steps", type=int, required=True)
    parser.add_argument("--sample-every-steps", type=int, required=True)
    parser.add_argument("--grad-clip-norm", type=float, required=True)
    parser.add_argument("--grad-clip-min-norm", type=float, default=None)

    parser.add_argument("--sample-steps", type=int, required=True)
    parser.add_argument("--p-mean", type=float, default=-0.8)
    parser.add_argument("--p-std", type=float, default=0.8)
    parser.add_argument("--t-eps", type=float, default=0.05)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--prediction-type", type=str, default="x", choices=["x", "noise", "velocity"])
    parser.add_argument("--ema-decay", type=float, required=True)
    parser.add_argument("--ema-start-step", type=int, default=40000)
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    configure_torch_cuda_performance()
    enable_torch_sdpa_backends()
    device = resolve_device(args.device)
    print(f"[train] mode=dit_xpred device={device} seed={int(args.seed)}")
    print(f"[train] attention_backend={describe_torch_sdpa_backends()}")
    if device.type == "cuda":
        print(
            "[train] cuda_performance="
            f"cudnn_benchmark={int(torch.backends.cudnn.benchmark)} "
            f"matmul_allow_tf32={int(torch.backends.cuda.matmul.allow_tf32)} "
            f"cudnn_allow_tf32={int(torch.backends.cudnn.allow_tf32)} "
            f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
        )

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
        train_ratio=float(args.train_ratio),
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
            train_ratio=float(args.train_ratio),
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
    trainer_val_max_batches = None
    if val_dataset is not None:
        val_dataloader = build_style_eval_dataloader(
            val_dataset,
            num_workers=int(args.num_workers),
            device=device,
            seed=int(args.seed) + 1,
            cartesian_fonts_per_batch=int(args.cartesian_fonts_per_batch),
            cartesian_chars_per_batch=int(args.cartesian_chars_per_batch),
            fixed_font_count=16,
            fixed_char_count=16,
        )
        trainer_val_max_batches = None
        print(
            "[val] fixed unseen eval grid "
            f"fonts={min(16, len(val_dataset.font_names))} "
            f"chars={min(16, len(val_dataset.split_char_indices))} "
            f"samples={sum(len(batch) for batch in val_dataloader.batch_sampler)}",
            flush=True,
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
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        total_steps=total_steps,
        lr_schedule=str(args.lr_schedule),
        lr_warmup_steps=int(args.lr_warmup_steps),
        lr_min_scale=float(args.lr_min_scale),
        p_mean=float(args.p_mean),
        p_std=float(args.p_std),
        t_eps=float(args.t_eps),
        noise_scale=float(args.noise_scale),
        prediction_type=str(args.prediction_type),
        sample_steps=int(args.sample_steps),
        ema_decay=float(args.ema_decay),
        ema_start_step=int(args.ema_start_step),
        log_every_steps=log_every_steps,
        save_every_steps=resolved_save_every_steps,
        val_every_steps=val_every_steps,
        val_max_batches=trainer_val_max_batches,
        grad_clip_norm=float(args.grad_clip_norm),
        grad_clip_min_norm=None if args.grad_clip_min_norm is None else float(args.grad_clip_min_norm),
    )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    trainer.sample_batch_builder = lambda: build_sample_batch(dataset, val_dataset, device=device, seed=int(args.seed))
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
    run_config["train_chars"] = int(len(dataset.split_char_indices))
    run_config["val_chars"] = 0 if val_dataset is None else int(len(val_dataset.split_char_indices))
    run_config["fixed_val_fonts"] = 0 if val_dataset is None else min(16, int(len(val_dataset.font_names)))
    run_config["fixed_val_chars"] = 0 if val_dataset is None else min(16, int(len(val_dataset.split_char_indices)))
    run_config["fixed_val_samples"] = 0 if val_dataloader is None else sum(len(batch) for batch in val_dataloader.batch_sampler)
    run_config["computed_total_steps"] = int(total_steps)
    run_config["model_type"] = "dit_xpred"
    run_config["prediction_type"] = str(args.prediction_type)
    run_config["loss_type"] = "jit_v_mse"
    run_config["lr_schedule"] = str(args.lr_schedule)
    run_config["lr_warmup_steps"] = int(args.lr_warmup_steps)
    run_config["lr_min_scale"] = float(args.lr_min_scale)
    run_config["weight_decay"] = float(args.weight_decay)
    run_config["adam_betas"] = [float(args.adam_beta1), float(args.adam_beta2)]
    run_config["p_mean"] = float(args.p_mean)
    run_config["p_std"] = float(args.p_std)
    run_config["t_eps"] = float(args.t_eps)
    run_config["noise_scale"] = float(args.noise_scale)
    run_config["ffn_activation"] = "swiglu"
    run_config["norm_variant"] = "rms"
    run_config["ode_solver"] = "heun_last_euler"
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
