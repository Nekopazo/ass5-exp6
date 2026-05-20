#!/usr/bin/env python3
"""Evaluate fixed train/val font-char grids for x-pred checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from dataset import FontImageDataset
from models.model import XPredTrainer
from models.sdpa_attention import enable_torch_sdpa_backends
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform
from train import (
    FixedFontCharBatchSampler,
    StyleEvalBatchCollator,
    _pack_unique_content,
    configure_torch_cuda_performance,
)


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str, fallback: str | None = None) -> torch.device:
    raw_device = str(raw_device)
    if raw_device == "auto":
        if fallback:
            return torch.device(fallback)
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(raw_device)


def load_eval_trainer(checkpoint_path: Path, device: torch.device) -> XPredTrainer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if checkpoint.get("stage") != "xpred":
        raise RuntimeError(f"Checkpoint is not an x-pred checkpoint: {checkpoint_path}")
    trainer_config = checkpoint.get("trainer_config", {})
    model = SourcePartRefDiT(**checkpoint["model_config"])
    trainer = XPredTrainer(
        model,
        device,
        total_steps=1,
        prediction_type=str(trainer_config["prediction_type"]),
        sample_steps=int(trainer_config.get("sample_steps", 20)),
        ema_decay=float(trainer_config.get("ema_decay", 0.9999)),
    )
    trainer.model.load_state_dict(checkpoint["model_state"], strict=True)
    trainer.global_step = int(checkpoint.get("step", 0))
    trainer.current_epoch = int(checkpoint.get("epoch", 0))
    ema_state = checkpoint.get("ema_model_state")
    if isinstance(ema_state, dict):
        trainer._ensure_ema_model()
        assert trainer.ema_model is not None
        trainer.ema_model.load_state_dict(ema_state, strict=True)
        trainer.ema_initialized = True
        trainer.ema_model.eval()
    trainer.model.eval()
    return trainer


def build_dataset(config: dict[str, Any], split: str, *, eval_style_ref_count: int) -> FontImageDataset:
    image_size = int(config["image_size"])
    glyph_transform = build_base_glyph_transform(image_size=image_size)
    style_ref_count = max(1, int(eval_style_ref_count))
    return FontImageDataset(
        project_root=Path(config["data_root"]),
        max_fonts=int(config.get("max_fonts", 0)),
        style_ref_count=style_ref_count,
        style_ref_count_min=int(config["style_ref_count_min"]),
        style_ref_count_max=int(config["style_ref_count_max"]),
        include_positive_style=False,
        random_seed=int(config["seed"]),
        font_split=split,
        font_split_seed=int(config["font_split_seed"]),
        train_ratio=float(config["train_ratio"]),
        transform=glyph_transform,
        style_transform=glyph_transform,
        load_style_refs=True,
    )


class FixedStyleEvalBatchCollator(StyleEvalBatchCollator):
    def __init__(self, dataset: FontImageDataset, *, fixed_style_indices: list[int]) -> None:
        super().__init__(dataset)
        self.fixed_style_indices = [int(idx) for idx in fixed_style_indices]
        if not self.fixed_style_indices:
            raise ValueError("FixedStyleEvalBatchCollator requires at least one fixed style char.")

    def _select_shared_style_indices(self, excluded_by_font: dict[str, list[int]]) -> list[int]:
        style_set = set(self.fixed_style_indices)
        for font_name, excluded_indices in excluded_by_font.items():
            candidates = set(self.dataset.list_style_candidate_indices(font_name, excluded_indices=excluded_indices))
            missing = sorted(style_set - candidates)
            if missing:
                raise RuntimeError(
                    f"Fixed style chars are unavailable for font '{font_name}': "
                    f"{[self.dataset.char_list[idx] for idx in missing]}"
                )
        return list(self.fixed_style_indices)

    def __call__(self, samples) -> dict[str, torch.Tensor]:
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
            "fixed_style_char_id": torch.tensor(shared_style_indices, dtype=torch.long),
        }


def build_eval_loader(
    dataset: FontImageDataset,
    *,
    device: torch.device,
    num_workers: int,
    fonts_per_batch: int,
    chars_per_batch: int,
    fixed_char_count: int,
    fixed_char_seed: int,
    fixed_style_seed: int,
    fixed_style_count: int,
):
    split_char_indices = list(dataset.split_char_indices)
    fixed_style_count = min(len(split_char_indices), max(1, int(fixed_style_count)))
    style_rng = random.Random(int(fixed_style_seed))
    fixed_style_indices = style_rng.sample(split_char_indices, fixed_style_count)

    target_candidates = [idx for idx in split_char_indices if idx not in set(fixed_style_indices)]
    fixed_char_count = min(len(split_char_indices), max(1, int(fixed_char_count)))
    fixed_char_count = min(len(target_candidates), fixed_char_count)
    rng = random.Random(int(fixed_char_seed))
    char_indices = rng.sample(target_candidates, fixed_char_count)
    batch_sampler = FixedFontCharBatchSampler(
        dataset,
        font_names=list(dataset.font_names),
        char_indices=char_indices,
        fonts_per_batch=int(fonts_per_batch),
        chars_per_batch=int(chars_per_batch),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=FixedStyleEvalBatchCollator(dataset, fixed_style_indices=fixed_style_indices),
    ), char_indices, fixed_style_indices


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x.float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_split(
    trainer: XPredTrainer,
    loader,
    *,
    split_name: str,
    step: int,
    inference_steps: int,
    device: torch.device,
    seed: int,
    log_every: int,
    metric_image_size: int = 0,
) -> dict[str, float | int | str]:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    l1_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    sample_count = 0
    batch_count = 0
    started = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        set_seed(int(seed) + int(step) * 10_000 + (0 if split_name == "train" else 1_000) + batch_idx)
        target = batch["target"].to(device, non_blocking=True)
        generated = trainer.sample(
            batch["content"],
            content_index=batch["content_index"],
            style_img=batch["style_img"],
            style_index=batch["style_index"],
            num_inference_steps=int(inference_steps),
        )
        pred01 = to_unit(generated)
        target01 = to_unit(target)
        if int(metric_image_size) > 0:
            size = int(metric_image_size)
            pred01 = F.interpolate(pred01, size=(size, size), mode="bilinear", align_corners=False)
            target01 = F.interpolate(target01, size=(size, size), mode="bilinear", align_corners=False)
        batch_size = int(target01.size(0))

        fid.update(target01, real=True)
        fid.update(pred01, real=False)

        l1_sum += float(F.l1_loss(pred01, target01, reduction="sum").item()) / float(
            target01[0].numel()
        )
        ssim_sum += float(ssim(pred01, target01).detach().item()) * batch_size
        lpips_sum += float(lpips(pred01, target01).detach().item()) * batch_size
        sample_count += batch_size
        batch_count += 1

        if log_every > 0 and (batch_idx % int(log_every) == 0 or batch_idx == len(loader)):
            elapsed = time.time() - started
            print(
                f"[eval] step={step} split={split_name} "
                f"batch={batch_idx}/{len(loader)} samples={sample_count} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    return {
        "step": int(step),
        "split": split_name,
        "samples": int(sample_count),
        "batches": int(batch_count),
        "fid": float(fid.compute().detach().cpu().item()),
        "l1": float(l1_sum / max(1, sample_count)),
        "ssim": float(ssim_sum / max(1, sample_count)),
        "lpips": float(lpips_sum / max(1, sample_count)),
        "elapsed_sec": float(time.time() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=str, default="100000,150000,200000")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fixed-char-count", type=int, default=1000)
    parser.add_argument("--fixed-char-seed", type=int, default=None)
    parser.add_argument("--fixed-style-count", type=int, default=8)
    parser.add_argument("--fixed-style-seed", type=int, default=None)
    parser.add_argument("--eval-fonts-per-batch", type=int, default=8)
    parser.add_argument("--eval-chars-per-batch", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--metric-image-size", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    fixed_char_seed = int(args.fixed_char_seed) if args.fixed_char_seed is not None else int(config["seed"])
    fixed_style_seed = int(args.fixed_style_seed) if args.fixed_style_seed is not None else fixed_char_seed + 8_191
    output_dir = (args.output_dir or (run_dir / "eval_fixed1000_random_chars_all_fonts")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = output_dir / "metrics.jsonl"
    metrics_jsonl.write_text("", encoding="utf-8")

    set_seed(int(config["seed"]))
    configure_torch_cuda_performance()
    enable_torch_sdpa_backends()
    device = resolve_device(args.device, fallback=str(config.get("resolved_device", "")) or None)

    print(f"[eval] device={device} output_dir={output_dir}", flush=True)
    datasets = {
        "train": build_dataset(config, "train", eval_style_ref_count=int(args.fixed_style_count)),
        "val": build_dataset(config, "test", eval_style_ref_count=int(args.fixed_style_count)),
    }
    loaders = {}
    split_meta = {}
    for split_name, dataset in datasets.items():
        loader, char_indices, fixed_style_indices = build_eval_loader(
            dataset,
            device=device,
            num_workers=int(args.num_workers),
            fonts_per_batch=int(args.eval_fonts_per_batch),
            chars_per_batch=int(args.eval_chars_per_batch),
            fixed_char_count=int(args.fixed_char_count),
            fixed_char_seed=fixed_char_seed,
            fixed_style_seed=fixed_style_seed,
            fixed_style_count=int(args.fixed_style_count),
        )
        loaders[split_name] = loader
        split_meta[split_name] = {
            "fonts": int(len(dataset.font_names)),
            "split_chars": int(len(dataset.split_char_indices)),
            "fixed_chars": int(len(char_indices)),
            "fixed_char_indices": [int(idx) for idx in char_indices],
            "fixed_chars_text": [dataset.char_list[int(idx)] for idx in char_indices],
            "fixed_style_chars": int(len(fixed_style_indices)),
            "fixed_style_char_indices": [int(idx) for idx in fixed_style_indices],
            "fixed_style_chars_text": [dataset.char_list[int(idx)] for idx in fixed_style_indices],
            "samples": int(sum(len(batch) for batch in loader.batch_sampler)),
            "batches": int(len(loader)),
        }
        print(f"[eval] split={split_name} meta={split_meta[split_name]}", flush=True)

    all_results = {
        "run_dir": str(run_dir),
        "device": str(device),
        "inference_steps": int(args.inference_steps),
        "fixed_char_count": int(args.fixed_char_count),
        "fixed_char_seed": int(fixed_char_seed),
        "fixed_style_count": int(args.fixed_style_count),
        "fixed_style_seed": int(fixed_style_seed),
        "eval_fonts_per_batch": int(args.eval_fonts_per_batch),
        "eval_chars_per_batch": int(args.eval_chars_per_batch),
        "train_ratio": float(config["train_ratio"]),
        "font_split_seed": int(config["font_split_seed"]),
        "splits": split_meta,
        "results": [],
    }

    steps = [int(item.strip()) for item in str(args.steps).split(",") if item.strip()]
    for step in steps:
        checkpoint_path = run_dir / f"ckpt_step_{step}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        print(f"[eval] loading checkpoint step={step} path={checkpoint_path}", flush=True)
        trainer = load_eval_trainer(checkpoint_path, device)
        for split_name in ("train", "val"):
            row = evaluate_split(
                trainer,
                loaders[split_name],
                split_name=split_name,
                step=step,
                inference_steps=int(args.inference_steps),
                device=device,
                seed=int(config["seed"]),
                log_every=int(args.log_every),
            )
            all_results["results"].append(row)
            with metrics_jsonl.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            (output_dir / "metrics.json").write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"[eval] result={json.dumps(row, ensure_ascii=False, sort_keys=True)}", flush=True)
        del trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (output_dir / "metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[eval] wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
