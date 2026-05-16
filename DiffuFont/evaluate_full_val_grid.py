#!/usr/bin/env python3
"""Evaluate every validation font/glyph combination for x-pred checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from evaluate_fixed_grid import (
    build_dataset,
    evaluate_split,
    load_eval_trainer,
    resolve_device,
    set_seed,
)
from models.sdpa_attention import enable_torch_sdpa_backends
from train import (
    FixedFontCharBatchSampler,
    StyleEvalBatchCollator,
    configure_torch_cuda_performance,
)


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    raw = str(checkpoint)
    path = Path(raw)
    if path.exists():
        return path.resolve()
    candidate = run_dir / raw
    if candidate.exists():
        return candidate.resolve()
    if raw.isdigit():
        step_candidate = run_dir / f"ckpt_step_{int(raw)}.pt"
        if step_candidate.exists():
            return step_candidate.resolve()
        epoch_candidate = run_dir / f"ckpt_epoch_{int(raw)}.pt"
        if epoch_candidate.exists():
            return epoch_candidate.resolve()
    raise FileNotFoundError(f"Could not resolve checkpoint {checkpoint!r} under {run_dir}")


def build_full_val_loader(
    dataset,
    *,
    device: torch.device,
    num_workers: int,
    fonts_per_batch: int,
    chars_per_batch: int,
):
    char_indices = list(dataset.split_char_indices)
    sampler = FixedFontCharBatchSampler(
        dataset,
        font_names=list(dataset.font_names),
        char_indices=char_indices,
        fonts_per_batch=int(fonts_per_batch),
        chars_per_batch=int(chars_per_batch),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=StyleEvalBatchCollator(dataset),
    )


def checkpoint_label(path: Path) -> str:
    stem = path.stem
    for prefix in ("ckpt_step_", "ckpt_epoch_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="latest.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-fonts-per-batch", type=int, default=8)
    parser.add_argument("--eval-chars-per-batch", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--style-ref-count", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir
        or (run_dir / f"eval_full_val_{checkpoint_label(resolve_checkpoint(run_dir, args.checkpoint))}")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    configure_torch_cuda_performance()
    enable_torch_sdpa_backends()
    device = resolve_device(args.device, fallback=str(config.get("resolved_device", "")) or None)

    val_dataset = build_dataset(config, "test", eval_style_ref_count=int(args.style_ref_count))
    val_loader = build_full_val_loader(
        val_dataset,
        device=device,
        num_workers=int(args.num_workers),
        fonts_per_batch=int(args.eval_fonts_per_batch),
        chars_per_batch=int(args.eval_chars_per_batch),
    )
    split_meta: dict[str, Any] = {
        "fonts": int(len(val_dataset.font_names)),
        "split_chars": int(len(val_dataset.split_char_indices)),
        "samples": int(sum(len(batch) for batch in val_loader.batch_sampler)),
        "batches": int(len(val_loader)),
        "font_names": list(val_dataset.font_names),
        "char_indices": [int(idx) for idx in val_dataset.split_char_indices],
        "chars_text": [val_dataset.char_list[int(idx)] for idx in val_dataset.split_char_indices],
    }
    print(f"[eval-full-val] device={device} output_dir={output_dir}", flush=True)
    print(f"[eval-full-val] val_meta={json.dumps(split_meta, ensure_ascii=False)}", flush=True)

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    print(f"[eval-full-val] loading checkpoint={checkpoint_path}", flush=True)
    started = time.time()
    trainer = load_eval_trainer(checkpoint_path, device)
    row = evaluate_split(
        trainer,
        val_loader,
        split_name="val",
        step=int(getattr(trainer, "global_step", 0)),
        inference_steps=int(args.inference_steps),
        device=device,
        seed=int(config["seed"]),
        log_every=int(args.log_every),
    )
    row["checkpoint"] = str(checkpoint_path)
    row["checkpoint_label"] = checkpoint_label(checkpoint_path)

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "inference_steps": int(args.inference_steps),
        "style_ref_count": int(args.style_ref_count),
        "eval_fonts_per_batch": int(args.eval_fonts_per_batch),
        "eval_chars_per_batch": int(args.eval_chars_per_batch),
        "train_ratio": float(config["train_ratio"]),
        "font_split_seed": int(config["font_split_seed"]),
        "split": split_meta,
        "result": row,
        "elapsed_sec_total": float(time.time() - started),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[eval-full-val] result={json.dumps(row, ensure_ascii=False, sort_keys=True)}", flush=True)
    print(f"[eval-full-val] wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
