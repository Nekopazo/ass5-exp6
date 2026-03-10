#!/usr/bin/env python3
"""Teacher training entry for DiffuFont (baseline / part_only / style_only)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from dataset import FontImageDataset
from models.model import DiffusionTrainer, FlowMatchingTrainer
from models.source_part_ref_unet import (
    FIXED_STYLE_LOCAL_MOD_SCALES,
    FIXED_STYLE_SITE_ARCH,
    FIXED_STYLE_TOKEN_CONSUMER_MAP,
    FIXED_STYLE_TRANSFORMER_SCALES,
    SourcePartRefUNet,
)
from models.source_fontdiffuser.attention import CrossAttention as SourceCrossAttention
from style_augment import build_base_glyph_transform, build_style_reference_transform


STYLE_ONLY_MAIN_DEFAULTS: Dict[str, float | int] = {
    "lambda_nce": 0.05,
    "aux_loss_warmup_steps": 5000,
    "lambda_slot_nce": 0.03,
    "lambda_cons": 0.02,
    "lambda_div": 0.01,
    "lambda_proxy_low": 0.05,
    "lambda_proxy_mid": 0.05,
    "lambda_proxy_high": 0.05,
    "lambda_attn_sep": 0.01,
    "lambda_attn_order": 0.0,
    "lambda_attn_role": 0.0,
    "attn_overlap_margin": 0.70,
    "attn_entropy_gap": 0.03,
    "style_ref_drop_prob": 0.15,
    "style_ref_drop_min_keep": 4,
    "style_site_drop_prob": 0.10,
    "style_site_drop_min_keep": 1,
    "freeze_style_backbone_steps": 5000,
    "style_backbone_lr_scale": 0.1,
    "style_memory_up16_count": 6,
    "style_memory_up32_count": 6,
    "style_memory_up16_pool_hw": 16,
    "style_memory_up32_pool_hw": 16,
    "content_query_up16_count": 2,
    "content_query_up32_count": 4,
}


def _try_enable_xformers(model: SourcePartRefUNet) -> bool:
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        enabled_count = sum(
            1
            for module in model.unet.modules()
            if isinstance(module, SourceCrossAttention)
            and bool(getattr(module, "_use_memory_efficient_attention_xformers", False))
        )
        print(
            f"[train] xformers memory-efficient attention enabled "
            f"(custom_cross_attn_modules={enabled_count})"
        )
        return True
    except Exception as e:
        print(f"[train] xformers not available, using default attention: {e}")
        return False


def _try_enable_gradient_checkpointing(model: SourcePartRefUNet) -> bool:
    try:
        model.unet.enable_gradient_checkpointing()
        print("[train] gradient checkpointing enabled")
        return True
    except Exception as e:
        print(f"[train] gradient checkpointing not available: {e}")
        return False


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    contents = [s["content"] for s in samples]
    targets = [s["input"] for s in samples]

    font_names = [s["font"] for s in samples]
    unique_fonts = sorted(set(font_names))
    font_to_id = {f: i for i, f in enumerate(unique_fonts)}
    font_ids = torch.tensor([font_to_id[f] for f in font_names], dtype=torch.long)

    batch: Dict[str, torch.Tensor] = {
        "content": torch.stack(contents),
        "target": torch.stack(targets),
        "font_ids": font_ids,
    }

    has_style = any("style_img" in s for s in samples)
    if has_style:
        if not all("style_img" in s for s in samples):
            raise ValueError("Inconsistent style_img presence in batch")
        style_img = torch.stack([s["style_img"] for s in samples])
        batch["style_img"] = style_img

        if all("style_ref_mask" in s for s in samples):
            batch["style_ref_mask"] = torch.stack([s["style_ref_mask"] for s in samples])
        elif style_img.dim() == 5:
            # (B, R, C, H, W)
            bsz, ref_n = int(style_img.size(0)), int(style_img.size(1))
            batch["style_ref_mask"] = torch.ones((bsz, ref_n), dtype=torch.float32)

    has_style_v2 = any("style_img_view2" in s for s in samples)
    if has_style_v2:
        if not all("style_img_view2" in s for s in samples):
            raise ValueError("Inconsistent style_img_view2 presence in batch")
        style_img_v2 = torch.stack([s["style_img_view2"] for s in samples])
        batch["style_img_view2"] = style_img_v2

        if all("style_ref_mask_view2" in s for s in samples):
            batch["style_ref_mask_view2"] = torch.stack([s["style_ref_mask_view2"] for s in samples])
        elif style_img_v2.dim() == 5:
            bsz2, ref_n2 = int(style_img_v2.size(0)), int(style_img_v2.size(1))
            batch["style_ref_mask_view2"] = torch.ones((bsz2, ref_n2), dtype=torch.float32)

    return batch


def normalize_conditioning_mode(raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    alias = {"parts_vector_only": "part_only"}
    mode = alias.get(mode, mode)
    valid = {"baseline", "part_only", "style_only"}
    if mode not in valid:
        raise ValueError(f"conditioning mode must be one of {sorted(valid)}, got '{raw_mode}'")
    return mode


def mode_uses_style(mode: str) -> bool:
    return normalize_conditioning_mode(mode) in {"part_only", "style_only"}


def resolve_device(device_arg: str) -> torch.device:
    def _probe_cuda(candidate: str) -> tuple[bool, str | None]:
        try:
            dev = torch.device(candidate)
            _ = torch.empty((1,), device=dev)
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    if device_arg == "auto":
        ok, _ = _probe_cuda("cuda:0")
        return torch.device("cuda:0" if ok else "cpu")

    req = torch.device(device_arg)
    if req.type == "cuda":
        ok, err = _probe_cuda(str(req))
        if not ok:
            raise ValueError(f"CUDA device requested but probe failed for {req}: {err}")
    return req


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    return v


def set_global_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _seed_worker(worker_id: int) -> None:
    _ = worker_id
    ws = torch.initial_seed() % 2**32
    random.seed(ws)
    np.random.seed(ws)
    # Avoid CPU thread oversubscription with many DataLoader workers.
    torch.set_num_threads(1)


def split_indices_by_font(
    dataset: FontImageDataset,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    total_samples = len(dataset.samples)
    if total_samples <= 0:
        return [], [], {"total_fonts": 0, "train_fonts": 0, "val_fonts": 0}

    fonts = sorted({font for font, _ in dataset.samples})
    if val_ratio <= 0.0 or len(fonts) < 2:
        return (
            list(range(total_samples)),
            [],
            {"total_fonts": len(fonts), "train_fonts": len(fonts), "val_fonts": 0},
        )

    rng = random.Random(int(seed) + 2027)
    rng.shuffle(fonts)
    val_n = int(round(len(fonts) * float(val_ratio)))
    val_n = max(1, min(len(fonts) - 1, val_n))
    val_fonts = set(fonts[:val_n])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for i, (font, _) in enumerate(dataset.samples):
        if font in val_fonts:
            val_indices.append(i)
        else:
            train_indices.append(i)

    if not train_indices:
        train_indices = [val_indices.pop()]
    if not val_indices:
        val_indices = [train_indices.pop()]

    return (
        train_indices,
        val_indices,
        {"total_fonts": len(fonts), "train_fonts": len(fonts) - val_n, "val_fonts": val_n},
    )


class NoReplacementFontBatchSampler(Sampler[list[int]]):
    """Batch sampler: each batch contains unique fonts (no replacement within batch)."""

    def __init__(
        self,
        subset: Subset,
        batch_size: int,
        seed: int,
        steps_per_epoch: int,
        shuffle: bool = True,
    ) -> None:
        self.subset = subset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.shuffle = bool(shuffle)
        self._epoch = 0

        if self.batch_size <= 0:
            raise ValueError("NoReplacementFontBatchSampler batch_size must be > 0")

        base = subset.dataset
        if not hasattr(base, "samples"):
            raise ValueError("subset.dataset must expose .samples for font-aware batching")

        font_to_local: dict[str, list[int]] = defaultdict(list)
        for local_i, global_i in enumerate(subset.indices):
            font_name = base.samples[int(global_i)][0]
            font_to_local[str(font_name)].append(int(local_i))

        self.font_to_local_indices = {
            k: list(v) for k, v in sorted(font_to_local.items(), key=lambda x: x[0])
        }
        self.font_pool = list(self.font_to_local_indices.keys())
        if len(self.font_pool) < self.batch_size:
            raise ValueError(
                f"batch_size={self.batch_size} exceeds train font count={len(self.font_pool)} "
                "(cannot satisfy no-replacement font batch)."
            )

    def __len__(self) -> int:
        return int(self.steps_per_epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch * 100003)
        self._epoch += 1

        produced = 0
        while produced < self.steps_per_epoch:
            order = list(self.font_pool)
            if self.shuffle:
                rng.shuffle(order)
            pos = 0
            while (pos + self.batch_size) <= len(order) and produced < self.steps_per_epoch:
                picked_fonts = order[pos : pos + self.batch_size]
                batch_indices: list[int] = []
                for f in picked_fonts:
                    local_choices = self.font_to_local_indices[f]
                    batch_indices.append(int(rng.choice(local_choices)))
                yield batch_indices
                pos += self.batch_size
                produced += 1


def build_mixed_visual_batch(
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
    total_batch_size: int,
) -> Dict[str, torch.Tensor]:
    train_bs = int(train_batch["content"].size(0))
    val_bs = int(val_batch["content"].size(0))
    target_bs = max(2, int(total_batch_size))
    n_train = min(max(1, target_bs // 2), train_bs)
    n_val = min(max(1, target_bs - n_train), val_bs)

    train_cut = {k: v[:n_train] for k, v in train_batch.items()}
    val_cut = {k: v[:n_val] for k, v in val_batch.items()}
    mixed: Dict[str, torch.Tensor] = {}

    for k in train_cut:
        if k not in val_cut:
            continue
        t = train_cut[k]
        v = val_cut[k]
        if not (torch.is_tensor(t) and torch.is_tensor(v)):
            continue
        mixed[k] = torch.cat([t, v], dim=0)

    mixed["viz_split_flag"] = torch.cat(
        [
            torch.ones((n_train,), dtype=torch.long),
            torch.zeros((n_val,), dtype=torch.long),
        ],
        dim=0,
    )

    if "content" not in mixed:
        raise RuntimeError("failed to build mixed visualization batch: missing content tensor")
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable UNet gradient checkpointing (default: disabled).",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--trainer", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-nce", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_nce"]))
    parser.add_argument("--lambda-slot-nce", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_slot_nce"]))
    parser.add_argument("--lambda-cons", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_cons"]))
    parser.add_argument("--lambda-div", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_div"]))
    parser.add_argument("--lambda-proxy-low", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_proxy_low"]))
    parser.add_argument("--lambda-proxy-mid", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_proxy_mid"]))
    parser.add_argument("--lambda-proxy-high", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_proxy_high"]))
    parser.add_argument("--lambda-attn-sep", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_attn_sep"]))
    parser.add_argument("--lambda-attn-order", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_attn_order"]))
    parser.add_argument("--lambda-attn-role", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["lambda_attn_role"]))
    parser.add_argument("--style-nce-temp", type=float, default=0.07)
    parser.add_argument("--aux-loss-warmup-steps", type=int, default=int(STYLE_ONLY_MAIN_DEFAULTS["aux_loss_warmup_steps"]))
    parser.add_argument("--attn-overlap-margin", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["attn_overlap_margin"]))
    parser.add_argument("--attn-entropy-gap", type=float, default=float(STYLE_ONLY_MAIN_DEFAULTS["attn_entropy_gap"]))
    parser.add_argument(
        "--style-memory-up16-count",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_memory_up16_count"]),
    )
    parser.add_argument(
        "--style-memory-up32-count",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_memory_up32_count"]),
    )
    parser.add_argument(
        "--style-memory-up16-pool-hw",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_memory_up16_pool_hw"]),
    )
    parser.add_argument(
        "--style-memory-up32-pool-hw",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_memory_up32_pool_hw"]),
    )
    parser.add_argument(
        "--content-query-up16-count",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["content_query_up16_count"]),
    )
    parser.add_argument(
        "--content-query-up32-count",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["content_query_up32_count"]),
    )
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Total training steps for OneCycleLR. <=0 means epochs*steps_per_epoch.",
    )

    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio by font identity. 0 disables val split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-line",
        type=str,
        default="part_only",
        choices=["baseline", "part_only", "style_only"],
        help="Teacher line to train.",
    )
    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default=None,
        choices=["baseline", "parts_vector_only", "part_only", "style_only"],
        help="Legacy alias; if set it overrides --teacher-line.",
    )
    parser.add_argument("--style-ref-count", type=int, default=8)
    parser.add_argument(
        "--reference-cluster-json",
        type=Path,
        default=Path("CharacterData/reference_cluster.json"),
    )
    parser.add_argument(
        "--style-ref-drop-prob",
        type=float,
        default=float(STYLE_ONLY_MAIN_DEFAULTS["style_ref_drop_prob"]),
        help="Reference dropout probability in main training (1.md suggests 0.1~0.2).",
    )
    parser.add_argument(
        "--style-ref-drop-min-keep",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_ref_drop_min_keep"]),
        help="Minimum kept references after dropout when style refs are used.",
    )
    parser.add_argument(
        "--style-site-drop-prob",
        type=float,
        default=float(STYLE_ONLY_MAIN_DEFAULTS["style_site_drop_prob"]),
        help="Drop probability for style injection sites during training.",
    )
    parser.add_argument(
        "--style-site-drop-min-keep",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["style_site_drop_min_keep"]),
        help="Minimum kept style injection sites after site dropout.",
    )
    parser.add_argument(
        "--freeze-style-backbone-steps",
        type=int,
        default=int(STYLE_ONLY_MAIN_DEFAULTS["freeze_style_backbone_steps"]),
        help=(
            "Freeze the entire style backbone until this step, then unfreeze low and high stages together."
        ),
    )
    parser.add_argument(
        "--style-backbone-lr-scale",
        type=float,
        default=float(STYLE_ONLY_MAIN_DEFAULTS["style_backbone_lr_scale"]),
        help="LR scale applied to the full style backbone parameter group after unfreezing.",
    )

    parser.add_argument("--sample-every-steps", type=int, default=300)
    parser.add_argument("--sample-solver", type=str, default="dpm", choices=["dpm", "ddim"])
    parser.add_argument("--sample-inference-steps", type=int, default=20)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--detailed-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--split-save-components", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--style-start-channel", type=int, default=16)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--style-token-count", type=int, default=3, help="Fixed at 3 (low/mid/high).")
    parser.add_argument("--pretrained-style-encoder", type=str, default=None)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch * grad_accum.",
    )
    args = parser.parse_args()

    active_mode = normalize_conditioning_mode(
        args.conditioning_profile if args.conditioning_profile is not None else args.teacher_line
    )
    use_style_image = mode_uses_style(active_mode)

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    print(f"[train] device={device} precision={args.precision} seed={int(args.seed)}")

    run_cfg: Dict[str, Any] = {k: _to_jsonable(v) for k, v in vars(args).items()}
    run_cfg.update(
        {
            "stage": "teacher_only",
            "active_conditioning_mode": active_mode,
            "use_style_image": bool(use_style_image),
            "use_part_bank": False,
            "fixed_style_transformer_scales": list(FIXED_STYLE_TRANSFORMER_SCALES),
            "fixed_style_local_mod_scales": list(FIXED_STYLE_LOCAL_MOD_SCALES),
            "fixed_style_token_consumers": {
                k: list(v) for k, v in FIXED_STYLE_TOKEN_CONSUMER_MAP.items()
            },
            "fixed_style_site_arch": dict(FIXED_STYLE_SITE_ARCH),
        }
    )

    glyph_transform = build_base_glyph_transform(image_size=128)
    style_transform = build_style_reference_transform(image_size=128)

    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=args.max_fonts,
        use_style_image=bool(use_style_image),
        use_part_bank=False,
        random_seed=int(args.seed),
        transform=glyph_transform,
        style_transform=style_transform,
        cache_style_image=False,
        style_ref_count=(int(args.style_ref_count) if use_style_image else 1),
        reference_cluster_json=args.reference_cluster_json,
    )
    run_cfg.update(
        {
            "style_ref_count_active": int(args.style_ref_count) if use_style_image else 0,
            "style_transform": "resize_only",
            "cache_style_image": False,
            "style_ref_drop_prob_active": float(args.style_ref_drop_prob) if use_style_image else 0.0,
            "style_ref_drop_min_keep_active": int(args.style_ref_drop_min_keep) if use_style_image else 0,
            "style_site_drop_prob_active": float(args.style_site_drop_prob) if use_style_image else 0.0,
            "style_site_drop_min_keep_active": int(args.style_site_drop_min_keep) if use_style_image else 0,
        }
    )

    train_indices, val_indices, split_stats = split_indices_by_font(
        dataset=dataset,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    print(
        "[train] split=by-font "
        f"train_fonts={split_stats['train_fonts']} val_fonts={split_stats['val_fonts']} "
        f"train_samples={len(train_indices)} val_samples={len(val_indices)}"
    )
    run_cfg.update(
        {
            "split_strategy": "by_font",
            "val_ratio": float(args.val_ratio),
            "split_total_fonts": int(split_stats["total_fonts"]),
            "split_train_fonts": int(split_stats["train_fonts"]),
            "split_val_fonts": int(split_stats["val_fonts"]),
            "split_train_samples": int(len(train_indices)),
            "split_val_samples": int(len(val_indices)),
        }
    )

    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed))

    target_steps_per_epoch = max(1, len(train_indices) // max(1, int(args.batch)))
    loader_kwargs: Dict[str, Any] = {
        "dataset": train_subset,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if use_style_image:
        train_batch_sampler = NoReplacementFontBatchSampler(
            subset=train_subset,
            batch_size=int(args.batch),
            seed=int(args.seed) + 101,
            steps_per_epoch=target_steps_per_epoch,
            shuffle=True,
        )
        loader_kwargs["batch_sampler"] = train_batch_sampler
    else:
        loader_kwargs["batch_size"] = int(args.batch)
        loader_kwargs["shuffle"] = True
        loader_kwargs["generator"] = loader_gen
    if int(args.num_workers) > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
    if int(args.num_workers) > 0 and int(args.prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(**loader_kwargs)
    run_cfg.update(
        {
            "train_batch_sampler": "no_replacement_font" if use_style_image else "random_sample_shuffle",
            "train_steps_per_epoch_target": int(target_steps_per_epoch),
        }
    )

    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader_kwargs: Dict[str, Any] = {
            "dataset": val_subset,
            "batch_size": min(int(args.batch), len(val_subset)),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": True,
            "collate_fn": collate_fn,
        }
        if int(args.num_workers) > 0:
            val_loader_kwargs["worker_init_fn"] = _seed_worker
        if int(args.num_workers) > 0 and int(args.prefetch_factor) > 0:
            val_loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        val_loader = DataLoader(**val_loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_train_steps = steps_per_epoch * args.epochs
    if total_train_steps <= 0:
        raise ValueError(f"Invalid steps: steps_per_epoch={steps_per_epoch}, epochs={args.epochs}")

    effective_train_steps = total_train_steps // max(1, int(args.grad_accum))
    total_steps = args.total_steps if args.total_steps > 0 else effective_train_steps
    print(
        "[train] "
        f"mode={active_mode} style_attn={FIXED_STYLE_TRANSFORMER_SCALES} "
        f"style_local_mod={FIXED_STYLE_LOCAL_MOD_SCALES} "
        f"style_memory=(up16:{int(args.style_memory_up16_count)},up32:{int(args.style_memory_up32_count)}) "
        f"content_queries=(up16:{int(args.content_query_up16_count)},up32:{int(args.content_query_up32_count)}) "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} grad_accum={args.grad_accum}"
    )

    model = SourcePartRefUNet(
        in_channels=1,
        image_size=args.image_size,
        content_start_channel=64,
        style_start_channel=int(args.style_start_channel),
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=4,
        channel_attn=True,
        conditioning_profile=active_mode,
        style_token_dim=int(args.style_token_dim),
        style_token_count=int(args.style_token_count),
        style_memory_up16_count=int(args.style_memory_up16_count),
        style_memory_up32_count=int(args.style_memory_up32_count),
        style_memory_up16_pool_hw=int(args.style_memory_up16_pool_hw),
        style_memory_up32_pool_hw=int(args.style_memory_up32_pool_hw),
        content_query_up16_count=int(args.content_query_up16_count),
        content_query_up32_count=int(args.content_query_up32_count),
    )

    _try_enable_xformers(model)
    if bool(args.gradient_checkpointing):
        _try_enable_gradient_checkpointing(model)
    else:
        print("[train] gradient checkpointing disabled")

    if args.pretrained_style_encoder:
        ckpt = torch.load(args.pretrained_style_encoder, map_location="cpu")
        route_meta = {}
        consumer_arch_meta = {}
        if isinstance(ckpt, dict) and isinstance(ckpt.get("extra"), dict):
            route_meta = ckpt["extra"].get("token_consumer_map", {}) or {}
            consumer_arch_meta = ckpt["extra"].get("style_site_arch", {}) or {}
        expected_route = {k: list(v) for k, v in FIXED_STYLE_TOKEN_CONSUMER_MAP.items()}
        expected_consumer_arch = dict(FIXED_STYLE_SITE_ARCH)
        if route_meta != expected_route:
            print(
                "[train] warning: pretrained style encoder routing metadata mismatch: "
                f"ckpt={route_meta or '<missing>'} current={expected_route}. "
                "Loading style encoder weights non-strictly anyway.",
                flush=True,
            )
        if consumer_arch_meta != expected_consumer_arch:
            print(
                "[train] warning: pretrained style encoder consumer metadata mismatch: "
                f"ckpt={consumer_arch_meta or '<missing>'} current={expected_consumer_arch}. "
                "Loading style encoder weights non-strictly anyway.",
                flush=True,
            )
        if isinstance(ckpt, dict) and isinstance(ckpt.get("style_encoder"), dict):
            sd = ckpt["style_encoder"]
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("model_state"), dict):
            sd = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise RuntimeError(f"Unsupported style checkpoint format: {type(ckpt)}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[train] Loaded pretrained style encoder from {args.pretrained_style_encoder}")
        print(f"  loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}")

    trainer_cls = DiffusionTrainer if args.trainer == "diffusion" else FlowMatchingTrainer
    trainer_kwargs: Dict[str, Any] = {
        "lr": args.lr,
        "lambda_nce": float(args.lambda_nce),
        "lambda_slot_nce": float(args.lambda_slot_nce),
        "lambda_cons": float(args.lambda_cons),
        "lambda_div": float(args.lambda_div),
        "lambda_proxy_low": float(args.lambda_proxy_low),
        "lambda_proxy_mid": float(args.lambda_proxy_mid),
        "lambda_proxy_high": float(args.lambda_proxy_high),
        "lambda_attn_sep": float(args.lambda_attn_sep),
        "lambda_attn_order": float(args.lambda_attn_order),
        "lambda_attn_role": float(args.lambda_attn_role),
        "nce_temperature": float(args.style_nce_temp),
        "aux_loss_warmup_steps": int(args.aux_loss_warmup_steps),
        "attn_overlap_margin": float(args.attn_overlap_margin),
        "attn_entropy_gap": float(args.attn_entropy_gap),
        "T": args.diffusion_steps,
        "total_steps": total_steps,
        "precision": args.precision,
        "save_every_steps": (args.save_every_steps if args.save_every_steps > 0 else None),
        "log_every_steps": (args.log_every_steps if args.log_every_steps > 0 else None),
        "detailed_log": args.detailed_log,
        "grad_accum_steps": max(1, int(args.grad_accum)),
        "conditioning_mode": active_mode,
        "part_drop_prob": 0.0,
        "style_ref_drop_prob": float(args.style_ref_drop_prob) if use_style_image else 0.0,
        "style_ref_drop_min_keep": int(args.style_ref_drop_min_keep) if use_style_image else 1,
        "style_site_drop_prob": float(args.style_site_drop_prob) if use_style_image else 0.0,
        "style_site_drop_min_keep": int(args.style_site_drop_min_keep) if use_style_image else 1,
        "freeze_part_encoder_steps": 0,
        "freeze_style_backbone_steps": int(args.freeze_style_backbone_steps) if use_style_image else 0,
        "style_backbone_lr_scale": float(args.style_backbone_lr_scale),
    }
    if args.trainer == "flow_matching":
        trainer_kwargs["lambda_fm"] = args.lambda_fm
    else:
        trainer_kwargs["lambda_mse"] = args.lambda_diff

    trainer = trainer_cls(model, device, **trainer_kwargs)
    trainer.sample_solver = str(args.sample_solver).lower()
    trainer.sample_inference_steps = int(args.sample_inference_steps)
    trainer.save_split_components = bool(args.split_save_components)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}")

    try:
        if args.trainer == "diffusion" and val_loader is not None:
            train_viz_batch = next(iter(train_loader))
            val_viz_batch = next(iter(val_loader))
            fixed_batch = build_mixed_visual_batch(
                train_batch=train_viz_batch,
                val_batch=val_viz_batch,
                total_batch_size=int(args.batch),
            )
        else:
            fixed_batch = next(iter(train_loader))
    except Exception as e:
        print(
            "[train] failed to fetch fixed sample batch from selected DataLoader; "
            f"fallback to single-process loader. reason={type(e).__name__}: {e}"
        )
        fallback_dataset = val_subset if (args.trainer == "diffusion" and val_subset is not None) else train_subset
        sample_loader = DataLoader(
            fallback_dataset,
            batch_size=min(int(args.batch), 8),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        fixed_batch = next(iter(sample_loader))

    trainer.sample_every_steps = args.sample_every_steps
    trainer.sample_batch = fixed_batch
    trainer.sample_dir = Path(args.save_dir) / "samples"

    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    (save_dir_path / "train_run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer.fit(
        train_loader,
        epochs=args.epochs,
        save_every=(args.save_every_epochs if args.save_every_epochs > 0 else None),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
