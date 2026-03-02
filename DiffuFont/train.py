#!/usr/bin/env python3
"""Training entry for DiffuFont (source-aligned UNet + PartBank + InfoNCE)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from dataset import FontImageDataset
from models.model import DiffusionTrainer, FlowMatchingTrainer
from models.source_part_ref_unet import SourcePartRefUNet


def _try_enable_xformers(model: SourcePartRefUNet) -> bool:
    """Try to enable xformers memory-efficient attention on the UNet."""
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        print("[train] xformers memory-efficient attention enabled")
        return True
    except Exception as e:
        print(f"[train] xformers not available, using default attention: {e}")
        return False


def _try_enable_gradient_checkpointing(model: SourcePartRefUNet) -> bool:
    """Try to enable gradient checkpointing on the UNet to save activation memory."""
    try:
        model.unet.enable_gradient_checkpointing()
        print("[train] gradient checkpointing enabled")
        return True
    except Exception as e:
        print(f"[train] gradient checkpointing not available: {e}")
        return False


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    def _pad_part_tensors(
        tensors: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_p = max(int(t.size(0)) for t in tensors)
        bsz = len(tensors)
        c = int(tensors[0].size(1))
        h = int(tensors[0].size(2))
        w = int(tensors[0].size(3))
        parts_out = torch.zeros((bsz, max_p, c, h, w), dtype=tensors[0].dtype)
        mask_out = torch.zeros((bsz, max_p), dtype=torch.float32)
        for i, (t, m) in enumerate(zip(tensors, masks)):
            p = int(t.size(0))
            parts_out[i, :p] = t
            mask_out[i, :p] = m.to(dtype=torch.float32)
        return parts_out, mask_out

    contents, targets = [], []
    style_imgs = []
    has_style = False
    parts_list, part_masks = [], []
    parts_list_b, part_masks_b = [], []
    has_parts_flags = []

    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])
        has_parts_flags.append(float(s.get("has_parts", 1.0 if "parts" in s else 0.0)))
        if "style_img" in s:
            has_style = True
            style_imgs.append(s["style_img"])

        if "parts" in s:
            parts_list.append(s["parts"])
            part_masks.append(s.get("part_mask", torch.ones((s["parts"].size(0),), dtype=torch.float32)))
        if "parts_b" in s:
            parts_list_b.append(s["parts_b"])
            part_masks_b.append(s.get("part_mask_b", torch.ones((s["parts_b"].size(0),), dtype=torch.float32)))

    n_samples = len(samples)

    # Build per-batch integer font IDs for InfoNCE.
    font_names = [s["font"] for s in samples]
    unique_fonts = sorted(set(font_names))
    font_to_id = {f: i for i, f in enumerate(unique_fonts)}
    font_ids = torch.tensor([font_to_id[f] for f in font_names], dtype=torch.long)

    batch: Dict[str, torch.Tensor] = {
        "content": torch.stack(contents),
        "target": torch.stack(targets),
        "font_ids": font_ids,
        "has_parts": torch.tensor(has_parts_flags, dtype=torch.float32),
    }
    if has_style and len(style_imgs) != len(samples):
        raise ValueError(
            f"Inconsistent style_img presence in batch: got {len(style_imgs)}/{len(samples)} samples."
        )
    if has_style and len(style_imgs) == len(samples):
        batch["style_img"] = torch.stack(style_imgs)

    if parts_list:
        batch["parts"], batch["part_mask"] = _pad_part_tensors(parts_list, part_masks)
    if parts_list_b:
        batch["parts_b"], batch["part_mask_b"] = _pad_part_tensors(parts_list_b, part_masks_b)
    return batch


def normalize_conditioning_mode(raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    alias = {
        "parts_vector_only": "part_only",
    }
    mode = alias.get(mode, mode)
    valid = {"baseline", "part_only", "style_only", "part_style"}
    if mode not in valid:
        raise ValueError(f"conditioning mode must be one of {sorted(valid)}, got '{raw_mode}'")
    return mode


def mode_uses_parts(mode: str) -> bool:
    return normalize_conditioning_mode(mode) in {"part_only", "part_style"}


def mode_uses_style(mode: str) -> bool:
    return normalize_conditioning_mode(mode) in {"style_only"}


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


def _parse_int_csv(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    vals = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        return None
    return tuple(vals)


def set_global_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _seed_worker(worker_id: int) -> None:
    # Derive per-worker seed from torch initial seed for reproducibility.
    ws = torch.initial_seed() % 2**32
    random.seed(ws)
    np.random.seed(ws)


def split_indices_by_font(
    dataset: FontImageDataset,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    """Split sample indices by font so train/val fonts are disjoint."""
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
        # Safety fallback: keep at least one sample in train.
        train_indices = [val_indices.pop()]
    if not val_indices:
        # Safety fallback: keep at least one sample in val.
        val_indices = [train_indices.pop()]

    return (
        train_indices,
        val_indices,
        {"total_fonts": len(fonts), "train_fonts": len(fonts) - val_n, "val_fonts": val_n},
    )


def build_mixed_visual_batch(
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
    total_batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Build a visualization batch by concatenating half train + half val."""
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

        if k in {"parts", "parts_b"} and t.dim() == 5 and v.dim() == 5:
            max_p = max(int(t.size(1)), int(v.size(1)))
            if int(t.size(1)) < max_p:
                pad = torch.zeros(
                    (int(t.size(0)), max_p - int(t.size(1)), int(t.size(2)), int(t.size(3)), int(t.size(4))),
                    dtype=t.dtype,
                    device=t.device,
                )
                t = torch.cat([t, pad], dim=1)
            if int(v.size(1)) < max_p:
                pad = torch.zeros(
                    (int(v.size(0)), max_p - int(v.size(1)), int(v.size(2)), int(v.size(3)), int(v.size(4))),
                    dtype=v.dtype,
                    device=v.device,
                )
                v = torch.cat([v, pad], dim=1)
        elif k in {"part_mask", "part_mask_b"} and t.dim() == 2 and v.dim() == 2:
            max_p = max(int(t.size(1)), int(v.size(1)))
            if int(t.size(1)) < max_p:
                pad = torch.zeros((int(t.size(0)), max_p - int(t.size(1))), dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=1)
            if int(v.size(1)) < max_p:
                pad = torch.zeros((int(v.size(0)), max_p - int(v.size(1))), dtype=v.dtype, device=v.device)
                v = torch.cat([v, pad], dim=1)

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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--trainer", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-nce", type=float, default=0.05)
    parser.add_argument(
        "--nce-warmup-steps", type=int, default=5000,
        help="Linearly ramp lambda_nce from 0 to its target over this many steps (0 = no warmup).",
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
    parser.add_argument("--stage", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument(
        "--teacher-line",
        type=str,
        default="part_style",
        choices=["baseline", "part_only", "style_only", "part_style"],
        help="Stage-A line for teacher training.",
    )

    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default=None,
        choices=["baseline", "parts_vector_only", "part_only", "style_only", "part_style"],
        help="Legacy alias; if set it overrides --teacher-line in teacher stage.",
    )
    parser.add_argument(
        "--attn-scales",
        type=str,
        default="16,32",
        help="Comma-separated resolutions where style attention is enabled, e.g. '32,64'. Empty means all scales.",
    )

    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument("--part-bank-lmdb", type=str, default="DataPreparation/LMDB/PartBank.lmdb")
    parser.add_argument("--part-set-min", type=int, default=None, help="Deprecated, ignored.")
    parser.add_argument("--part-set-max", type=int, default=0, help="Deprecated, ignored.")
    parser.add_argument("--part-image-size", type=int, default=40)
    parser.add_argument("--part-image-cache-size", type=int, default=4096)
    parser.add_argument("--lmdb-decode-cache-size", type=int, default=1024)

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
    parser.add_argument(
        "--part-encode-chunk-size",
        type=int,
        default=128,
        help="Chunk size for part encoder forward over B*P samples. 0 disables chunking.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps. Effective batch = batch * grad_accum.",
    )
    parser.add_argument(
        "--pretrained-part-encoder", type=str, default=None,
        help="Path to pretrained part encoder checkpoint from pretrain_part_style_encoder.py.",
    )
    parser.add_argument("--part-drop-prob", type=float, default=0.0)
    parser.add_argument("--lambda-kd", type=float, default=0.0)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument(
        "--teacher-distill-mode",
        type=str,
        default="part_style",
        choices=["baseline", "part_only", "style_only", "part_style"],
        help="Teacher conditioning mode used to generate KD targets in student stage.",
    )
    args = parser.parse_args()

    stage = str(args.stage).strip().lower()
    if stage not in {"teacher", "student"}:
        raise ValueError(f"Unsupported --stage: {stage}")

    teacher_mode = normalize_conditioning_mode(
        args.conditioning_profile if args.conditioning_profile is not None else args.teacher_line
    )
    student_mode = "style_only"
    teacher_distill_mode = normalize_conditioning_mode(args.teacher_distill_mode)

    if stage == "teacher":
        active_mode = teacher_mode
        use_part_bank = mode_uses_parts(active_mode)
        use_style_image = mode_uses_style(active_mode)
    else:
        active_mode = student_mode
        use_part_bank = mode_uses_parts(teacher_distill_mode)
        use_style_image = True
        if not args.teacher_ckpt:
            raise ValueError("--teacher-ckpt is required when --stage student")

    attn_scales = _parse_int_csv(args.attn_scales)

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    print(f"[train] device={device} precision={args.precision} seed={int(args.seed)}")

    run_cfg: Dict[str, Any] = {k: _to_jsonable(v) for k, v in vars(args).items()}
    run_cfg.update(
        {
            "stage": stage,
            "active_conditioning_mode": active_mode,
            "teacher_mode": teacher_mode,
            "student_mode": student_mode,
            "teacher_distill_mode": teacher_distill_mode,
            "use_part_bank": bool(use_part_bank),
            "use_style_image": bool(use_style_image),
        }
    )
    transform = T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=args.max_fonts,
        use_style_image=bool(use_style_image),
        enforce_part_bank_font_match=True,
        use_part_bank=bool(use_part_bank),
        part_bank_manifest=args.part_bank_manifest,
        part_bank_lmdb=args.part_bank_lmdb,
        part_set_min=args.part_set_min,
        part_set_max=args.part_set_max,
        part_image_size=args.part_image_size,
        part_image_cache_size=args.part_image_cache_size,
        lmdb_decode_cache_size=args.lmdb_decode_cache_size,
        random_seed=int(args.seed),
        transform=transform,
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

    loader_kwargs: Dict[str, Any] = {
        "dataset": train_subset,
        "batch_size": args.batch,
        "shuffle": True,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
        "collate_fn": collate_fn,
        "generator": loader_gen,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
    if int(args.num_workers) > 0 and int(args.prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(**loader_kwargs)
    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=min(int(args.batch), len(val_subset)),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    steps_per_epoch = len(train_loader)
    total_train_steps = steps_per_epoch * args.epochs
    if total_train_steps <= 0:
        raise ValueError(f"Invalid steps: steps_per_epoch={steps_per_epoch}, epochs={args.epochs}")

    # OneCycleLR counts optimizer steps, not mini-batch steps.
    # With gradient accumulation, optimizer steps = batches / grad_accum.
    effective_train_steps = total_train_steps // max(1, int(args.grad_accum))
    total_steps = args.total_steps if args.total_steps > 0 else effective_train_steps
    print(
        "[train] "
        f"stage={stage} mode={active_mode} distill_teacher_mode={teacher_distill_mode} "
        f"attn_scales={attn_scales} "
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
        attn_scales=attn_scales,
        style_token_dim=int(args.style_token_dim),
        part_encode_chunk_size=int(args.part_encode_chunk_size),
    )

    _try_enable_xformers(model)
    _try_enable_gradient_checkpointing(model)

    if args.pretrained_part_encoder:
        ckpt = torch.load(args.pretrained_part_encoder, map_location="cpu", weights_only=True)
        sd = ckpt.get("part_encoder", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded_keys = [k for k in sd if k not in unexpected]
        print(f"[train] Loaded pretrained part encoder from {args.pretrained_part_encoder}")
        print(f"  loaded={len(loaded_keys)} missing={len(missing)} unexpected={len(unexpected)}")

    trainer_cls = DiffusionTrainer if args.trainer == "diffusion" else FlowMatchingTrainer
    teacher_for_distill = None
    if stage == "student":
        teacher_model = SourcePartRefUNet(
            in_channels=1,
            image_size=args.image_size,
            content_start_channel=64,
            style_start_channel=int(args.style_start_channel),
            unet_channels=(64, 128, 256, 512),
            content_encoder_downsample_size=4,
            channel_attn=True,
            conditioning_profile=teacher_distill_mode,
            attn_scales=attn_scales,
            style_token_dim=int(args.style_token_dim),
            part_encode_chunk_size=int(args.part_encode_chunk_size),
        )
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                teacher_state = ckpt["model_state"]
            elif "state_dict" in ckpt:
                teacher_state = ckpt["state_dict"]
            else:
                teacher_state = ckpt
        else:
            teacher_state = ckpt
        teacher_model.load_state_dict(teacher_state, strict=True)
        print(f"[train] loaded teacher ckpt={args.teacher_ckpt} (strict=True)")
        teacher_for_distill = teacher_model

    eff_lambda_nce = float(args.lambda_nce if stage == "teacher" else 0.0)
    trainer_kwargs: Dict[str, Any] = {
        "lr": args.lr,
        "lambda_nce": eff_lambda_nce,
        "nce_warmup_steps": args.nce_warmup_steps,
        "T": args.diffusion_steps,
        "total_steps": total_steps,
        "precision": args.precision,
        "save_every_steps": (args.save_every_steps if args.save_every_steps > 0 else None),
        "log_every_steps": (args.log_every_steps if args.log_every_steps > 0 else None),
        "detailed_log": args.detailed_log,
        "grad_accum_steps": max(1, int(args.grad_accum)),
        "conditioning_mode": active_mode,
        "part_drop_prob": float(args.part_drop_prob),
        "lambda_kd": float(args.lambda_kd if stage == "student" else 0.0),
        "teacher_model": teacher_for_distill,
        "teacher_conditioning_mode": teacher_distill_mode,
    }
    if args.trainer == "flow_matching":
        trainer_kwargs["lambda_fm"] = args.lambda_fm
    else:
        trainer_kwargs["lambda_mse"] = args.lambda_diff

    trainer = trainer_cls(
        model,
        device,
        **trainer_kwargs,
    )
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
            num_workers=0,
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
