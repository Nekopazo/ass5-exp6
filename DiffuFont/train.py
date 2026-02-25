#!/usr/bin/env python3
"""Training entry for source-aligned FontDiffuser RSI model + PartBank retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

# Work around hosts with broken NVML by avoiding the native caching allocator path.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import FontImageDataset
from models.model import DiffusionTrainer, FlowMatchingTrainer
from models.source_part_ref_unet import SourcePartRefUNet


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    contents, styles, targets = [], [], []
    parts_list, part_masks = [], []
    parts_list_b, part_masks_b = [], []

    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])

        style_imgs = s["styles"]
        if not style_imgs:
            raise ValueError("Dataset returned empty style image list for one sample.")
        # Single-reference style path: keep exactly one style image per sample.
        styles.append(style_imgs[0])

        if "parts" in s:
            parts_list.append(s["parts"])
            part_masks.append(s.get("part_mask", torch.ones((s["parts"].size(0),), dtype=torch.float32)))
        if "parts_b" in s:
            parts_list_b.append(s["parts_b"])
            part_masks_b.append(s.get("part_mask_b", torch.ones((s["parts_b"].size(0),), dtype=torch.float32)))

    batch: Dict[str, torch.Tensor] = {
        "content": torch.stack(contents),
        "style": torch.stack(styles),
        "target": torch.stack(targets),
    }

    def _pad_parts(plist, mlist):
        max_parts = max(int(x.size(0)) for x in plist)
        c = int(plist[0].size(1))
        h = int(plist[0].size(2))
        w = int(plist[0].size(3))
        parts = torch.zeros((len(plist), max_parts, c, h, w), dtype=plist[0].dtype)
        mask = torch.zeros((len(plist), max_parts), dtype=torch.float32)

        for i, p in enumerate(plist):
            n = int(p.size(0))
            parts[i, :n] = p
            m = mlist[i]
            if m.dim() == 1 and m.size(0) == n:
                mask[i, :n] = m.float()
            else:
                mask[i, :n] = 1.0
        return parts, mask

    if parts_list:
        parts, mask = _pad_parts(parts_list, part_masks)
        batch["parts"] = parts
        batch["part_mask"] = mask
    if parts_list_b:
        parts_b, mask_b = _pad_parts(parts_list_b, part_masks_b)
        batch["parts_b"] = parts_b
        batch["part_mask_b"] = mask_b
    return batch


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--trainer", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    parser.add_argument("--lambda-fm", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-off", type=float, default=0.5)
    parser.add_argument("--lambda-style", type=float, default=0.0)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-tmax-steps",
        type=int,
        default=0,
        help="CosineAnnealingLR T_max in optimizer steps. <=0 means epochs*steps_per_epoch.",
    )

    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-name", type=str, default=None)
    parser.add_argument("--font-mode", type=str, default="random", choices=["fixed", "random"])
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--auto-select-font", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-target-in-style", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default="parts_vector_only",
        choices=["baseline", "parts_vector_only", "rsi_only", "full"],
        help="baseline=no parts_vector/no RSI; parts_vector_only=parts_vector on RSI off; rsi_only=RSI on parts_vector off; full=both on.",
    )
    parser.add_argument(
        "--attn-scales",
        type=str,
        default="16,32",
        help="Comma-separated resolutions where style attention is enabled, e.g. '32,64'. Empty means all scales.",
    )

    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument("--part-bank-lmdb", type=str, default="DataPreparation/LMDB/PartBank.lmdb")
    parser.add_argument("--part-retrieval-ep-ckpt", type=str, default=None)
    parser.add_argument("--part-retrieval-device", type=str, default="",
                        help="Device for online part-retrieval CNN inference. Empty means follow --device.")
    parser.add_argument("--part-set-size", type=int, default=32)
    parser.add_argument("--part-set-min-size", type=int, default=32)
    parser.add_argument("--part-set-sampling", type=str, default="random", choices=["deterministic", "random"])
    parser.add_argument("--part-target-char-priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part-image-size", type=int, default=64)
    parser.add_argument("--part-image-cache-size", type=int, default=4096)
    parser.add_argument("--lmdb-decode-cache-size", type=int, default=1024)
    parser.add_argument("--use-style-plan-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--style-prefetch-limit", type=int, default=1024)

    parser.add_argument("--sample-every-steps", type=int, default=300)
    parser.add_argument("--sample-solver", type=str, default="dpm", choices=["dpm", "ddim"])
    parser.add_argument("--sample-guidance-scale", type=float, default=7.5)
    parser.add_argument("--sample-use-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-inference-steps", type=int, default=20)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--detailed-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--split-save-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frozen-retrieval-copy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-vector-pretrain-ckpt", type=str, default=None)
    parser.add_argument("--style-token-count", type=int, default=8)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.part_set_min_size > args.part_set_size:
        raise ValueError(
            f"--part-set-min-size ({args.part_set_min_size}) must be <= --part-set-size ({args.part_set_size})"
        )
    profile = str(args.conditioning_profile).strip().lower()
    enable_parts_vector = profile in {"parts_vector_only", "full"}
    enable_rsi = profile in {"rsi_only", "full"}
    use_part_bank = bool(enable_parts_vector)
    if not enable_parts_vector and args.part_vector_pretrain_ckpt:
        raise ValueError("part_vector_pretrain_ckpt is only valid when part-vector conditioning is enabled.")
    if args.sample_solver == "ddim" and args.sample_use_cfg:
        raise ValueError("sample_use_cfg is unsupported with sample_solver=ddim. Use dpm for CFG sampling.")
    if args.trainer == "flow_matching" and args.sample_solver != "dpm":
        raise ValueError("flow_matching trainer supports only sample_solver=dpm.")

    resolved_ep_ckpt: Path | None = None
    resolved_part_vector_ckpt: Path | None = None
    attn_scales = _parse_int_csv(args.attn_scales)
    if use_part_bank:
        if not args.part_retrieval_ep_ckpt:
            raise ValueError("--part-retrieval-ep-ckpt is required when parts_vector conditioning is enabled.")
        resolved_ep_ckpt = Path(args.part_retrieval_ep_ckpt)
        if not resolved_ep_ckpt.is_absolute():
            resolved_ep_ckpt = (args.data_root / resolved_ep_ckpt).resolve()
        if not resolved_ep_ckpt.exists():
            raise FileNotFoundError(f"E_p checkpoint not found: {resolved_ep_ckpt}")
    if args.part_vector_pretrain_ckpt:
        resolved_part_vector_ckpt = Path(args.part_vector_pretrain_ckpt)
        if not resolved_part_vector_ckpt.is_absolute():
            resolved_part_vector_ckpt = (args.data_root / resolved_part_vector_ckpt).resolve()
        if not resolved_part_vector_ckpt.exists():
            raise FileNotFoundError(f"Part vector pretrain checkpoint not found: {resolved_part_vector_ckpt}")

    device = resolve_device(args.device)
    print(f"[train] device={device} precision={args.precision}")
    retrieval_device = args.part_retrieval_device if str(args.part_retrieval_device).strip() else str(device)
    if str(retrieval_device).startswith("cuda") and int(args.num_workers) > 0:
        raise ValueError(
            "part-retrieval on CUDA requires --num-workers 0, "
            "because Dataset-side CUDA in multiprocessing workers is unsupported."
        )

    run_cfg: Dict[str, Any] = {k: _to_jsonable(v) for k, v in vars(args).items()}
    run_cfg.update(
        {
            "enable_parts_vector_condition": bool(enable_parts_vector),
            "enable_rsi_condition": bool(enable_rsi),
            "use_part_bank": bool(use_part_bank),
        }
    )
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    dataset = FontImageDataset(
        project_root=args.data_root,
        font_index=args.font_index,
        font_name=args.font_name,
        font_mode=args.font_mode,
        max_fonts=args.max_fonts,
        auto_select_font=args.auto_select_font,
        num_style_refs=1,
        include_target_in_style=args.include_target_in_style,
        use_part_bank=bool(use_part_bank),
        part_bank_manifest=args.part_bank_manifest,
        part_bank_lmdb=args.part_bank_lmdb,
        part_retrieval_ep_ckpt=args.part_retrieval_ep_ckpt,
        part_retrieval_device=retrieval_device,
        part_set_size=args.part_set_size,
        part_set_min_size=args.part_set_min_size,
        part_set_sampling=args.part_set_sampling,
        part_target_char_priority=args.part_target_char_priority,
        part_image_size=args.part_image_size,
        part_image_cache_size=args.part_image_cache_size,
        lmdb_decode_cache_size=args.lmdb_decode_cache_size,
        use_style_plan_cache=args.use_style_plan_cache,
        style_prefetch_limit=args.style_prefetch_limit,
        transform=transform,
    )

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch,
        "shuffle": True,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if int(args.num_workers) > 0 and int(args.prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    steps_per_epoch = len(loader)
    total_train_steps = steps_per_epoch * args.epochs
    if total_train_steps <= 0:
        raise ValueError(f"Invalid steps: steps_per_epoch={steps_per_epoch}, epochs={args.epochs}")

    lr_tmax_steps = args.lr_tmax_steps if args.lr_tmax_steps > 0 else total_train_steps
    print(
        "[train] "
        f"conditioning_profile={args.conditioning_profile} "
        f"attn_scales={attn_scales} "
        "part_retrieval_policy=top1_gate_top3_fixed "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_train_steps} lr_tmax_steps={lr_tmax_steps}"
    )

    model = SourcePartRefUNet(
        in_channels=3,
        image_size=args.image_size,
        content_start_channel=64,
        style_start_channel=64,
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=4,
        channel_attn=True,
        conditioning_profile=args.conditioning_profile,
        attn_scales=attn_scales,
        style_token_count=int(args.style_token_count),
        style_token_dim=int(args.style_token_dim),
    )
    if resolved_part_vector_ckpt is not None:
        model.load_part_vector_pretrained(str(resolved_part_vector_ckpt))
        print(f"[train] loaded part vector pretrain from {resolved_part_vector_ckpt}")

    trainer_cls = DiffusionTrainer if args.trainer == "diffusion" else FlowMatchingTrainer
    trainer_kwargs: Dict[str, Any] = {
        "lr": args.lr,
        "lambda_off": args.lambda_off,
        "lambda_style": args.lambda_style,
        "cfg_drop_prob": args.cfg_drop_prob,
        "T": args.diffusion_steps,
        "lr_tmax": lr_tmax_steps,
        "precision": args.precision,
        "save_every_steps": (args.save_every_steps if args.save_every_steps > 0 else None),
        "log_every_steps": (args.log_every_steps if args.log_every_steps > 0 else None),
        "detailed_log": args.detailed_log,
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
    trainer.sample_guidance_scale = float(args.sample_guidance_scale)
    trainer.sample_use_cfg = bool(args.sample_use_cfg)
    trainer.sample_inference_steps = int(args.sample_inference_steps)
    trainer.save_split_components = bool(args.split_save_components)
    trainer.save_frozen_retrieval_copy = bool(args.save_frozen_retrieval_copy)
    trainer.frozen_retrieval_ckpt_path = str(resolved_ep_ckpt) if resolved_ep_ckpt is not None else None

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}")

    try:
        fixed_batch = next(iter(loader))
    except Exception as e:
        print(
            "[train] failed to fetch fixed sample batch from training DataLoader; "
            f"fallback to single-process loader. reason={type(e).__name__}: {e}"
        )
        sample_loader = DataLoader(
            dataset,
            batch_size=min(int(args.batch), 8),
            shuffle=True,
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
        loader,
        epochs=args.epochs,
        save_every=(args.save_every_epochs if args.save_every_epochs > 0 else None),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
