#!/usr/bin/env python3
"""Training entry for DiffuFont (source-aligned UNet + PartBank + InfoNCE)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch
from torch.utils.data import DataLoader
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


def collate_fn(samples) -> Dict[str, torch.Tensor]:
    contents, targets = [], []
    parts_list, part_masks = [], []
    parts_list_b, part_masks_b = [], []

    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])

        if "parts" in s:
            parts_list.append(s["parts"])
            part_masks.append(s.get("part_mask", torch.ones((s["parts"].size(0),), dtype=torch.float32)))
        if "parts_b" in s:
            parts_list_b.append(s["parts_b"])
            part_masks_b.append(s.get("part_mask_b", torch.ones((s["parts_b"].size(0),), dtype=torch.float32)))

    # Build per-batch integer font IDs for InfoNCE.
    font_names = [s["font"] for s in samples]
    unique_fonts = sorted(set(font_names))
    font_to_id = {f: i for i, f in enumerate(unique_fonts)}
    font_ids = torch.tensor([font_to_id[f] for f in font_names], dtype=torch.long)

    batch: Dict[str, torch.Tensor] = {
        "content": torch.stack(contents),
        "target": torch.stack(targets),
        "font_ids": font_ids,
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
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Total training steps for OneCycleLR. <=0 means epochs*steps_per_epoch.",
    )

    parser.add_argument("--max-fonts", type=int, default=0)

    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default="parts_vector_only",
        choices=["baseline", "parts_vector_only"],
        help="baseline=no parts_vector; parts_vector_only=parts_vector conditioning on.",
    )
    parser.add_argument(
        "--attn-scales",
        type=str,
        default="16,32",
        help="Comma-separated resolutions where style attention is enabled, e.g. '32,64'. Empty means all scales.",
    )

    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument("--part-bank-lmdb", type=str, default="DataPreparation/LMDB/PartBank.lmdb")
    parser.add_argument("--part-set-max", type=int, default=8)
    parser.add_argument("--part-set-min", type=int, default=1)
    parser.add_argument("--part-image-size", type=int, default=64)
    parser.add_argument("--part-image-cache-size", type=int, default=4096)
    parser.add_argument("--lmdb-decode-cache-size", type=int, default=1024)

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
    parser.add_argument("--style-token-count", type=int, default=8)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--pretrained-part-encoder", type=str, default=None,
        help="Path to pretrained part encoder checkpoint from pretrain_part_style_encoder.py.",
    )
    args = parser.parse_args()

    if args.part_set_min > args.part_set_max:
        raise ValueError(
            f"--part-set-min ({args.part_set_min}) must be <= --part-set-max ({args.part_set_max})"
        )
    profile = str(args.conditioning_profile).strip().lower()
    enable_parts_vector = profile == "parts_vector_only"
    use_part_bank = bool(enable_parts_vector)
    attn_scales = _parse_int_csv(args.attn_scales)

    device = resolve_device(args.device)
    print(f"[train] device={device} precision={args.precision}")

    run_cfg: Dict[str, Any] = {k: _to_jsonable(v) for k, v in vars(args).items()}
    run_cfg.update(
        {
            "enable_parts_vector_condition": bool(enable_parts_vector),
            "use_part_bank": bool(use_part_bank),
        }
    )
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=args.max_fonts,
        use_part_bank=bool(use_part_bank),
        part_bank_manifest=args.part_bank_manifest,
        part_bank_lmdb=args.part_bank_lmdb,
        part_set_max=args.part_set_max,
        part_set_min=args.part_set_min,
        part_image_size=args.part_image_size,
        part_image_cache_size=args.part_image_cache_size,
        lmdb_decode_cache_size=args.lmdb_decode_cache_size,
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

    total_steps = args.total_steps if args.total_steps > 0 else total_train_steps
    print(
        "[train] "
        f"conditioning_profile={args.conditioning_profile} "
        f"attn_scales={attn_scales} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}"
    )

    model = SourcePartRefUNet(
        in_channels=1,
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

    _try_enable_xformers(model)

    if args.pretrained_part_encoder:
        ckpt = torch.load(args.pretrained_part_encoder, map_location="cpu", weights_only=True)
        sd = ckpt.get("part_encoder", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded_keys = [k for k in sd if k not in unexpected]
        print(f"[train] Loaded pretrained part encoder from {args.pretrained_part_encoder}")
        print(f"  loaded={len(loaded_keys)} missing={len(missing)} unexpected={len(unexpected)}")

    trainer_cls = DiffusionTrainer if args.trainer == "diffusion" else FlowMatchingTrainer
    trainer_kwargs: Dict[str, Any] = {
        "lr": args.lr,
        "lambda_nce": args.lambda_nce,
        "cfg_drop_prob": args.cfg_drop_prob,
        "T": args.diffusion_steps,
        "total_steps": total_steps,
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
