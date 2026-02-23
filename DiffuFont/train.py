#!/usr/bin/env python3
"""Training script for FontDiffusionUNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import FontImageDataset
from models.font_diffusion_unet import FontDiffusionUNet
from models.model import DiffusionTrainer


def parse_bool_list(s: str) -> List[bool]:
    vals: List[bool] = []
    for p in s.split(","):
        p = p.strip().lower()
        if p in {"1", "true", "t", "yes", "y", "on"}:
            vals.append(True)
        elif p in {"0", "false", "f", "no", "n", "off"}:
            vals.append(False)
        elif p == "":
            continue
        else:
            raise ValueError(f"invalid bool token in list: {p}")
    if not vals:
        raise ValueError("empty bool list")
    return vals


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("empty int list")
    return vals


def parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("empty float list")
    return vals


def collate_fn(samples, style_k: int):
    """Convert a list of dataset samples to batch tensors."""
    contents, styles, targets = [], [], []
    parts_list, part_masks = [], []
    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])  # training target glyph

        style_imgs = s["styles"][:style_k]
        if len(style_imgs) < style_k:
            # Repeat when references are insufficient.
            style_imgs = (style_imgs * style_k)[:style_k]
        styles.append(torch.cat(style_imgs, dim=0))

        if "parts" in s:
            parts_list.append(s["parts"])
            if "part_mask" in s:
                part_masks.append(s["part_mask"])
            else:
                part_masks.append(torch.ones((s["parts"].size(0),), dtype=torch.float32))

    batch = {
        "content": torch.stack(contents),
        "style": torch.stack(styles),
        "target": torch.stack(targets),
    }
    if parts_list:
        max_parts = max(int(x.size(0)) for x in parts_list)
        c = int(parts_list[0].size(1))
        h = int(parts_list[0].size(2))
        w = int(parts_list[0].size(3))
        parts = torch.zeros((len(parts_list), max_parts, c, h, w), dtype=parts_list[0].dtype)
        mask = torch.zeros((len(parts_list), max_parts), dtype=torch.float32)
        for i, p in enumerate(parts_list):
            n = int(p.size(0))
            parts[i, :n] = p
            if i < len(part_masks):
                m = part_masks[i]
                if m.dim() == 1 and m.size(0) == n:
                    mask[i, :n] = m.float()
                else:
                    mask[i, :n] = 1.0
            else:
                mask[i, :n] = 1.0
        batch["parts"] = parts
        batch["part_mask"] = mask
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-tmax-steps",
        type=int,
        default=0,
        help="CosineAnnealingLR T_max in optimizer steps. <=0 means epochs*steps_per_epoch.",
    )
    parser.add_argument("--style-k", type=int, default=3)

    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-name", type=str, default=None)
    parser.add_argument("--font-mode", type=str, default="random", choices=["fixed", "random"])
    parser.add_argument("--max-fonts", type=int, default=0, help="Used when --font-mode=random, 0 means all usable fonts")
    parser.add_argument("--auto-select-font", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-target-in-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--component-guided-style", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--style-overlap-topk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pick style chars by highest component overlap first (global style set).",
    )
    parser.add_argument("--decomposition-json", type=str, default="CharacterData/decomposition.json")

    parser.add_argument("--daca-layers", type=str, default="0,1,1,0")
    parser.add_argument("--fgsa-layers", type=str, default="1,1,1,0")
    parser.add_argument("--attnx-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attnx-positions", type=str, default="bottleneck_16,up0_16to32")

    parser.add_argument("--use-global-style", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-part-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part-patch-size", type=int, default=64)
    parser.add_argument("--part-patch-stride", type=int, default=32)
    parser.add_argument("--part-min-patches-per-style", type=int, default=1)
    parser.add_argument("--part-max-patches-per-style", type=int, default=8)
    parser.add_argument("--part-fuse-strength", type=float, default=1.0)
    parser.add_argument(
        "--part-fuse-scales",
        type=str,
        default=None,
        help="Comma-separated encoder scale indices for part fusion, e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--part-fuse-scale-gains",
        type=str,
        default=None,
        help="Comma-separated gains aligned with --part-fuse-scales, e.g. '0.2,1.0,1.0'.",
    )
    parser.add_argument("--part-style-pretrained", type=str, default=None)
    parser.add_argument("--freeze-part-style", action="store_true")
    parser.add_argument("--use-part-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument("--part-set-size", type=int, default=32)
    parser.add_argument(
        "--part-set-sampling",
        type=str,
        default="deterministic",
        choices=["deterministic", "random"],
    )
    parser.add_argument("--part-target-char-priority", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part-image-size", type=int, default=64)

    parser.add_argument("--sample-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--detailed-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-report-samples", type=int, default=0)
    parser.add_argument("--overlap-report-seed", type=int, default=42)
    parser.add_argument("--overlap-report-json", type=str, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        req = torch.device(args.device)
        if req.type == "cuda":
            if not torch.cuda.is_available():
                print("[train] CUDA unavailable, fallback to CPU")
                device = torch.device("cpu")
            elif req.index is not None and req.index >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid --device {args.device}, cuda device count={torch.cuda.device_count()}"
                )
            else:
                device = req
        else:
            device = req
    print(f"[train] device={device} precision={args.precision}")

    transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
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
        num_style_refs=args.style_k,
        include_target_in_style=args.include_target_in_style,
        component_guided_style=args.component_guided_style,
        style_overlap_topk=args.style_overlap_topk,
        decomposition_json=args.decomposition_json,
        use_part_bank=bool(args.use_part_style and args.use_part_bank),
        part_bank_manifest=args.part_bank_manifest,
        part_set_size=args.part_set_size,
        part_set_sampling=args.part_set_sampling,
        part_target_char_priority=args.part_target_char_priority,
        part_image_size=args.part_image_size,
        transform=transform,
    )

    if args.overlap_report_samples > 0:
        report = dataset.component_overlap_stats(
            num_samples=args.overlap_report_samples,
            random_seed=args.overlap_report_seed,
            top_char_k=20,
            top_pair_k=50,
        )
        print(
            "[component-overlap] "
            f"positive_rate={report.get('positive_rate', 0.0):.4f} "
            f"mean={report.get('mean_overlap', 0.0):.4f} "
            f"p90={report.get('p90_overlap', 0.0):.4f} "
            f"pairs={report.get('total_pairs', 0)}"
        )
        if args.overlap_report_json:
            out_path = Path(args.overlap_report_json)
            if not out_path.is_absolute():
                out_path = args.data_root / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[component-overlap] saved report json: {out_path}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, args.style_k),
    )
    steps_per_epoch = len(loader)
    total_train_steps = steps_per_epoch * args.epochs
    if total_train_steps <= 0:
        raise ValueError(
            f"Invalid training steps: steps_per_epoch={steps_per_epoch}, epochs={args.epochs}"
        )
    lr_tmax_steps = args.lr_tmax_steps if args.lr_tmax_steps > 0 else total_train_steps
    print(
        "[train] "
        f"diffusion_steps={args.diffusion_steps} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_train_steps} "
        f"lr_tmax_steps={lr_tmax_steps}"
    )

    part_fuse_scales = parse_int_list(args.part_fuse_scales) if args.part_fuse_scales else None
    part_fuse_scale_gains = parse_float_list(args.part_fuse_scale_gains) if args.part_fuse_scale_gains else None
    if part_fuse_scale_gains is not None and part_fuse_scales is None:
        raise ValueError("--part-fuse-scale-gains requires --part-fuse-scales")

    model = FontDiffusionUNet(
        in_channels=3,
        style_k=args.style_k,
        daca_layers=parse_bool_list(args.daca_layers),
        fgsa_layers=parse_bool_list(args.fgsa_layers),
        attnx_enabled=args.attnx_enabled,
        attnx_positions=parse_csv(args.attnx_positions),
        use_global_style=args.use_global_style,
        use_part_style=args.use_part_style,
        part_patch_size=args.part_patch_size,
        part_patch_stride=args.part_patch_stride,
        part_min_patches_per_style=args.part_min_patches_per_style,
        part_max_patches_per_style=args.part_max_patches_per_style,
        part_fuse_scales=part_fuse_scales,
        part_fuse_scale_gains=part_fuse_scale_gains,
        part_fuse_strength=args.part_fuse_strength,
    )
    print(
        "[train] style switches "
        f"use_global_style={args.use_global_style} use_part_style={args.use_part_style}"
    )

    if args.part_style_pretrained:
        if not args.use_part_style:
            raise ValueError("--part-style-pretrained requires --use-part-style")
        model.load_part_style_pretrained(args.part_style_pretrained, strict=False)
        print(f"Loaded part-style pretrained weights: {args.part_style_pretrained}")

    if args.freeze_part_style:
        if not args.use_part_style:
            raise ValueError("--freeze-part-style requires --use-part-style")
        for p in model.part_style_encoder.parameters():
            p.requires_grad = False
        print("Part-style encoder parameters frozen")

    trainer = DiffusionTrainer(
        model,
        device,
        lr=args.lr,
        T=args.diffusion_steps,
        lr_tmax=lr_tmax_steps,
        precision=args.precision,
        save_every_steps=(args.save_every_steps if args.save_every_steps > 0 else None),
        log_every_steps=(args.log_every_steps if args.log_every_steps > 0 else None),
        detailed_log=args.detailed_log,
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}")

    fixed_batch = next(iter(loader))
    out_dir = Path(args.save_dir) / "samples"

    trainer.sample_every_steps = args.sample_every_steps
    trainer.sample_batch = fixed_batch
    trainer.sample_dir = out_dir

    trainer.fit(
        loader,
        epochs=args.epochs,
        save_every=(args.save_every_epochs if args.save_every_epochs > 0 else None),
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
