#!/usr/bin/env python3
"""Training entry for source-aligned FontDiffuser RSI model + PartBank retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import FontImageDataset
from models.model import DiffusionTrainer
from models.source_part_ref_unet import SourcePartRefUNet


def collate_fn(samples, style_k: int) -> Dict[str, torch.Tensor]:
    contents, styles, targets = [], [], []
    char_indices = []
    parts_list, part_masks = [], []
    parts_list_b, part_masks_b = [], []

    for s in samples:
        contents.append(s["content"])
        targets.append(s["input"])
        char_indices.append(int(s.get("char_index", 0)))

        style_imgs = s["styles"][:style_k]
        if len(style_imgs) < style_k:
            style_imgs = (style_imgs * style_k)[:style_k]
        styles.append(torch.cat(style_imgs, dim=0))

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
        "char_index": torch.tensor(char_indices, dtype=torch.long),
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
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    req = torch.device(device_arg)
    if req.type == "cuda":
        if not torch.cuda.is_available():
            print("[train] CUDA unavailable, fallback to CPU")
            return torch.device("cpu")
        if req.index is not None and req.index >= torch.cuda.device_count():
            raise ValueError(f"Invalid --device {device_arg}, cuda count={torch.cuda.device_count()}")
    return req


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cons", type=float, default=0.05)
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
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--auto-select-font", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-target-in-style", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--component-guided-style", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--style-overlap-topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decomposition-json", type=str, default="CharacterData/decomposition.json")

    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default="full",
        choices=["baseline", "token_only", "rsi_only", "full"],
        help="baseline=no token/no RSI; token_only=token on RSI off; rsi_only=RSI on token off; full=both on.",
    )

    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument(
        "--part-retrieval-mode",
        type=str,
        default="font_softmax_top1",
        choices=["none", "font_softmax_top1"],
    )
    parser.add_argument("--part-retrieval-ep-ckpt", type=str, default=None)
    parser.add_argument("--part-set-size", type=int, default=10)
    parser.add_argument("--part-set-min-size", type=int, default=2)
    parser.add_argument("--part-set-sampling", type=str, default="random", choices=["deterministic", "random"])
    parser.add_argument("--part-target-char-priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part-image-size", type=int, default=64)

    parser.add_argument("--sample-every-steps", type=int, default=300)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--detailed-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-report-samples", type=int, default=0)
    parser.add_argument("--overlap-report-seed", type=int, default=42)
    parser.add_argument("--overlap-report-json", type=str, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.part_set_min_size > args.part_set_size:
        raise ValueError(
            f"--part-set-min-size ({args.part_set_min_size}) must be <= --part-set-size ({args.part_set_size})"
        )

    profile = str(args.conditioning_profile).strip().lower()
    enable_token = profile in {"token_only", "full"}
    enable_rsi = profile in {"rsi_only", "full"}
    use_part_style = enable_token or enable_rsi
    if use_part_style and args.part_retrieval_mode == "none":
        args.part_retrieval_mode = "font_softmax_top1"
    if not use_part_style:
        args.part_retrieval_mode = "none"
    if not enable_token:
        args.lambda_cons = 0.0

    if use_part_style and args.part_retrieval_mode == "font_softmax_top1":
        if not args.part_retrieval_ep_ckpt:
            raise ValueError("--part-retrieval-ep-ckpt is required for part_retrieval_mode=font_softmax_top1")
        ep_ckpt = Path(args.part_retrieval_ep_ckpt)
        if not ep_ckpt.is_absolute():
            ep_ckpt = (args.data_root / ep_ckpt).resolve()
        if not ep_ckpt.exists():
            raise FileNotFoundError(f"E_p checkpoint not found: {ep_ckpt}")

    device = resolve_device(args.device)
    print(f"[train] device={device} precision={args.precision}")

    run_cfg: Dict[str, Any] = {k: _to_jsonable(v) for k, v in vars(args).items()}
    run_cfg.update(
        {
            "resolved_device": str(device),
            "enable_token_condition": bool(enable_token),
            "enable_rsi_condition": bool(enable_rsi),
            "use_part_style": bool(use_part_style),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    print("[train-config] " + json.dumps(run_cfg, ensure_ascii=False, sort_keys=True), flush=True)

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
        use_part_bank=bool(use_part_style),
        part_bank_manifest=args.part_bank_manifest,
        part_retrieval_mode=args.part_retrieval_mode,
        part_retrieval_ep_ckpt=args.part_retrieval_ep_ckpt,
        part_set_size=args.part_set_size,
        part_set_min_size=args.part_set_min_size,
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
        raise ValueError(f"Invalid steps: steps_per_epoch={steps_per_epoch}, epochs={args.epochs}")

    lr_tmax_steps = args.lr_tmax_steps if args.lr_tmax_steps > 0 else total_train_steps
    print(
        "[train] "
        f"conditioning_profile={args.conditioning_profile} "
        f"part_retrieval_mode={args.part_retrieval_mode} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_train_steps} lr_tmax_steps={lr_tmax_steps}"
    )

    model = SourcePartRefUNet(
        in_channels=3,
        image_size=96,
        style_k=args.style_k,
        content_start_channel=64,
        style_start_channel=64,
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=3,
        channel_attn=True,
        conditioning_profile=args.conditioning_profile,
    )

    trainer = DiffusionTrainer(
        model,
        device,
        lr=args.lr,
        lambda_cons=args.lambda_cons,
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
