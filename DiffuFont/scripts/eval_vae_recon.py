#!/usr/bin/env python3
"""Evaluate VAE reconstruction quality by sampling a few chars per font."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import FontImageDataset
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform


def resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    probe_count = min(2, torch.cuda.device_count())
    best_idx = 0
    best_free = -1
    for idx in range(probe_count):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_free = int(free_bytes)
            best_idx = idx
    return torch.device(f"cuda:{best_idx}")


def load_model(checkpoint_path: Path, device: torch.device) -> SourcePartRefDiT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("stage") != "vae":
        raise RuntimeError(f"Checkpoint is not a VAE checkpoint: {checkpoint_path}")
    if "model_config" not in checkpoint or "vae_state" not in checkpoint:
        raise RuntimeError(f"Malformed VAE checkpoint: {checkpoint_path}")
    model = SourcePartRefDiT(**checkpoint["model_config"])
    model.vae.load_state_dict(checkpoint["vae_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def _tensor01(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().clamp(-1.0, 1.0).add(1.0).mul_(0.5)


def _tensor_to_u8_image(x: torch.Tensor, size: int) -> Image.Image:
    arr = (_tensor01(x).squeeze(0).mul(255.0).round().byte().cpu().numpy())
    image = Image.fromarray(arr, mode="L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image


def _choose_font(root: Path, size: int) -> ImageFont.ImageFont:
    candidates = [
        root / "fonts" / "SourceHanSansCN-Regular#1.otf",
        root / "fonts" / "SourceHanSerifCN-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_pages(
    root: Path,
    font_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    chars_per_font: int,
    cell_size: int = 96,
    rows_per_page: int = 8,
) -> list[str]:
    label_w = 260
    pair_gap = 8
    group_gap = 24
    info_h = 28
    row_gap = 18
    margin = 12
    page_paths: list[str] = []
    header_font = _choose_font(root, 16)
    body_font = _choose_font(root, 15)
    small_font = _choose_font(root, 13)
    row_h = info_h + cell_size + row_gap
    page_w = label_w + chars_per_font * (cell_size * 2 + pair_gap) + (chars_per_font - 1) * group_gap + margin * 2

    for page_idx in range(0, len(font_rows), rows_per_page):
        chunk = font_rows[page_idx : page_idx + rows_per_page]
        page_h = margin * 2 + len(chunk) * row_h
        canvas = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for row_idx, row in enumerate(chunk):
            y0 = margin + row_idx * row_h
            draw.text((12, y0 + 2), row["font"], fill=(0, 0, 0), font=body_font)
            draw.text(
                (12, y0 + 18),
                f"mae={row['mean_mae']:.4f} mse={row['mean_mse']:.5f} psnr={row['mean_psnr']:.2f}",
                fill=(80, 80, 80),
                font=small_font,
            )
            y_img = y0 + info_h
            for char_idx, sample in enumerate(row["samples"]):
                x_pair = label_w + char_idx * (cell_size * 2 + pair_gap + group_gap)
                draw.text((x_pair, y0 + 2), f"{sample['char']} gt", fill=(0, 0, 0), font=header_font)
                draw.text((x_pair + cell_size + pair_gap, y0 + 2), "recon", fill=(0, 0, 0), font=header_font)
                canvas.paste(sample["target_img"].convert("RGB"), (x_pair, y_img))
                canvas.paste(sample["recon_img"].convert("RGB"), (x_pair + cell_size + pair_gap, y_img))

        page_path = out_dir / f"page_{(page_idx // rows_per_page) + 1:02d}.png"
        canvas.save(page_path)
        page_paths.append(str(page_path))
    return page_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--font-split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument(
        "--font-name",
        type=str,
        action="append",
        default=None,
        help="Restrict evaluation to explicit font names. Repeat the flag for multiple fonts.",
    )
    parser.add_argument(
        "--font-list-file",
        type=Path,
        default=None,
        help="Optional newline-delimited font list file used to restrict evaluation to a subset of fonts.",
    )
    parser.add_argument("--chars-per-font", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)

    if args.out_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = args.data_root / "analysis" / f"vae_eval_{args.font_split}_fonts_3chars_{stamp}"
    else:
        out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    glyph_transform = build_base_glyph_transform(image_size=int(model.image_size))
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=1,
        random_seed=int(args.seed),
        font_split=str(args.font_split),
        font_split_seed=int(args.seed),
        font_train_ratio=0.95,
        transform=glyph_transform,
        style_transform=glyph_transform,
    )
    dataset._ensure_txns()

    font_names = sorted(dataset.font_names)
    selected_font_order: list[str] = []
    selected_font_names: set[str] = set()
    if args.font_name:
        for raw_name in args.font_name:
            font_name = str(raw_name).strip()
            if not font_name or font_name in selected_font_names:
                continue
            selected_font_order.append(font_name)
            selected_font_names.add(font_name)
    if args.font_list_file is not None:
        lines = args.font_list_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            font_name = line.strip()
            if not font_name or font_name in selected_font_names:
                continue
            selected_font_order.append(font_name)
            selected_font_names.add(font_name)
    if selected_font_names:
        available_font_names = set(font_names)
        missing_font_names = sorted(selected_font_names.difference(available_font_names))
        if selected_font_order:
            font_names = [name for name in selected_font_order if name in available_font_names]
        else:
            font_names = [name for name in font_names if name in selected_font_names]
        if missing_font_names:
            print(f"[vae_eval] skipped_missing_fonts={missing_font_names}")
        if not font_names:
            raise RuntimeError("No fonts matched the requested subset.")
    rng = random.Random(int(args.seed))
    eval_entries: list[dict[str, Any]] = []
    for font_name in font_names:
        valid_indices = [int(idx) for idx in dataset.valid_indices_by_font[font_name]]
        sample_count = min(int(args.chars_per_font), len(valid_indices))
        if sample_count <= 0:
            continue
        chosen = sorted(rng.sample(valid_indices, k=sample_count))
        for char_index in chosen:
            char = dataset.char_list[char_index]
            target = dataset._load_tensor(dataset._t_txn, f"{font_name}@{char}", style=False)
            eval_entries.append(
                {
                    "font": font_name,
                    "char": char,
                    "target": target,
                }
            )

    by_font: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": []})

    with torch.no_grad():
        for start in range(0, len(eval_entries), int(args.batch_size)):
            batch_entries = eval_entries[start : start + int(args.batch_size)]
            target_batch = torch.stack([entry["target"] for entry in batch_entries], dim=0).to(device)
            recon_batch, _, _, _ = model.vae_forward(target_batch, sample_posterior=False)
            recon_batch = recon_batch.cpu()
            target_batch = target_batch.cpu()

            for entry, target, recon in zip(batch_entries, target_batch, recon_batch):
                target01 = _tensor01(target)
                recon01 = _tensor01(recon)
                mae = float(F.l1_loss(recon01, target01).item())
                mse = float(F.mse_loss(recon01, target01).item())
                psnr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
                by_font[entry["font"]]["font"] = entry["font"]
                by_font[entry["font"]]["samples"].append(
                    {
                        "char": entry["char"],
                        "mae": mae,
                        "mse": mse,
                        "psnr": psnr,
                        "target_img": _tensor_to_u8_image(target, size=int(model.image_size)),
                        "recon_img": _tensor_to_u8_image(recon, size=int(model.image_size)),
                    }
                )

    font_rows: list[dict[str, Any]] = []
    all_mae: list[float] = []
    all_mse: list[float] = []
    all_psnr: list[float] = []
    for font_name in font_names:
        row = by_font.get(font_name)
        if not row or not row["samples"]:
            continue
        samples = row["samples"]
        mean_mae = float(sum(sample["mae"] for sample in samples) / len(samples))
        mean_mse = float(sum(sample["mse"] for sample in samples) / len(samples))
        mean_psnr = float(sum(sample["psnr"] for sample in samples) / len(samples))
        row["mean_mae"] = mean_mae
        row["mean_mse"] = mean_mse
        row["mean_psnr"] = mean_psnr
        font_rows.append(row)
        all_mae.extend(sample["mae"] for sample in samples)
        all_mse.extend(sample["mse"] for sample in samples)
        all_psnr.extend(sample["psnr"] for sample in samples)

    page_paths = build_pages(
        PROJECT_ROOT,
        font_rows,
        out_dir,
        chars_per_font=int(args.chars_per_font),
    )

    sortable_rows = sorted(font_rows, key=lambda item: item["mean_mae"])
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "font_split": str(args.font_split),
        "seed": int(args.seed),
        "chars_per_font": int(args.chars_per_font),
        "font_count": int(len(font_rows)),
        "sample_count": int(len(all_mae)),
        "mean_mae_all_fonts": float(sum(all_mae) / len(all_mae)),
        "mean_mse_all_fonts": float(sum(all_mse) / len(all_mse)),
        "mean_psnr_all_fonts": float(sum(all_psnr) / len(all_psnr)),
        "best10_by_mae": [
            {
                "font": row["font"],
                "mean_mae": row["mean_mae"],
                "mean_mse": row["mean_mse"],
                "mean_psnr": row["mean_psnr"],
                "chars": [sample["char"] for sample in row["samples"]],
            }
            for row in sortable_rows[:10]
        ],
        "worst10_by_mae": [
            {
                "font": row["font"],
                "mean_mae": row["mean_mae"],
                "mean_mse": row["mean_mse"],
                "mean_psnr": row["mean_psnr"],
                "chars": [sample["char"] for sample in row["samples"]],
            }
            for row in sortable_rows[-10:]
        ],
        "per_font": [
            {
                "font": row["font"],
                "mean_mae": row["mean_mae"],
                "mean_mse": row["mean_mse"],
                "mean_psnr": row["mean_psnr"],
                "samples": [
                    {
                        "char": sample["char"],
                        "mae": sample["mae"],
                        "mse": sample["mse"],
                        "psnr": sample["psnr"],
                    }
                    for sample in row["samples"]
                ],
            }
            for row in font_rows
        ],
        "pages": page_paths,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        f"checkpoint: {args.checkpoint.resolve()}",
        f"font_split: {args.font_split}",
        f"font_count: {len(font_rows)}",
        f"sample_count: {len(all_mae)}",
        f"mean_mae_all_fonts: {report['mean_mae_all_fonts']:.6f}",
        f"mean_mse_all_fonts: {report['mean_mse_all_fonts']:.6f}",
        f"mean_psnr_all_fonts: {report['mean_psnr_all_fonts']:.4f}",
        "",
        "best10_by_mae:",
    ]
    for row in report["best10_by_mae"]:
        summary_lines.append(
            f"  {row['font']}: mae={row['mean_mae']:.6f} mse={row['mean_mse']:.6f} "
            f"psnr={row['mean_psnr']:.4f} chars={','.join(row['chars'])}"
        )
    summary_lines.append("")
    summary_lines.append("worst10_by_mae:")
    for row in report["worst10_by_mae"]:
        summary_lines.append(
            f"  {row['font']}: mae={row['mean_mae']:.6f} mse={row['mean_mse']:.6f} "
            f"psnr={row['mean_psnr']:.4f} chars={','.join(row['chars'])}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[vae_eval] device={device}")
    print(f"[vae_eval] report={out_dir / 'report.json'}")
    print(f"[vae_eval] summary={out_dir / 'summary.txt'}")
    print(f"[vae_eval] pages={len(page_paths)}")


if __name__ == "__main__":
    main()
