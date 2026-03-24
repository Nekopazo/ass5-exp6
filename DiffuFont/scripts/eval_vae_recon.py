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

import cv2
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


def normalize_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = dict(checkpoint["model_config"])
    if "content_cross_attn_blocks" not in config and "content_cross_attn_layers" in config:
        config["content_cross_attn_blocks"] = int(config.pop("content_cross_attn_layers"))
    if "content_cross_attn_indices" not in config:
        depth = int(config.get("dit_depth", 16))
        block_count = int(config.pop("content_cross_attn_blocks", depth))
        block_count = max(0, min(depth, block_count))
        config["content_cross_attn_indices"] = list(range(block_count))
    if "style_token_cross_attn_indices" not in config:
        depth = int(config.get("dit_depth", 16))
        every_n = max(1, int(config.pop("style_cross_attn_every_n_layers", 1)))
        config["style_token_cross_attn_indices"] = list(range(0, depth, every_n))
    config.pop("content_cross_attn_schedule", None)
    config.pop("content_cross_attn_prefix_blocks", None)
    if "style_tokens_per_ref" not in config:
        if "style_token_count" in config:
            config["style_tokens_per_ref"] = max(1, int(config.pop("style_token_count")) // 8)
        else:
            local_tokens = int(config.pop("local_style_tokens_per_ref", 24))
            residual_tokens = int(config.pop("style_residual_tokens", 0))
            config["style_tokens_per_ref"] = max(1, (local_tokens + residual_tokens) // 8)
        config.pop("style_mid_tokens_per_ref", None)
        config.pop("style_residual_gate_init", None)
    if "vae_bottleneck_channels" not in config:
        if "to_stats.weight" in checkpoint["vae_state"]:
            config["vae_bottleneck_channels"] = int(checkpoint["vae_state"]["to_stats.weight"].shape[1])
        else:
            config["vae_bottleneck_channels"] = 128
    if "vae_encoder_16x16_blocks" not in config:
        config["vae_encoder_16x16_blocks"] = 1
    if "vae_decoder_16x16_blocks" not in config:
        config["vae_decoder_16x16_blocks"] = 1
    if "vae_decoder_tail_blocks" not in config:
        if "vae_decoder_extra_block" in config:
            config["vae_decoder_tail_blocks"] = 1 if bool(config.pop("vae_decoder_extra_block")) else 0
        else:
            config["vae_decoder_tail_blocks"] = 1 if any(
                key.startswith("dec_block128.") for key in checkpoint["vae_state"].keys()
            ) else 0
    config.pop("vae_detail_residual_enabled", None)
    config.pop("vae_detail_residual_blocks", None)
    config.pop("vae_lambda_detail", None)
    config.pop("vae_detail_mask_threshold", None)
    config.pop("vae_detail_mask_dilation", None)
    return config


def normalize_vae_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = dict(checkpoint["vae_state"])
    has_extra_block = any(key.startswith("dec_block128.") for key in state.keys())
    if not has_extra_block and "dec_up128.2.weight" in state and "dec_norm128.weight" not in state:
        state["dec_norm128.weight"] = state.pop("dec_up128.2.weight")
        state["dec_norm128.bias"] = state.pop("dec_up128.2.bias")
    return state


def load_model(checkpoint_path: Path, device: torch.device) -> SourcePartRefDiT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("stage") != "vae":
        raise RuntimeError(f"Checkpoint is not a VAE checkpoint: {checkpoint_path}")
    if "model_config" not in checkpoint or "vae_state" not in checkpoint:
        raise RuntimeError(f"Malformed VAE checkpoint: {checkpoint_path}")
    model = SourcePartRefDiT(**normalize_model_config(checkpoint))
    model.vae.load_state_dict(normalize_vae_state(checkpoint), strict=True)
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


def _foreground_mask(x01: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    return x01.squeeze(0).lt(float(threshold))


def _binary_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    union = torch.logical_or(pred_mask, target_mask).sum().item()
    if union <= 0:
        return 1.0
    return float(inter) / float(union)


def _binary_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    total = pred_mask.sum().item() + target_mask.sum().item()
    if total <= 0:
        return 1.0
    return float(2.0 * inter) / float(total)


def _edge_mask(binary_mask: torch.Tensor) -> torch.Tensor:
    x = binary_mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).squeeze(0).squeeze(0).gt(0.0)


def _component_count(binary_mask: torch.Tensor) -> int:
    arr = binary_mask.to(dtype=torch.uint8).cpu().numpy()
    count, _ = cv2.connectedComponents(arr)
    return max(0, int(count) - 1)


def compute_structure_metrics(target01: torch.Tensor, recon01: torch.Tensor) -> dict[str, float]:
    target_mask = _foreground_mask(target01)
    recon_mask = _foreground_mask(recon01)
    target_edge = _edge_mask(target_mask)
    recon_edge = _edge_mask(recon_mask)
    target_components = _component_count(target_mask)
    recon_components = _component_count(recon_mask)
    fg_ratio_abs_diff = abs(float(recon_mask.float().mean().item()) - float(target_mask.float().mean().item()))
    bin_iou = _binary_iou(recon_mask, target_mask)
    bin_dice = _binary_dice(recon_mask, target_mask)
    edge_dice = _binary_dice(recon_edge, target_edge)
    component_abs_diff = abs(recon_components - target_components)
    structure_score = 0.6 * bin_dice + 0.4 * edge_dice
    return {
        "bin_iou": float(bin_iou),
        "bin_dice": float(bin_dice),
        "edge_dice": float(edge_dice),
        "component_abs_diff": float(component_abs_diff),
        "fg_ratio_abs_diff": float(fg_ratio_abs_diff),
        "structure_score": float(structure_score),
    }


def compute_latent_sample_metrics(mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, float]:
    mu = mu.detach().float()
    logvar = logvar.detach().float()
    latent_std = float(mu.std(unbiased=False).item())
    latent_scale = max(latent_std, 1e-6)

    dx = mu[:, :, 1:] - mu[:, :, :-1]
    dy = mu[:, 1:, :] - mu[:, :-1, :]
    spatial_tv = 0.5 * (dx.abs().mean() + dy.abs().mean())

    lap_xx = mu[:, :, 2:] - 2.0 * mu[:, :, 1:-1] + mu[:, :, :-2]
    lap_yy = mu[:, 2:, :] - 2.0 * mu[:, 1:-1, :] + mu[:, :-2, :]
    laplacian = 0.0
    if lap_xx.numel() > 0:
        laplacian += 0.5 * float(lap_xx.abs().mean().item())
    if lap_yy.numel() > 0:
        laplacian += 0.5 * float(lap_yy.abs().mean().item())

    if mu.size(2) > 1:
        horiz_a = mu[:, :, :-1].permute(1, 2, 0).reshape(-1, mu.size(0))
        horiz_b = mu[:, :, 1:].permute(1, 2, 0).reshape(-1, mu.size(0))
        horiz_cos = F.cosine_similarity(horiz_a, horiz_b, dim=-1).mean()
    else:
        horiz_cos = torch.tensor(1.0)
    if mu.size(1) > 1:
        vert_a = mu[:, :-1, :].permute(1, 2, 0).reshape(-1, mu.size(0))
        vert_b = mu[:, 1:, :].permute(1, 2, 0).reshape(-1, mu.size(0))
        vert_cos = F.cosine_similarity(vert_a, vert_b, dim=-1).mean()
    else:
        vert_cos = torch.tensor(1.0)
    posterior_std = torch.exp(0.5 * logvar)
    channel_std = mu.view(mu.size(0), -1).std(dim=1, unbiased=False)
    channel_std_mean = float(channel_std.mean().item())
    channel_std_cv = float(channel_std.std(unbiased=False).item() / max(channel_std_mean, 1e-6))
    return {
        "latent_abs_mean": float(mu.mean().abs().item()),
        "latent_std": latent_std,
        "spatial_tv_norm": float(spatial_tv.item() / latent_scale),
        "spatial_laplacian_norm": float(laplacian / latent_scale),
        "neighbor_cosine_mean": float(0.5 * (float(horiz_cos.item()) + float(vert_cos.item()))),
        "posterior_std_mean": float(posterior_std.mean().item()),
        "posterior_std_std": float(posterior_std.std(unbiased=False).item()),
        "channel_std_cv": channel_std_cv,
    }


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(torch.sqrt((x.square().sum()) * (y.square().sum())).item())
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum().item() / denom)


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
    parser.add_argument("--latent-semantic-samples", type=int, default=256)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)

    if args.out_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = args.data_root / "analysis" / f"vae_eval_{args.font_split}_fonts_{int(args.chars_per_font)}chars_{stamp}"
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

    semantic_sample_count = min(int(args.latent_semantic_samples), len(eval_entries))
    semantic_indices = set(rng.sample(range(len(eval_entries)), k=semantic_sample_count)) if semantic_sample_count > 1 else set()

    by_font: dict[str, dict[str, Any]] = defaultdict(lambda: {"samples": []})
    all_mae: list[float] = []
    all_mse: list[float] = []
    all_psnr: list[float] = []
    all_bin_iou: list[float] = []
    all_bin_dice: list[float] = []
    all_edge_dice: list[float] = []
    all_component_abs_diff: list[float] = []
    all_fg_ratio_abs_diff: list[float] = []
    all_structure_score: list[float] = []
    all_latent_abs_mean: list[float] = []
    all_latent_std: list[float] = []
    all_spatial_tv_norm: list[float] = []
    all_spatial_laplacian_norm: list[float] = []
    all_neighbor_cosine: list[float] = []
    all_posterior_std_mean: list[float] = []
    all_posterior_std_std: list[float] = []
    all_channel_std_cv: list[float] = []

    latent_element_count = 0
    latent_raw_sum = 0.0
    latent_raw_sq_sum = 0.0
    latent_raw_cube_sum = 0.0
    latent_raw_quad_sum = 0.0
    channel_sum: torch.Tensor | None = None
    channel_sq_sum: torch.Tensor | None = None
    channel_cross_sum: torch.Tensor | None = None
    channel_count = 0
    semantic_latents: list[torch.Tensor] = []
    semantic_images: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(eval_entries), int(args.batch_size)):
            batch_entries = eval_entries[start : start + int(args.batch_size)]
            target_batch = torch.stack([entry["target"] for entry in batch_entries], dim=0).to(device)
            recon_batch, _, mu_batch, logvar_batch = model.vae_forward(target_batch, sample_posterior=False)
            recon_batch = recon_batch.cpu()
            target_batch = target_batch.cpu()
            mu_batch = mu_batch.cpu()
            logvar_batch = logvar_batch.cpu()

            flat_mu = mu_batch.permute(0, 2, 3, 1).reshape(-1, mu_batch.size(1)).double()
            if channel_sum is None:
                channel_sum = torch.zeros(mu_batch.size(1), dtype=torch.float64)
                channel_sq_sum = torch.zeros(mu_batch.size(1), dtype=torch.float64)
                channel_cross_sum = torch.zeros((mu_batch.size(1), mu_batch.size(1)), dtype=torch.float64)
            channel_sum += flat_mu.sum(dim=0)
            channel_sq_sum += flat_mu.square().sum(dim=0)
            channel_cross_sum += flat_mu.T @ flat_mu
            channel_count += int(flat_mu.size(0))

            mu_all = mu_batch.reshape(-1).double()
            latent_element_count += int(mu_all.numel())
            latent_raw_sum += float(mu_all.sum().item())
            latent_raw_sq_sum += float(mu_all.square().sum().item())
            latent_raw_cube_sum += float(mu_all.pow(3).sum().item())
            latent_raw_quad_sum += float(mu_all.pow(4).sum().item())

            for local_idx, (entry, target, recon, mu, logvar) in enumerate(
                zip(batch_entries, target_batch, recon_batch, mu_batch, logvar_batch)
            ):
                global_idx = start + local_idx
                target01 = _tensor01(target)
                recon01 = _tensor01(recon)
                mae = float(F.l1_loss(recon01, target01).item())
                mse = float(F.mse_loss(recon01, target01).item())
                psnr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
                structure_metrics = compute_structure_metrics(target01, recon01)
                latent_metrics = compute_latent_sample_metrics(mu, logvar)
                by_font[entry["font"]]["font"] = entry["font"]
                by_font[entry["font"]]["samples"].append(
                    {
                        "char": entry["char"],
                        "mae": mae,
                        "mse": mse,
                        "psnr": psnr,
                        **structure_metrics,
                        **latent_metrics,
                        "target_img": _tensor_to_u8_image(target, size=int(model.image_size)),
                        "recon_img": _tensor_to_u8_image(recon, size=int(model.image_size)),
                    }
                )
                all_mae.append(mae)
                all_mse.append(mse)
                all_psnr.append(psnr)
                all_bin_iou.append(structure_metrics["bin_iou"])
                all_bin_dice.append(structure_metrics["bin_dice"])
                all_edge_dice.append(structure_metrics["edge_dice"])
                all_component_abs_diff.append(structure_metrics["component_abs_diff"])
                all_fg_ratio_abs_diff.append(structure_metrics["fg_ratio_abs_diff"])
                all_structure_score.append(structure_metrics["structure_score"])
                all_latent_abs_mean.append(latent_metrics["latent_abs_mean"])
                all_latent_std.append(latent_metrics["latent_std"])
                all_spatial_tv_norm.append(latent_metrics["spatial_tv_norm"])
                all_spatial_laplacian_norm.append(latent_metrics["spatial_laplacian_norm"])
                all_neighbor_cosine.append(latent_metrics["neighbor_cosine_mean"])
                all_posterior_std_mean.append(latent_metrics["posterior_std_mean"])
                all_posterior_std_std.append(latent_metrics["posterior_std_std"])
                all_channel_std_cv.append(latent_metrics["channel_std_cv"])

                if global_idx in semantic_indices:
                    semantic_latents.append(mu.reshape(-1).float())
                    semantic_images.append(
                        F.avg_pool2d(target01.unsqueeze(0), kernel_size=max(1, int(model.image_size) // int(model.latent_size)))
                        .reshape(-1)
                        .float()
                    )

    font_rows: list[dict[str, Any]] = []
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
        row["mean_bin_iou"] = float(sum(sample["bin_iou"] for sample in samples) / len(samples))
        row["mean_bin_dice"] = float(sum(sample["bin_dice"] for sample in samples) / len(samples))
        row["mean_edge_dice"] = float(sum(sample["edge_dice"] for sample in samples) / len(samples))
        row["mean_component_abs_diff"] = float(sum(sample["component_abs_diff"] for sample in samples) / len(samples))
        row["mean_fg_ratio_abs_diff"] = float(sum(sample["fg_ratio_abs_diff"] for sample in samples) / len(samples))
        row["mean_structure_score"] = float(sum(sample["structure_score"] for sample in samples) / len(samples))
        row["mean_latent_abs_mean"] = float(sum(sample["latent_abs_mean"] for sample in samples) / len(samples))
        row["mean_latent_std"] = float(sum(sample["latent_std"] for sample in samples) / len(samples))
        row["mean_spatial_tv_norm"] = float(sum(sample["spatial_tv_norm"] for sample in samples) / len(samples))
        row["mean_spatial_laplacian_norm"] = float(
            sum(sample["spatial_laplacian_norm"] for sample in samples) / len(samples)
        )
        row["mean_neighbor_cosine"] = float(sum(sample["neighbor_cosine_mean"] for sample in samples) / len(samples))
        row["mean_posterior_std_mean"] = float(sum(sample["posterior_std_mean"] for sample in samples) / len(samples))
        font_rows.append(row)

    if channel_sum is None or channel_sq_sum is None or channel_cross_sum is None or channel_count <= 0:
        raise RuntimeError("No evaluation samples were processed.")

    latent_raw_mean = latent_raw_sum / float(latent_element_count)
    latent_raw_second = latent_raw_sq_sum / float(latent_element_count)
    latent_raw_third = latent_raw_cube_sum / float(latent_element_count)
    latent_raw_fourth = latent_raw_quad_sum / float(latent_element_count)
    latent_global_var = max(latent_raw_second - latent_raw_mean * latent_raw_mean, 1e-12)
    latent_global_std = math.sqrt(latent_global_var)
    latent_central_third = latent_raw_third - 3.0 * latent_raw_mean * latent_raw_second + 2.0 * (latent_raw_mean ** 3)
    latent_central_fourth = (
        latent_raw_fourth
        - 4.0 * latent_raw_mean * latent_raw_third
        + 6.0 * (latent_raw_mean ** 2) * latent_raw_second
        - 3.0 * (latent_raw_mean ** 4)
    )
    latent_skew = latent_central_third / max(latent_global_std ** 3, 1e-12)
    latent_excess_kurtosis = latent_central_fourth / max(latent_global_var ** 2, 1e-12) - 3.0

    channel_mean = channel_sum / float(channel_count)
    channel_var = (channel_sq_sum / float(channel_count)) - channel_mean.square()
    channel_var = torch.clamp(channel_var, min=1e-12)
    channel_std = torch.sqrt(channel_var)
    channel_std_cv_global = float(channel_std.std(unbiased=False).item() / max(channel_std.mean().item(), 1e-12))
    channel_cov = (channel_cross_sum / float(channel_count)) - torch.outer(channel_mean, channel_mean)
    channel_denom = torch.sqrt(torch.outer(channel_var, channel_var)).clamp_min(1e-12)
    channel_corr = channel_cov / channel_denom
    offdiag_mask = ~torch.eye(channel_corr.size(0), dtype=torch.bool)
    channel_corr_abs_mean = float(channel_corr[offdiag_mask].abs().mean().item()) if offdiag_mask.any() else 0.0

    latent_image_distance_corr = 0.0
    if len(semantic_latents) >= 2 and len(semantic_images) >= 2:
        latent_matrix = torch.stack(semantic_latents, dim=0)
        image_matrix = torch.stack(semantic_images, dim=0)
        latent_dist = torch.cdist(latent_matrix, latent_matrix, p=2)
        image_dist = torch.cdist(image_matrix, image_matrix, p=2)
        tri = torch.triu_indices(latent_dist.size(0), latent_dist.size(1), offset=1)
        latent_image_distance_corr = _pearson_corr(
            latent_dist[tri[0], tri[1]],
            image_dist[tri[0], tri[1]],
        )

    page_paths = build_pages(
        PROJECT_ROOT,
        font_rows,
        out_dir,
        chars_per_font=int(args.chars_per_font),
    )

    sortable_rows = sorted(font_rows, key=lambda item: item["mean_mae"])
    structure_rows = sorted(font_rows, key=lambda item: item["mean_structure_score"], reverse=True)
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
        "mean_bin_iou_all_fonts": float(sum(all_bin_iou) / len(all_bin_iou)),
        "mean_bin_dice_all_fonts": float(sum(all_bin_dice) / len(all_bin_dice)),
        "mean_edge_dice_all_fonts": float(sum(all_edge_dice) / len(all_edge_dice)),
        "mean_component_abs_diff_all_fonts": float(sum(all_component_abs_diff) / len(all_component_abs_diff)),
        "mean_fg_ratio_abs_diff_all_fonts": float(sum(all_fg_ratio_abs_diff) / len(all_fg_ratio_abs_diff)),
        "mean_structure_score_all_fonts": float(sum(all_structure_score) / len(all_structure_score)),
        "latent_global_metrics": {
            "latent_abs_mean": float(sum(all_latent_abs_mean) / len(all_latent_abs_mean)),
            "latent_std_sample_mean": float(sum(all_latent_std) / len(all_latent_std)),
            "spatial_tv_norm_mean": float(sum(all_spatial_tv_norm) / len(all_spatial_tv_norm)),
            "spatial_laplacian_norm_mean": float(sum(all_spatial_laplacian_norm) / len(all_spatial_laplacian_norm)),
            "neighbor_cosine_mean": float(sum(all_neighbor_cosine) / len(all_neighbor_cosine)),
            "posterior_std_mean": float(sum(all_posterior_std_mean) / len(all_posterior_std_mean)),
            "posterior_std_std_mean": float(sum(all_posterior_std_std) / len(all_posterior_std_std)),
            "channel_std_cv_mean": float(sum(all_channel_std_cv) / len(all_channel_std_cv)),
            "latent_global_mean": float(latent_raw_mean),
            "latent_global_std": float(latent_global_std),
            "latent_skew": float(latent_skew),
            "latent_excess_kurtosis": float(latent_excess_kurtosis),
            "channel_std_cv_global": float(channel_std_cv_global),
            "channel_corr_abs_mean": float(channel_corr_abs_mean),
            "latent_image_distance_corr": float(latent_image_distance_corr),
            "semantic_pair_count": int(len(semantic_latents)),
        },
        "best10_by_mae": [
            {
                "font": row["font"],
                "mean_mae": row["mean_mae"],
                "mean_mse": row["mean_mse"],
                "mean_psnr": row["mean_psnr"],
                "mean_structure_score": row["mean_structure_score"],
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
                "mean_structure_score": row["mean_structure_score"],
                "chars": [sample["char"] for sample in row["samples"]],
            }
            for row in sortable_rows[-10:]
        ],
        "best10_by_structure_score": [
            {
                "font": row["font"],
                "mean_structure_score": row["mean_structure_score"],
                "mean_bin_dice": row["mean_bin_dice"],
                "mean_edge_dice": row["mean_edge_dice"],
                "mean_mae": row["mean_mae"],
                "chars": [sample["char"] for sample in row["samples"]],
            }
            for row in structure_rows[:10]
        ],
        "worst10_by_structure_score": [
            {
                "font": row["font"],
                "mean_structure_score": row["mean_structure_score"],
                "mean_bin_dice": row["mean_bin_dice"],
                "mean_edge_dice": row["mean_edge_dice"],
                "mean_mae": row["mean_mae"],
                "chars": [sample["char"] for sample in row["samples"]],
            }
            for row in structure_rows[-10:]
        ],
        "per_font": [
            {
                "font": row["font"],
                "mean_mae": row["mean_mae"],
                "mean_mse": row["mean_mse"],
                "mean_psnr": row["mean_psnr"],
                "mean_bin_iou": row["mean_bin_iou"],
                "mean_bin_dice": row["mean_bin_dice"],
                "mean_edge_dice": row["mean_edge_dice"],
                "mean_component_abs_diff": row["mean_component_abs_diff"],
                "mean_fg_ratio_abs_diff": row["mean_fg_ratio_abs_diff"],
                "mean_structure_score": row["mean_structure_score"],
                "mean_latent_abs_mean": row["mean_latent_abs_mean"],
                "mean_latent_std": row["mean_latent_std"],
                "mean_spatial_tv_norm": row["mean_spatial_tv_norm"],
                "mean_spatial_laplacian_norm": row["mean_spatial_laplacian_norm"],
                "mean_neighbor_cosine": row["mean_neighbor_cosine"],
                "mean_posterior_std_mean": row["mean_posterior_std_mean"],
                "samples": [
                    {
                        "char": sample["char"],
                        "mae": sample["mae"],
                        "mse": sample["mse"],
                        "psnr": sample["psnr"],
                        "bin_iou": sample["bin_iou"],
                        "bin_dice": sample["bin_dice"],
                        "edge_dice": sample["edge_dice"],
                        "component_abs_diff": sample["component_abs_diff"],
                        "fg_ratio_abs_diff": sample["fg_ratio_abs_diff"],
                        "structure_score": sample["structure_score"],
                        "latent_abs_mean": sample["latent_abs_mean"],
                        "latent_std": sample["latent_std"],
                        "spatial_tv_norm": sample["spatial_tv_norm"],
                        "spatial_laplacian_norm": sample["spatial_laplacian_norm"],
                        "neighbor_cosine_mean": sample["neighbor_cosine_mean"],
                        "posterior_std_mean": sample["posterior_std_mean"],
                        "posterior_std_std": sample["posterior_std_std"],
                        "channel_std_cv": sample["channel_std_cv"],
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
        f"mean_bin_iou_all_fonts: {report['mean_bin_iou_all_fonts']:.6f}",
        f"mean_bin_dice_all_fonts: {report['mean_bin_dice_all_fonts']:.6f}",
        f"mean_edge_dice_all_fonts: {report['mean_edge_dice_all_fonts']:.6f}",
        f"mean_component_abs_diff_all_fonts: {report['mean_component_abs_diff_all_fonts']:.6f}",
        f"mean_fg_ratio_abs_diff_all_fonts: {report['mean_fg_ratio_abs_diff_all_fonts']:.6f}",
        f"mean_structure_score_all_fonts: {report['mean_structure_score_all_fonts']:.6f}",
        "",
        "latent_global_metrics:",
        f"  latent_abs_mean: {report['latent_global_metrics']['latent_abs_mean']:.6f}",
        f"  latent_std_sample_mean: {report['latent_global_metrics']['latent_std_sample_mean']:.6f}",
        f"  spatial_tv_norm_mean: {report['latent_global_metrics']['spatial_tv_norm_mean']:.6f}",
        f"  spatial_laplacian_norm_mean: {report['latent_global_metrics']['spatial_laplacian_norm_mean']:.6f}",
        f"  neighbor_cosine_mean: {report['latent_global_metrics']['neighbor_cosine_mean']:.6f}",
        f"  posterior_std_mean: {report['latent_global_metrics']['posterior_std_mean']:.6f}",
        f"  posterior_std_std_mean: {report['latent_global_metrics']['posterior_std_std_mean']:.6f}",
        f"  channel_std_cv_mean: {report['latent_global_metrics']['channel_std_cv_mean']:.6f}",
        f"  latent_global_mean: {report['latent_global_metrics']['latent_global_mean']:.6f}",
        f"  latent_global_std: {report['latent_global_metrics']['latent_global_std']:.6f}",
        f"  latent_skew: {report['latent_global_metrics']['latent_skew']:.6f}",
        f"  latent_excess_kurtosis: {report['latent_global_metrics']['latent_excess_kurtosis']:.6f}",
        f"  channel_std_cv_global: {report['latent_global_metrics']['channel_std_cv_global']:.6f}",
        f"  channel_corr_abs_mean: {report['latent_global_metrics']['channel_corr_abs_mean']:.6f}",
        f"  latent_image_distance_corr: {report['latent_global_metrics']['latent_image_distance_corr']:.6f}",
        f"  semantic_pair_count: {report['latent_global_metrics']['semantic_pair_count']}",
        "",
        "best10_by_mae:",
    ]
    for row in report["best10_by_mae"]:
        summary_lines.append(
            f"  {row['font']}: mae={row['mean_mae']:.6f} mse={row['mean_mse']:.6f} "
            f"psnr={row['mean_psnr']:.4f} structure={row['mean_structure_score']:.6f} chars={','.join(row['chars'])}"
        )
    summary_lines.append("")
    summary_lines.append("worst10_by_mae:")
    for row in report["worst10_by_mae"]:
        summary_lines.append(
            f"  {row['font']}: mae={row['mean_mae']:.6f} mse={row['mean_mse']:.6f} "
            f"psnr={row['mean_psnr']:.4f} structure={row['mean_structure_score']:.6f} chars={','.join(row['chars'])}"
        )
    summary_lines.append("")
    summary_lines.append("worst10_by_structure_score:")
    for row in report["worst10_by_structure_score"]:
        summary_lines.append(
            f"  {row['font']}: structure={row['mean_structure_score']:.6f} "
            f"bin_dice={row['mean_bin_dice']:.6f} edge_dice={row['mean_edge_dice']:.6f} "
            f"mae={row['mean_mae']:.6f} chars={','.join(row['chars'])}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[vae_eval] device={device}")
    print(f"[vae_eval] report={out_dir / 'report.json'}")
    print(f"[vae_eval] summary={out_dir / 'summary.txt'}")
    print(f"[vae_eval] pages={len(page_paths)}")


if __name__ == "__main__":
    main()
