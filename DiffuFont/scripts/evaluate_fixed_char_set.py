#!/usr/bin/env python3
"""Evaluate x-pred checkpoints on a fixed char set across all fonts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import numpy as np
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import linalg
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import FontImageDataset
from inference import load_trainer
from models.font_perceptor import load_font_perceptor_from_checkpoint
from style_augment import build_base_glyph_transform


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    font_name: str
    font_id: int
    split: str
    char: str
    char_id: int
    style_chars: tuple[str, ...]
    style_char_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed chars x all fonts for DiffuFont x-pred checkpoints.")
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to the run's train_config.json",
    )
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--final-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--perceptor-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--num-target-chars", type=int, default=30)
    parser.add_argument("--target-char-source", type=str, default="all", choices=["all", "style_pool"])
    parser.add_argument("--style-ref-count", type=int, default=8)
    parser.add_argument("--style-pool-file", type=Path, default=None)
    parser.add_argument("--fonts-per-batch", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--cell-size", type=int, default=96)
    parser.add_argument("--limit-fonts", type=int, default=0)
    parser.add_argument(
        "--worst-rank-by",
        type=str,
        default="perceptor_perceptual",
        choices=["perceptor_perceptual", "mae", "mse", "rmse", "neg_ssim"],
    )
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        return torch.device("cpu")
    return torch.device(raw_device)


def resolve_run_paths(args: argparse.Namespace, train_config: dict[str, Any]) -> dict[str, Path]:
    run_dir = args.train_config.resolve().parent
    total_steps = int(train_config["total_steps"])
    best_checkpoint = args.best_checkpoint.resolve() if args.best_checkpoint is not None else (run_dir / "best.pt")
    final_checkpoint = (
        args.final_checkpoint.resolve()
        if args.final_checkpoint is not None
        else (run_dir / f"ckpt_step_{total_steps}.pt")
    )
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = run_dir / "analysis_fixed30_allfonts"
    return {
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "output_dir": output_dir,
    }


def load_style_pool(path: Path) -> list[str]:
    chars = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chars:
        raise RuntimeError(f"Style pool file is empty: {path}")
    return chars


def infer_checkpoint_step(alias: str, checkpoint_path: Path, run_dir: Path, train_config: dict[str, Any]) -> int:
    if alias == "best":
        best_metrics_path = run_dir / "best_val_metrics.json"
        if best_metrics_path.exists():
            return int(read_json(best_metrics_path)["step"])
    match = re.search(r"step_(\d+)\.pt$", checkpoint_path.name)
    if match:
        return int(match.group(1))
    return int(train_config["total_steps"])


def load_datasets(root: Path, train_config: dict[str, Any]) -> tuple[FontImageDataset, set[str], set[str]]:
    transform = build_base_glyph_transform(image_size=int(train_config["image_size"]))
    dataset_common_kwargs = {
        "project_root": root,
        "max_fonts": int(train_config["max_fonts"]),
        "style_ref_count": int(train_config["style_ref_count_max"]),
        "style_ref_count_min": int(train_config["style_ref_count_max"]),
        "style_ref_count_max": int(train_config["style_ref_count_max"]),
        "random_seed": int(train_config["seed"]),
        "font_split_seed": int(train_config["font_split_seed"]),
        "font_train_ratio": float(train_config["font_train_ratio"]),
        "transform": transform,
        "style_transform": transform,
        "load_style_refs": False,
    }
    dataset_all = FontImageDataset(font_split="all", **dataset_common_kwargs)
    dataset_train = FontImageDataset(font_split="train", **dataset_common_kwargs)
    dataset_val = FontImageDataset(font_split="test", **dataset_common_kwargs)
    return dataset_all, set(dataset_train.font_names), set(dataset_val.font_names)


def sample_target_chars(
    dataset: FontImageDataset,
    *,
    num_target_chars: int,
    eval_seed: int,
    target_char_source: str,
    style_pool_chars: list[str],
) -> list[str]:
    char_pool = list(dataset.char_list)
    if target_char_source == "style_pool":
        char_pool = [char for char in style_pool_chars if char in set(dataset.char_list)]
    if len(char_pool) < int(num_target_chars):
        raise RuntimeError(f"Not enough target chars to sample: requested={num_target_chars} available={len(char_pool)}")
    rng = random.Random(int(eval_seed))
    return rng.sample(char_pool, k=int(num_target_chars))


def infer_split(font_name: str, train_fonts: set[str], val_fonts: set[str]) -> str:
    if font_name in train_fonts:
        return "train"
    if font_name in val_fonts:
        return "val"
    return "unknown"


def build_sample_specs(
    dataset: FontImageDataset,
    *,
    font_names: list[str],
    target_chars: list[str],
    style_pool_chars: list[str],
    style_ref_count: int,
    eval_seed: int,
    train_fonts: set[str],
    val_fonts: set[str],
) -> list[SampleSpec]:
    char_id_by_char = {char: idx for idx, char in enumerate(dataset.char_list)}
    style_pool_ids = [int(char_id_by_char[char]) for char in style_pool_chars]
    specs: list[SampleSpec] = []
    for font_name in font_names:
        font_id = int(dataset.font_id_by_name[font_name])
        split = infer_split(font_name, train_fonts, val_fonts)
        for char in target_chars:
            char_id = int(char_id_by_char[char])
            candidates = [idx for idx in style_pool_ids if idx != char_id]
            if len(candidates) < int(style_ref_count):
                raise RuntimeError(
                    f"Style pool too small for font={font_name} char={char} "
                    f"requested_refs={style_ref_count} candidates={len(candidates)}"
                )
            rng = random.Random(f"{int(eval_seed)}|{font_name}|{char_id}")
            style_char_ids = tuple(sorted(rng.sample(candidates, k=int(style_ref_count))))
            style_chars = tuple(dataset.char_list[idx] for idx in style_char_ids)
            specs.append(
                SampleSpec(
                    sample_id=f"{font_name}::{char}",
                    font_name=font_name,
                    font_id=font_id,
                    split=split,
                    char=char,
                    char_id=char_id,
                    style_chars=style_chars,
                    style_char_ids=style_char_ids,
                )
            )
    return specs


def specs_by_font(specs: Iterable[SampleSpec]) -> dict[str, list[SampleSpec]]:
    output: dict[str, list[SampleSpec]] = {}
    for spec in specs:
        output.setdefault(spec.font_name, []).append(spec)
    return output


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[idx: idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def tensor_to_u8_rgb(tensor: torch.Tensor) -> torch.Tensor:
    x = ((tensor.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return x.repeat(1, 3, 1, 1)


def tensor_to_pil(tensor: torch.Tensor, cell_size: int) -> Image.Image:
    x = tensor.detach().cpu().float()
    if x.dim() == 3:
        x = x[0]
    x = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte().numpy()
    return Image.fromarray(x, mode="L").resize((cell_size, cell_size), Image.Resampling.NEAREST).convert("RGB")


def prepare_batch(
    dataset: FontImageDataset,
    batch_specs: list[SampleSpec],
) -> dict[str, Any]:
    content_slot_by_char_id: dict[int, int] = {}
    content_tensors: list[torch.Tensor] = []
    content_index: list[int] = []
    target_tensors: list[torch.Tensor] = []
    style_img_tensors: list[torch.Tensor] = []
    style_ref_mask_tensors: list[torch.Tensor] = []
    for spec in batch_specs:
        sample_index = int(dataset.sample_index_by_font_char[spec.font_name][spec.char_id])
        sample = dataset[sample_index]
        slot = content_slot_by_char_id.get(spec.char_id)
        if slot is None:
            slot = len(content_tensors)
            content_slot_by_char_id[spec.char_id] = slot
            content_tensors.append(sample["content"])
        content_index.append(slot)
        target_tensors.append(sample["target"])
        style_img, style_ref_mask, _ = dataset.load_style_refs_by_indices(spec.font_name, list(spec.style_char_ids))
        style_img_tensors.append(style_img)
        style_ref_mask_tensors.append(style_ref_mask)
    style_count = len(style_img_tensors)
    return {
        "content": torch.stack(content_tensors, dim=0),
        "content_index": torch.tensor(content_index, dtype=torch.long),
        "target": torch.stack(target_tensors, dim=0),
        "style_img": torch.stack(style_img_tensors, dim=0),
        "style_ref_mask": torch.stack(style_ref_mask_tensors, dim=0),
        "style_index": torch.arange(style_count, dtype=torch.long),
    }


def compute_perceptor_metrics(
    perceptor_model,
    generation: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device_type = "cuda" if generation.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        gen_outputs = perceptor_model(generation.float())
        tgt_outputs = perceptor_model(target.float())
        perceptual_per_sample = generation.new_zeros(generation.size(0), dtype=torch.float32)
        for pred_feat, target_feat in zip(gen_outputs["feature_maps"], tgt_outputs["feature_maps"], strict=True):
            perceptual_per_sample = perceptual_per_sample + (pred_feat - target_feat).abs().flatten(1).mean(dim=1)
    return {
        "perceptor_perceptual": perceptual_per_sample.detach(),
    }


def compute_image_metrics(
    generation: torch.Tensor,
    target: torch.Tensor,
    perceptor_model,
) -> dict[str, torch.Tensor]:
    generation01 = generation.detach().float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    target01 = target.detach().float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    diff = generation01 - target01
    mae = diff.abs().flatten(1).mean(dim=1)
    mse = diff.pow(2).flatten(1).mean(dim=1)
    rmse = torch.sqrt(mse.clamp_min(1e-12))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ssim = structural_similarity_index_measure(
        generation01,
        target01,
        data_range=1.0,
        reduction="none",
    )
    perceptor_metrics = compute_perceptor_metrics(perceptor_model, generation, target)
    return {
        "mae": mae.detach(),
        "mse": mse.detach(),
        "rmse": rmse.detach(),
        "psnr": psnr.detach(),
        "ssim": ssim.detach(),
        **perceptor_metrics,
    }


def init_fid(device: torch.device, feature: int) -> FrechetInceptionDistance:
    metric = FrechetInceptionDistance(feature=int(feature), normalize=False)
    return metric.to(device)


def compute_fid_from_metric(metric: FrechetInceptionDistance) -> float:
    if int(metric.real_features_num_samples) < 2 or int(metric.fake_features_num_samples) < 2:
        raise RuntimeError("More than one real/fake sample is required to compute FID.")

    mean_real = (metric.real_features_sum / metric.real_features_num_samples).unsqueeze(0)
    mean_fake = (metric.fake_features_sum / metric.fake_features_num_samples).unsqueeze(0)
    cov_real_num = metric.real_features_cov_sum - metric.real_features_num_samples * mean_real.t().mm(mean_real)
    cov_fake_num = metric.fake_features_cov_sum - metric.fake_features_num_samples * mean_fake.t().mm(mean_fake)
    cov_real = cov_real_num / (metric.real_features_num_samples - 1)
    cov_fake = cov_fake_num / (metric.fake_features_num_samples - 1)

    mu1 = mean_real.squeeze(0).detach().cpu().double().numpy()
    mu2 = mean_fake.squeeze(0).detach().cpu().double().numpy()
    sigma1 = cov_real.detach().cpu().double().numpy()
    sigma2 = cov_fake.detach().cpu().double().numpy()
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(np.real(fid))


def update_fids(
    fid_by_split: dict[str, FrechetInceptionDistance],
    generation: torch.Tensor,
    target: torch.Tensor,
    split_labels: list[str],
) -> None:
    fake_u8 = tensor_to_u8_rgb(generation)
    real_u8 = tensor_to_u8_rgb(target)
    fid_by_split["all"].update(real_u8, real=True)
    fid_by_split["all"].update(fake_u8, real=False)
    index_by_split: dict[str, list[int]] = {}
    for idx, split in enumerate(split_labels):
        index_by_split.setdefault(split, []).append(idx)
    for split_name, indices in index_by_split.items():
        if split_name not in fid_by_split:
            continue
        batch_index = torch.tensor(indices, device=generation.device, dtype=torch.long)
        fid_by_split[split_name].update(real_u8.index_select(0, batch_index), real=True)
        fid_by_split[split_name].update(fake_u8.index_select(0, batch_index), real=False)


def summarize_metrics(
    df: pd.DataFrame,
    fid_results: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    metric_columns = ["mae", "mse", "rmse", "psnr", "ssim", "perceptor_perceptual"]
    output: dict[str, dict[str, float | int]] = {}
    for split_name in ["all", "train", "val"]:
        split_df = df if split_name == "all" else df[df["split"] == split_name]
        if split_df.empty:
            continue
        row: dict[str, float | int] = {
            "num_samples": int(len(split_df)),
            "num_fonts": int(split_df["font_name"].nunique()),
            "num_chars": int(split_df["char"].nunique()),
        }
        if split_name in fid_results:
            row["fid"] = float(fid_results[split_name])
        for metric_name in metric_columns:
            row[f"{metric_name}_mean"] = float(split_df[metric_name].mean())
            row[f"{metric_name}_std"] = float(split_df[metric_name].std(ddof=0))
        output[split_name] = row
    return output


def _pair_sum(normalized_vectors: torch.Tensor) -> tuple[float, int]:
    count = int(normalized_vectors.size(0))
    if count < 2:
        return 0.0, 0
    summed = normalized_vectors.sum(dim=0)
    pair_sum = float((summed.dot(summed) - float(count)) / 2.0)
    pair_count = count * (count - 1) // 2
    return pair_sum, pair_count


def _group_pair_sum(normalized_vectors: torch.Tensor, labels: list[Any]) -> tuple[float, int]:
    group_indices: dict[Any, list[int]] = {}
    for idx, label in enumerate(labels):
        group_indices.setdefault(label, []).append(idx)
    total_sum = 0.0
    total_count = 0
    for indices in group_indices.values():
        if len(indices) < 2:
            continue
        subset = normalized_vectors.index_select(0, torch.tensor(indices, dtype=torch.long))
        pair_sum, pair_count = _pair_sum(subset)
        total_sum += pair_sum
        total_count += pair_count
    return total_sum, total_count


def build_worst_grid(
    dataset: FontImageDataset,
    trainer,
    worst_specs: list[SampleSpec],
    metrics_df: pd.DataFrame,
    *,
    cell_size: int,
    inference_steps: int,
    checkpoint_alias: str,
    rank_by: str,
    output_path: Path,
) -> None:
    if not worst_specs:
        return
    batch = prepare_batch(dataset, worst_specs)
    generation = trainer.sample(
        batch["content"],
        content_index=batch["content_index"],
        style_img=batch["style_img"],
        style_index=batch["style_index"],
        style_ref_mask=batch["style_ref_mask"],
        num_inference_steps=inference_steps,
    ).cpu()
    target = batch["target"].cpu()
    content_unique = batch["content"].cpu()
    content_index = batch["content_index"].cpu()
    style_img = batch["style_img"].cpu()

    label_width = 380
    header_height = 40
    columns = ["content", "style1", "style2", "style3", "style4", "style5", "style6", "target", "gen"]
    width = label_width + len(columns) * cell_size
    height = header_height + len(worst_specs) * cell_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for col_idx, label in enumerate(columns):
        draw.text((label_width + col_idx * cell_size + 4, 10), label, fill=(0, 0, 0), font=font)

    metric_rows = {
        f"{row['font_name']}::{row['char']}": row
        for _, row in metrics_df.iterrows()
    }
    for row_idx, spec in enumerate(worst_specs):
        y0 = header_height + row_idx * cell_size
        row = metric_rows[spec.sample_id]
        info = (
            f"#{row_idx + 1} {spec.font_name}\n"
            f"split={spec.split} char_id={spec.char_id} rank={row[rank_by]:.6f}\n"
            f"mae={row['mae']:.4f} ssim={row['ssim']:.4f}"
        )
        draw.text((8, y0 + 10), info, fill=(0, 0, 0), font=small_font)
        images = [content_unique[content_index[row_idx]]]
        images.extend(style_img[row_idx, style_idx] for style_idx in range(int(style_img.size(1))))
        images.extend([target[row_idx], generation[row_idx]])
        for col_idx, image_tensor in enumerate(images):
            canvas.paste(tensor_to_pil(image_tensor, cell_size=cell_size), (label_width + col_idx * cell_size, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"[evaluate] saved worst-grid {checkpoint_alias} -> {output_path}", flush=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    *,
    run_metadata: dict[str, Any],
    summary_by_checkpoint: dict[str, dict[str, Any]],
    target_chars: list[str],
    style_pool_path: Path,
    worst_grid_paths: dict[str, str],
) -> None:
    lines: list[str] = []
    lines.append("# X-Pred Evaluation Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- run_dir: `{run_metadata['run_dir']}`")
    lines.append(f"- generated_at_utc: `{run_metadata['generated_at_utc']}`")
    lines.append(f"- device: `{run_metadata['device']}`")
    lines.append(f"- eval_seed: `{run_metadata['eval_seed']}`")
    lines.append(f"- fid_feature_dim: `{run_metadata['fid_feature']}`")
    lines.append(f"- target_chars ({len(target_chars)}): {' '.join(target_chars)}")
    lines.append(f"- style_pool_file: `{style_pool_path}`")
    lines.append(f"- style_ref_count_per_sample: `{run_metadata['style_ref_count']}`")
    lines.append(f"- fonts_evaluated: `{run_metadata['fonts_evaluated']}`")
    lines.append(f"- samples_per_checkpoint: `{run_metadata['samples_per_checkpoint']}`")
    lines.append("")

    overall_rows: list[list[str]] = []
    for checkpoint_alias in ["best", "final"]:
        summary = summary_by_checkpoint[checkpoint_alias]["summary"]["all"]
        style_stats = summary_by_checkpoint[checkpoint_alias]["style_vector_stats"]["all"]
        overall_rows.append(
            [
                checkpoint_alias,
                str(summary_by_checkpoint[checkpoint_alias]["step"]),
                str(summary["num_samples"]),
                f"{summary['fid']:.4f}",
                f"{summary['mae_mean']:.5f}",
                f"{summary['ssim_mean']:.5f}",
                f"{summary['psnr_mean']:.4f}",
                f"{summary['perceptor_perceptual_mean']:.5f}",
                f"{style_stats['same_font_pair_cos_mean']:.5f}",
                f"{style_stats['different_font_pair_cos_mean']:.5f}",
                f"{style_stats['same_font_margin']:.5f}",
            ]
        )
    lines.append("## Overall Comparison")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "checkpoint",
                "step",
                "samples",
                "FID",
                "MAE",
                "SSIM",
                "PSNR",
                "Perceptor",
                "StyleVec SameFont",
                "StyleVec DiffFont",
                "StyleVec Margin",
            ],
            overall_rows,
        )
    )
    lines.append("")

    split_rows: list[list[str]] = []
    for checkpoint_alias in ["best", "final"]:
        for split_name in ["train", "val"]:
            if split_name not in summary_by_checkpoint[checkpoint_alias]["summary"]:
                continue
            summary = summary_by_checkpoint[checkpoint_alias]["summary"][split_name]
            split_rows.append(
                [
                    checkpoint_alias,
                    split_name,
                    str(summary["num_samples"]),
                    f"{summary['fid']:.4f}" if "fid" in summary else "-",
                    f"{summary['mae_mean']:.5f}",
                    f"{summary['ssim_mean']:.5f}",
                    f"{summary['psnr_mean']:.4f}",
                    f"{summary['perceptor_perceptual_mean']:.5f}",
                ]
            )
    lines.append("## Split Breakdown")
    lines.append("")
    lines.append(
        markdown_table(
            ["checkpoint", "split", "samples", "FID", "MAE", "SSIM", "PSNR", "Perceptor"],
            split_rows,
        )
    )
    lines.append("")

    lines.append("## Worst 20 Visualizations")
    lines.append("")
    for checkpoint_alias in ["best", "final"]:
        lines.append(f"- {checkpoint_alias}: `{worst_grid_paths[checkpoint_alias]}`")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_checkpoint(
    *,
    checkpoint_alias: str,
    checkpoint_path: Path,
    checkpoint_step: int,
    dataset: FontImageDataset,
    sample_specs: list[SampleSpec],
    font_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    perceptor_model,
    output_dir: Path,
) -> dict[str, Any]:
    trainer = load_trainer(checkpoint_path, device)

    fid_by_split: dict[str, FrechetInceptionDistance] = {"all": init_fid(device, int(args.fid_feature))}

    rows: list[dict[str, Any]] = []

    grouped = specs_by_font(sample_specs)
    total_chunks = math.ceil(len(font_names) / int(args.fonts_per_batch))
    start_time = time.time()
    for chunk_idx, font_chunk in enumerate(chunk_list(font_names, int(args.fonts_per_batch)), start=1):
        batch_specs = [spec for font_name in font_chunk for spec in grouped[font_name]]
        batch = prepare_batch(dataset, batch_specs)
        generation = trainer.sample(
            batch["content"],
            content_index=batch["content_index"],
            style_img=batch["style_img"],
            style_index=batch["style_index"],
            style_ref_mask=batch["style_ref_mask"],
            num_inference_steps=int(args.inference_steps),
        )
        target = batch["target"].to(device)
        metrics = compute_image_metrics(generation, target, perceptor_model)
        split_batch_labels = [spec.split for spec in batch_specs]
        update_fids(fid_by_split, generation, target, split_batch_labels)

        for sample_idx, spec in enumerate(batch_specs):
            row = {
                "checkpoint": checkpoint_alias,
                "checkpoint_path": str(checkpoint_path),
                "step": int(checkpoint_step),
                "font_name": spec.font_name,
                "font_id": int(spec.font_id),
                "split": spec.split,
                "char": spec.char,
                "char_id": int(spec.char_id),
                "style_chars": "".join(spec.style_chars),
                "style_char_ids": ",".join(str(idx) for idx in spec.style_char_ids),
                "sample_id": spec.sample_id,
            }
            for metric_name, metric_values in metrics.items():
                value = float(metric_values[sample_idx].detach().cpu().item())
                row[metric_name] = value
            row["neg_ssim"] = float(-row["ssim"])
            rows.append(row)

        elapsed = time.time() - start_time
        print(
            f"[evaluate] {checkpoint_alias} chunk={chunk_idx}/{total_chunks} "
            f"fonts={len(font_chunk)} samples={len(batch_specs)} elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    fid_results = {split_name: float(compute_fid_from_metric(metric)) for split_name, metric in fid_by_split.items()}
    print(f"[evaluate] {checkpoint_alias} fid_all={fid_results['all']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.sort_values(["split", "font_name", "char_id"], inplace=True, ignore_index=True)
    csv_path = output_dir / f"{checkpoint_alias}_per_sample_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = summarize_metrics(df, fid_results)

    worst_df = df.sort_values(args.worst_rank_by, ascending=False).head(20).copy()
    worst_specs_lookup = {spec.sample_id: spec for spec in sample_specs}
    worst_specs = [worst_specs_lookup[sample_id] for sample_id in worst_df["sample_id"].tolist()]
    worst_grid_path = output_dir / f"{checkpoint_alias}_worst20_{args.worst_rank_by}.png"
    build_worst_grid(
        dataset,
        trainer,
        worst_specs,
        worst_df,
        cell_size=int(args.cell_size),
        inference_steps=int(args.inference_steps),
        checkpoint_alias=checkpoint_alias,
        rank_by=args.worst_rank_by,
        output_path=worst_grid_path,
    )

    result = {
        "checkpoint": checkpoint_alias,
        "checkpoint_path": str(checkpoint_path),
        "step": int(checkpoint_step),
        "summary": summary,
        "per_sample_metrics_csv": str(csv_path),
        "worst_grid_path": str(worst_grid_path),
        "worst_samples": worst_df.to_dict(orient="records"),
    }
    save_json(output_dir / f"{checkpoint_alias}_summary.json", result)

    del trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    train_config = read_json(args.train_config.resolve())
    run_paths = resolve_run_paths(args, train_config)
    output_dir = run_paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if args.inference_steps is None:
        args.inference_steps = int(train_config["sample_steps"])

    root = Path(train_config["data_root"]).resolve()
    style_pool_path = (
        args.style_pool_file.resolve()
        if args.style_pool_file is not None
        else (root / "fontprocessing" / "outputs_notosanssc_3500_k16_final" / "reference_400.txt")
    )
    style_pool_chars = load_style_pool(style_pool_path)
    dataset, train_fonts, val_fonts = load_datasets(root, train_config)

    font_names = sorted(dataset.font_names)
    if int(args.limit_fonts) > 0:
        font_names = font_names[: int(args.limit_fonts)]
    target_chars = sample_target_chars(
        dataset,
        num_target_chars=int(args.num_target_chars),
        eval_seed=int(args.eval_seed),
        target_char_source=str(args.target_char_source),
        style_pool_chars=style_pool_chars,
    )
    sample_specs = build_sample_specs(
        dataset,
        font_names=font_names,
        target_chars=target_chars,
        style_pool_chars=style_pool_chars,
        style_ref_count=int(args.style_ref_count),
        eval_seed=int(args.eval_seed),
        train_fonts=train_fonts,
        val_fonts=val_fonts,
    )
    save_json(output_dir / "sample_specs.json", [asdict(spec) for spec in sample_specs])

    resolved_perceptor_checkpoint = args.perceptor_checkpoint
    if resolved_perceptor_checkpoint is None:
        train_config_perceptor = train_config.get("perceptor_checkpoint")
        if train_config_perceptor is None:
            raise KeyError(
                "Missing perceptor checkpoint. Pass --perceptor-checkpoint explicitly because train_config.json no longer stores it."
            )
        resolved_perceptor_checkpoint = Path(train_config_perceptor)
    perceptor_checkpoint = resolved_perceptor_checkpoint.resolve()
    perceptor_model, _, perceptor_report = load_font_perceptor_from_checkpoint(perceptor_checkpoint, map_location="cpu")
    perceptor_model = perceptor_model.to(device).eval()
    for param in perceptor_model.parameters():
        param.requires_grad_(False)

    results: dict[str, dict[str, Any]] = {}
    checkpoint_meta = {
        "best": {
            "path": run_paths["best_checkpoint"],
            "step": infer_checkpoint_step("best", run_paths["best_checkpoint"], run_paths["run_dir"], train_config),
        },
        "final": {
            "path": run_paths["final_checkpoint"],
            "step": infer_checkpoint_step("final", run_paths["final_checkpoint"], run_paths["run_dir"], train_config),
        },
    }

    for checkpoint_alias in ["best", "final"]:
        meta = checkpoint_meta[checkpoint_alias]
        results[checkpoint_alias] = evaluate_checkpoint(
            checkpoint_alias=checkpoint_alias,
            checkpoint_path=meta["path"],
            checkpoint_step=int(meta["step"]),
            dataset=dataset,
            sample_specs=sample_specs,
            font_names=font_names,
            args=args,
            device=device,
            perceptor_model=perceptor_model,
            output_dir=output_dir,
        )

    generated_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_payload = {
        "run_dir": str(run_paths["run_dir"]),
        "train_config_path": str(args.train_config.resolve()),
        "device": str(device),
        "generated_at_utc": generated_at_utc,
        "eval_seed": int(args.eval_seed),
        "inference_steps": int(args.inference_steps),
        "fid_feature": int(args.fid_feature),
        "style_ref_count": int(args.style_ref_count),
        "target_char_source": str(args.target_char_source),
        "target_chars": target_chars,
        "fonts_evaluated": int(len(font_names)),
        "samples_per_checkpoint": int(len(sample_specs)),
        "style_pool_path": str(style_pool_path),
        "style_pool_count": int(len(style_pool_chars)),
        "perceptor_checkpoint": str(perceptor_checkpoint),
        "perceptor_report": perceptor_report,
        "checkpoints": {
            alias: {"path": str(meta["path"]), "step": int(meta["step"])}
            for alias, meta in checkpoint_meta.items()
        },
        "results": results,
    }
    save_json(output_dir / "report.json", report_payload)
    write_markdown_report(
        output_path=output_dir / "report.md",
        run_metadata=report_payload,
        summary_by_checkpoint=results,
        target_chars=target_chars,
        style_pool_path=style_pool_path,
        worst_grid_paths={alias: results[alias]["worst_grid_path"] for alias in ["best", "final"]},
    )
    print(f"[evaluate] report_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
