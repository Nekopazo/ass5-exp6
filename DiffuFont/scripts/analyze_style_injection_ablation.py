#!/usr/bin/env python3
"""Analyze style injection sites with attention heatmaps and single-site ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import FontImageDataset
from models.model import DiffusionTrainer, FlowMatchingTrainer
from models.source_part_ref_unet import SourcePartRefUNet
from style_augment import build_base_glyph_transform, build_style_reference_transform
from train import resolve_device, set_global_seed, split_indices_by_font


SITE_TO_TOKEN = {
    "mid": 0,
    "up_16": 1,
    "up_32": 2,
}


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")

    def _step_of(path: Path) -> int:
        m = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_step_of)


def _safe_name(text: str) -> str:
    out = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", str(text))
    return out.strip("_") or "sample"


def _tensor_gray_to_u8(x: torch.Tensor) -> np.ndarray:
    t = x.detach().to(dtype=torch.float32, device="cpu")
    if t.dim() == 3:
        t = t.squeeze(0)
    arr = ((t.clamp(-1.0, 1.0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return arr


def _mask_to_u8(x: torch.Tensor) -> np.ndarray:
    t = x.detach().to(dtype=torch.float32, device="cpu")
    arr = (t.clamp(0.0, 1.0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr


def _float_map_to_u8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _make_overlay(base_u8: np.ndarray, heat_u8: np.ndarray, title: str, size: int = 256) -> np.ndarray:
    base = cv2.resize(base_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    heat = cv2.resize(heat_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.45, color, 0.55, 0.0)
    cv2.putText(overlay, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _stack_images_h(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _stack_images_v(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)


def _style_preview(style_img: torch.Tensor, style_ref_mask: torch.Tensor) -> np.ndarray:
    refs = style_img.detach().to(dtype=torch.float32, device="cpu")
    mask = style_ref_mask.detach().to(dtype=torch.float32, device="cpu")
    w = mask / mask.sum().clamp_min(1.0)
    avg = (refs * w.view(-1, 1, 1, 1)).sum(dim=0)
    return _tensor_gray_to_u8(avg)


def _aggregate_site_heatmaps(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    # token_attn: (3, R, 32, 32)
    mask = style_ref_mask.detach().to(dtype=torch.float32, device="cpu")
    weights = mask / mask.sum().clamp_min(1.0)
    token_maps = []
    for token_idx in range(int(token_attn.size(0))):
        weighted = (token_attn[token_idx].detach().to(dtype=torch.float32, device="cpu") * weights.view(-1, 1, 1)).sum(dim=0)
        weighted = weighted / weighted.max().clamp_min(1e-8)
        token_maps.append(_mask_to_u8(weighted))
    blank = np.zeros_like(token_maps[0], dtype=np.uint8)
    return {
        site: (blank if token_idx is None else token_maps[token_idx])
        for site, token_idx in SITE_TO_TOKEN.items()
    }


def _aggregate_site_heatmaps_float(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    mask = style_ref_mask.detach().to(dtype=torch.float32, device="cpu")
    weights = mask / mask.sum().clamp_min(1.0)
    token_maps: list[np.ndarray] = []
    for token_idx in range(int(token_attn.size(0))):
        weighted = (token_attn[token_idx].detach().to(dtype=torch.float32, device="cpu") * weights.view(-1, 1, 1)).sum(dim=0)
        weighted = weighted / weighted.max().clamp_min(1e-8)
        token_maps.append(weighted.numpy().astype(np.float32))
    return {
        site: token_maps[token_idx]
        for site, token_idx in SITE_TO_TOKEN.items()
    }


def _build_heatmap_grid(
    style_preview_u8: np.ndarray,
    site_heatmaps: dict[str, np.ndarray],
) -> np.ndarray:
    tiles = [_make_overlay(style_preview_u8, site_heatmaps[site], site) for site in ("mid", "up_16", "up_32")]
    return _stack_images_h(tiles)


def _build_gen_panel(
    content_u8: np.ndarray,
    target_u8: np.ndarray,
    style_u8: np.ndarray,
    outputs: dict[str, np.ndarray],
    size: int = 256,
) -> np.ndarray:
    def _tile(img_u8: np.ndarray, title: str) -> np.ndarray:
        img = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas

    top = _stack_images_h([
        _tile(content_u8, "content"),
        _tile(style_u8, "style_avg"),
        _tile(target_u8, "target"),
    ])
    bottom = _stack_images_h([
        _tile(outputs["full"], "full"),
        _tile(outputs["drop_mid"], "drop_mid"),
        _tile(outputs["drop_up_16"], "drop_up_16"),
        _tile(outputs["drop_up_32"], "drop_up_32"),
    ])
    if top.shape[1] < bottom.shape[1]:
        pad = np.full((top.shape[0], bottom.shape[1] - top.shape[1], 3), 255, dtype=np.uint8)
        top = np.concatenate([top, pad], axis=1)
    return _stack_images_v([top, bottom])


def _mean_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(x - y)).item())


def _mean_sq(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((x - y) ** 2).item())


def _normalize_heatmap_prob(arr: np.ndarray) -> np.ndarray:
    flat = np.clip(arr.astype(np.float64), 0.0, None)
    total = float(flat.sum())
    if total <= 1e-12:
        return np.full_like(flat, 1.0 / flat.size, dtype=np.float64)
    return flat / total


def _heatmap_entropy(arr: np.ndarray) -> float:
    prob = _normalize_heatmap_prob(arr)
    ent = -np.sum(prob * np.log(np.clip(prob, 1e-12, None)))
    return float(ent / math.log(prob.size))


def _heatmap_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _heatmap_overlap(a: np.ndarray, b: np.ndarray) -> float:
    pa = _normalize_heatmap_prob(a)
    pb = _normalize_heatmap_prob(b)
    return float(np.minimum(pa, pb).sum())


def _mean_std_dict(entries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics = sorted(entries[0].keys()) if entries else []
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = np.asarray([entry[metric] for entry in entries], dtype=np.float64)
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


class AttentionStatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.mean_maps = {site: np.zeros((32, 32), dtype=np.float64) for site in SITE_TO_TOKEN}
        self.entropy = {site: [] for site in SITE_TO_TOKEN}
        self.pair_cosine = {"mid__up_16": [], "mid__up_32": [], "up_16__up_32": []}
        self.pair_overlap = {"mid__up_16": [], "mid__up_32": [], "up_16__up_32": []}

    def update(self, site_maps: dict[str, np.ndarray]) -> None:
        self.count += 1
        for site, arr in site_maps.items():
            self.mean_maps[site] += arr.astype(np.float64)
            self.entropy[site].append(_heatmap_entropy(arr))
        pairs = [("mid", "up_16"), ("mid", "up_32"), ("up_16", "up_32")]
        for a, b in pairs:
            key = f"{a}__{b}"
            self.pair_cosine[key].append(_heatmap_cosine(site_maps[a], site_maps[b]))
            self.pair_overlap[key].append(_heatmap_overlap(site_maps[a], site_maps[b]))

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("attention accumulator is empty")
        mean_maps = {
            site: (arr / float(self.count)).astype(np.float32)
            for site, arr in self.mean_maps.items()
        }
        return {
            "count": self.count,
            "entropy": {
                site: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                }
                for site, vals in self.entropy.items()
            },
            "pair_cosine": {
                key: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                }
                for key, vals in self.pair_cosine.items()
            },
            "pair_overlap": {
                key: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                }
                for key, vals in self.pair_overlap.items()
            },
            "mean_maps": mean_maps,
        }


def _write_attention_summary(out_dir: Path, attention_summary: dict[str, Any]) -> dict[str, str]:
    mean_maps = attention_summary.pop("mean_maps")
    tiles = [_make_overlay(np.full((32, 32), 255, dtype=np.uint8), _float_map_to_u8(mean_maps[site]), f"mean_{site}") for site in ("mid", "up_16", "up_32")]
    heatmap_grid = _stack_images_h(tiles)
    avg_heatmap_path = out_dir / "attention_mean_heatmaps.png"
    cv2.imwrite(str(avg_heatmap_path), heatmap_grid)

    npz_path = out_dir / "attention_mean_maps.npz"
    np.savez_compressed(npz_path, **mean_maps)

    json_path = out_dir / "attention_stats.json"
    json_path.write_text(json.dumps(attention_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "attention_stats_json": str(json_path.resolve()),
        "attention_mean_heatmaps": str(avg_heatmap_path.resolve()),
        "attention_mean_maps_npz": str(npz_path.resolve()),
    }


def _run_sample(
    trainer: DiffusionTrainer | FlowMatchingTrainer,
    sample: dict[str, Any],
    out_dir: Path,
    sample_seed: int,
    inference_steps: int,
    repeat_seeds: int,
) -> dict[str, Any]:
    model = trainer.model
    content = sample["content"].unsqueeze(0).to(trainer.device)
    target = sample["input"].unsqueeze(0).to(trainer.device)
    style_img = sample["style_img"].unsqueeze(0).to(trainer.device)
    style_ref_mask = sample["style_ref_mask"].unsqueeze(0).to(trainer.device)

    with torch.no_grad():
        _, token_attn = model.encode_style_tokens_with_attention(style_img, style_ref_mask=style_ref_mask)
    site_heatmaps = _aggregate_site_heatmaps(token_attn[0], style_ref_mask[0])
    site_heatmaps_float = _aggregate_site_heatmaps_float(token_attn[0], style_ref_mask[0])

    def _sample_with_drop(drop_site: str | None, active_seed: int) -> torch.Tensor:
        original_force_keep = getattr(model, "_style_site_force_keep", None)
        try:
            if drop_site is None:
                model.set_style_site_force_keep(None)
            else:
                keep_sites = tuple(site for site in SITE_TO_TOKEN if site != drop_site)
                model.set_style_site_force_keep(keep_sites)
            torch.manual_seed(int(active_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(active_seed))
            return trainer.dpm_solver_sample(
                content_img=content,
                style_img=style_img,
                style_ref_mask=style_ref_mask,
                num_inference_steps=int(inference_steps),
                condition_mode="style_only",
            ).detach().cpu()[0]
        finally:
            if original_force_keep is None:
                model.set_style_site_force_keep(None)
            else:
                model.set_style_site_force_keep(tuple(sorted(original_force_keep)))

    outputs_repeats: dict[str, list[torch.Tensor]] = {k: [] for k in ("full", "drop_mid", "drop_up_16", "drop_up_32")}
    for repeat_idx in range(max(1, int(repeat_seeds))):
        seed_base = int(sample_seed) + repeat_idx * 100003
        outputs_repeats["full"].append(_sample_with_drop(None, seed_base + 11))
        outputs_repeats["drop_mid"].append(_sample_with_drop("mid", seed_base + 23))
        outputs_repeats["drop_up_16"].append(_sample_with_drop("up_16", seed_base + 37))
        outputs_repeats["drop_up_32"].append(_sample_with_drop("up_32", seed_base + 53))

    outputs_t = {
        key: torch.stack(vals, dim=0).mean(dim=0)
        for key, vals in outputs_repeats.items()
    }

    content_u8 = _tensor_gray_to_u8(content[0])
    target_u8 = _tensor_gray_to_u8(target[0])
    style_u8 = _style_preview(style_img[0], style_ref_mask[0])
    outputs_u8 = {k: _tensor_gray_to_u8(v) for k, v in outputs_t.items()}

    heatmap_grid = _build_heatmap_grid(style_u8, site_heatmaps)
    gen_grid = _build_gen_panel(content_u8, target_u8, style_u8, outputs_u8)

    sample_name = _safe_name(f"{sample['font']}__{sample['char']}")
    cv2.imwrite(str(out_dir / "heatmaps" / f"{sample_name}.png"), heatmap_grid)
    cv2.imwrite(str(out_dir / "generations" / f"{sample_name}.png"), gen_grid)

    metrics: dict[str, dict[str, float]] = {}
    for key, preds in outputs_repeats.items():
        per_repeat_metrics = [
            {
                "pixel_mse": _mean_sq(pred, target[0].cpu()),
                "pixel_l1": _mean_abs(pred, target[0].cpu()),
            }
            for pred in preds
        ]
        stats = _mean_std_dict(per_repeat_metrics)
        metrics[key] = {
            "pixel_mse": stats["pixel_mse"]["mean"],
            "pixel_l1": stats["pixel_l1"]["mean"],
            "pixel_mse_std": stats["pixel_mse"]["std"],
            "pixel_l1_std": stats["pixel_l1"]["std"],
        }

    return {
        "sample_name": sample_name,
        "font": sample["font"],
        "char": sample["char"],
        "metrics": metrics,
        "repeat_seeds": int(repeat_seeds),
        "attention_entropy": {
            site: _heatmap_entropy(arr)
            for site, arr in site_heatmaps_float.items()
        },
        "heatmap_path": str((out_dir / "heatmaps" / f"{sample_name}.png").resolve()),
        "generation_path": str((out_dir / "generations" / f"{sample_name}.png").resolve()),
        "site_heatmaps_float": site_heatmaps_float,
    }


def _build_trainer_from_config(cfg: dict[str, Any], device: torch.device, total_steps: int) -> DiffusionTrainer | FlowMatchingTrainer:
    mode = str(cfg.get("active_conditioning_mode") or cfg.get("teacher_line") or "style_only").strip().lower()
    model = SourcePartRefUNet(
        in_channels=1,
        image_size=int(cfg.get("image_size", 256)),
        content_start_channel=64,
        style_start_channel=int(cfg.get("style_start_channel", 16)),
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=4,
        channel_attn=True,
        conditioning_profile=mode,
        style_token_dim=int(cfg.get("style_token_dim", 256)),
        style_token_count=int(cfg.get("style_token_count", 3)),
    )

    trainer_cls = DiffusionTrainer if str(cfg.get("trainer", "diffusion")) == "diffusion" else FlowMatchingTrainer
    kwargs: dict[str, Any] = {
        "lr": float(cfg.get("lr", 2e-4)),
        "lambda_nce": float(cfg.get("lambda_nce", 0.0)),
        "lambda_slot_nce": float(cfg.get("lambda_slot_nce", 0.0)),
        "lambda_cons": float(cfg.get("lambda_cons", 0.0)),
        "lambda_div": float(cfg.get("lambda_div", 0.0)),
        "lambda_proxy_low": float(cfg.get("lambda_proxy_low", 0.0)),
        "lambda_proxy_mid": float(cfg.get("lambda_proxy_mid", 0.0)),
        "lambda_proxy_high": float(cfg.get("lambda_proxy_high", 0.0)),
        "lambda_attn_sep": float(cfg.get("lambda_attn_sep", 0.0)),
        "lambda_attn_order": float(cfg.get("lambda_attn_order", 0.0)),
        "lambda_attn_role": float(cfg.get("lambda_attn_role", 0.0)),
        "nce_temperature": float(cfg.get("style_nce_temp", 0.07)),
        "aux_loss_warmup_steps": int(cfg.get("aux_loss_warmup_steps", 0)),
        "attn_overlap_margin": float(cfg.get("attn_overlap_margin", 0.80)),
        "attn_entropy_gap": float(cfg.get("attn_entropy_gap", 0.03)),
        "T": int(cfg.get("diffusion_steps", 1000)),
        "total_steps": int(total_steps),
        "lr_warmup_steps": int(cfg.get("lr_warmup_steps", 0)),
        "lr_min_scale": float(cfg.get("lr_min_scale", 1e-3)),
        "save_every_steps": None,
        "log_every_steps": None,
        "detailed_log": False,
        "grad_accum_steps": 1,
        "conditioning_mode": mode,
        "part_drop_prob": 0.0,
        "style_ref_drop_prob": 0.0,
        "style_ref_drop_min_keep": 1,
        "style_site_drop_prob": float(cfg.get("style_site_drop_prob", 0.0)),
        "style_site_drop_min_keep": int(cfg.get("style_site_drop_min_keep", 1)),
        "freeze_part_encoder_steps": 0,
        "freeze_style_backbone_steps": 0,
        "style_backbone_lr_scale": float(cfg.get("style_backbone_lr_scale", 0.1)),
    }
    if trainer_cls is FlowMatchingTrainer:
        kwargs["lambda_fm"] = float(cfg.get("lambda_fm", 1.0))

    trainer = trainer_cls(model=model, device=device, **kwargs)
    trainer.model.eval()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze style injection site ablations.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--repeat-seeds", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "train_run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    checkpoint = _latest_checkpoint(run_dir)

    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = args.out_dir.resolve()
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    (out_dir / "generations").mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer_from_config(cfg, device=device, total_steps=max(1, int(cfg.get("total_steps", 1))))
    trainer.load(checkpoint)
    trainer.model.eval()

    glyph_transform = build_base_glyph_transform(image_size=128)
    style_transform = build_style_reference_transform(image_size=128)
    dataset = FontImageDataset(
        project_root=cfg.get("data_root", "."),
        max_fonts=int(cfg.get("max_fonts", 0)),
        use_style_image=bool(cfg.get("use_style_image", True)),
        use_part_bank=False,
        random_seed=int(cfg.get("seed", 42)),
        transform=glyph_transform,
        style_transform=style_transform,
        cache_style_image=False,
        style_ref_count=int(cfg.get("style_ref_count_active") or cfg.get("style_ref_count") or 1),
        reference_cluster_json=cfg.get("reference_cluster_json"),
    )
    train_indices, val_indices, split_stats = split_indices_by_font(
        dataset=dataset,
        val_ratio=float(cfg.get("val_ratio", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    chosen_pool = val_indices if args.split == "val" and val_indices else train_indices
    if not chosen_pool:
        raise RuntimeError("no samples available after split")

    sample_count = max(1, min(len(chosen_pool), int(args.num_samples)))
    rng = random.Random(int(args.seed))
    chosen = rng.sample(list(chosen_pool), k=sample_count)
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, float]]] = {}
    attention_acc = AttentionStatsAccumulator()

    for rank, sample_index in enumerate(chosen):
        sample = dataset[int(sample_index)]
        row = _run_sample(
            trainer=trainer,
            sample=sample,
            out_dir=out_dir,
            sample_seed=int(args.seed) + rank,
            inference_steps=int(args.inference_steps),
            repeat_seeds=int(args.repeat_seeds),
        )
        site_heatmaps_float = row.pop("site_heatmaps_float")
        attention_acc.update(site_heatmaps_float)
        rows.append(row)
        for condition, metrics in row["metrics"].items():
            aggregate.setdefault(condition, []).append(metrics)
        print(
            f"[analyze] sample={row['sample_name']} "
            + " ".join(
                f"{cond}:mse={row['metrics'][cond]['pixel_mse']:.4f},l1={row['metrics'][cond]['pixel_l1']:.4f}"
                for cond in ("full", "drop_mid", "drop_up_16", "drop_up_32")
            ),
            flush=True,
        )

    attention_summary = attention_acc.finalize()
    attention_paths = _write_attention_summary(out_dir, attention_summary)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "split": args.split,
        "repeat_seeds": int(args.repeat_seeds),
        "split_stats": split_stats,
        "num_samples": len(rows),
        "samples": rows,
        "aggregate_metrics": {
            condition: {
                metric: {
                    "mean": float(np.mean([entry[metric] for entry in values])),
                    "std": float(np.std([entry[metric] for entry in values], ddof=0)),
                }
                for metric in ("pixel_mse", "pixel_l1", "pixel_mse_std", "pixel_l1_std")
            }
            for condition, values in aggregate.items()
        },
        "attention_summary": attention_summary,
        "attention_paths": attention_paths,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
