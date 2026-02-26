#!/usr/bin/env python3
"""Contrastive pretraining for the PartBank encoder head.

Trains part_patch_encoder, part_set_norm, style_token_mlp, and contrastive_head
with the **same architecture** as SourcePartRefUNet so that the pretrained
weights can be loaded directly into the full model.

Input:  DataPreparation/PartBank/manifest.json
Output: checkpoint with keys that map 1-to-1 to SourcePartRefUNet state dict.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))


# ---------------------------------------------------------------------------
# Lightweight replica of the part-encoder from SourcePartRefUNet
# ---------------------------------------------------------------------------

class PartEncoderModule(nn.Module):
    """Minimal standalone module matching SourcePartRefUNet part-related layers."""

    def __init__(
        self,
        in_channels: int = 1,
        style_token_dim: int = 256,
        style_token_count: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.style_token_dim = style_token_dim
        self.style_token_count = style_token_count

        # Same architecture as SourcePartRefUNet.part_patch_encoder
        self.part_patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, style_token_dim),
        )
        self.part_set_norm = nn.LayerNorm(style_token_dim)
        self.style_token_mlp = nn.Sequential(
            nn.Linear(style_token_dim, style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(style_token_dim * 4, style_token_count * style_token_dim),
        )
        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(style_token_dim),
            nn.Linear(style_token_dim, style_token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(style_token_dim, style_token_dim),
        )

    def encode_contrastive_z(
        self,
        part_imgs: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        """(B, P, C, H, W) + mask (B, P) -> L2-normalised embedding (B, D)."""
        b, p, c, h, w = part_imgs.shape
        x = part_imgs.view(b * p, c, h, w)
        z = self.part_patch_encoder(x).view(b, p, self.style_token_dim)  # (B, P, D)

        mask_f = part_mask.to(dtype=z.dtype, device=z.device).unsqueeze(-1)  # (B, P, 1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = (z * mask_f).sum(dim=1) / denom  # (B, D)
        pooled = self.part_set_norm(pooled)

        tokens = self.style_token_mlp(pooled).view(b, self.style_token_count, self.style_token_dim)
        pooled2 = tokens.mean(dim=1)  # (B, D)
        out = self.contrastive_head(pooled2)  # (B, D)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_part_bank(manifest_path: Path, root: Path) -> Dict[str, List[Path]]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    fonts = obj.get("fonts", {})
    out: Dict[str, List[Path]] = {}
    for font_name, info in fonts.items():
        parts = info.get("parts", [])
        paths: List[Path] = []
        for p in parts:
            rel = p.get("path")
            if not rel:
                continue
            fp = (root / rel).resolve()
            if fp.exists():
                paths.append(fp)
        if paths:
            out[font_name] = paths
    return out


def load_part_image(path: Path, patch_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((patch_size, patch_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # match main training normalization range [-1, 1]
    return torch.from_numpy(arr)[None, :, :].contiguous()  # (1, H, W)


def sample_paths(paths: List[Path], n: int, rng: random.Random) -> Tuple[List[Path], List[float]]:
    """Return n paths (with replacement if needed) and corresponding mask."""
    if len(paths) >= n:
        chosen = rng.sample(paths, n)
    else:
        chosen = list(paths)
        while len(chosen) < n:
            chosen.append(rng.choice(paths))
    mask = [1.0] * len(chosen)
    return chosen, mask


def split_parts_within_font(
    bank: Dict[str, List[Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Split each font's part paths into train/val by ratio (e.g. 80/20)."""
    if val_ratio <= 0.0:
        return {k: list(v) for k, v in bank.items()}, {}

    train_bank: Dict[str, List[Path]] = {}
    val_bank: Dict[str, List[Path]] = {}
    for i, (font, paths) in enumerate(sorted(bank.items(), key=lambda x: x[0])):
        pool = list(paths)
        if len(pool) < 2:
            train_bank[font] = pool
            continue
        rng = random.Random(seed + 1009 + i * 17)
        rng.shuffle(pool)
        val_n = int(round(len(pool) * float(val_ratio)))
        val_n = max(1, min(len(pool) - 1, val_n))
        val_split = pool[:val_n]
        train_split = pool[val_n:]
        if not train_split:
            train_split = [val_split.pop()]
        train_bank[font] = train_split
        if val_split:
            val_bank[font] = val_split
    return train_bank, val_bank


def resolve_path(root: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (root / path_value).resolve()


def should_maximize(metric_name: str) -> bool:
    return metric_name.endswith("_acc")


def is_improved(metric_name: str, new_value: float, best_value: float | None, min_delta: float) -> bool:
    if best_value is None:
        return True
    if should_maximize(metric_name):
        return new_value > (best_value + min_delta)
    return new_value < (best_value - min_delta)


def make_checkpoint(
    encoder: PartEncoderModule,
    args: argparse.Namespace,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "part_encoder": encoder.state_dict(),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "extra": extra,
    }


def run_one_batch(
    encoder: PartEncoderModule,
    bank: Dict[str, List[Path]],
    font_pool: List[str],
    batch_size: int,
    min_k: int,
    max_k: int,
    patch_size: int,
    temperature: float,
    rng: random.Random,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    if len(font_pool) >= batch_size:
        batch_fonts = rng.sample(font_pool, batch_size)
    else:
        batch_fonts = [rng.choice(font_pool) for _ in range(batch_size)]

    imgs1_list: List[torch.Tensor] = []
    mask1_list: List[torch.Tensor] = []
    imgs2_list: List[torch.Tensor] = []
    mask2_list: List[torch.Tensor] = []

    for font in batch_fonts:
        part_paths = bank[font]
        ub = min(max_k, len(part_paths))
        lb = min(min_k, ub)
        n1 = rng.randint(lb, ub)
        n2 = rng.randint(lb, ub)

        set1, m1 = sample_paths(part_paths, n1, rng)
        set2, m2 = sample_paths(part_paths, n2, rng)

        # Pad to max_k so we can batch
        while len(set1) < max_k:
            set1.append(set1[0])
            m1.append(0.0)
        while len(set2) < max_k:
            set2.append(set2[0])
            m2.append(0.0)

        imgs1_list.append(torch.stack([load_part_image(p, patch_size) for p in set1]))
        mask1_list.append(torch.tensor(m1, dtype=torch.float32))
        imgs2_list.append(torch.stack([load_part_image(p, patch_size) for p in set2]))
        mask2_list.append(torch.tensor(m2, dtype=torch.float32))

    imgs1 = torch.stack(imgs1_list).to(device)  # (B, max_k, C, H, W)
    mask1 = torch.stack(mask1_list).to(device)   # (B, max_k)
    imgs2 = torch.stack(imgs2_list).to(device)
    mask2 = torch.stack(mask2_list).to(device)

    z1 = encoder.encode_contrastive_z(imgs1, mask1)  # (B, D)
    z2 = encoder.encode_contrastive_z(imgs2, mask2)

    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(logits.size(0), device=device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_12 + loss_21)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
    return float(loss.item()), float(acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("DataPreparation/PartBank/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/part_style_encoder_pretrain.pt"))
    parser.add_argument("--best-out", type=Path, default=Path("checkpoints/part_style_encoder_pretrain_best.pt"))
    parser.add_argument("--log-file", type=Path, default=Path("checkpoints/part_style_encoder_pretrain.log"))
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("checkpoints/part_style_encoder_pretrain.metrics.jsonl"))

    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-set-size", type=int, default=1)
    parser.add_argument("--max-set-size", type=int, default=8)
    parser.add_argument("--warmup-max-set-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_acc", "train_loss", "train_acc"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=0, help="In units of log events. 0 disables early stop.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--style-token-count", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = resolve_path(root, args.manifest)
    out_path = resolve_path(root, args.out)
    best_out_path = resolve_path(root, args.best_out)
    log_path = resolve_path(root, args.log_file)
    metrics_path = resolve_path(root, args.metrics_jsonl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 17)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bank = load_part_bank(manifest_path, root)
    # keep fonts with enough parts for two random views
    bank = {k: v for k, v in bank.items() if len(v) >= max(2, args.min_set_size)}
    if len(bank) < 2:
        raise RuntimeError("Part bank has too few fonts for contrastive pretraining.")

    font_names = sorted(bank.keys())
    train_bank, val_bank = split_parts_within_font(bank, args.val_ratio, args.seed)
    has_val = len(val_bank) >= 2

    if len(train_bank) < 2:
        raise RuntimeError("Train split has too few fonts for contrastive pretraining.")

    monitor_name = args.monitor
    if monitor_name.startswith("val_") and not has_val:
        monitor_name = "train_loss"

    log_fp = log_path.open("w", encoding="utf-8")
    metrics_fp = metrics_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log(
        f"Loaded part bank: total_fonts={len(font_names)} train_fonts={len(train_bank)} val_fonts={len(val_bank)} "
        f"split=per-font-part(1-{int((1.0-args.val_ratio)*100)}/{int(args.val_ratio*100)})"
    )
    log(f"Using device={device}, monitor={monitor_name}, steps={args.steps}, batch_size={args.batch_size}")

    min_k = max(1, args.min_set_size)
    final_max_k = max(min_k, args.max_set_size)
    warmup_max_k = max(min_k, min(args.warmup_max_set_size, final_max_k))
    log(
        f"Part-set size schedule: min={min_k}, warmup_max={warmup_max_k}, "
        f"final_max={final_max_k}, warmup_steps={args.warmup_steps}"
    )

    encoder = PartEncoderModule(
        in_channels=1,
        style_token_dim=args.style_token_dim,
        style_token_count=args.style_token_count,
    ).to(device)

    params = list(encoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    best_metric: float | None = None
    best_step = 0
    no_improve_count = 0
    last_step = 0

    try:
        encoder.train()
        for step in range(1, args.steps + 1):
            current_max_k = warmup_max_k if step <= args.warmup_steps else final_max_k

            train_loss, train_acc = run_one_batch(
                encoder=encoder,
                bank=train_bank,
                font_pool=list(train_bank.keys()),
                batch_size=args.batch_size,
                min_k=min_k,
                max_k=current_max_k,
                patch_size=args.patch_size,
                temperature=args.temperature,
                rng=train_rng,
                device=device,
                opt=opt,
            )
            last_step = step

            should_log = (step % args.log_every == 0) or (step == 1) or (step == args.steps)
            if not should_log:
                continue

            val_loss = None
            val_acc = None
            if has_val:
                encoder.eval()
                val_losses: List[float] = []
                val_accs: List[float] = []
                with torch.no_grad():
                    for _ in range(max(1, args.val_batches)):
                        vl, va = run_one_batch(
                            encoder=encoder,
                            bank=val_bank,
                            font_pool=list(val_bank.keys()),
                            batch_size=min(args.batch_size, len(val_bank)),
                            min_k=min_k,
                            max_k=current_max_k,
                            patch_size=args.patch_size,
                            temperature=args.temperature,
                            rng=eval_rng,
                            device=device,
                            opt=None,
                        )
                        val_losses.append(vl)
                        val_accs.append(va)
                val_loss = float(np.mean(val_losses))
                val_acc = float(np.mean(val_accs))
                encoder.train()

            metrics = {
                "step": step,
                "max_set_size": current_max_k,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            metrics_fp.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metrics_fp.flush()

            current_metric = metrics.get(monitor_name)
            if current_metric is None:
                current_metric = train_loss if monitor_name.endswith("loss") else train_acc

            improved = is_improved(
                metric_name=monitor_name,
                new_value=float(current_metric),
                best_value=best_metric,
                min_delta=args.early_stop_min_delta,
            )

            if improved:
                best_metric = float(current_metric)
                best_step = step
                no_improve_count = 0
                best_ckpt = make_checkpoint(
                    encoder=encoder,
                    args=args,
                    extra={
                        "step": step,
                        "best_step": best_step,
                        "best_metric_name": monitor_name,
                        "best_metric_value": best_metric,
                        "metrics": metrics,
                        "split": {
                            "train_fonts": len(train_bank),
                            "val_fonts": len(val_bank),
                        },
                    },
                )
                torch.save(best_ckpt, best_out_path)
            else:
                no_improve_count += 1

            msg = (
                f"step={step:05d} "
                f"max_set={current_max_k} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
            )
            if val_loss is not None and val_acc is not None:
                msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            msg += (
                f" monitor={monitor_name}:{float(current_metric):.4f} "
                f"best_step={best_step} best={best_metric:.4f} "
                f"patience={no_improve_count}/{args.early_stop_patience}"
            )
            log(msg)

            if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                log(f"Early stop at step={step}, no improvement for {no_improve_count} log events.")
                break
    finally:
        final_ckpt = make_checkpoint(
            encoder=encoder,
            args=args,
            extra={
                "step": last_step,
                "best_step": best_step,
                "best_metric_name": monitor_name,
                "best_metric_value": best_metric,
                "split": {
                    "train_fonts": len(train_bank),
                    "val_fonts": len(val_bank),
                },
            },
        )
        torch.save(final_ckpt, out_path)
        log(f"Saved final checkpoint: {out_path}")
        if best_step > 0:
            log(f"Saved best checkpoint: {best_out_path} ({monitor_name}={best_metric:.4f} @ step={best_step})")
        else:
            log("Best checkpoint was not updated during training.")
        log(f"Metrics jsonl: {metrics_path}")
        log_fp.close()
        metrics_fp.close()


if __name__ == "__main__":
    main()
