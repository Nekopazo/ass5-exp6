#!/usr/bin/env python3
"""Pretrain E_p as a font classifier on full glyph images.

E_p pipeline:
- Backbone CNN (ResNet-18/34 or LightCNN)
- Global average pooling inside backbone
- Linear -> font logits
- Softmax for p(font|x_s)

Training loss: CrossEntropy (optional label smoothing).
"""

from __future__ import annotations

import argparse
import io
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import sys

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from models.style_encoders import FontClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def scan_lmdb_font_chars(
    env: lmdb.Environment,
    scan_limit: int = 0,
    min_chars_per_font: int = 8,
) -> Dict[str, List[str]]:
    chars_by_font: Dict[str, set[str]] = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, kv in enumerate(cursor):
            key = kv[0]
            if b"@" not in key:
                continue
            try:
                k = key.decode("utf-8")
            except Exception:
                continue
            font, ch = k.split("@", 1)
            if len(ch) != 1:
                continue
            chars_by_font.setdefault(font, set()).add(ch)
            if scan_limit > 0 and i >= scan_limit:
                break
    out = {f: sorted(list(chars)) for f, chars in chars_by_font.items() if len(chars) >= int(min_chars_per_font)}
    return out


def split_chars_within_font(
    chars_by_font: Dict[str, List[str]],
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split each font's chars into train/val by ratio (e.g. 80/20)."""
    if val_ratio <= 0.0:
        return {k: list(v) for k, v in chars_by_font.items()}, {}

    train_chars: Dict[str, List[str]] = {}
    val_chars: Dict[str, List[str]] = {}
    for i, (font, chars) in enumerate(sorted(chars_by_font.items(), key=lambda x: x[0])):
        pool = list(chars)
        if len(pool) < 2:
            train_chars[font] = pool
            continue
        rng = random.Random(seed + 1009 + i * 17)
        rng.shuffle(pool)
        val_n = int(round(len(pool) * float(val_ratio)))
        val_n = max(1, min(len(pool) - 1, val_n))
        val_split = sorted(pool[:val_n])
        train_split = sorted(pool[val_n:])
        if not train_split:
            train_split = [val_split.pop()]
        if val_split:
            val_chars[font] = val_split
        train_chars[font] = train_split
    return train_chars, val_chars


def pick_font_classes(chars_by_font: Dict[str, List[str]], num_classes: int) -> List[str]:
    scored = sorted(chars_by_font.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)
    if int(num_classes) <= 0:
        k = len(scored)
    else:
        k = min(len(scored), max(2, int(num_classes)))
    return sorted([x[0] for x in scored[:k]])


def load_glyph_image(env: lmdb.Environment, font_name: str, ch: str, transform) -> torch.Tensor:
    key = f"{font_name}@{ch}".encode("utf-8")
    with env.begin() as txn:
        b = txn.get(key)
    if b is None:
        raise KeyError(f"Missing glyph in lmdb: {font_name}@{ch}")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return transform(img)


def run_batch(
    model: FontClassifier,
    env: lmdb.Environment,
    fonts: Sequence[str],
    chars_by_font: Dict[str, List[str]],
    font_to_label: Dict[str, int],
    batch_size: int,
    transform,
    rng: random.Random,
    device: torch.device,
    label_smoothing: float,
    opt: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    if not fonts:
        raise RuntimeError("empty fonts for batch")
    picked_fonts = [rng.choice(list(fonts)) for _ in range(batch_size)]

    x_list: List[torch.Tensor] = []
    y_list: List[int] = []
    for font in picked_fonts:
        chars = chars_by_font[font]
        ch = rng.choice(chars)
        x_list.append(load_glyph_image(env, font, ch, transform))
        y_list.append(int(font_to_label[font]))

    x = torch.stack(x_list, dim=0).to(device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)

    logits = model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=float(max(0.0, label_smoothing)))

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = float((pred == y).float().mean().item())

    if opt is not None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return {"loss": float(loss.item()), "acc": float(acc)}


def make_ckpt(
    model: FontClassifier,
    args: argparse.Namespace,
    font_names: Sequence[str],
    extra: Dict,
) -> Dict:
    return {
        "task": "font_classification",
        "e_p": model.state_dict(),
        "font_names": list(font_names),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "extra": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/e_p_font_encoder.pt"))
    parser.add_argument("--best-out", type=Path, default=Path("checkpoints/e_p_font_encoder_best.pt"))
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("checkpoints/e_p_font_encoder.metrics.jsonl"))
    parser.add_argument("--log-file", type=Path, default=Path("checkpoints/e_p_font_encoder.log"))

    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--monitor", type=str, default="val_loss", choices=["val_loss", "train_loss"])
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument(
        "--num-font-classes",
        type=int,
        default=0,
        help="<=0 means use all usable fonts from LMDB; >0 keeps top-K fonts by available chars.",
    )
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "light_cnn"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lmdb-scan-limit", type=int, default=0)
    parser.add_argument("--min-chars-per-font", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root.resolve()
    train_lmdb = resolve_path(root, args.train_lmdb)
    out_path = resolve_path(root, args.out)
    best_path = resolve_path(root, args.best_out)
    metrics_path = resolve_path(root, args.metrics_jsonl)
    log_path = resolve_path(root, args.log_file)
    for p in [out_path, best_path, metrics_path, log_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 17)

    env = lmdb.open(str(train_lmdb), readonly=True, lock=False, readahead=False)
    all_chars_by_font = scan_lmdb_font_chars(
        env,
        scan_limit=int(args.lmdb_scan_limit),
        min_chars_per_font=int(args.min_chars_per_font),
    )
    usable_fonts = pick_font_classes(all_chars_by_font, int(args.num_font_classes))
    if len(usable_fonts) < 4:
        raise RuntimeError("Too few usable fonts in TrainFont.lmdb for font classification.")

    chars_by_font = {f: all_chars_by_font[f] for f in usable_fonts}
    font_to_label = {f: i for i, f in enumerate(usable_fonts)}
    train_fonts = list(usable_fonts)
    train_chars, val_chars = split_chars_within_font(chars_by_font, args.val_ratio, args.seed)
    val_fonts = sorted([f for f in train_fonts if f in val_chars and len(val_chars[f]) > 0])
    has_val = len(val_fonts) >= 2
    monitor_name = args.monitor if has_val else "train_loss"

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        T.RandomAffine(degrees=7.0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    e_p = FontClassifier(
        in_channels=3,
        num_fonts=len(usable_fonts),
        backbone=args.backbone,
    ).to(device)
    opt = torch.optim.AdamW(e_p.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_fp = log_path.open("w", encoding="utf-8")
    metrics_fp = metrics_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log(
        f"device={device} usable_fonts={len(usable_fonts)} train_fonts={len(train_fonts)} "
        f"val_fonts={len(val_fonts)} split=per-font-char(1-{int((1.0-args.val_ratio)*100)}/"
        f"{int(args.val_ratio*100)}) backbone={args.backbone} monitor={monitor_name}"
    )

    best_metric: float | None = None
    best_step = 0
    no_improve = 0
    last_step = 0

    try:
        for step in range(1, args.steps + 1):
            e_p.train()
            train_stats = run_batch(
                model=e_p,
                env=env,
                fonts=train_fonts,
                chars_by_font=train_chars,
                font_to_label=font_to_label,
                batch_size=args.batch_size,
                transform=transform,
                rng=train_rng,
                device=device,
                label_smoothing=args.label_smoothing,
                opt=opt,
            )

            val_stats: Dict[str, float] = {}
            if has_val and args.val_batches > 0 and (step % max(1, args.log_every) == 0):
                e_p.eval()
                agg_loss = 0.0
                agg_acc = 0.0
                with torch.no_grad():
                    for _ in range(args.val_batches):
                        s = run_batch(
                            model=e_p,
                            env=env,
                            fonts=val_fonts,
                            chars_by_font=val_chars,
                            font_to_label=font_to_label,
                            batch_size=args.batch_size,
                            transform=transform,
                            rng=eval_rng,
                            device=device,
                            label_smoothing=0.0,
                            opt=None,
                        )
                        agg_loss += s["loss"]
                        agg_acc += s["acc"]
                val_stats = {
                    "val_loss": agg_loss / float(args.val_batches),
                    "val_acc": agg_acc / float(args.val_batches),
                }

            metrics_row = {
                "step": step,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                **val_stats,
            }
            metrics_fp.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")
            metrics_fp.flush()

            should_log = step % max(1, args.log_every) == 0 or step == 1 or step == args.steps
            if should_log:
                if val_stats:
                    log(
                        f"step={step} train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
                        f"val_loss={val_stats['val_loss']:.4f} val_acc={val_stats['val_acc']:.4f}"
                    )
                else:
                    log(
                        f"step={step} train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f}"
                    )

            monitor_value = float(metrics_row.get(monitor_name, train_stats["loss"]))
            if best_metric is None or monitor_value < (best_metric - float(args.early_stop_min_delta)):
                best_metric = monitor_value
                best_step = step
                no_improve = 0
                torch.save(
                    make_ckpt(
                        e_p,
                        args,
                        font_names=usable_fonts,
                        extra={
                            "step": step,
                            "best_step": best_step,
                            "best_monitor": monitor_name,
                            "best_metric": best_metric,
                        },
                    ),
                    best_path,
                )
            else:
                no_improve += 1
            last_step = step

            if args.early_stop_patience > 0 and no_improve >= int(args.early_stop_patience):
                log(
                    f"early-stop at step={step} no_improve={no_improve} "
                    f"best_step={best_step} best_{monitor_name}={best_metric:.6f}"
                )
                break

        torch.save(
            make_ckpt(
                e_p,
                args,
                font_names=usable_fonts,
                extra={
                    "last_step": last_step,
                    "best_step": best_step,
                    "best_monitor": monitor_name,
                    "best_metric": best_metric,
                },
            ),
            out_path,
        )
        log(
            f"finished last_step={last_step} best_step={best_step} "
            f"best_{monitor_name}={(best_metric if best_metric is not None else float('nan')):.6f}"
        )
    finally:
        metrics_fp.close()
        log_fp.close()
        env.close()


if __name__ == "__main__":
    main()
