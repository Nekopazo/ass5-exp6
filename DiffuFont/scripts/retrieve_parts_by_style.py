#!/usr/bin/env python3
"""Retrieve top-1 PartBank font from style image(s) using E_p softmax classifier."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Dict, List

import lmdb
import torch
from PIL import Image
from torchvision import transforms as T

import sys

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from models.style_encoders import FontClassifier


def resolve_path(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def load_partbank(manifest_path: Path, root: Path) -> Dict[str, List[Dict]]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    fonts = obj.get("fonts", {})
    out: Dict[str, List[Dict]] = {}
    for font_name, info in fonts.items():
        rows: List[Dict] = []
        for row in info.get("parts", []):
            rel = row.get("path")
            if not rel:
                continue
            p = resolve_path(root, Path(rel))
            if not p.exists():
                continue
            rr = dict(row)
            rr["abs_path"] = str(p)
            rows.append(rr)
        if rows:
            out[font_name] = rows
    return out


def load_style_from_file(path: Path, transform) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img)


def load_style_from_lmdb(env: lmdb.Environment, key: str, transform) -> torch.Tensor:
    with env.begin() as txn:
        b = txn.get(key.encode("utf-8"))
    if b is None:
        raise KeyError(f"missing lmdb key: {key}")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return transform(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("DataPreparation/PartBank/manifest.json"))
    parser.add_argument("--ep-ckpt", type=Path, required=True, help="E_p classifier checkpoint")
    parser.add_argument("--style-image", type=str, default="", help="Comma-separated style image paths")
    parser.add_argument("--style-key", type=str, default="", help="Comma-separated lmdb keys, e.g. 'FZBANGSKFW@你'")
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--style-image-size", type=int, default=256)
    parser.add_argument("--font-topk", type=int, default=3)
    parser.add_argument("--part-min-size", type=int, default=2)
    parser.add_argument("--part-max-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/retrieved_parts.json"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.part_min_size > args.part_max_size:
        raise ValueError("--part-min-size must be <= --part-max-size")

    root = args.project_root.resolve()
    manifest_path = resolve_path(root, args.manifest)
    ep_ckpt = resolve_path(root, args.ep_ckpt)
    out_json = resolve_path(root, args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    partbank = load_partbank(manifest_path, root)
    if not partbank:
        raise RuntimeError("empty partbank from manifest")

    tfm = T.Compose(
        [
            T.Resize((args.style_image_size, args.style_image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )

    style_imgs: List[torch.Tensor] = []
    if args.style_image.strip():
        for p in [x.strip() for x in args.style_image.split(",") if x.strip()]:
            style_imgs.append(load_style_from_file(resolve_path(root, Path(p)), tfm))

    if args.style_key.strip():
        env = lmdb.open(str(resolve_path(root, args.train_lmdb)), readonly=True, lock=False, readahead=False)
        try:
            for key in [x.strip() for x in args.style_key.split(",") if x.strip()]:
                style_imgs.append(load_style_from_lmdb(env, key, tfm))
        finally:
            env.close()

    if not style_imgs:
        raise RuntimeError("provide at least one --style-image or --style-key")

    try:
        ckpt = torch.load(ep_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ep_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"invalid checkpoint: {ep_ckpt}")

    class_font_names = [str(x) for x in ckpt.get("font_names", [])]
    if not class_font_names:
        raise RuntimeError("checkpoint missing 'font_names'; expected classifier-pretrained E_p")
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    backbone = str(cfg.get("backbone", "resnet18"))

    e_p = FontClassifier(in_channels=3, num_fonts=len(class_font_names), backbone=backbone).to(device)
    state = ckpt.get("e_p", ckpt)
    e_p.load_state_dict(state, strict=False)
    e_p.eval()

    candidate_fonts = [f for f in class_font_names if f in partbank]
    if not candidate_fonts:
        raise RuntimeError("no overlap between PartBank fonts and E_p classifier font classes")

    label_of_font = {f: i for i, f in enumerate(class_font_names)}
    candidate_labels = torch.tensor([label_of_font[f] for f in candidate_fonts], dtype=torch.long, device=device)

    with torch.no_grad():
        x = torch.stack(style_imgs, dim=0).to(device)
        logits = e_p(x)
        probs = torch.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        cand_probs = mean_probs.index_select(0, candidate_labels)

    order = torch.argsort(cand_probs, descending=True)
    topk = max(1, min(int(args.font_topk), int(order.numel())))
    top_indices = order[:topk].detach().cpu().tolist()
    top_fonts = [(candidate_fonts[int(i)], float(cand_probs[int(i)].item())) for i in top_indices]
    top1_font = top_fonts[0][0]

    rows = partbank[top1_font]
    rng = random.Random(args.seed)
    hi = min(int(args.part_max_size), len(rows))
    lo = min(int(args.part_min_size), hi)
    if hi <= 0:
        raise RuntimeError(f"selected font has no parts: {top1_font}")
    pick_n = rng.randint(lo, hi)
    picked = rng.sample(rows, k=pick_n) if len(rows) >= pick_n else [rng.choice(rows) for _ in range(pick_n)]

    result = {
        "retrieval_method": "E_p softmax font classification (top-1)",
        "topk_fonts": [{"font": f, "prob": float(s)} for f, s in top_fonts],
        "selected_font": top1_font,
        "selected_parts": [
            {
                "path": r["abs_path"],
                "char": r.get("char", ""),
                "x": int(r.get("x", 0)),
                "y": int(r.get("y", 0)),
                "response": float(r.get("response", 0.0)),
            }
            for r in picked
        ],
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
