#!/usr/bin/env python3
"""Build top-1 font retrieval cache using E_p softmax classifier.

Output npz fields:
- source_fonts: (N,) source fonts from LMDB
- retrieved_fonts: (N,) top-1 retrieved fonts for PartBank lookup
- top1_scores: (N,) softmax probability of retrieved font
- counts: (N,) number of glyphs used per source font
- class_font_names: (C,) classifier label->font mapping
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import lmdb
import numpy as np
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


def scan_lmdb_font_chars(
    env: lmdb.Environment,
    scan_limit: int = 0,
    min_chars_per_font: int = 2,
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
    return {f: sorted(list(chars)) for f, chars in chars_by_font.items() if len(chars) >= int(min_chars_per_font)}


def load_partbank_fonts(manifest_path: Path) -> List[str]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    fonts = obj.get("fonts", {}) if isinstance(obj, dict) else {}
    if not isinstance(fonts, dict):
        return []
    return sorted([str(k) for k, v in fonts.items() if isinstance(k, str) and isinstance(v, dict)])


def load_glyph_image(env: lmdb.Environment, font_name: str, ch: str, transform) -> torch.Tensor:
    key = f"{font_name}@{ch}".encode("utf-8")
    with env.begin() as txn:
        b = txn.get(key)
    if b is None:
        raise KeyError(f"missing glyph in lmdb: {font_name}@{ch}")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return transform(img)


def pick_chars(chars: Sequence[str], chars_per_font: int, rng: random.Random) -> List[str]:
    if chars_per_font <= 0:
        return list(chars)
    k = min(int(chars_per_font), len(chars))
    if len(chars) == k:
        return list(chars)
    return rng.sample(list(chars), k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--manifest", type=Path, default=Path("DataPreparation/PartBank/manifest.json"))
    parser.add_argument("--ep-ckpt", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, default=Path("checkpoints/font_retrieval_cache.npz"))
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/font_retrieval_cache.summary.json"))

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chars-per-font", type=int, default=48)
    parser.add_argument("--min-chars-per-font", type=int, default=8)
    parser.add_argument("--lmdb-scan-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    root = args.project_root.resolve()
    lmdb_path = resolve_path(root, args.train_lmdb)
    manifest_path = resolve_path(root, args.manifest)
    ckpt_path = resolve_path(root, args.ep_ckpt)
    out_npz = resolve_path(root, args.out_npz)
    out_json = resolve_path(root, args.out_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    chars_by_font = scan_lmdb_font_chars(
        env,
        scan_limit=int(args.lmdb_scan_limit),
        min_chars_per_font=int(args.min_chars_per_font),
    )
    source_fonts = sorted(chars_by_font.keys())
    if not source_fonts:
        raise RuntimeError("no usable fonts found in TrainFont.lmdb")

    partbank_fonts = set(load_partbank_fonts(manifest_path))
    if not partbank_fonts:
        raise RuntimeError(f"no fonts found in PartBank manifest: {manifest_path}")

    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid checkpoint format: {ckpt_path}")
    state = obj.get("e_p", obj)
    class_font_names = [str(x) for x in obj.get("font_names", [])]
    if not class_font_names:
        raise RuntimeError("checkpoint missing 'font_names'; please use classifier-pretrained E_p checkpoint")

    cfg = obj.get("config", {}) if isinstance(obj.get("config", {}), dict) else {}
    backbone = str(cfg.get("backbone", "resnet18"))
    model = FontClassifier(in_channels=3, num_fonts=len(class_font_names), backbone=backbone).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    label_of_font = {f: i for i, f in enumerate(class_font_names)}
    candidate_labels = [label_of_font[f] for f in class_font_names if f in partbank_fonts and f in label_of_font]
    if not candidate_labels:
        raise RuntimeError("no overlap between classifier classes and PartBank fonts")

    tfm = T.Compose(
        [
            T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    rng = random.Random(args.seed)
    bs = max(1, int(args.batch_size))

    retrieved_fonts: List[str] = []
    top1_scores: List[float] = []
    counts = np.zeros((len(source_fonts),), dtype=np.int32)

    with torch.no_grad():
        for i, font in enumerate(source_fonts):
            chars = pick_chars(chars_by_font[font], int(args.chars_per_font), rng)
            counts[i] = int(len(chars))
            prob_sum = None
            n_rows = 0
            for st in range(0, len(chars), bs):
                ed = min(len(chars), st + bs)
                x = torch.stack(
                    [load_glyph_image(env, font, ch, tfm) for ch in chars[st:ed]],
                    dim=0,
                ).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                p_sum = probs.sum(dim=0)
                prob_sum = p_sum if prob_sum is None else (prob_sum + p_sum)
                n_rows += int(x.size(0))
            if prob_sum is None or n_rows <= 0:
                fallback_font = font if font in partbank_fonts else class_font_names[candidate_labels[0]]
                retrieved_fonts.append(fallback_font)
                top1_scores.append(0.0)
                continue

            mean_probs = prob_sum / float(n_rows)
            cand = torch.tensor(candidate_labels, dtype=torch.long, device=mean_probs.device)
            cand_probs = mean_probs.index_select(0, cand)
            local_idx = int(torch.argmax(cand_probs).item())
            best_label = int(candidate_labels[local_idx])
            retrieved_fonts.append(class_font_names[best_label])
            top1_scores.append(float(cand_probs[local_idx].item()))

            if (i + 1) % 50 == 0 or (i + 1) == len(source_fonts):
                print(f"[build-cache] {i + 1}/{len(source_fonts)} fonts processed", flush=True)

    np.savez(
        out_npz,
        source_fonts=np.array(source_fonts, dtype=object),
        retrieved_fonts=np.array(retrieved_fonts, dtype=object),
        top1_scores=np.array(top1_scores, dtype=np.float32),
        counts=counts,
        class_font_names=np.array(class_font_names, dtype=object),
    )
    summary = {
        "lmdb": str(lmdb_path),
        "manifest": str(manifest_path),
        "ep_ckpt": str(ckpt_path),
        "num_source_fonts": int(len(source_fonts)),
        "num_classifier_fonts": int(len(class_font_names)),
        "num_candidate_partbank_fonts": int(len(candidate_labels)),
        "chars_per_font": int(args.chars_per_font),
        "min_chars_per_font": int(args.min_chars_per_font),
        "out_npz": str(out_npz),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()


if __name__ == "__main__":
    main()
