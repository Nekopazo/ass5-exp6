#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def generate_gb2312_hanzi() -> List[str]:
    chars = []
    for high in range(0xA1, 0xF8):
        for low in range(0xA1, 0xFF):
            raw = bytes([high, low])
            try:
                ch = raw.decode("gb2312")
            except UnicodeDecodeError:
                continue
            code = ord(ch)
            if 0x4E00 <= code <= 0x9FFF:
                chars.append(ch)
    # Stable de-dup
    seen = set()
    uniq = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            uniq.append(ch)
    return uniq


def load_charset_from_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    chars: List[str] = []
    for line in text.splitlines():
        token = line.strip()
        if not token:
            continue
        if len(token) == 1:
            chars.append(token)
        else:
            # Support packed format where multiple chars are placed on one line.
            chars.extend(list(token))

    seen = set()
    uniq: List[str] = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            uniq.append(ch)
    return uniq


def render_char_image(ch: str, font: ImageFont.FreeTypeFont, size: int = 224, pad: int = 18) -> Image.Image:
    canvas = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(canvas)

    # Use a temporary drawing to estimate tight bbox for centering.
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    max_w = size - 2 * pad
    max_h = size - 2 * pad
    if w <= 0 or h <= 0:
        return canvas.convert("RGB")

    sx = max_w / w
    sy = max_h / h
    scale = min(1.0, sx, sy)

    if scale < 0.999:
        dyn_font_size = max(8, int(font.size * scale))
        font = ImageFont.truetype(font.path, dyn_font_size)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    return canvas.convert("RGB")


def build_clip_embeddings(
    chars: Sequence[str],
    font_path: Path,
    font_size: int,
    batch_size: int,
    device: str,
    clip_model: str,
) -> np.ndarray:
    model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
    processor = CLIPImageProcessor.from_pretrained(clip_model)
    model.to(device)
    model.eval()

    font = ImageFont.truetype(str(font_path), font_size)

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(chars), batch_size), desc="Encoding chars"):
            batch_chars = chars[i : i + batch_size]
            imgs = [render_char_image(ch, font=font, size=224) for ch in batch_chars]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model(**inputs).image_embeds
            feats = torch.nn.functional.normalize(feats, dim=-1)
            outputs.append(feats.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def farthest_point_sampling(points: np.ndarray, n_select: int, seed_index: int = 0) -> List[int]:
    if n_select <= 0:
        return []
    n = points.shape[0]
    if n_select >= n:
        return list(range(n))

    selected = [seed_index]
    distances = np.linalg.norm(points - points[seed_index], axis=1)

    for _ in range(1, n_select):
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)

    return selected


def quota_per_cluster(k: int, target_total: int = 400) -> List[int]:
    base = target_total // k
    rem = target_total % k
    quotas = [base] * k
    for i in range(rem):
        quotas[i] += 1
    return quotas


def select_references_for_k(
    chars: Sequence[str],
    embeds: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    k: int,
    target_total: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, float]]:
    quotas = quota_per_cluster(k, target_total=target_total)

    selected_global_indices: List[int] = []
    selected_set = set()
    mapping: Dict[str, int] = {}

    cluster_sizes = []
    d_to_center = np.zeros(len(chars), dtype=np.float32)

    for cid in range(k):
        idx = np.where(labels == cid)[0]
        cluster_sizes.append(int(len(idx)))
        if len(idx) == 0:
            continue

        cluster_points = embeds[idx]
        center = centers[cid]
        dists = np.linalg.norm(cluster_points - center[None, :], axis=1)
        d_to_center[idx] = dists

        quota = min(quotas[cid], len(idx))
        if quota <= 0:
            continue

        center_local = int(np.argmin(dists))
        center_global = int(idx[center_local])

        local_selected = [center_local]
        if quota > 1:
            fps_local = farthest_point_sampling(cluster_points, n_select=quota, seed_index=center_local)
            local_selected = fps_local

        for li in local_selected:
            gi = int(idx[li])
            if gi in selected_set:
                continue
            selected_set.add(gi)
            selected_global_indices.append(gi)
            mapping[chars[gi]] = cid

    if len(selected_global_indices) < target_total:
        deficit = target_total - len(selected_global_indices)
        candidates = [
            (i, float(d_to_center[i]))
            for i in range(len(chars))
            if i not in selected_set
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for gi, _ in candidates[:deficit]:
            selected_set.add(gi)
            selected_global_indices.append(gi)
            mapping[chars[gi]] = int(labels[gi])

    selected_global_indices = selected_global_indices[:target_total]
    selected_chars = [chars[i] for i in selected_global_indices]

    # Quality metrics
    p95 = float(np.percentile(d_to_center, 95))
    min_size = int(min(cluster_sizes)) if cluster_sizes else 0
    max_size = int(max(cluster_sizes)) if cluster_sizes else 0
    ratio = float(max_size / max(min_size, 1))

    metrics = {
        "k": int(k),
        "selected": int(len(selected_chars)),
        "p95_d": p95,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "max_min_ratio": ratio,
    }
    return selected_chars, mapping, metrics


def choose_best_k(results: List[Dict]) -> Dict:
    # Priority: larger min cluster size, then lower p95 distance, then lower ratio.
    return sorted(
        results,
        key=lambda x: (-x["metrics"]["min_cluster_size"], x["metrics"]["p95_d"], x["metrics"]["max_min_ratio"]),
    )[0]


def save_text(path: Path, chars: Sequence[str]) -> None:
    path.write_text("\n".join(chars) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 400-char reference set using CLIP + KMeans.")
    parser.add_argument("--font", type=Path, required=True, help="Path to base TTF font.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--charset-file", type=Path, default=None, help="Optional charset file path.")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--k-list", type=int, nargs="+", default=[12, 16, 20, 24, 28])
    parser.add_argument("--target-total", type=int, default=400)
    parser.add_argument("--font-size", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.charset_file is not None:
        chars = load_charset_from_file(args.charset_file)
        charset_path = args.out_dir / "charset_input.txt"
        embed_path = args.out_dir / "charset_clip_embeds.npy"
    else:
        chars = generate_gb2312_hanzi()
        charset_path = args.out_dir / "gb2312_hanzi.txt"
        embed_path = args.out_dir / "gb2312_clip_embeds.npy"
    save_text(charset_path, chars)

    print(f"Loaded charset size: {len(chars)}")
    print(f"Encoding with CLIP on {args.device}...")
    embeds = build_clip_embeddings(
        chars=chars,
        font_path=args.font,
        font_size=args.font_size,
        batch_size=args.batch_size,
        device=args.device,
        clip_model=args.clip_model,
    )

    np.save(embed_path, embeds)

    all_results = []
    for k in args.k_list:
        print(f"Running KMeans with K={k}")
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        labels = kmeans.fit_predict(embeds)

        selected_chars, mapping, metrics = select_references_for_k(
            chars=chars,
            embeds=embeds,
            labels=labels,
            centers=kmeans.cluster_centers_,
            k=k,
            target_total=args.target_total,
        )

        k_dir = args.out_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        save_text(k_dir / "reference_400.txt", selected_chars)
        (k_dir / "reference_cluster.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (k_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        all_results.append({
            "k": k,
            "selected_chars": selected_chars,
            "mapping": mapping,
            "metrics": metrics,
        })

    best = choose_best_k(all_results)
    (args.out_dir / "k_search_metrics.json").write_text(
        json.dumps([r["metrics"] for r in all_results], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_text(args.out_dir / "reference_400.txt", best["selected_chars"])
    (args.out_dir / "reference_cluster.json").write_text(
        json.dumps(best["mapping"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "best_k.json").write_text(
        json.dumps({"k": best["k"], "metrics": best["metrics"]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Done.")
    print(f"Best K={best['k']}")
    print(f"reference_400.txt: {args.out_dir / 'reference_400.txt'}")
    print(f"reference_cluster.json: {args.out_dir / 'reference_cluster.json'}")


if __name__ == "__main__":
    main()
