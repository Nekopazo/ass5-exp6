#!/usr/bin/env python3
"""Export token-attention visualizations from a pretrained style encoder checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import FontImageDataset
from scripts.pretrain_style_encoder import StyleEncoderModule
from style_augment import build_base_glyph_transform, build_style_reference_transform


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and isinstance(obj.get("style_encoder"), dict):
        return obj["style_encoder"]
    if isinstance(obj, dict) and isinstance(obj.get("model_state"), dict):
        return obj["model_state"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {type(obj)}")


def _infer_arch(sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    query_keys = ("t_low_query", "t_mid_query", "t_high_query")
    if all(k in sd for k in query_keys):
        dims = {int(sd[k].shape[-1]) for k in query_keys}
        if len(dims) != 1:
            raise ValueError(f"inconsistent token dims in checkpoint queries: {sorted(dims)}")
        return int(next(iter(dims))), len(query_keys), 3

    if "style_queries" in sd:
        style_token_count = int(sd["style_queries"].shape[0])
        style_token_dim = int(sd["style_queries"].shape[1])
        local_token_count = int(sd["style_local_queries"].shape[0]) if "style_local_queries" in sd else 3
        return style_token_dim, style_token_count, local_token_count

    raise KeyError(
        "state_dict missing token query keys; expected one of: "
        "('t_low_query','t_mid_query','t_high_query') or 'style_queries'"
    )


def _tensor_to_u8(img: torch.Tensor) -> np.ndarray:
    t = img.detach().to(dtype=torch.float32, device="cpu")
    if t.dim() == 3:
        t = t.squeeze(0)
    arr = t.numpy()
    arr = ((arr + 1.0) * 127.5).clip(0.0, 255.0).astype(np.uint8)
    return arr


def _resize_u8(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _make_token_tile(base_u8: np.ndarray, heat: np.ndarray, title: str, size: int) -> np.ndarray:
    base = _resize_u8(base_u8, size)
    h = cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    h = np.clip(h, 0.0, 1.0)
    h_u8 = (h * 255.0).astype(np.uint8)
    h_color = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.50, h_color, 0.50, 0.0)
    cv2.putText(overlay, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _token_title(token_idx: int, token_count: int) -> str:
    if int(token_count) == 3:
        names = ["low", "mid", "high"]
        return names[token_idx]
    return f"tok{token_idx}"


def _stack_row(tiles: list[np.ndarray], width: int, tile_size: int) -> np.ndarray:
    blank = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    if len(tiles) < width:
        tiles = tiles + [blank] * (width - len(tiles))
    return np.concatenate(tiles, axis=1)


def _normalize_prob_map(x: np.ndarray) -> np.ndarray:
    z = np.clip(x.astype(np.float64), 0.0, None)
    s = float(z.sum())
    if s <= 0.0:
        z = np.ones_like(z, dtype=np.float64)
        s = float(z.sum())
    return z / s


def _norm_entropy(prob: np.ndarray) -> float:
    p = np.clip(prob.astype(np.float64), 1e-12, None)
    h = float(-(p * np.log(p)).sum())
    hmax = float(np.log(max(2, p.size)))
    return float(h / hmax)


def _vis_heat_from_prob(prob: np.ndarray, mode: str) -> np.ndarray:
    mode_key = str(mode).strip().lower()
    if mode_key == "max":
        return prob / max(1e-8, float(prob.max()))
    if mode_key == "deviation":
        base = 1.0 / float(prob.size)
        pos = np.clip(prob - base, 0.0, None)
        scale = float(pos.max())
        if scale <= 0.0:
            return np.zeros_like(prob, dtype=np.float64)
        return pos / scale
    raise ValueError(f"unsupported vis mode: {mode}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 0.0 or bn <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _pairwise_cosine_rows(mat: np.ndarray) -> np.ndarray:
    # mat: (N, D)
    x = mat.astype(np.float64, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    y = x / n
    return y @ y.T


def _analyze_token_roles(token_vecs: np.ndarray) -> dict[str, Any]:
    # token_vecs: (S, T, P), each vec is a probability map flattened.
    s_cnt, t_cnt, _ = token_vecs.shape
    within = np.zeros((t_cnt, t_cnt), dtype=np.float64)
    for s in range(s_cnt):
        within += _pairwise_cosine_rows(token_vecs[s])
    within /= float(max(1, s_cnt))

    # Per-token stability across samples: pairwise cosine over sample axis.
    token_stability: list[float] = []
    for t in range(t_cnt):
        c = _pairwise_cosine_rows(token_vecs[:, t, :])
        if s_cnt <= 1:
            token_stability.append(1.0)
        else:
            mask = ~np.eye(s_cnt, dtype=bool)
            token_stability.append(float(c[mask].mean()))

    # Separation margin: same-token vs different-token template matching.
    centers = token_vecs.mean(axis=0)  # (T, P)
    same_vals: list[float] = []
    diff_vals: list[float] = []
    for s in range(s_cnt):
        for t in range(t_cnt):
            v = token_vecs[s, t]
            same_vals.append(_cosine(v, centers[t]))
            for u in range(t_cnt):
                if u == t:
                    continue
                diff_vals.append(_cosine(v, centers[u]))
    same_mean = float(np.mean(same_vals)) if same_vals else 0.0
    diff_mean = float(np.mean(diff_vals)) if diff_vals else 0.0

    offdiag_mask = ~np.eye(t_cnt, dtype=bool)
    offdiag_vals = within[offdiag_mask]

    return {
        "within_sample_token_overlap_matrix": within.tolist(),
        "within_sample_overlap_offdiag_mean": float(offdiag_vals.mean()) if offdiag_vals.size else 0.0,
        "within_sample_overlap_offdiag_p90": float(np.percentile(offdiag_vals, 90)) if offdiag_vals.size else 0.0,
        "per_token_stability": [float(x) for x in token_stability],
        "stability_mean": float(np.mean(token_stability)) if token_stability else 0.0,
        "stability_min": float(np.min(token_stability)) if token_stability else 0.0,
        "same_token_center_cosine_mean": same_mean,
        "diff_token_center_cosine_mean": diff_mean,
        "role_separation_margin": float(same_mean - diff_mean),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export 50 random-char token-attention visualizations.")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--style-ref-count", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tile-size", type=int, default=160)
    p.add_argument("--vis-mode", type=str, default="deviation", choices=["deviation", "max"])
    args = p.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    obj = torch.load(str(args.checkpoint), map_location="cpu")
    sd = _extract_state_dict(obj)
    style_token_dim, style_token_count, local_token_count = _infer_arch(sd)

    model = StyleEncoderModule(
        in_channels=1,
        style_token_dim=style_token_dim,
        style_token_count=style_token_count,
        local_token_count=local_token_count,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] non-strict load missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    content_transform = build_base_glyph_transform(image_size=128)
    style_transform = build_style_reference_transform(image_size=128)

    ds = FontImageDataset(
        project_root=args.project_root,
        use_style_image=True,
        use_part_bank=False,
        random_seed=int(args.seed),
        transform=content_transform,
        style_transform=style_transform,
        cache_style_image=False,
        style_ref_count=max(1, int(args.style_ref_count)),
    )

    n = min(max(1, int(args.num_samples)), len(ds))
    sampled_indices = random.sample(range(len(ds)), k=n)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    rows_meta: list[dict[str, Any]] = []
    token_vecs_all: list[np.ndarray] = []
    with torch.no_grad():
        for rank, idx in enumerate(sampled_indices, start=1):
            sample = ds[idx]
            style_refs = sample["style_img"].unsqueeze(0).to(device)          # (1,R,1,128,128)
            style_ref_mask = sample["style_ref_mask"].unsqueeze(0).to(device)  # (1,R)
            _, token_attn = model.encode_style_tokens_with_attention(
                style_refs,
                style_ref_mask,
            )
            # token_attn: (1,T,R,Hf,Wf)
            token_attn = token_attn[0].detach().to(dtype=torch.float32, device="cpu")

            refs_u8 = [_tensor_to_u8(x) for x in sample["style_img"]]
            base_u8 = refs_u8[0]
            if len(refs_u8) > 1:
                base_u8 = np.mean(np.stack(refs_u8, axis=0), axis=0).astype(np.uint8)

            ref_tiles: list[np.ndarray] = []
            for i, ref in enumerate(refs_u8):
                tile = cv2.cvtColor(_resize_u8(ref, int(args.tile_size)), cv2.COLOR_GRAY2BGR)
                label = f"ref{i}:{sample.get('style_chars', ['?'])[i]}"
                cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
                ref_tiles.append(tile)

            token_tiles: list[np.ndarray] = []
            token_peaks: list[float] = []
            token_peak_mean_ratios: list[float] = []
            token_entropies: list[float] = []
            token_vecs_sample: list[np.ndarray] = []
            for t_i in range(int(token_attn.shape[0])):
                heat_raw = token_attn[t_i].mean(dim=0).numpy()  # mean over refs -> (Hf,Wf)
                heat_prob = _normalize_prob_map(heat_raw)
                token_vecs_sample.append(heat_prob.reshape(-1).astype(np.float64))
                token_peaks.append(float(heat_prob.max()))
                token_peak_mean_ratios.append(float(heat_prob.max() / max(1e-12, float(heat_prob.mean()))))
                token_entropies.append(_norm_entropy(heat_prob))
                heat_vis = _vis_heat_from_prob(heat_prob, mode=str(args.vis_mode))
                token_tiles.append(
                    _make_token_tile(
                        base_u8,
                        heat_vis,
                        _token_title(t_i, int(token_attn.shape[0])),
                        int(args.tile_size),
                    )
                )

            token_vecs_arr = np.stack(token_vecs_sample, axis=0)  # (T,P)
            token_vecs_all.append(token_vecs_arr)
            sample_overlap = _pairwise_cosine_rows(token_vecs_arr)
            if int(token_vecs_arr.shape[0]) > 1:
                m = ~np.eye(int(token_vecs_arr.shape[0]), dtype=bool)
                sample_overlap_offdiag = float(sample_overlap[m].mean())
            else:
                sample_overlap_offdiag = 0.0

            width = max(len(ref_tiles), len(token_tiles))
            row1 = _stack_row(ref_tiles, width=width, tile_size=int(args.tile_size))
            row2 = _stack_row(token_tiles, width=width, tile_size=int(args.tile_size))
            panel = np.concatenate([row1, row2], axis=0)

            header = np.full((40, panel.shape[1], 3), 255, dtype=np.uint8)
            title = f"#{rank:02d} font={sample['font']} char={sample['char']} refs={len(ref_tiles)} tokens={len(token_tiles)}"
            cv2.putText(header, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
            out_img = np.concatenate([header, panel], axis=0)

            out_name = f"{rank:03d}_{sample['font']}_{ord(sample['char']):X}.png"
            out_path = out_dir / "images" / out_name
            cv2.imwrite(str(out_path), out_img)

            rows_meta.append(
                {
                    "rank": int(rank),
                    "dataset_index": int(idx),
                    "font": str(sample["font"]),
                    "char": str(sample["char"]),
                    "char_codepoint": f"U+{ord(sample['char']):04X}",
                    "style_chars": [str(x) for x in sample.get("style_chars", [])],
                    "token_peaks": [float(x) for x in token_peaks],
                    "token_peak_mean_ratios": [float(x) for x in token_peak_mean_ratios],
                    "token_entropies": [float(x) for x in token_entropies],
                    "sample_token_overlap_offdiag": float(sample_overlap_offdiag),
                    "image": str(Path("images") / out_name),
                }
            )
            print(f"[{rank:03d}/{n}] saved {out_path}")

    token_role_analysis: dict[str, Any] = {}
    if token_vecs_all:
        token_vecs_np = np.stack(token_vecs_all, axis=0)  # (S,T,P)
        token_role_analysis = _analyze_token_roles(token_vecs_np)

    meta = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "num_samples": int(n),
        "seed": int(args.seed),
        "style_ref_count": int(args.style_ref_count),
        "style_transform": "resize_only",
        "vis_mode": str(args.vis_mode),
        "style_token_dim": int(style_token_dim),
        "style_token_count": int(style_token_count),
        "local_token_count": int(local_token_count),
        "token_role_analysis": token_role_analysis,
        "samples": rows_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if token_role_analysis:
        (out_dir / "analysis.json").write_text(
            json.dumps(token_role_analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] exported {n} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
