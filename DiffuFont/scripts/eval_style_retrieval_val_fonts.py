#!/usr/bin/env python3
"""Style consistency + retrieval evaluation on validation fonts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from style_augment import build_style_reference_transform
from scripts.pretrain_style_encoder import (
    StyleEncoderModule,
    load_ref_tensor,
    sample_refs_from_buckets,
    scan_train_lmdb_font_chars,
    split_fonts,
)


def set_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_path(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def build_font_infos(
    env: lmdb.Environment,
    cluster_json: Path,
    min_keep: int,
) -> Dict[str, Dict[str, Any]]:
    cluster_raw = json.loads(cluster_json.read_text(encoding="utf-8"))
    char_to_bucket: Dict[str, int] = {str(k): int(v) for k, v in cluster_raw.items()}

    font_chars = scan_train_lmdb_font_chars(env)
    font_infos: Dict[str, Dict[str, Any]] = {}
    for font, chars in font_chars.items():
        bucket_to_chars: Dict[int, List[str]] = {}
        for ch in chars:
            if ch not in char_to_bucket:
                continue
            b = int(char_to_bucket[ch])
            bucket_to_chars.setdefault(b, []).append(ch)
        all_chars = sorted({ch for arr in bucket_to_chars.values() for ch in arr})
        if len(all_chars) >= max(1, int(min_keep)):
            font_infos[font] = {"bucket_to_chars": bucket_to_chars, "all_chars": all_chars}
    return font_infos


def make_encoder_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    style_start_channel: int,
    style_token_dim: int,
    style_token_count: int,
) -> StyleEncoderModule:
    encoder = StyleEncoderModule(
        in_channels=1,
        style_start_channel=int(style_start_channel),
        style_token_dim=int(style_token_dim),
        style_token_count=int(style_token_count),
    ).to(device)

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and isinstance(obj.get("style_encoder"), dict):
        sd = obj["style_encoder"]
    elif isinstance(obj, dict) and isinstance(obj.get("model_state"), dict):
        sd = obj["model_state"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError(f"unsupported checkpoint format: {type(obj)}")

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[retrieval] non-strict load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    encoder.eval()
    return encoder


def build_worker_init_fn(base_seed: int, worker_torch_threads: int):
    seed0 = int(base_seed)
    threads = int(worker_torch_threads)

    def _init(worker_id: int) -> None:
        ws = (seed0 + int(worker_id) * 100003) % (2**32)
        random.seed(ws)
        np.random.seed(ws)
        torch.manual_seed(ws)
        if threads > 0:
            torch.set_num_threads(threads)

    return _init


class FontRetrievalDataset(Dataset):
    def __init__(
        self,
        lmdb_path: Path,
        fonts: List[str],
        font_infos: Dict[str, Dict[str, Any]],
        ref_per_style: int,
        sets_per_font: int,
        transform,
        seed_base: int,
        decode_cache_size: int,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.fonts = list(fonts)
        self.font_infos = font_infos
        self.ref_per_style = int(ref_per_style)
        self.sets_per_font = max(1, int(sets_per_font))
        self.transform = transform
        self.seed_base = int(seed_base)
        self.decode_cache_size = max(0, int(decode_cache_size))

        self._env: lmdb.Environment | None = None
        self._txn = None
        self._worker_id = -1
        self._decode_cache: OrderedDict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.fonts) * self.sets_per_font

    def _ensure_worker_state(self) -> None:
        wi = get_worker_info()
        worker_id = int(wi.id) if wi is not None else 0
        if self._env is not None and self._worker_id == worker_id:
            return
        if self._env is not None:
            self._env.close()
        self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        self._txn = self._env.begin(buffers=True)
        self._worker_id = worker_id
        self._decode_cache = OrderedDict() if self.decode_cache_size > 0 else None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_worker_state()
        assert self._txn is not None

        idx = int(index)
        font_idx = idx // self.sets_per_font
        set_idx = idx % self.sets_per_font
        font = self.fonts[font_idx]
        info = self.font_infos[font]
        rng = random.Random(self.seed_base + font_idx * 1000003 + set_idx * 10007)
        chars = sample_refs_from_buckets(
            info["bucket_to_chars"],
            info["all_chars"],
            self.ref_per_style,
            rng,
        )
        refs = torch.stack(
            [
                load_ref_tensor(
                    self._txn,
                    font,
                    ch,
                    self.transform,
                    decode_cache=self._decode_cache,
                    decode_cache_size=self.decode_cache_size,
                )
                for ch in chars
            ],
            dim=0,
        )
        mask = torch.ones((int(self.ref_per_style),), dtype=torch.float32)
        return refs, mask, torch.tensor(font_idx, dtype=torch.long)


def build_retrieval_loader(
    *,
    lmdb_path: Path,
    fonts: List[str],
    font_infos: Dict[str, Dict[str, Any]],
    ref_per_style: int,
    sets_per_font: int,
    transform,
    seed_base: int,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    worker_torch_threads: int,
    decode_cache_size: int,
) -> DataLoader:
    ds = FontRetrievalDataset(
        lmdb_path=lmdb_path,
        fonts=fonts,
        font_infos=font_infos,
        ref_per_style=ref_per_style,
        sets_per_font=sets_per_font,
        transform=transform,
        seed_base=seed_base,
        decode_cache_size=decode_cache_size,
    )
    nw = max(0, int(workers))
    kwargs: Dict[str, Any] = {}
    if nw > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = build_worker_init_fn(seed_base, int(worker_torch_threads))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )


@torch.no_grad()
def encode_font_set(
    *,
    encoder: StyleEncoderModule,
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emb_chunks: List[torch.Tensor] = []
    lbl_chunks: List[torch.Tensor] = []
    collapse_chunks: List[torch.Tensor] = []
    dev = next(encoder.parameters()).device
    for refs, masks, labels in loader:
        r = refs.to(dev, non_blocking=True)
        m = masks.to(dev, non_blocking=True)
        tok = encoder.encode_style_tokens(r, m)
        tnorm = F.normalize(tok, dim=-1)
        tsim = torch.matmul(tnorm, tnorm.transpose(1, 2))
        ntok = int(tsim.size(-1))
        eye = torch.eye(ntok, device=tsim.device, dtype=torch.bool).unsqueeze(0)
        off = tsim.masked_select(~eye).view(int(tsim.size(0)), -1)
        collapse = off.mean(dim=1)
        z = torch.nn.functional.normalize(tok.mean(dim=1), dim=-1)
        emb_chunks.append(z.detach().cpu())
        lbl_chunks.append(labels.detach().cpu())
        collapse_chunks.append(collapse.detach().cpu())
    return (
        torch.cat(emb_chunks, dim=0),
        torch.cat(lbl_chunks, dim=0).long(),
        torch.cat(collapse_chunks, dim=0),
    )


def retrieval_metrics_with_labels(
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    topk: int = 5,
) -> Dict[str, float]:
    query_emb = F.normalize(query_emb, dim=-1)
    gallery_emb = F.normalize(gallery_emb, dim=-1)
    sim = query_emb @ gallery_emb.t()
    n = int(query_emb.size(0))
    if n <= 0:
        raise RuntimeError("empty query set")
    order = torch.argsort(sim, dim=1, descending=True)
    top1_hit = []
    topk_hit = []
    rr = []
    k = min(int(topk), int(order.size(1)))
    for i in range(n):
        ql = int(query_labels[i].item())
        ranked_idx = order[i]
        ranked_lbl = gallery_labels.index_select(0, ranked_idx)
        match = ranked_lbl.eq(ql)
        top1_hit.append(float(bool(match[:1].any().item())))
        topk_hit.append(float(bool(match[:k].any().item())))
        pos = torch.nonzero(match, as_tuple=False)
        if int(pos.numel()) > 0:
            rank = int(pos[0, 0].item()) + 1
            rr.append(1.0 / float(rank))
        else:
            rr.append(0.0)
    return {
        "top1": float(np.mean(top1_hit)),
        "top5": float(np.mean(topk_hit)),
        "mrr": float(np.mean(rr)),
    }


def _safe_stats(values: torch.Tensor) -> Dict[str, float]:
    if int(values.numel()) <= 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(q[0].item()),
        "p50": float(q[1].item()),
        "p90": float(q[2].item()),
    }


def consistency_metrics(
    emb: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    z = F.normalize(emb, dim=-1)
    sim = z @ z.t()
    y = labels.view(-1).long()
    same = y.unsqueeze(0).eq(y.unsqueeze(1))
    eye = torch.eye(int(sim.size(0)), dtype=torch.bool)
    same = same & (~eye)
    diff = ~same & (~eye)
    same_vals = sim[same]
    diff_vals = sim[diff]
    s = _safe_stats(same_vals)
    d = _safe_stats(diff_vals)
    out = {
        "same_mean": s["mean"],
        "same_std": s["std"],
        "same_p10": s["p10"],
        "same_p50": s["p50"],
        "same_p90": s["p90"],
        "diff_mean": d["mean"],
        "diff_std": d["std"],
        "diff_p10": d["p10"],
        "diff_p50": d["p50"],
        "diff_p90": d["p90"],
    }
    out["gap_same_minus_diff"] = float(out["same_mean"] - out["diff_mean"])
    return out


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def token_collapse_metrics(collapse_vals: torch.Tensor) -> Dict[str, float]:
    st = _safe_stats(collapse_vals)
    if int(collapse_vals.numel()) <= 0:
        frac = 0.0
    else:
        frac = float((collapse_vals > 0.8).float().mean().item())
    return {
        "mean": float(st["mean"]),
        "std": float(st["std"]),
        "p90": float(st["p90"]),
        "ratio_gt_0p8": float(frac),
    }


@torch.no_grad()
def run_trials(
    *,
    encoder: StyleEncoderModule,
    lmdb_path: Path,
    fonts: List[str],
    font_infos: Dict[str, Dict[str, Any]],
    ref_per_style: int,
    trials: int,
    batch_size: int,
    consistency_sets_per_font: int,
    gallery_sets_per_font: int,
    query_sets_per_font: int,
    transform_query,
    transform_gallery,
    transform_consistency,
    seed: int,
    workers: int,
    prefetch_factor: int,
    worker_torch_threads: int,
    decode_cache_size: int,
) -> Dict[str, Any]:
    out_rows: List[Dict[str, float]] = []
    for t in range(int(trials)):
        print(f"[retrieval] trial {t + 1}/{int(trials)}", flush=True)
        cons_loader = build_retrieval_loader(
            lmdb_path=lmdb_path,
            fonts=fonts,
            font_infos=font_infos,
            ref_per_style=int(ref_per_style),
            sets_per_font=int(consistency_sets_per_font),
            transform=transform_consistency,
            seed_base=int(seed) + t * 2003 + 7,
            batch_size=int(batch_size),
            workers=int(workers),
            prefetch_factor=int(prefetch_factor),
            worker_torch_threads=int(worker_torch_threads),
            decode_cache_size=int(decode_cache_size),
        )
        cons_emb, cons_labels, cons_collapse = encode_font_set(encoder=encoder, loader=cons_loader)
        cons = consistency_metrics(cons_emb, cons_labels)
        coll = token_collapse_metrics(cons_collapse)

        gallery_loader = build_retrieval_loader(
            lmdb_path=lmdb_path,
            fonts=fonts,
            font_infos=font_infos,
            ref_per_style=int(ref_per_style),
            sets_per_font=int(gallery_sets_per_font),
            transform=transform_gallery,
            seed_base=int(seed) + t * 2003 + 11,
            batch_size=int(batch_size),
            workers=int(workers),
            prefetch_factor=int(prefetch_factor),
            worker_torch_threads=int(worker_torch_threads),
            decode_cache_size=int(decode_cache_size),
        )
        query_loader = build_retrieval_loader(
            lmdb_path=lmdb_path,
            fonts=fonts,
            font_infos=font_infos,
            ref_per_style=int(ref_per_style),
            sets_per_font=int(query_sets_per_font),
            transform=transform_query,
            seed_base=int(seed) + t * 2003 + 97,
            batch_size=int(batch_size),
            workers=int(workers),
            prefetch_factor=int(prefetch_factor),
            worker_torch_threads=int(worker_torch_threads),
            decode_cache_size=int(decode_cache_size),
        )
        gallery_emb, gallery_labels, _ = encode_font_set(
            encoder=encoder,
            loader=gallery_loader,
        )
        query_emb, query_labels, _ = encode_font_set(
            encoder=encoder,
            loader=query_loader,
        )
        retr = retrieval_metrics_with_labels(
            query_emb=query_emb,
            query_labels=query_labels,
            gallery_emb=gallery_emb,
            gallery_labels=gallery_labels,
            topk=5,
        )
        row = {
            "retr_top1": float(retr["top1"]),
            "retr_top5": float(retr["top5"]),
            "retr_mrr": float(retr["mrr"]),
            "cons_same_mean": float(cons["same_mean"]),
            "cons_diff_mean": float(cons["diff_mean"]),
            "cons_gap_same_minus_diff": float(cons["gap_same_minus_diff"]),
            "cons_same_p50": float(cons["same_p50"]),
            "cons_diff_p50": float(cons["diff_p50"]),
            "token_collapse_mean": float(coll["mean"]),
            "token_collapse_p90": float(coll["p90"]),
            "token_collapse_ratio_gt_0p8": float(coll["ratio_gt_0p8"]),
        }
        out_rows.append(row)

    return {"summary": aggregate_rows(out_rows), "trials": out_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--cluster-json", type=Path, default=Path("CharacterData/reference_cluster.json"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/style_retrieval_val_metrics.json"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--ref-per-style", type=int, default=12)
    parser.add_argument("--min-keep", type=int, default=4)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--consistency-sets-per-font", type=int, default=20)
    parser.add_argument("--gallery-sets-per-font", type=int, default=10)
    parser.add_argument("--query-sets-per-font", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--decode-cache-size", type=int, default=3000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--style-token-count", type=int, default=8)
    parser.add_argument("--style-start-channel", type=int, default=16)

    parser.add_argument("--style-aug-canvas-size", type=int, default=256)
    parser.add_argument("--style-aug-crop-min", type=float, default=0.6)
    parser.add_argument("--style-aug-crop-max", type=float, default=0.9)
    parser.add_argument("--style-aug-mask-prob", type=float, default=0.5)
    parser.add_argument("--style-aug-mask-min", type=float, default=0.15)
    parser.add_argument("--style-aug-mask-max", type=float, default=0.3)
    parser.add_argument("--style-aug-affine-deg", type=float, default=5.0)
    parser.add_argument("--style-aug-translate", type=float, default=0.05)
    parser.add_argument("--style-aug-scale-min", type=float, default=1.0)
    parser.add_argument("--style-aug-scale-max", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    root = args.project_root.resolve()
    lmdb_path = resolve_path(root, args.train_lmdb)
    cluster_path = resolve_path(root, args.cluster_json)
    ckpt_path = resolve_path(root, args.ckpt)
    out_json = resolve_path(root, args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    font_infos = build_font_infos(env, cluster_path, min_keep=int(args.min_keep))
    train_fonts, val_fonts = split_fonts(list(font_infos.keys()), float(args.val_ratio), int(args.seed))
    if len(val_fonts) < 2:
        raise RuntimeError(f"val fonts too few for retrieval: {len(val_fonts)}")

    encoder = make_encoder_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        style_start_channel=int(args.style_start_channel),
        style_token_dim=int(args.style_token_dim),
        style_token_count=int(args.style_token_count),
    )

    clean_transform = build_style_reference_transform(image_size=128, augment=False)
    aug_transform = build_style_reference_transform(
        image_size=128,
        augment=True,
        pre_resize=int(args.style_aug_canvas_size),
        crop_scale_min=float(args.style_aug_crop_min),
        crop_scale_max=float(args.style_aug_crop_max),
        mask_prob=float(args.style_aug_mask_prob),
        mask_area_min=float(args.style_aug_mask_min),
        mask_area_max=float(args.style_aug_mask_max),
        affine_degrees=float(args.style_aug_affine_deg),
        affine_translate=float(args.style_aug_translate),
        affine_scale_min=float(args.style_aug_scale_min),
        affine_scale_max=float(args.style_aug_scale_max),
    )

    clean = run_trials(
        encoder=encoder,
        lmdb_path=lmdb_path,
        fonts=val_fonts,
        font_infos=font_infos,
        ref_per_style=int(args.ref_per_style),
        trials=int(args.trials),
        batch_size=int(args.batch_size),
        consistency_sets_per_font=int(args.consistency_sets_per_font),
        gallery_sets_per_font=int(args.gallery_sets_per_font),
        query_sets_per_font=int(args.query_sets_per_font),
        transform_query=clean_transform,
        transform_gallery=clean_transform,
        transform_consistency=clean_transform,
        seed=int(args.seed) + 11,
        workers=int(args.workers),
        prefetch_factor=int(args.prefetch_factor),
        worker_torch_threads=int(args.worker_torch_threads),
        decode_cache_size=int(args.decode_cache_size),
    )
    aug = run_trials(
        encoder=encoder,
        lmdb_path=lmdb_path,
        fonts=val_fonts,
        font_infos=font_infos,
        ref_per_style=int(args.ref_per_style),
        trials=int(args.trials),
        batch_size=int(args.batch_size),
        consistency_sets_per_font=int(args.consistency_sets_per_font),
        gallery_sets_per_font=int(args.gallery_sets_per_font),
        query_sets_per_font=int(args.query_sets_per_font),
        transform_query=aug_transform,
        transform_gallery=aug_transform,
        transform_consistency=aug_transform,
        seed=int(args.seed) + 97,
        workers=int(args.workers),
        prefetch_factor=int(args.prefetch_factor),
        worker_torch_threads=int(args.worker_torch_threads),
        decode_cache_size=int(args.decode_cache_size),
    )

    payload = {
        "ckpt": str(ckpt_path),
        "lmdb": str(lmdb_path),
        "cluster_json": str(cluster_path),
        "device": str(device),
        "val_ratio": float(args.val_ratio),
        "val_fonts": int(len(val_fonts)),
        "ref_per_style": int(args.ref_per_style),
        "trials": int(args.trials),
        "workers": int(args.workers),
        "consistency_sets_per_font": int(args.consistency_sets_per_font),
        "gallery_sets_per_font": int(args.gallery_sets_per_font),
        "query_sets_per_font": int(args.query_sets_per_font),
        "clean": clean["summary"],
        "aug": aug["summary"],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[retrieval] saved: {out_json}")
    env.close()


if __name__ == "__main__":
    main()
