#!/usr/bin/env python3
"""Style encoder pretraining for low/mid/high semantic tokens."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import lmdb
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from models.hierarchical_style_encoder import HierarchicalStyleEncoderMixin
from style_augment import build_style_reference_transform


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def resolve_path(root: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (root / path_value).resolve()


def scan_train_lmdb_font_chars(env: lmdb.Environment) -> Dict[str, List[str]]:
    out: Dict[str, set[str]] = {}
    txn = env.begin(buffers=True)
    cursor = txn.cursor()
    for raw_key, _ in cursor:
        key = bytes(raw_key) if isinstance(raw_key, memoryview) else raw_key
        if b"@" not in key:
            continue
        try:
            text = key.decode("utf-8")
        except UnicodeDecodeError:
            continue
        font, ch = text.split("@", 1)
        if not font or not ch:
            continue
        out.setdefault(font, set()).add(ch)
    return {k: sorted(v) for k, v in sorted(out.items())}


def split_fonts(font_names: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    fonts = list(sorted(font_names))
    if val_ratio <= 0.0 or len(fonts) < 2:
        return fonts, []
    rng = random.Random(int(seed) + 1009)
    rng.shuffle(fonts)
    val_n = int(round(len(fonts) * float(val_ratio)))
    val_n = max(1, min(len(fonts) - 1, val_n))
    return fonts[val_n:], fonts[:val_n]


def split_fonts_by_fixed_counts(
    font_names: List[str],
    train_count: int,
    val_count: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    fonts = list(sorted(font_names))
    if len(fonts) < 2:
        return fonts, []
    if train_count <= 0 or val_count <= 0:
        raise ValueError("fixed split requires train_count>0 and val_count>0")
    if train_count + val_count > len(fonts):
        raise ValueError(
            f"fixed split too large: train+val={train_count + val_count} > total={len(fonts)}"
        )
    rng = random.Random(int(seed) + 1009)
    rng.shuffle(fonts)
    val_fonts = fonts[:val_count]
    train_fonts = fonts[val_count : val_count + train_count]
    return train_fonts, val_fonts


class StyleEncoderModule(nn.Module, HierarchicalStyleEncoderMixin):
    """Standalone style branch with the same core as SourcePartRefUNet."""

    def __init__(
        self,
        in_channels: int = 1,
        style_start_channel: int = 16,
        style_token_dim: int = 256,
        style_token_count: int = 3,
        local_token_count: int = 3,
    ):
        super().__init__()
        _ = style_start_channel
        self._init_hierarchical_style_encoder(
            in_channels=int(in_channels),
            style_token_dim=int(style_token_dim),
            style_token_count=int(style_token_count),
            local_token_count=int(local_token_count),
        )

    def _normalize_style_inputs(
        self,
        style_refs: torch.Tensor,
        style_ref_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if style_refs.dim() != 5:
            raise ValueError(f"style_refs must be 5D (B,R,C,H,W), got {tuple(style_refs.shape)}")
        b, r, c, _, _ = style_refs.shape
        if c != self.in_channels:
            raise ValueError(f"style_refs channels mismatch: got {c}, expected {self.in_channels}")
        if style_ref_mask is None:
            style_ref_mask = torch.ones((b, r), device=style_refs.device, dtype=torch.float32)
        else:
            style_ref_mask = style_ref_mask.to(device=style_refs.device, dtype=torch.float32)
        return style_refs, style_ref_mask


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (z1 @ z2.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_fill(eye, 0.0)
    denom = max(1, int(sim.size(-1) * (sim.size(-1) - 1)))
    return (off_diag.pow(2).sum(dim=(1, 2)) / float(denom)).mean()


def token_collapse_score(tokens: torch.Tensor) -> torch.Tensor:
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(int(sim.size(0)), -1)
    return off_diag.mean()


def slot_consistency_loss(tok1: torch.Tensor, tok2: torch.Tensor) -> torch.Tensor:
    if tok1.shape != tok2.shape:
        raise ValueError(f"slot consistency shape mismatch: {tuple(tok1.shape)} vs {tuple(tok2.shape)}")
    t1 = F.normalize(tok1, dim=-1)
    t2 = F.normalize(tok2, dim=-1)
    return (1.0 - (t1 * t2).sum(dim=-1)).mean()


def cosine_same_diff(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = F.cosine_similarity(z1, z2, dim=-1).mean()
    sim = z1 @ z2.t()
    b = int(sim.size(0))
    if b <= 1:
        diff = torch.zeros((), device=sim.device, dtype=sim.dtype)
    else:
        eye = torch.eye(b, device=sim.device, dtype=torch.bool)
        diff = sim.masked_select(~eye).mean()
    return same, diff


def retrieval_topk(z1: torch.Tensor, z2: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    logits = z1 @ z2.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    pred1 = logits.argmax(dim=1)
    top1 = (pred1 == labels).float().mean()
    kk = min(max(1, int(k)), int(logits.size(1)))
    predk = torch.topk(logits, k=kk, dim=1, largest=True).indices
    topk = predk.eq(labels.unsqueeze(1)).any(dim=1).float().mean()
    return top1, topk


def proxy_band_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def sample_refs_from_buckets(
    bucket_to_chars: Dict[int, List[str]],
    all_chars: List[str],
    ref_per_style: int,
    rng: random.Random,
    avoid_chars: set[str] | None = None,
) -> List[str]:
    if not all_chars:
        raise RuntimeError("sample_refs_from_buckets got empty char list")

    target = max(1, int(ref_per_style))
    avoid = set() if avoid_chars is None else set(avoid_chars)
    buckets = [b for b, chars in bucket_to_chars.items() if chars]
    rng.shuffle(buckets)

    chosen: List[str] = []
    chosen_set: set[str] = set()
    for b in buckets:
        if len(chosen) >= target:
            break
        chars = bucket_to_chars[b]
        cand = [ch for ch in chars if ch not in chosen_set and ch not in avoid]
        if not cand:
            cand = [ch for ch in chars if ch not in chosen_set]
        if not cand:
            cand = list(chars)
        pick = rng.choice(cand)
        chosen.append(pick)
        chosen_set.add(pick)

    prefer = [ch for ch in all_chars if ch not in chosen_set and ch not in avoid]
    rng.shuffle(prefer)
    while len(chosen) < target and prefer:
        pick = prefer.pop()
        chosen.append(pick)
        chosen_set.add(pick)

    backup = [ch for ch in all_chars if ch not in chosen_set]
    rng.shuffle(backup)
    while len(chosen) < target and backup:
        pick = backup.pop()
        chosen.append(pick)
        chosen_set.add(pick)

    while len(chosen) < target:
        chosen.append(rng.choice(all_chars))
    return chosen


def apply_reference_dropout(
    n_ref: int,
    p_ref_drop: float,
    min_keep: int,
    rng: random.Random,
) -> torch.Tensor:
    keep = torch.ones((int(n_ref),), dtype=torch.float32)
    for i in range(int(n_ref)):
        if rng.random() < float(p_ref_drop):
            keep[i] = 0.0

    cur_keep = int((keep > 0).sum().item())
    need = max(1, int(min_keep)) - cur_keep
    if need > 0:
        dropped = [i for i in range(int(n_ref)) if keep[i] <= 0]
        rng.shuffle(dropped)
        for i in dropped[:need]:
            keep[i] = 1.0
    return keep


def load_ref_tensor(
    txn,
    font_name: str,
    ch: str,
    transform,
    decode_cache: OrderedDict[str, np.ndarray] | None = None,
    decode_cache_size: int = 0,
) -> torch.Tensor:
    key_text = f"{font_name}@{ch}"
    base_u8: np.ndarray | None = None
    if decode_cache is not None:
        base_u8 = decode_cache.get(key_text, None)
        if base_u8 is not None:
            decode_cache.move_to_end(key_text, last=True)

    if base_u8 is None:
        value = txn.get(key_text.encode("utf-8"))
        if value is None:
            raise KeyError(f"Missing LMDB key: {key_text}")
        value_u8 = np.frombuffer(value, dtype=np.uint8)
        decoded = cv2.imdecode(value_u8, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            img = Image.open(io.BytesIO(bytes(value))).convert("L")
            base_u8 = np.asarray(img, dtype=np.uint8)
        else:
            base_u8 = np.ascontiguousarray(decoded)
        if decode_cache is not None and decode_cache_size > 0:
            decode_cache[key_text] = base_u8
            decode_cache.move_to_end(key_text, last=True)
            while len(decode_cache) > int(decode_cache_size):
                decode_cache.popitem(last=False)

    return transform(base_u8)


class StylePretrainDataset(Dataset):
    """Randomized style dual-view sampling from TrainFont LMDB."""

    def __init__(
        self,
        lmdb_path: Path,
        font_infos: Dict[str, Dict[str, Any]],
        font_pool: List[str],
        ref_per_style: int,
        p_ref_drop: float,
        min_keep: int,
        transform,
        decode_cache_size: int,
        seed: int,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.font_infos = font_infos
        self.font_pool = list(font_pool)
        self.ref_per_style = int(ref_per_style)
        self.p_ref_drop = float(p_ref_drop)
        self.min_keep = int(min_keep)
        self.transform = transform
        self.decode_cache_size = max(0, int(decode_cache_size))
        self.seed = int(seed)

        self._env: lmdb.Environment | None = None
        self._txn = None
        self._worker_id = -1
        self._decode_cache: OrderedDict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.font_pool)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_worker_state()
        assert self._txn is not None

        font_idx = int(index) % len(self.font_pool)
        font = self.font_pool[font_idx]
        info = self.font_infos[font]
        rng = random.Random(self.seed + font_idx * 1000003)

        chars_v1 = sample_refs_from_buckets(
            info["bucket_to_chars"],
            info["all_chars"],
            self.ref_per_style,
            rng,
        )
        chars_v2 = sample_refs_from_buckets(
            info["bucket_to_chars"],
            info["all_chars"],
            self.ref_per_style,
            rng,
            avoid_chars=set(chars_v1),
        )
        keep_v1 = apply_reference_dropout(len(chars_v1), self.p_ref_drop, self.min_keep, rng)
        keep_v2 = apply_reference_dropout(len(chars_v2), self.p_ref_drop, self.min_keep, rng)

        refs_v1 = torch.stack(
            [
                load_ref_tensor(
                    self._txn,
                    font,
                    ch,
                    self.transform,
                    decode_cache=self._decode_cache,
                    decode_cache_size=self.decode_cache_size,
                )
                for ch in chars_v1
            ],
            dim=0,
        )
        refs_v2 = torch.stack(
            [
                load_ref_tensor(
                    self._txn,
                    font,
                    ch,
                    self.transform,
                    decode_cache=self._decode_cache,
                    decode_cache_size=self.decode_cache_size,
                )
                for ch in chars_v2
            ],
            dim=0,
        )
        return refs_v1, keep_v1, refs_v2, keep_v2


class NoReplacementBatchSampler(Sampler[List[int]]):
    def __init__(self, item_count: int, batch_size: int, steps_per_epoch: int, seed: int, shuffle: bool = True):
        self.item_count = int(item_count)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        if self.batch_size <= 0 or self.batch_size > self.item_count:
            raise ValueError(f"invalid batch_size={self.batch_size} for item_count={self.item_count}")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch * 100003)
        self._epoch += 1
        produced = 0
        while produced < self.steps_per_epoch:
            order = list(range(self.item_count))
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    break
                yield batch
                produced += 1
                if produced >= self.steps_per_epoch:
                    break


def build_style_loader(
    lmdb_path: Path,
    font_infos: Dict[str, Dict[str, Any]],
    font_pool: List[str],
    batch_size: int,
    ref_per_style: int,
    p_ref_drop: float,
    min_keep: int,
    transform,
    decode_workers: int,
    decode_cache_size: int,
    seed: int,
    shuffle: bool,
    steps_per_epoch: int,
    prefetch_factor: int,
    worker_torch_threads: int,
) -> DataLoader:
    if int(batch_size) > len(font_pool):
        raise ValueError(
            f"batch_size={int(batch_size)} exceeds font_pool={len(font_pool)}; "
            "this violates no-replacement batch sampling."
        )
    ds = StylePretrainDataset(
        lmdb_path=lmdb_path,
        font_infos=font_infos,
        font_pool=font_pool,
        ref_per_style=ref_per_style,
        p_ref_drop=p_ref_drop,
        min_keep=min_keep,
        transform=transform,
        decode_cache_size=decode_cache_size,
        seed=seed,
    )
    nw = max(0, int(decode_workers))
    batch_sampler = NoReplacementBatchSampler(
        item_count=len(font_pool),
        batch_size=int(batch_size),
        steps_per_epoch=max(1, int(steps_per_epoch)),
        seed=int(seed) + 17,
        shuffle=bool(shuffle),
    )
    kwargs: Dict[str, Any] = {}
    if nw > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = build_worker_init_fn(seed, int(worker_torch_threads))
    return DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=nw,
        pin_memory=True,
        **kwargs,
    )


def make_style_encoder_state_dict(encoder: StyleEncoderModule) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}


def next_loader_batch(loader: DataLoader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def compute_proxy_losses(proxy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "loss_proxy_low": proxy_band_loss(proxy["pred_low"], proxy["target_low"]),
        "loss_proxy_mid": proxy_band_loss(proxy["pred_mid"], proxy["target_mid"]),
        "loss_proxy_high": proxy_band_loss(proxy["pred_high"], proxy["target_high"]),
    }


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = sorted({k for row in metrics_list for k in row.keys()})
    return {k: float(np.mean([row[k] for row in metrics_list if k in row])) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--cluster-json", type=Path, default=Path("CharacterData/reference_cluster.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/style_encoder_pretrain.pt"))
    parser.add_argument("--log-file", type=Path, default=Path("checkpoints/style_encoder_pretrain.log"))
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("checkpoints/style_encoder_pretrain.metrics.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-steps-per-epoch", type=int, default=2048)
    parser.add_argument("--val-loader-steps-per-epoch", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-font-count", type=int, default=0)
    parser.add_argument("--val-font-count", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ref-per-style", type=int, default=12)
    parser.add_argument("--style-token-count", type=int, default=3)
    parser.add_argument("--local-token-count", type=int, default=3)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--style-start-channel", type=int, default=16)
    parser.add_argument("--decode-cache-size", type=int, default=3000)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--worker-torch-threads", type=int, default=1)
    parser.add_argument("--p-ref-drop", type=float, default=0.25)
    parser.add_argument("--min-keep", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-cons", type=float, default=0.15)
    parser.add_argument("--lambda-div", type=float, default=0.0)
    parser.add_argument("--lambda-proxy-low", type=float, default=0.20)
    parser.add_argument("--lambda-proxy-mid", type=float, default=0.20)
    parser.add_argument("--lambda-proxy-high", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    root = args.project_root.resolve()
    lmdb_path = resolve_path(root, args.train_lmdb)
    cluster_path = resolve_path(root, args.cluster_json)
    out_path = resolve_path(root, args.out)
    log_path = resolve_path(root, args.log_file)
    metrics_path = resolve_path(root, args.metrics_jsonl)
    init_ckpt_path = resolve_path(root, args.init_checkpoint) if args.init_checkpoint is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    if not lmdb_path.exists():
        raise FileNotFoundError(f"TrainFont LMDB not found: {lmdb_path}")
    if not cluster_path.exists():
        raise FileNotFoundError(f"reference cluster json not found: {cluster_path}")

    cluster_raw = json.loads(cluster_path.read_text(encoding="utf-8"))
    char_to_bucket: Dict[str, int] = {str(k): int(v) for k, v in cluster_raw.items()}

    env_scan = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    font_chars = scan_train_lmdb_font_chars(env_scan)
    env_scan.close()
    if len(font_chars) < 2:
        raise RuntimeError("Train LMDB has too few fonts for style pretraining.")

    font_infos: Dict[str, Dict[str, Any]] = {}
    for font, chars in font_chars.items():
        bucket_to_chars: Dict[int, List[str]] = {}
        for ch in chars:
            if ch not in char_to_bucket:
                continue
            bucket_to_chars.setdefault(int(char_to_bucket[ch]), []).append(ch)
        all_chars = sorted({ch for arr in bucket_to_chars.values() for ch in arr})
        if len(all_chars) >= max(1, int(args.min_keep)):
            font_infos[font] = {"bucket_to_chars": bucket_to_chars, "all_chars": all_chars}

    if len(font_infos) < 2:
        raise RuntimeError("Not enough fonts with clustered refs for style pretraining.")

    if int(args.train_font_count) > 0 or int(args.val_font_count) > 0:
        train_fonts, val_fonts = split_fonts_by_fixed_counts(
            list(font_infos.keys()),
            train_count=int(args.train_font_count),
            val_count=int(args.val_font_count),
            seed=int(args.seed),
        )
    else:
        train_fonts, val_fonts = split_fonts(list(font_infos.keys()), float(args.val_ratio), int(args.seed))
    if len(train_fonts) < 2:
        raise RuntimeError("Train split has too few fonts for style pretraining.")
    has_val = len(val_fonts) >= 2

    transform = build_style_reference_transform(image_size=int(args.image_size))

    encoder = StyleEncoderModule(
        in_channels=1,
        style_start_channel=int(args.style_start_channel),
        style_token_dim=int(args.style_token_dim),
        style_token_count=int(args.style_token_count),
        local_token_count=int(args.local_token_count),
    ).to(device)

    if init_ckpt_path is not None:
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_ckpt_path}")
        init_state = torch.load(init_ckpt_path, map_location="cpu")
        if isinstance(init_state, dict) and "style_encoder" in init_state:
            init_state = init_state["style_encoder"]
        if not isinstance(init_state, dict):
            raise RuntimeError(f"unexpected style_encoder checkpoint format: {type(init_state)!r}")
        missing, unexpected = encoder.load_state_dict(init_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "init checkpoint mismatch: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )

    opt = torch.optim.AdamW(encoder.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = build_style_loader(
        lmdb_path=lmdb_path,
        font_infos=font_infos,
        font_pool=train_fonts,
        batch_size=int(args.batch_size),
        ref_per_style=int(args.ref_per_style),
        p_ref_drop=float(args.p_ref_drop),
        min_keep=int(args.min_keep),
        transform=transform,
        decode_workers=int(args.decode_workers),
        decode_cache_size=int(args.decode_cache_size),
        seed=int(args.seed) + 101,
        shuffle=True,
        steps_per_epoch=int(args.loader_steps_per_epoch),
        prefetch_factor=int(args.prefetch_factor),
        worker_torch_threads=int(args.worker_torch_threads),
    )
    train_iter = iter(train_loader)

    val_loader = None
    val_iter = None
    if has_val:
        val_loader = build_style_loader(
            lmdb_path=lmdb_path,
            font_infos=font_infos,
            font_pool=val_fonts,
            batch_size=min(int(args.batch_size), len(val_fonts)),
            ref_per_style=int(args.ref_per_style),
            p_ref_drop=float(args.p_ref_drop),
            min_keep=int(args.min_keep),
            transform=transform,
            decode_workers=int(args.decode_workers),
            decode_cache_size=int(args.decode_cache_size),
            seed=int(args.seed) + 211,
            shuffle=True,
            steps_per_epoch=int(args.val_loader_steps_per_epoch),
            prefetch_factor=int(args.prefetch_factor),
            worker_torch_threads=int(args.worker_torch_threads),
        )
        val_iter = iter(val_loader)

    log_fp = log_path.open("w", encoding="utf-8")
    metrics_fp = metrics_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log(
        f"device={device} train_fonts={len(train_fonts)} val_fonts={len(val_fonts)} "
        f"steps={int(args.steps)} batch={int(args.batch_size)} ref={int(args.ref_per_style)} "
        f"lambda_cons={float(args.lambda_cons):g} lambda_div={float(args.lambda_div):g} "
        f"lambda_proxy_low={float(args.lambda_proxy_low):g} "
        f"lambda_proxy_mid={float(args.lambda_proxy_mid):g} "
        f"lambda_proxy_high={float(args.lambda_proxy_high):g}"
    )

    best_val_loss: float | None = None
    best_step = 0
    last_step = 0

    try:
        for step in range(1, int(args.steps) + 1):
            encoder.train()
            (v1, m1, v2, m2), train_iter = next_loader_batch(train_loader, train_iter)
            v1 = v1.to(device, non_blocking=True)
            m1 = m1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            m2 = m2.to(device, non_blocking=True)

            tok1, proxy1 = encoder.encode_style_tokens_with_proxy(v1, m1)
            tok2, proxy2 = encoder.encode_style_tokens_with_proxy(v2, m2)
            z1 = F.normalize(tok1.mean(dim=1), dim=-1)
            z2 = F.normalize(tok2.mean(dim=1), dim=-1)

            loss_nce = info_nce_loss(z1, z2, temperature=float(args.tau))
            loss_cons = slot_consistency_loss(tok1, tok2)
            loss_div = 0.5 * (token_diversity_loss(tok1) + token_diversity_loss(tok2))
            loss_collapse = 0.5 * (token_collapse_score(tok1) + token_collapse_score(tok2))
            proxy_loss1 = compute_proxy_losses(proxy1)
            proxy_loss2 = compute_proxy_losses(proxy2)
            loss_proxy_low = 0.5 * (proxy_loss1["loss_proxy_low"] + proxy_loss2["loss_proxy_low"])
            loss_proxy_mid = 0.5 * (proxy_loss1["loss_proxy_mid"] + proxy_loss2["loss_proxy_mid"])
            loss_proxy_high = 0.5 * (proxy_loss1["loss_proxy_high"] + proxy_loss2["loss_proxy_high"])
            loss = (
                loss_nce
                + float(args.lambda_cons) * loss_cons
                + float(args.lambda_div) * loss_div
                + float(args.lambda_proxy_low) * loss_proxy_low
                + float(args.lambda_proxy_mid) * loss_proxy_mid
                + float(args.lambda_proxy_high) * loss_proxy_high
            )
            train_cos_same, train_cos_diff = cosine_same_diff(z1, z2)
            train_retr_top1, train_retr_top5 = retrieval_topk(z1, z2, k=5)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            last_step = step
            should_log = (step % int(args.log_every) == 0) or (step == 1) or (step == int(args.steps))
            if not should_log:
                continue

            val_metrics: Dict[str, float] = {}
            if has_val and val_loader is not None and val_iter is not None:
                encoder.eval()
                rows: List[Dict[str, float]] = []
                with torch.no_grad():
                    for _ in range(int(args.val_batches)):
                        (ev1, em1, ev2, em2), val_iter = next_loader_batch(val_loader, val_iter)
                        ev1 = ev1.to(device, non_blocking=True)
                        em1 = em1.to(device, non_blocking=True)
                        ev2 = ev2.to(device, non_blocking=True)
                        em2 = em2.to(device, non_blocking=True)

                        et1, eproxy1 = encoder.encode_style_tokens_with_proxy(ev1, em1)
                        et2, eproxy2 = encoder.encode_style_tokens_with_proxy(ev2, em2)
                        ez1 = F.normalize(et1.mean(dim=1), dim=-1)
                        ez2 = F.normalize(et2.mean(dim=1), dim=-1)

                        el_nce = info_nce_loss(ez1, ez2, temperature=float(args.tau))
                        el_cons = slot_consistency_loss(et1, et2)
                        el_div = 0.5 * (token_diversity_loss(et1) + token_diversity_loss(et2))
                        el_collapse = 0.5 * (token_collapse_score(et1) + token_collapse_score(et2))
                        eproxy_loss1 = compute_proxy_losses(eproxy1)
                        eproxy_loss2 = compute_proxy_losses(eproxy2)
                        el_proxy_low = 0.5 * (eproxy_loss1["loss_proxy_low"] + eproxy_loss2["loss_proxy_low"])
                        el_proxy_mid = 0.5 * (eproxy_loss1["loss_proxy_mid"] + eproxy_loss2["loss_proxy_mid"])
                        el_proxy_high = 0.5 * (eproxy_loss1["loss_proxy_high"] + eproxy_loss2["loss_proxy_high"])
                        el_total = (
                            el_nce
                            + float(args.lambda_cons) * el_cons
                            + float(args.lambda_div) * el_div
                            + float(args.lambda_proxy_low) * el_proxy_low
                            + float(args.lambda_proxy_mid) * el_proxy_mid
                            + float(args.lambda_proxy_high) * el_proxy_high
                        )
                        ecos_same, ecos_diff = cosine_same_diff(ez1, ez2)
                        etop1, etop5 = retrieval_topk(ez1, ez2, k=5)
                        rows.append(
                            {
                                "val_loss": float(el_total.item()),
                                "val_loss_nce": float(el_nce.item()),
                                "val_loss_cons": float(el_cons.item()),
                                "val_loss_div": float(el_div.item()),
                                "val_loss_proxy_low": float(el_proxy_low.item()),
                                "val_loss_proxy_mid": float(el_proxy_mid.item()),
                                "val_loss_proxy_high": float(el_proxy_high.item()),
                                "val_retr_top1": float(etop1.item()),
                                "val_retr_top5": float(etop5.item()),
                                "val_cos_same": float(ecos_same.item()),
                                "val_cos_diff": float(ecos_diff.item()),
                                "val_token_collapse": float(el_collapse.item()),
                            }
                        )
                val_metrics = summarize_metrics(rows)
                val_loss = float(val_metrics.get("val_loss", 0.0))
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    torch.save(
                        {
                            "style_encoder": make_style_encoder_state_dict(encoder),
                            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                            "extra": {"step": int(step), "best_val_loss": float(val_loss)},
                        },
                        out_path.with_name("style_encoder_pretrain_best.pt"),
                    )

            metrics_row: Dict[str, float | int] = {
                "step": int(step),
                "loss": float(loss.item()),
                "loss_nce": float(loss_nce.item()),
                "loss_cons": float(loss_cons.item()),
                "loss_div": float(loss_div.item()),
                "loss_proxy_low": float(loss_proxy_low.item()),
                "loss_proxy_mid": float(loss_proxy_mid.item()),
                "loss_proxy_high": float(loss_proxy_high.item()),
                "retr_top1": float(train_retr_top1.item()),
                "retr_top5": float(train_retr_top5.item()),
                "cos_same": float(train_cos_same.item()),
                "cos_diff": float(train_cos_diff.item()),
                "token_collapse": float(loss_collapse.item()),
            }
            metrics_row.update(val_metrics)
            metrics_fp.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")
            metrics_fp.flush()

            msg = (
                f"step={step}/{int(args.steps)} loss={loss.item():.4f} "
                f"nce={loss_nce.item():.4f} cons={loss_cons.item():.4f} div={loss_div.item():.4f} "
                f"proxy_low={loss_proxy_low.item():.4f} proxy_mid={loss_proxy_mid.item():.4f} proxy_high={loss_proxy_high.item():.4f} "
                f"retr1={train_retr_top1.item():.4f} retr5={train_retr_top5.item():.4f} "
                f"cos_same={train_cos_same.item():.4f} cos_diff={train_cos_diff.item():.4f} "
                f"tok_coll={loss_collapse.item():.4f}"
            )
            if val_metrics:
                msg += (
                    f" val_loss={val_metrics['val_loss']:.4f} "
                    f"val_retr1={val_metrics['val_retr_top1']:.4f} "
                    f"val_retr5={val_metrics['val_retr_top5']:.4f}"
                )
            if best_val_loss is not None:
                msg += f" best_val={best_val_loss:.4f}@{best_step}"
            log(msg)
    finally:
        final_ckpt = {
            "style_encoder": make_style_encoder_state_dict(encoder),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "extra": {
                "step": int(last_step),
                "best_step": int(best_step),
                "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
                "ran_full_steps": bool(last_step == int(args.steps)),
            },
        }
        torch.save(final_ckpt, out_path)
        log(f"saved final checkpoint: {out_path}")
        log_fp.close()
        metrics_fp.close()


if __name__ == "__main__":
    main()
