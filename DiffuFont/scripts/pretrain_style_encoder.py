#!/usr/bin/env python3
"""Style encoder pretraining with bucket references and dual-view contrastive loss."""

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

import lmdb
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from PIL import Image

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from style_augment import build_style_reference_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def split_fonts(
    font_names: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
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
        raise ValueError(
            f"fixed split requires train_count>0 and val_count>0, got {train_count}/{val_count}"
        )
    if train_count + val_count > len(fonts):
        raise ValueError(
            f"fixed split too large: train+val={train_count + val_count} > total={len(fonts)}"
        )
    rng = random.Random(int(seed) + 1009)
    rng.shuffle(fonts)
    val_fonts = fonts[:val_count]
    train_fonts = fonts[val_count : val_count + train_count]
    return train_fonts, val_fonts


class StyleEncoderModule(nn.Module):
    """Standalone style branch with the same key names as SourcePartRefUNet."""

    def __init__(
        self,
        in_channels: int = 1,
        style_start_channel: int = 16,
        style_token_dim: int = 256,
        style_token_count: int = 8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.style_token_dim = int(style_token_dim)
        self.style_token_count = max(1, int(style_token_count))

        c1 = int(style_start_channel)
        c2 = c1 * 4
        c3 = c1 * 16

        self.style_img_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.style_feat_to_token = (
            nn.Identity() if c3 == self.style_token_dim else nn.Linear(c3, self.style_token_dim)
        )

        heads = 8 if (self.style_token_dim % 8 == 0) else (4 if (self.style_token_dim % 4 == 0) else 1)
        self.style_queries = nn.Parameter(torch.randn(self.style_token_count, self.style_token_dim) * 0.02)
        self.style_token_attn = nn.MultiheadAttention(
            embed_dim=self.style_token_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.style_token_norm = nn.LayerNorm(self.style_token_dim)
        self.style_token_ffn = nn.Sequential(
            nn.LayerNorm(self.style_token_dim),
            nn.Linear(self.style_token_dim, self.style_token_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.style_token_dim * 4, self.style_token_dim),
        )
        self.style_token_out_norm = nn.LayerNorm(self.style_token_dim)

    def encode_style_tokens(
        self,
        style_refs: torch.Tensor,
        style_ref_mask: torch.Tensor,
    ) -> torch.Tensor:
        # style_refs: (B, R, 1, 128, 128), style_ref_mask: (B, R)
        b, r, c, h, w = style_refs.shape
        if c != self.in_channels:
            raise ValueError(f"style_refs channels mismatch: got {c}, expected {self.in_channels}")

        x = style_refs.view(b * r, c, h, w)
        feat = self.style_img_encoder(x)
        bf, cf, hf, wf = feat.shape
        patch = feat.view(bf, cf, hf * wf).transpose(1, 2).contiguous()
        patch = self.style_feat_to_token(patch)
        patch = patch.view(b, r * hf * wf, self.style_token_dim)

        valid = (style_ref_mask > 0)
        patch_valid = valid.unsqueeze(-1).expand(b, r, hf * wf).reshape(b, r * hf * wf)
        key_padding_mask = ~patch_valid

        queries = self.style_queries.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.style_token_attn(
            query=queries,
            key=patch,
            value=patch,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = self.style_token_norm(attn_out + queries)
        tokens = self.style_token_out_norm(tokens + self.style_token_ffn(tokens))
        return tokens

    def encode_style_embedding(
        self,
        style_refs: torch.Tensor,
        style_ref_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.encode_style_tokens(style_refs, style_ref_mask)
        z = tokens.mean(dim=1)
        return F.normalize(z, dim=-1)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (z1 @ z2.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    # tokens: (B, T, D)
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_fill(eye, 0.0)
    denom = max(1, int(sim.size(-1) * (sim.size(-1) - 1)))
    return (off_diag.pow(2).sum(dim=(1, 2)) / float(denom)).mean()


def token_collapse_score(tokens: torch.Tensor) -> torch.Tensor:
    # Mean off-diagonal token cosine; high means collapse.
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(int(sim.size(0)), -1)
    return off_diag.mean()


def retrieval_topk(z1: torch.Tensor, z2: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    logits = z1 @ z2.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    pred1 = logits.argmax(dim=1)
    top1 = (pred1 == labels).float().mean()
    kk = min(max(1, int(k)), int(logits.size(1)))
    predk = torch.topk(logits, k=kk, dim=1, largest=True).indices
    topk = predk.eq(labels.unsqueeze(1)).any(dim=1).float().mean()
    return top1, topk


def cosine_same_diff(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # z1/z2: (B,D), already normalized.
    same = F.cosine_similarity(z1, z2, dim=-1).mean()
    sim = z1 @ z2.t()
    b = int(sim.size(0))
    if b <= 1:
        diff = torch.zeros((), device=sim.device, dtype=sim.dtype)
    else:
        eye = torch.eye(b, device=sim.device, dtype=torch.bool)
        diff = sim.masked_select(~eye).mean()
    return same, diff


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
    k_text = f"{font_name}@{ch}"
    base_u8: np.ndarray | None = None
    if decode_cache is not None:
        base_u8 = decode_cache.get(k_text, None)
        if base_u8 is not None:
            decode_cache.move_to_end(k_text, last=True)

    if base_u8 is None:
        key = k_text.encode("utf-8")
        value = txn.get(key)
        if value is None:
            raise KeyError(f"Missing LMDB key: {k_text}")
        value_u8 = np.frombuffer(value, dtype=np.uint8)
        decoded = cv2.imdecode(value_u8, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            img = Image.open(io.BytesIO(bytes(value))).convert("L")
            base_u8 = np.asarray(img, dtype=np.uint8)
        else:
            base_u8 = np.ascontiguousarray(decoded)
        if decode_cache is not None and decode_cache_size > 0:
            decode_cache[k_text] = base_u8
            decode_cache.move_to_end(k_text, last=True)
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
        dataset_size: int = 200000,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.font_infos = font_infos
        self.font_pool = list(font_pool)
        self.ref_per_style = int(ref_per_style)
        self.p_ref_drop = float(p_ref_drop)
        self.min_keep = int(min_keep)
        self.transform = transform
        self.decode_cache_size = max(0, int(decode_cache_size))
        self.base_seed = int(seed)

        self._env: lmdb.Environment | None = None
        self._txn = None
        self._rng = random.Random(self.base_seed)
        self._worker_id = -1
        self._decode_cache: OrderedDict[str, np.ndarray] | None = None
        self._font_count = len(self.font_pool)
        if self._font_count <= 0:
            raise ValueError("font_pool is empty for StylePretrainDataset")

    def __len__(self) -> int:
        return self._font_count

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
        self._rng = random.Random(self.base_seed + worker_id * 100003)
        self._decode_cache = OrderedDict() if self.decode_cache_size > 0 else None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_worker_state()
        assert self._txn is not None

        font = self.font_pool[int(index) % self._font_count]
        info = self.font_infos[font]
        chars1 = sample_refs_from_buckets(
            info["bucket_to_chars"],
            info["all_chars"],
            self.ref_per_style,
            self._rng,
        )
        chars2 = sample_refs_from_buckets(
            info["bucket_to_chars"],
            info["all_chars"],
            self.ref_per_style,
            self._rng,
            avoid_chars=set(chars1),
        )

        refs1 = torch.stack(
            [
                load_ref_tensor(
                    self._txn,
                    font,
                    ch,
                    self.transform,
                    decode_cache=self._decode_cache,
                    decode_cache_size=self.decode_cache_size,
                )
                for ch in chars1
            ],
            dim=0,
        )
        refs2 = torch.stack(
            [
                load_ref_tensor(
                    self._txn,
                    font,
                    ch,
                    self.transform,
                    decode_cache=self._decode_cache,
                    decode_cache_size=self.decode_cache_size,
                )
                for ch in chars2
            ],
            dim=0,
        )

        m1 = apply_reference_dropout(self.ref_per_style, self.p_ref_drop, self.min_keep, self._rng)
        m2 = apply_reference_dropout(self.ref_per_style, self.p_ref_drop, self.min_keep, self._rng)
        return refs1, m1, refs2, m2


class NoReplacementBatchSampler(Sampler[List[int]]):
    """Generate fixed-size batches with no repeated font index inside each batch."""

    def __init__(
        self,
        item_count: int,
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.item_count = int(item_count)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        if self.item_count <= 0:
            raise ValueError("NoReplacementBatchSampler item_count must be > 0")
        if self.batch_size <= 0:
            raise ValueError("NoReplacementBatchSampler batch_size must be > 0")
        if self.batch_size > self.item_count:
            raise ValueError(
                f"batch_size={self.batch_size} exceeds item_count={self.item_count}; "
                "cannot sample no-replacement batch"
            )

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch * 100003)
        self._epoch += 1

        produced = 0
        while produced < self.steps_per_epoch:
            if self.shuffle:
                order = torch.randperm(self.item_count, generator=gen).tolist()
            else:
                order = list(range(self.item_count))
            pos = 0
            while (pos + self.batch_size) <= self.item_count and produced < self.steps_per_epoch:
                yield order[pos : pos + self.batch_size]
                pos += self.batch_size
                produced += 1

    def __len__(self) -> int:
        return int(self.steps_per_epoch)


def next_loader_batch(
    loader: DataLoader,
    iterator,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Any]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


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
    pf = max(1, int(prefetch_factor))
    batch_sampler = NoReplacementBatchSampler(
        item_count=len(font_pool),
        batch_size=int(batch_size),
        steps_per_epoch=max(1, int(steps_per_epoch)),
        seed=int(seed) + 17,
        shuffle=bool(shuffle),
    )
    kwargs: Dict[str, Any] = {}
    if nw > 0:
        kwargs["prefetch_factor"] = pf
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
    return {
        k: v.detach().cpu().clone()
        for k, v in encoder.state_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-lmdb", type=Path, default=Path("DataPreparation/LMDB/TrainFont.lmdb"))
    parser.add_argument("--cluster-json", type=Path, default=Path("CharacterData/reference_cluster.json"))

    parser.add_argument("--out", type=Path, default=Path("checkpoints/style_encoder_pretrain.pt"))
    parser.add_argument("--log-file", type=Path, default=Path("checkpoints/style_encoder_pretrain.log"))
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("checkpoints/style_encoder_pretrain.metrics.jsonl"))

    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--loader-steps-per-epoch",
        type=int,
        default=2048,
        help="Number of training batches per DataLoader epoch to avoid frequent iterator resets.",
    )
    parser.add_argument(
        "--val-loader-steps-per-epoch",
        type=int,
        default=512,
        help="Number of validation batches per DataLoader epoch.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--train-font-count",
        type=int,
        default=0,
        help="Fixed train font count. >0 enables fixed-count split.",
    )
    parser.add_argument(
        "--val-font-count",
        type=int,
        default=0,
        help="Fixed val font count. >0 enables fixed-count split.",
    )
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--ref-per-style", type=int, default=12)
    parser.add_argument("--style-token-count", type=int, default=8)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--style-start-channel", type=int, default=16)
    parser.add_argument(
        "--decode-cache-size",
        type=int,
        default=3000,
        help="LRU cache size for decoded LMDB glyphs (uint8 tensors). 0 disables.",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=8,
        help="DataLoader workers for LMDB decode + augmentation.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor when decode-workers>0.",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="torch CPU threads per worker process (1 avoids CPU oversubscription).",
    )

    parser.add_argument("--p-ref-drop", type=float, default=0.25)
    parser.add_argument("--min-keep", type=int, default=4)

    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-cons", type=float, default=0.2)
    parser.add_argument("--lambda-div", type=float, default=0.05)
    parser.add_argument("--monitor-weight-retr", type=float, default=0.60)
    parser.add_argument("--monitor-weight-cons", type=float, default=0.30)
    parser.add_argument(
        "--time-ema-decay",
        type=float,
        default=0.90,
        help="EMA decay for data/train step time logs.",
    )

    parser.add_argument("--image-size", type=int, default=128)
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

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if int(args.views) != 2:
        raise ValueError("this pretrain script uses fixed dual views; set --views 2")

    root = args.project_root.resolve()
    lmdb_path = resolve_path(root, args.train_lmdb)
    cluster_path = resolve_path(root, args.cluster_json)

    out_path = resolve_path(root, args.out)
    log_path = resolve_path(root, args.log_file)
    metrics_path = resolve_path(root, args.metrics_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
            b = int(char_to_bucket[ch])
            bucket_to_chars.setdefault(b, []).append(ch)
        all_chars = sorted({ch for arr in bucket_to_chars.values() for ch in arr})
        if len(all_chars) >= max(1, int(args.min_keep)):
            font_infos[font] = {
                "bucket_to_chars": bucket_to_chars,
                "all_chars": all_chars,
            }

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

    transform = build_style_reference_transform(
        image_size=int(args.image_size),
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

    encoder = StyleEncoderModule(
        in_channels=1,
        style_start_channel=int(args.style_start_channel),
        style_token_dim=int(args.style_token_dim),
        style_token_count=int(args.style_token_count),
    ).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    decode_cache_size = max(0, int(args.decode_cache_size))
    decode_workers = max(0, int(args.decode_workers))
    batch_size = int(args.batch_size)
    val_batch_size = min(batch_size, len(val_fonts)) if has_val else batch_size

    train_loader = build_style_loader(
        lmdb_path=lmdb_path,
        font_infos=font_infos,
        font_pool=train_fonts,
        batch_size=batch_size,
        ref_per_style=int(args.ref_per_style),
        p_ref_drop=float(args.p_ref_drop),
        min_keep=int(args.min_keep),
        transform=transform,
        decode_cache_size=decode_cache_size,
        decode_workers=decode_workers,
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
            batch_size=val_batch_size,
            ref_per_style=int(args.ref_per_style),
            p_ref_drop=float(args.p_ref_drop),
            min_keep=int(args.min_keep),
            transform=transform,
            decode_cache_size=decode_cache_size,
            decode_workers=decode_workers,
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
        f"steps={int(args.steps)} batch={batch_size} ref={int(args.ref_per_style)} "
        f"loader_steps(train/val)={int(args.loader_steps_per_epoch)}/{int(args.val_loader_steps_per_epoch)} "
        f"decode_cache={decode_cache_size} workers={decode_workers} "
        f"prefetch={int(args.prefetch_factor)} worker_threads={int(args.worker_torch_threads)} "
        f"lambda_cons={float(args.lambda_cons):g} lambda_div={float(args.lambda_div):g}"
    )

    time_decay = float(max(0.0, min(0.999, float(args.time_ema_decay))))
    data_time_ema = 0.0
    train_time_ema = 0.0
    data_time_window = 0.0
    train_time_window = 0.0
    window_steps = 0

    best_val_loss: float | None = None
    best_val_monitor: float | None = None
    best_val_retr_top1: float | None = None
    best_step = 0
    last_step = 0

    try:
        for step in range(1, int(args.steps) + 1):
            t_data0 = time.perf_counter()
            encoder.train()
            (v1, m1, v2, m2), train_iter = next_loader_batch(train_loader, train_iter)
            v1 = v1.to(device, non_blocking=True)
            m1 = m1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            m2 = m2.to(device, non_blocking=True)
            data_t = float(time.perf_counter() - t_data0)

            t_train0 = time.perf_counter()
            tok1 = encoder.encode_style_tokens(v1, m1)
            tok2 = encoder.encode_style_tokens(v2, m2)
            z1 = F.normalize(tok1.mean(dim=1), dim=-1)
            z2 = F.normalize(tok2.mean(dim=1), dim=-1)
            train_cos_same, train_cos_diff = cosine_same_diff(z1, z2)

            loss_nce = info_nce_loss(z1, z2, temperature=float(args.tau))
            loss_cons = 1.0 - train_cos_same
            loss_div = 0.5 * (token_diversity_loss(tok1) + token_diversity_loss(tok2))
            loss_collapse = 0.5 * (token_collapse_score(tok1) + token_collapse_score(tok2))
            loss = (
                loss_nce
                + float(args.lambda_cons) * loss_cons
                + float(args.lambda_div) * loss_div
            )
            train_retr_top1, train_retr_top5 = retrieval_topk(z1, z2, k=5)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_t = float(time.perf_counter() - t_train0)

            last_step = step
            window_steps += 1
            data_time_window += data_t
            train_time_window += train_t
            if step == 1:
                data_time_ema = data_t
                train_time_ema = train_t
            else:
                data_time_ema = time_decay * data_time_ema + (1.0 - time_decay) * data_t
                train_time_ema = time_decay * train_time_ema + (1.0 - time_decay) * train_t

            should_log = (step % int(args.log_every) == 0) or (step == 1) or (step == int(args.steps))
            if not should_log:
                continue

            avg_data_t = float(data_time_window / max(1, window_steps))
            avg_train_t = float(train_time_window / max(1, window_steps))
            data_time_window = 0.0
            train_time_window = 0.0
            window_steps = 0

            val_loss = None
            val_retr_top1 = None
            val_retr_top5 = None
            val_cons = None
            val_cos_same = None
            val_cos_diff = None
            val_div = None
            val_collapse = None
            val_monitor = None
            eval_t = 0.0

            if has_val:
                t_eval0 = time.perf_counter()
                encoder.eval()
                val_loss_list: List[float] = []
                val_retr_top1_list: List[float] = []
                val_retr_top5_list: List[float] = []
                val_cons_list: List[float] = []
                val_cos_same_list: List[float] = []
                val_cos_diff_list: List[float] = []
                val_div_list: List[float] = []
                val_collapse_list: List[float] = []
                with torch.no_grad():
                    for _ in range(max(1, int(args.val_batches))):
                        assert val_loader is not None
                        assert val_iter is not None
                        (ev1, em1, ev2, em2), val_iter = next_loader_batch(val_loader, val_iter)
                        ev1 = ev1.to(device, non_blocking=True)
                        em1 = em1.to(device, non_blocking=True)
                        ev2 = ev2.to(device, non_blocking=True)
                        em2 = em2.to(device, non_blocking=True)
                        et1 = encoder.encode_style_tokens(ev1, em1)
                        et2 = encoder.encode_style_tokens(ev2, em2)
                        ez1 = F.normalize(et1.mean(dim=1), dim=-1)
                        ez2 = F.normalize(et2.mean(dim=1), dim=-1)
                        ev_cos_same, ev_cos_diff = cosine_same_diff(ez1, ez2)
                        el_nce = info_nce_loss(ez1, ez2, temperature=float(args.tau))
                        el_cons = 1.0 - ev_cos_same
                        el_div = 0.5 * (token_diversity_loss(et1) + token_diversity_loss(et2))
                        el_collapse = 0.5 * (token_collapse_score(et1) + token_collapse_score(et2))
                        er_top1, er_top5 = retrieval_topk(ez1, ez2, k=5)
                        eval_loss = (
                            el_nce
                            + float(args.lambda_cons) * el_cons
                            + float(args.lambda_div) * el_div
                        )
                        val_loss_list.append(float(eval_loss.item()))
                        val_retr_top1_list.append(float(er_top1.item()))
                        val_retr_top5_list.append(float(er_top5.item()))
                        val_cons_list.append(float(el_cons.item()))
                        val_cos_same_list.append(float(ev_cos_same.item()))
                        val_cos_diff_list.append(float(ev_cos_diff.item()))
                        val_div_list.append(float(el_div.item()))
                        val_collapse_list.append(float(el_collapse.item()))

                eval_t = float(time.perf_counter() - t_eval0)
                val_loss = float(np.mean(val_loss_list))
                val_retr_top1 = float(np.mean(val_retr_top1_list))
                val_retr_top5 = float(np.mean(val_retr_top5_list))
                val_cons = float(np.mean(val_cons_list))
                val_cos_same = float(np.mean(val_cos_same_list))
                val_cos_diff = float(np.mean(val_cos_diff_list))
                val_div = float(np.mean(val_div_list))
                val_collapse = float(np.mean(val_collapse_list))
                val_cons_sim = float(val_cos_same)
                val_monitor = (
                    float(args.monitor_weight_retr) * float(val_retr_top1)
                    + float(args.monitor_weight_cons) * val_cons_sim
                )

            metrics = {
                "step": int(step),
                "loss": float(loss.item()),
                "loss_nce": float(loss_nce.item()),
                "loss_cons": float(loss_cons.item()),
                "loss_div": float(loss_div.item()),
                "token_collapse": float(loss_collapse.item()),
                "train_retr_top1": float(train_retr_top1.item()),
                "train_retr_top5": float(train_retr_top5.item()),
                "train_cos_same": float(train_cos_same.item()),
                "train_cos_diff": float(train_cos_diff.item()),
                "data_time": float(avg_data_t),
                "data_time_ema": float(data_time_ema),
                "train_time": float(avg_train_t),
                "train_time_ema": float(train_time_ema),
                "eval_time": float(eval_t),
                "val_loss": val_loss,
                "val_retr_top1": val_retr_top1,
                "val_retr_top5": val_retr_top5,
                "val_cons": val_cons,
                "val_cos_same": val_cos_same,
                "val_cos_diff": val_cos_diff,
                "val_div": val_div,
                "val_collapse": val_collapse,
                "val_token_collapse": val_collapse,
                "val_monitor": val_monitor,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            metrics_fp.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metrics_fp.flush()

            if val_loss is not None:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = float(val_loss)
                if val_retr_top1 is not None and (best_val_retr_top1 is None or val_retr_top1 > best_val_retr_top1):
                    best_val_retr_top1 = float(val_retr_top1)
                if val_monitor is not None and (best_val_monitor is None or val_monitor > best_val_monitor):
                    best_val_monitor = float(val_monitor)
                    best_step = int(step)

            msg = (
                f"step={step:05d} "
                f"loss={loss.item():.4f} nce={loss_nce.item():.4f} "
                f"cons={loss_cons.item():.4f} cos_same={train_cos_same.item():.4f} cos_diff={train_cos_diff.item():.4f} "
                f"div={loss_div.item():.4f} token_coll={loss_collapse.item():.4f} "
                f"retr1={train_retr_top1.item():.3f} retr5={train_retr_top5.item():.3f} "
                f"data_t={avg_data_t:.3f}s train_t={avg_train_t:.3f}s "
                f"data_ema={data_time_ema:.3f}s train_ema={train_time_ema:.3f}s"
            )
            if val_loss is not None:
                msg += (
                    f" val_loss={float(val_loss):.4f} val_retr1={float(val_retr_top1):.3f} val_retr5={float(val_retr_top5):.3f} "
                    f"val_cos_same={float(val_cos_same):.4f} val_cos_diff={float(val_cos_diff):.4f} "
                    f"val_token_coll={float(val_collapse):.4f} val_mon={float(val_monitor):.4f} "
                    f"eval_t={eval_t:.3f}s"
                )
            if best_val_monitor is not None:
                msg += f" best_val_monitor={best_val_monitor:.4f}@{best_step}"
            log(msg)
    finally:
        final_ckpt = {
            "style_encoder": make_style_encoder_state_dict(encoder),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "extra": {
                "step": int(last_step),
                "best_step": int(best_step),
                "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
                "best_val_retr_top1": float(best_val_retr_top1) if best_val_retr_top1 is not None else None,
                "best_val_monitor": float(best_val_monitor) if best_val_monitor is not None else None,
                "ran_full_steps": bool(last_step == int(args.steps)),
            },
        }
        torch.save(final_ckpt, out_path)
        log(f"saved final checkpoint: {out_path}")
        log_fp.close()
        metrics_fp.close()


if __name__ == "__main__":
    main()
