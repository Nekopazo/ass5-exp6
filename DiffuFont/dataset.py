#!/usr/bin/env python3
"""LMDB-backed dataset for content+style glyph flow training."""

from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path
import io
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler


class FontImageDataset(Dataset):
    """Returns one content glyph, one target glyph, and N style references."""

    def __init__(
        self,
        project_root: Union[str, Path] = ".",
        *,
        max_fonts: int = 0,
        style_ref_count: int = 8,
        random_seed: int = 42,
        font_split: str = "all",
        font_split_seed: int = 42,
        font_train_ratio: float = 0.9,
        content_lmdb: Optional[Union[str, Path]] = None,
        train_lmdb: Optional[Union[str, Path]] = None,
        transform=None,
        style_transform=None,
        lmdb_decode_cache_size: int = 20_000,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.max_fonts = max(0, int(max_fonts))
        self.style_ref_count = max(1, int(style_ref_count))
        self.font_split = str(font_split).strip().lower()
        if self.font_split not in {"train", "test", "all"}:
            raise ValueError(f"font_split must be one of train/test/all, got {font_split!r}")
        self.font_split_seed = int(font_split_seed)
        self.font_train_ratio = float(font_train_ratio)
        if not (0.0 < self.font_train_ratio < 1.0):
            raise ValueError(f"font_train_ratio must be in (0, 1), got {font_train_ratio}")
        self.transform = transform
        self.style_transform = style_transform if style_transform is not None else transform
        self._random_seed = int(random_seed)
        self.rng = random.Random(self._random_seed)
        self.lmdb_decode_cache_size = max(0, int(lmdb_decode_cache_size))
        self._glyph_decode_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        char_list_path = self.root / "CharacterData" / "CharList.json"
        font_list_path = self.root / "DataPreparation" / "FontList.json"
        self.char_list: List[str] = json.loads(char_list_path.read_text(encoding="utf-8"))
        raw_font_list: List[str] = json.loads(font_list_path.read_text(encoding="utf-8"))
        self.font_list_stems: List[str] = [Path(x).stem for x in raw_font_list]

        if content_lmdb is None:
            content_lmdb = self.root / "DataPreparation" / "LMDB" / "ContentFont.lmdb"
        if train_lmdb is None:
            train_lmdb = self.root / "DataPreparation" / "LMDB" / "TrainFont.lmdb"

        self._content_lmdb_path = str(Path(content_lmdb).resolve())
        self._train_lmdb_path = str(Path(train_lmdb).resolve())
        self.content_env = lmdb.open(self._content_lmdb_path, readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(self._train_lmdb_path, readonly=True, lock=False, readahead=False)
        self._c_txn = self.content_env.begin(buffers=True)
        self._t_txn = self.train_env.begin(buffers=True)
        self._worker_pid: Optional[int] = os.getpid()

        lmdb_fonts = self._scan_lmdb_font_names(self._t_txn)
        entries = self._build_multi_font_entries(self._c_txn, self._t_txn, lmdb_fonts)
        entries = self._apply_font_split(entries)
        if self.max_fonts > 0 and len(entries) > self.max_fonts:
            entries = sorted(entries, key=lambda item: len(item[1]), reverse=True)[: self.max_fonts]

        self.font_names = [name for name, _ in entries]
        self.valid_indices_by_font = {name: indices for name, indices in entries}
        self.samples: List[Tuple[str, int]] = []
        self.sample_indices_by_font: Dict[str, List[int]] = {name: [] for name in self.font_names}
        self.sample_indices_by_char: Dict[int, List[int]] = {idx: [] for idx in range(len(self.char_list))}
        self.sample_index_by_char_font: Dict[int, Dict[str, int]] = {idx: {} for idx in range(len(self.char_list))}
        for font_name, indices in entries:
            for char_index in indices:
                self.sample_indices_by_font[font_name].append(len(self.samples))
                self.sample_indices_by_char[int(char_index)].append(len(self.samples))
                self.sample_index_by_char_font[int(char_index)][font_name] = len(self.samples)
                self.samples.append((font_name, char_index))

        if not self.samples:
            raise RuntimeError("No valid samples found in ContentFont/TrainFont LMDB.")

        print(
            f"[FontImageDataset] samples={len(self.samples)} "
            f"fonts={len(self.font_names)} style_ref_count={self.style_ref_count} "
            f"font_split={self.font_split} font_split_seed={self.font_split_seed} "
            f"font_train_ratio={self.font_train_ratio:.2f}"
        )

    def _scan_lmdb_font_names(self, txn) -> List[str]:
        names = set()
        for raw_key, _ in txn.cursor():
            if b"@" not in raw_key:
                continue
            prefix = bytes(raw_key).split(b"@", 1)[0]
            try:
                names.add(prefix.decode("utf-8"))
            except UnicodeDecodeError:
                continue
        return sorted(names)

    def _build_indices_for_font(self, font_name: str, c_txn, t_txn) -> List[int]:
        valid_indices: List[int] = []
        for idx, ch in enumerate(self.char_list):
            content_key = f"ContentFont@{ch}".encode("utf-8")
            target_key = f"{font_name}@{ch}".encode("utf-8")
            if c_txn.get(content_key) is None:
                continue
            if t_txn.get(target_key) is None:
                continue
            valid_indices.append(idx)
        if len(valid_indices) <= 1:
            return []
        return valid_indices

    def _build_multi_font_entries(self, c_txn, t_txn, lmdb_fonts: List[str]) -> List[Tuple[str, List[int]]]:
        entries: List[Tuple[str, List[int]]] = []
        seen: set[str] = set()
        for font_name in self.font_list_stems + lmdb_fonts:
            if font_name in seen:
                continue
            seen.add(font_name)
            valid = self._build_indices_for_font(font_name, c_txn, t_txn)
            if valid:
                entries.append((font_name, valid))
        if not entries:
            raise RuntimeError("Could not find any usable font in TrainFont.lmdb.")
        return entries

    def _apply_font_split(self, entries: List[Tuple[str, List[int]]]) -> List[Tuple[str, List[int]]]:
        if self.font_split == "all" or len(entries) <= 1:
            return entries

        font_names = [name for name, _ in entries]
        shuffled = list(font_names)
        split_rng = random.Random(self.font_split_seed)
        split_rng.shuffle(shuffled)

        train_count = int(len(shuffled) * self.font_train_ratio)
        train_count = max(1, min(len(shuffled) - 1, train_count))
        if self.font_split == "train":
            selected = set(shuffled[:train_count])
        else:
            selected = set(shuffled[train_count:])
        filtered = [item for item in entries if item[0] in selected]
        if not filtered:
            raise RuntimeError(
                f"Font split '{self.font_split}' is empty. "
                f"fonts={len(entries)} seed={self.font_split_seed} train_ratio={self.font_train_ratio}"
            )
        return filtered

    def _decode_u8(self, image_bytes: bytes) -> np.ndarray:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is not None:
            return np.ascontiguousarray(decoded)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        return np.asarray(pil_img, dtype=np.uint8)

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        if self.lmdb_decode_cache_size <= 0:
            return None
        hit = self._glyph_decode_cache.get(key)
        if hit is None:
            return None
        self._glyph_decode_cache.move_to_end(key, last=True)
        return hit

    def _cache_put(self, key: str, value: np.ndarray) -> np.ndarray:
        if self.lmdb_decode_cache_size <= 0:
            return value
        self._glyph_decode_cache[key] = value
        self._glyph_decode_cache.move_to_end(key, last=True)
        while len(self._glyph_decode_cache) > self.lmdb_decode_cache_size:
            self._glyph_decode_cache.popitem(last=False)
        return value

    def _bytes_to_u8(self, image_bytes: bytes, *, cache_key: str) -> np.ndarray:
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        decoded = self._decode_u8(image_bytes)
        return self._cache_put(cache_key, decoded)

    def _ensure_txns(self) -> None:
        pid = os.getpid()
        if self._worker_pid == pid:
            return
        self.content_env = lmdb.open(self._content_lmdb_path, readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(self._train_lmdb_path, readonly=True, lock=False, readahead=False)
        self._c_txn = self.content_env.begin(buffers=True)
        self._t_txn = self.train_env.begin(buffers=True)
        self._glyph_decode_cache.clear()
        worker_info = torch.utils.data.get_worker_info()
        worker_offset = 0 if worker_info is None else int(worker_info.id)
        self.rng = random.Random(self._random_seed + worker_offset)
        self._worker_pid = pid

    def _load_tensor(self, txn, key: str, *, style: bool) -> torch.Tensor:
        image_bytes = txn.get(key.encode("utf-8"))
        if image_bytes is None:
            raise KeyError(f"Missing LMDB key: {key}")
        array = self._bytes_to_u8(bytes(image_bytes), cache_key=key)
        transform = self.style_transform if style else self.transform
        if transform is None:
            return torch.from_numpy(array.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return transform(array)

    def _sample_style_indices(
        self,
        font_name: str,
        target_index: int,
    ) -> List[int]:
        target_index = int(target_index)
        candidates = [int(idx) for idx in self.valid_indices_by_font[font_name] if int(idx) != target_index]
        if not candidates:
            raise RuntimeError(f"Font '{font_name}' has no alternative glyph for style reference.")
        if len(candidates) >= self.style_ref_count:
            return self.rng.sample(candidates, k=self.style_ref_count)
        return candidates + self.rng.choices(candidates, k=self.style_ref_count - len(candidates))

    def _content_key(self, char_index: int) -> str:
        return f"ContentFont@{self.char_list[int(char_index)]}"

    def _font_char_key(self, font_name: str, char_index: int) -> str:
        return f"{font_name}@{self.char_list[int(char_index)]}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_txns()
        font_name, char_index = self.samples[index]
        char = self.char_list[char_index]

        content_img = self._load_tensor(self._c_txn, self._content_key(char_index), style=False)
        target_img = self._load_tensor(self._t_txn, self._font_char_key(font_name, char_index), style=False)

        style_indices = self._sample_style_indices(font_name, char_index)
        style_img = torch.stack(
            [self._load_tensor(self._t_txn, self._font_char_key(font_name, idx), style=True) for idx in style_indices],
            dim=0,
        )

        return {
            "font": font_name,
            "char": char,
            "content": content_img,
            "target": target_img,
            "style_img": style_img,
        }

    def close(self) -> None:
        for attr in ("_c_txn", "_t_txn"):
            txn = getattr(self, attr, None)
            if txn is not None:
                try:
                    txn.abort()
                except Exception:
                    pass
                setattr(self, attr, None)
        for attr in ("content_env", "train_env"):
            env = getattr(self, attr, None)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        self._worker_pid = None

    def __del__(self) -> None:
        self.close()


class UniqueFontBatchSampler(Sampler[List[int]]):
    """Yields batches with at most one sample per font while covering all samples."""

    def __init__(self, dataset: FontImageDataset, batch_size: int, *, seed: int = 42, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._sample_counts = [
            len(self.dataset.sample_indices_by_font[font_name]) for font_name in self.dataset.font_names
        ]
        self._batch_count = self._compute_batch_count()

    def _compute_batch_count(self) -> int:
        remaining = deque(count for count in self._sample_counts if count > 0)
        yielded = 0
        while remaining:
            take_count = min(self.batch_size, len(remaining))
            next_round: List[int] = []
            for _ in range(take_count):
                count = remaining.popleft() - 1
                if count > 0:
                    next_round.append(count)
            if take_count == self.batch_size or (take_count > 0 and not self.drop_last):
                yielded += 1
            remaining.extend(next_round)
        return yielded

    def __len__(self) -> int:
        return self._batch_count

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        font_queues: Dict[str, deque[int]] = {}
        active_fonts: List[str] = []
        for font_name in self.dataset.font_names:
            sample_indices = list(self.dataset.sample_indices_by_font[font_name])
            if not sample_indices:
                continue
            rng.shuffle(sample_indices)
            font_queues[font_name] = deque(sample_indices)
            active_fonts.append(font_name)
        rng.shuffle(active_fonts)
        active_queue = deque(active_fonts)

        while active_queue:
            batch: List[int] = []
            next_round: List[str] = []
            take_count = min(self.batch_size, len(active_queue))
            for _ in range(take_count):
                font_name = active_queue.popleft()
                sample_queue = font_queues[font_name]
                batch.append(sample_queue.popleft())
                if sample_queue:
                    next_round.append(font_name)
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch
            rng.shuffle(next_round)
            active_queue.extend(next_round)
        self._epoch += 1


class GroupedCharFontBatchSampler(Sampler[List[int]]):
    """Yields batches as a fixed char set x fixed font set Cartesian product."""

    def __init__(
        self,
        dataset: FontImageDataset,
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
        grouped_char_count: int = 8,
        grouped_fonts_per_char: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.grouped_char_count = max(1, int(grouped_char_count))
        inferred_fonts_per_char = int(grouped_fonts_per_char)
        if inferred_fonts_per_char <= 0:
            if self.batch_size % self.grouped_char_count != 0:
                raise ValueError(
                    "grouped_char_count must divide batch_size when grouped_fonts_per_char is not set: "
                    f"batch_size={self.batch_size} grouped_char_count={self.grouped_char_count}"
                )
            inferred_fonts_per_char = self.batch_size // self.grouped_char_count
        self.grouped_fonts_per_char = max(1, inferred_fonts_per_char)
        expected_batch = self.grouped_char_count * self.grouped_fonts_per_char
        if expected_batch != self.batch_size:
            raise ValueError(
                "grouped_char_count * grouped_fonts_per_char must equal batch_size: "
                f"{self.grouped_char_count} * {self.grouped_fonts_per_char} != {self.batch_size}"
            )
        self._epoch = 0
        self._char_indices = [
            int(char_index) for char_index, indices in self.dataset.sample_indices_by_char.items() if indices
        ]
        self._font_names = [
            str(font_name) for font_name in self.dataset.font_names if self.dataset.sample_indices_by_font[font_name]
        ]
        self._batch_count = self._compute_batch_count()

    def _compute_batch_count(self) -> int:
        char_group_count = len(self._char_indices) // self.grouped_char_count
        font_group_count = len(self._font_names) // self.grouped_fonts_per_char
        if not self.drop_last:
            char_group_count = int(math.ceil(len(self._char_indices) / float(self.grouped_char_count)))
            font_group_count = int(math.ceil(len(self._font_names) / float(self.grouped_fonts_per_char)))
        return int(char_group_count * font_group_count)

    def __len__(self) -> int:
        return self._batch_count

    @staticmethod
    def _chunk_items(items: List[Any], chunk_size: int, *, drop_last: bool) -> List[List[Any]]:
        chunks = [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]
        if drop_last:
            chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]
        return chunks

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        char_indices = list(self._char_indices)
        font_names = list(self._font_names)
        rng.shuffle(char_indices)
        rng.shuffle(font_names)
        char_groups = self._chunk_items(char_indices, self.grouped_char_count, drop_last=self.drop_last)
        font_groups = self._chunk_items(font_names, self.grouped_fonts_per_char, drop_last=self.drop_last)
        batch_pairs = [(char_group, font_group) for char_group in char_groups for font_group in font_groups]
        rng.shuffle(batch_pairs)

        for char_group, font_group in batch_pairs:
            batch: List[int] = []
            for char_index in char_group:
                sample_lookup = self.dataset.sample_index_by_char_font[int(char_index)]
                for font_name in font_group:
                    sample_index = sample_lookup.get(font_name)
                    if sample_index is not None:
                        batch.append(int(sample_index))
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch
        self._epoch += 1


if __name__ == "__main__":
    dataset = FontImageDataset(project_root=Path("."))
    sample = dataset[0]
    print(
        sample["font"],
        sample["char"],
        tuple(sample["content"].shape),
        tuple(sample["target"].shape),
        tuple(sample["style_img"].shape),
    )
