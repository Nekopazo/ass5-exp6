#!/usr/bin/env python3
"""LMDB-backed dataset for content+style glyph flow training."""

from __future__ import annotations

from collections import OrderedDict
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
        include_positive_style: bool = False,
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
        self.include_positive_style = bool(include_positive_style)
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
        for font_name, indices in entries:
            for char_index in indices:
                self.sample_indices_by_font[font_name].append(len(self.samples))
                self.samples.append((font_name, char_index))

        if not self.samples:
            raise RuntimeError("No valid samples found in ContentFont/TrainFont LMDB.")

        print(
            f"[FontImageDataset] samples={len(self.samples)} "
            f"fonts={len(self.font_names)} style_ref_count={self.style_ref_count} "
            f"positive_pairs={self.include_positive_style} "
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
        *,
        excluded_indices: Optional[List[int]] = None,
    ) -> List[int]:
        target_index = int(target_index)
        excluded = {target_index}
        if excluded_indices is not None:
            excluded.update(int(idx) for idx in excluded_indices)

        all_candidates = [int(idx) for idx in self.valid_indices_by_font[font_name] if int(idx) != target_index]
        preferred_candidates = [idx for idx in all_candidates if idx not in excluded]
        if not all_candidates:
            raise RuntimeError(f"Font '{font_name}' has no alternative glyph for style reference.")
        self.rng.shuffle(preferred_candidates)
        self.rng.shuffle(all_candidates)

        chosen = preferred_candidates[: self.style_ref_count]
        if len(chosen) < self.style_ref_count:
            remainder = [idx for idx in all_candidates if idx not in chosen]
            self.rng.shuffle(remainder)
            chosen.extend(remainder[: self.style_ref_count - len(chosen)])
        if len(chosen) < self.style_ref_count:
            chosen.extend(self.rng.choices(all_candidates, k=self.style_ref_count - len(chosen)))
        return chosen

    def _sample_style_pair(self, font_name: str, target_index: int) -> Tuple[List[int], List[int]]:
        anchor_indices = self._sample_style_indices(font_name, target_index)
        positive_indices = self._sample_style_indices(
            font_name,
            target_index,
            excluded_indices=anchor_indices,
        )
        return anchor_indices, positive_indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_txns()
        font_name, char_index = self.samples[index]
        char = self.char_list[char_index]

        content_key = f"ContentFont@{char}"
        target_key = f"{font_name}@{char}"
        content_img = self._load_tensor(self._c_txn, content_key, style=False)
        target_img = self._load_tensor(self._t_txn, target_key, style=False)

        style_indices = self._sample_style_indices(font_name, char_index)
        style_chars = [self.char_list[idx] for idx in style_indices]
        style_imgs = [
            self._load_tensor(self._t_txn, f"{font_name}@{style_char}", style=True)
            for style_char in style_chars
        ]

        sample = {
            "font": font_name,
            "char": char,
            "content": content_img,
            "target": target_img,
            "style_img": torch.stack(style_imgs, dim=0),
            "style_ref_mask": torch.ones((len(style_imgs),), dtype=torch.float32),
            "style_chars": style_chars,
            "style_char": style_chars[0],
        }
        if self.include_positive_style:
            positive_style_indices = self._sample_style_indices(
                font_name,
                char_index,
                excluded_indices=style_indices,
            )
            positive_style_chars = [self.char_list[idx] for idx in positive_style_indices]
            positive_style_imgs = [
                self._load_tensor(self._t_txn, f"{font_name}@{style_char}", style=True)
                for style_char in positive_style_chars
            ]
            sample["style_img_pos"] = torch.stack(positive_style_imgs, dim=0)
            sample["style_ref_mask_pos"] = torch.ones((len(positive_style_imgs),), dtype=torch.float32)
            sample["style_chars_pos"] = positive_style_chars
        return sample

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
    """Yields batches with at most one sample per font."""

    def __init__(self, dataset: FontImageDataset, batch_size: int, *, seed: int = 42, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __len__(self) -> int:
        font_count = len(self.dataset.font_names)
        if self.drop_last:
            return font_count // self.batch_size
        return int(math.ceil(font_count / self.batch_size))

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        font_names = list(self.dataset.font_names)
        rng.shuffle(font_names)

        batch: List[int] = []
        for font_name in font_names:
            sample_indices = self.dataset.sample_indices_by_font[font_name]
            batch.append(sample_indices[rng.randrange(len(sample_indices))])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
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
