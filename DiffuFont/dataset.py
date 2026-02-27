#!/usr/bin/env python3
"""
FontImageDataset
================
Multi-font dataset.  Always operates in random-font mode.
Returns content glyph, training target, and two random
part-bank views (for InfoNCE contrastive learning).

Data sources
------------
* CharacterData/CharList.json
* DataPreparation/FontList.json
* DataPreparation/LMDB/{ContentFont,TrainFont,PartBank}.lmdb
"""

from pathlib import Path
import json
import io
import random
import re
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union, Tuple

import lmdb
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore
        pass


class FontImageDataset(Dataset):
    """Multi-font dataset.  Always operates in random-font mode."""

    def __init__(
        self,
        project_root: Union[str, Path] = ".",
        max_fonts: int = 0,
        lmdb_font_scan_limit: int = 100_000,
        # ---- PartBank (direct label-based sampling, no retrieval CNN) ----
        use_part_bank: bool = False,
        part_bank_manifest: Optional[Union[str, Path]] = None,
        part_bank_lmdb: Optional[Union[str, Path]] = None,
        part_set_max: int = 8,
        part_set_min: int = 1,
        part_image_size: int = 64,
        part_image_cache_size: int = 50_000,
        lmdb_decode_cache_size: int = 20_000,
        random_seed: int = 42,
        content_lmdb: Optional[Union[str, Path]] = None,
        train_lmdb: Optional[Union[str, Path]] = None,
        transform=None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.max_fonts = max(0, int(max_fonts))
        self.rng = random.Random(random_seed)
        self.lmdb_font_scan_limit = int(max(1000, lmdb_font_scan_limit))
        self.use_part_bank = bool(use_part_bank)
        self.part_env: Any = None
        self.part_set_max = max(1, int(part_set_max))
        self.part_set_min = max(1, int(part_set_min))
        if self.part_set_min > self.part_set_max:
            raise ValueError(
                f"part_set_min ({self.part_set_min}) must be <= part_set_max ({self.part_set_max})"
            )
        self.part_image_size = max(8, int(part_image_size))
        self.part_image_cache_size = max(0, int(part_image_cache_size))
        self.lmdb_decode_cache_size = max(0, int(lmdb_decode_cache_size))
        self.part_bank_by_font: Dict[str, List[Dict[str, Any]]] = {}
        self._part_tensor_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._glyph_decode_cache: "OrderedDict[str, Any]" = OrderedDict()

        # ---- Memory footprint warning ----
        _glyph_bytes_each = 3 * 256 * 256 * 4  # float32 tensor per cached glyph
        _part_bytes_each = 3 * part_image_size * part_image_size * 4
        _est_gb = (
            self.lmdb_decode_cache_size * _glyph_bytes_each
            + self.part_image_cache_size * _part_bytes_each
        ) / (1024 ** 3)
        if _est_gb > 4.0:
            print(
                f"[FontImageDataset] WARNING: in-memory cache may use ~{_est_gb:.1f} GB "
                f"(glyph_cache={self.lmdb_decode_cache_size} x {_glyph_bytes_each//1024}KB, "
                f"part_cache={self.part_image_cache_size} x {_part_bytes_each//1024}KB). "
                f"Reduce --lmdb-decode-cache-size / --part-image-cache-size to save RAM."
            )

        # ---- JSON paths ----
        char_list_path = self.root / "CharacterData" / "CharList.json"
        font_list_path = self.root / "DataPreparation" / "FontList.json"

        self.char_list: List[str] = json.load(char_list_path.open("r", encoding="utf-8"))
        font_list: List[str] = json.load(font_list_path.open("r", encoding="utf-8"))
        self.font_list_stems: List[str] = [Path(x).stem for x in font_list]

        # ---- LMDB ----
        if content_lmdb is None:
            content_lmdb = self.root / "DataPreparation" / "LMDB" / "ContentFont.lmdb"
        if train_lmdb is None:
            train_lmdb = self.root / "DataPreparation" / "LMDB" / "TrainFont.lmdb"

        self.content_env = lmdb.open(str(content_lmdb), readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(str(train_lmdb), readonly=True, lock=False, readahead=False)
        self.transform = transform

        # Keep persistent read-only transactions (avoids per-getitem txn overhead).
        self._c_txn = self.content_env.begin(buffers=True)
        self._t_txn = self.train_env.begin(buffers=True)

        # ---- Build multi-font entries ----
        lmdb_fonts = self._scan_lmdb_font_names(self._t_txn)
        entries = self._build_multi_font_entries(self._c_txn, self._t_txn, lmdb_fonts)

        self.font_names = [x[0] for x in entries]
        self.valid_indices_by_font = {x[0]: x[1] for x in entries}
        self.samples: List[Tuple[str, int]] = []
        for name, idxs in entries:
            self.samples.extend((name, idx) for idx in idxs)

        # ---- PartBank (direct label-based) ----
        if self.use_part_bank:
            if part_bank_manifest is None:
                part_bank_manifest = self.root / "DataPreparation" / "PartBank" / "manifest.json"
            if part_bank_lmdb is None:
                part_bank_lmdb = self.root / "DataPreparation" / "LMDB" / "PartBank.lmdb"
            part_bank_lmdb_path = Path(part_bank_lmdb)
            if not part_bank_lmdb_path.is_absolute():
                part_bank_lmdb_path = (self.root / part_bank_lmdb_path).resolve()
            if not part_bank_lmdb_path.exists():
                raise FileNotFoundError(f"PartBank LMDB not found: {part_bank_lmdb_path}")
            self.part_env = lmdb.open(str(part_bank_lmdb_path), readonly=True, lock=False, readahead=False)
            self._p_txn = self.part_env.begin(buffers=True)
            self.part_bank_by_font = self._load_part_bank_from_lmdb(self.part_env)
            self._filter_samples_by_part_bank()

        if not self.samples:
            raise RuntimeError("No valid samples found -- please check LMDB integrity and paths.")

        n_fonts = len(set(f for f, _ in self.samples))
        print(f"[FontImageDataset] {len(self.samples)} samples across {n_fonts} fonts.")

    # ------------------------------------------------------------------ #
    #  LMDB scanning
    # ------------------------------------------------------------------ #
    def _scan_lmdb_font_names(self, t_txn) -> List[str]:
        names = set()
        cursor = t_txn.cursor()
        for i, kv in enumerate(cursor):
            key = kv[0]
            if b"@" not in key:
                if i >= self.lmdb_font_scan_limit:
                    break
                continue
            prefix = key.split(b"@", 1)[0]
            try:
                names.add(prefix.decode("utf-8"))
            except UnicodeDecodeError:
                pass
            if i >= self.lmdb_font_scan_limit:
                break
        return sorted(names)

    def _build_indices_for_font(self, font_name: str, c_txn, t_txn) -> List[int]:
        valid_indices: List[int] = []
        for idx, ch in enumerate(self.char_list):
            content_key = f"ContentFont@{ch}".encode("utf-8")
            if c_txn.get(content_key) is None:
                continue
            input_key = f"{font_name}@{ch}".encode("utf-8")
            if t_txn.get(input_key) is None:
                continue
            valid_indices.append(idx)
        return valid_indices

    def _build_multi_font_entries(self, c_txn, t_txn, lmdb_fonts: List[str]):
        entries: List[Tuple[str, List[int]]] = []
        seen: set = set()
        for n in self.font_list_stems + lmdb_fonts:
            if n in seen:
                continue
            seen.add(n)
            valid = self._build_indices_for_font(n, c_txn, t_txn)
            if valid:
                entries.append((n, valid))
        if not entries:
            raise RuntimeError(
                "Could not find any usable font in TrainFont.lmdb. "
                "Please verify LMDB keys format '<FontName>@<char>'."
            )
        if self.max_fonts > 0 and len(entries) > self.max_fonts:
            entries = sorted(entries, key=lambda x: len(x[1]), reverse=True)[:self.max_fonts]
        print(f"[FontImageDataset] Multi-font mode with {len(entries)} fonts.")
        return entries

    # ------------------------------------------------------------------ #
    #  PartBank – scan LMDB directly (no manifest.json dependency)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_part_index(path_like: str) -> int:
        m = re.search(r"part_(\d+)", path_like)
        if m is None:
            return 1_000_000_000
        try:
            return int(m.group(1))
        except ValueError:
            return 1_000_000_000

    @staticmethod
    def _unicode_key_to_char(lmdb_key: str) -> str:
        """Extract the character from a key like .../part_002_U4EE5.png -> '以'."""
        m = re.search(r"_U([0-9A-Fa-f]{4,6})", lmdb_key)
        if m is None:
            return ""
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return ""

    def _load_part_bank_from_lmdb(self, part_env) -> Dict[str, List[Dict[str, Any]]]:
        """Scan PartBank LMDB keys directly and return {font: [row, ...]}.

        This avoids any mismatch between manifest.json and the actual LMDB
        content. Keys are expected to look like:
            DataPreparation/PartBank/<font_name>/part_NNN_UXXXX.png
        """
        from collections import defaultdict
        tmp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        txn = part_env.begin()
        cursor = txn.cursor()
        for raw_key, _ in cursor:
            key = raw_key.decode("utf-8") if isinstance(raw_key, (bytes, memoryview)) else raw_key
            parts = key.split("/")
            if len(parts) < 4:
                continue
            font_name = parts[2]
            ch = self._unicode_key_to_char(key)
            tmp[font_name].append({
                "lmdb_key": key,
                "char": ch,
                "index": self._parse_part_index(key),
            })
        # Sort for determinism
        out: Dict[str, List[Dict[str, Any]]] = {}
        for font_name in sorted(tmp):
            rows = tmp[font_name]
            rows.sort(key=lambda x: (int(x["index"]), str(x["lmdb_key"])))
            out[font_name] = rows
        return out

    def _filter_samples_by_part_bank(self) -> None:
        available_fonts = set(self.part_bank_by_font.keys())
        before = len(self.samples)
        self.samples = [x for x in self.samples if x[0] in available_fonts]
        after = len(self.samples)
        if after <= 0:
            raise RuntimeError("No samples left after applying PartBank font filter.")
        if after < before:
            print(f"[FontImageDataset] PartBank filter removed {before - after} samples.")

    # ------------------------------------------------------------------ #
    #  Part sampling -- direct label-based, random min..max per view
    # ------------------------------------------------------------------ #
    def _sample_parts_for_font(self, font_name: str) -> List[Dict[str, Any]]:
        """Random-sample part_set_min..part_set_max parts from the font's PartBank."""
        rows = self.part_bank_by_font.get(font_name, [])
        if not rows:
            return []
        k = self.rng.randint(self.part_set_min, min(self.part_set_max, len(rows)))
        return self.rng.sample(rows, k) if len(rows) >= k else rows[:]

    def _part_array_from_bytes(self, b: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(b)).convert("L")
        if self.part_image_size > 0:
            img = img.resize((self.part_image_size, self.part_image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return arr[None, :, :].astype(np.float32, copy=False)  # (1, H, W)

    def _load_part_tensor_uncached(self, part_txn, lmdb_key: str):
        b = part_txn.get(lmdb_key.encode("utf-8"))
        if b is None:
            raise KeyError(f"missing PartBank lmdb key: {lmdb_key}")
        arr = self._part_array_from_bytes(bytes(b))
        return torch.from_numpy(arr.copy()).contiguous()

    def _load_part_tensor(self, part_txn, lmdb_key: str):
        k = str(lmdb_key)
        if self.part_image_cache_size > 0:
            hit = self._part_tensor_cache.get(k)
            if hit is not None:
                self._part_tensor_cache.move_to_end(k, last=True)
                return hit
        t = self._load_part_tensor_uncached(part_txn, lmdb_key)
        if self.part_image_cache_size > 0:
            self._part_tensor_cache[k] = t
            self._part_tensor_cache.move_to_end(k, last=True)
            while len(self._part_tensor_cache) > self.part_image_cache_size:
                self._part_tensor_cache.popitem(last=False)
        return t

    # ------------------------------------------------------------------ #
    #  Image decode helpers
    # ------------------------------------------------------------------ #
    def _decode_img(self, b: bytes):
        img = Image.open(io.BytesIO(b)).convert("L")
        return self.transform(img) if self.transform else img

    def _cache_get(self, key: str):
        if self.lmdb_decode_cache_size <= 0:
            return None
        v = self._glyph_decode_cache.get(key)
        if v is None:
            return None
        self._glyph_decode_cache.move_to_end(key, last=True)
        return v

    def _cache_put(self, key: str, value: Any) -> Any:
        if self.lmdb_decode_cache_size <= 0:
            return value
        self._glyph_decode_cache[key] = value
        self._glyph_decode_cache.move_to_end(key, last=True)
        while len(self._glyph_decode_cache) > self.lmdb_decode_cache_size:
            self._glyph_decode_cache.popitem(last=False)
        return value

    def _bytes_to_img(self, b: bytes, cache_key: Optional[str] = None):
        if cache_key:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit
        img = self._decode_img(b)
        if cache_key:
            self._cache_put(cache_key, img)
        return img

    # ------------------------------------------------------------------ #
    #  __len__ / __getitem__
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        font_name, real_idx = self.samples[index]
        ch = self.char_list[real_idx]

        content_key = f"ContentFont@{ch}"
        input_key = f"{font_name}@{ch}"

        content_bytes = self._c_txn.get(content_key.encode("utf-8"))
        input_bytes = self._t_txn.get(input_key.encode("utf-8"))
        if content_bytes is None or input_bytes is None:
            raise KeyError(f"Missing required glyph images for {font_name}@{ch}")

        # buffers=True returns memoryview; copy before txn reuse
        content_img = self._bytes_to_img(bytes(content_bytes), cache_key=f"C@{content_key}")
        input_img = self._bytes_to_img(bytes(input_bytes), cache_key=f"T@{input_key}")

        sample: Dict[str, Any] = {
            "font": font_name,
            "char": ch,
            "content": content_img,
            "input": input_img,
        }

        if self.use_part_bank:
            if self._p_txn is None:
                raise RuntimeError("PartBank enabled but _p_txn is not initialized.")
            part_rows_a = self._sample_parts_for_font(font_name)
            part_rows_b = self._sample_parts_for_font(font_name)
            if not part_rows_a or not part_rows_b:
                raise KeyError(f"Missing PartBank parts for font: {font_name}")
            part_imgs_a = [self._load_part_tensor(self._p_txn, str(r["lmdb_key"])) for r in part_rows_a]
            part_imgs_b = [self._load_part_tensor(self._p_txn, str(r["lmdb_key"])) for r in part_rows_b]
            sample["parts"] = torch.stack(part_imgs_a, dim=0)
            sample["parts_b"] = torch.stack(part_imgs_b, dim=0)
            sample["part_mask"] = torch.ones((len(part_imgs_a),), dtype=torch.float32)
            sample["part_mask_b"] = torch.ones((len(part_imgs_b),), dtype=torch.float32)

        return sample

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        # Abort persistent read transactions before closing environments.
        for attr in ("_c_txn", "_t_txn", "_p_txn"):
            txn = getattr(self, attr, None)
            if txn is not None:
                try:
                    txn.abort()
                except Exception:
                    pass
                setattr(self, attr, None)
        if hasattr(self, "content_env"):
            self.content_env.close()
        if hasattr(self, "train_env"):
            self.train_env.close()
        if hasattr(self, "part_env") and self.part_env is not None:
            self.part_env.close()
            self.part_env = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    ds = FontImageDataset()
    print(f"Valid samples: {len(ds)}")
    sample = ds[0]
    print(sample["char"])
