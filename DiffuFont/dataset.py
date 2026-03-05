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
import os
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
        use_style_image: bool = False,
        enforce_part_bank_font_match: bool = False,
        # ---- PartBank (direct label-based sampling, no retrieval CNN) ----
        use_part_bank: bool = False,
        part_bank_manifest: Optional[Union[str, Path]] = None,
        part_bank_lmdb: Optional[Union[str, Path]] = None,
        part_set_min: Optional[int] = None,
        part_set_max: int = 8,
        part_pick_count: int = 0,
        part_image_size: int = 40,
        part_image_cache_size: int = 50_000,
        lmdb_decode_cache_size: int = 20_000,
        random_seed: int = 42,
        content_lmdb: Optional[Union[str, Path]] = None,
        train_lmdb: Optional[Union[str, Path]] = None,
        transform=None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.max_fonts = max(0, int(max_fonts))
        self._random_seed = int(random_seed)
        self.rng = random.Random(self._random_seed)
        self.lmdb_font_scan_limit = int(max(1000, lmdb_font_scan_limit))
        self.use_style_image = bool(use_style_image)
        self.enforce_part_bank_font_match = bool(enforce_part_bank_font_match)
        self.use_part_bank = bool(use_part_bank)
        self.part_env: Any = None
        # Deprecated: part count is now determined by actual part files per font+char.
        self.part_set_max = max(1, int(part_set_max))
        self.part_set_min = self.part_set_max if part_set_min is None else max(1, int(part_set_min))
        # 0 means keep all available parts for the selected font+char.
        self.part_pick_count = max(0, int(part_pick_count))
        self.part_image_size = max(8, int(part_image_size))
        self.part_image_cache_size = max(0, int(part_image_cache_size))
        self.lmdb_decode_cache_size = max(0, int(lmdb_decode_cache_size))
        self.part_bank_by_font: Dict[str, List[Dict[str, Any]]] = {}
        self.part_bank_by_font_char: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._part_tensor_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._glyph_decode_cache: "OrderedDict[str, Any]" = OrderedDict()

        # ---- Memory footprint warning ----
        _glyph_bytes_each = 1 * 128 * 128 * 4  # float32 tensor per cached glyph (1-ch grayscale, 128×128)
        _part_bytes_each = 1 * part_image_size * part_image_size * 4
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

        # Store LMDB paths for lazy re-open in forked worker processes.
        self._content_lmdb_path = str(content_lmdb)
        self._train_lmdb_path = str(train_lmdb)
        self.content_env = lmdb.open(self._content_lmdb_path, readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(self._train_lmdb_path, readonly=True, lock=False, readahead=False)
        self.transform = transform
        self._worker_pid: Optional[int] = os.getpid()  # tracks which PID owns the current txns

        # Keep persistent read-only transactions (avoids per-getitem txn overhead).
        self._c_txn = self.content_env.begin(buffers=True)
        self._t_txn = self.train_env.begin(buffers=True)

        # ---- Build multi-font entries ----
        lmdb_fonts = self._scan_lmdb_font_names(self._t_txn)
        all_entries = self._build_multi_font_entries(self._c_txn, self._t_txn, lmdb_fonts, apply_max_fonts=False)
        self._validate_part_bank_font_set(all_entries, part_bank_lmdb)
        entries = list(all_entries)
        if self.max_fonts > 0 and len(entries) > self.max_fonts:
            entries = sorted(entries, key=lambda x: len(x[1]), reverse=True)[:self.max_fonts]

        self.font_names = [x[0] for x in entries]
        self.valid_indices_by_font = {x[0]: x[1] for x in entries}
        self.samples: List[Tuple[str, int]] = []
        for name, idxs in entries:
            self.samples.extend((name, idx) for idx in idxs)

        # ---- PartBank (direct label-based) ----
        if self.use_part_bank:
            if self.part_pick_count > 0:
                print(
                    "[FontImageDataset] Part sampling mode: pick-"
                    f"{self.part_pick_count}-parts-per-font-char (part_set_min/max ignored)."
                )
            else:
                print("[FontImageDataset] Part sampling mode: use-all-parts-per-font-char (part_set_min/max ignored).")
            if part_bank_manifest is None:
                part_bank_manifest = self.root / "DataPreparation" / "PartBank" / "manifest.json"
            if part_bank_lmdb is None:
                part_bank_lmdb = self.root / "DataPreparation" / "LMDB" / "PartBank.lmdb"
            part_bank_lmdb_path = Path(part_bank_lmdb)
            if not part_bank_lmdb_path.is_absolute():
                part_bank_lmdb_path = (self.root / part_bank_lmdb_path).resolve()
            if not part_bank_lmdb_path.exists():
                raise FileNotFoundError(f"PartBank LMDB not found: {part_bank_lmdb_path}")
            self._part_bank_lmdb_path = str(part_bank_lmdb_path)
            self.part_env = lmdb.open(self._part_bank_lmdb_path, readonly=True, lock=False, readahead=False)
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
        """Scan all LMDB keys to extract font names (no limit)."""
        names = set()
        cursor = t_txn.cursor()
        for kv in cursor:
            key = kv[0]
            if b"@" not in key:
                continue
            prefix = key.split(b"@", 1)[0]
            try:
                names.add(prefix.decode("utf-8"))
            except UnicodeDecodeError:
                pass
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

    def _build_multi_font_entries(self, c_txn, t_txn, lmdb_fonts: List[str], apply_max_fonts: bool = True):
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
        if apply_max_fonts and self.max_fonts > 0 and len(entries) > self.max_fonts:
            entries = sorted(entries, key=lambda x: len(x[1]), reverse=True)[:self.max_fonts]
        print(f"[FontImageDataset] Multi-font mode with {len(entries)} fonts.")
        return entries

    @staticmethod
    def _scan_part_bank_font_names(part_bank_lmdb_path: Path) -> set[str]:
        env = lmdb.open(str(part_bank_lmdb_path), readonly=True, lock=False, readahead=False)
        names: set[str] = set()
        try:
            txn = env.begin()
            cursor = txn.cursor()
            for raw_key, _ in cursor:
                key = raw_key.decode("utf-8") if isinstance(raw_key, (bytes, memoryview)) else raw_key
                parts = str(key).split("/")
                if len(parts) >= 3:
                    names.add(parts[2])
        finally:
            env.close()
        return names

    def _resolve_part_bank_lmdb_path(self, part_bank_lmdb: Optional[Union[str, Path]]) -> Path:
        if part_bank_lmdb is None:
            part_bank_lmdb = self.root / "DataPreparation" / "LMDB" / "PartBank.lmdb"
        p = Path(part_bank_lmdb)
        if not p.is_absolute():
            p = (self.root / p).resolve()
        return p

    def _validate_part_bank_font_set(
        self,
        all_entries: List[Tuple[str, List[int]]],
        part_bank_lmdb: Optional[Union[str, Path]],
    ) -> None:
        if not self.enforce_part_bank_font_match:
            return

        part_bank_lmdb_path = self._resolve_part_bank_lmdb_path(part_bank_lmdb)
        if not part_bank_lmdb_path.exists():
            raise FileNotFoundError(
                "Font-set consistency check failed because PartBank LMDB does not exist: "
                f"{part_bank_lmdb_path}"
            )

        train_fonts = set(name for name, _ in all_entries)
        part_fonts = self._scan_part_bank_font_names(part_bank_lmdb_path)

        if train_fonts == part_fonts:
            print(f"[FontImageDataset] Font-set consistency check passed: {len(train_fonts)} fonts.")
            return

        only_train = sorted(train_fonts - part_fonts)
        only_part = sorted(part_fonts - train_fonts)
        msg = (
            "PartBank font set mismatch. Training aborted.\n"
            f"  train_fonts={len(train_fonts)} part_bank_fonts={len(part_fonts)}\n"
            f"  only_in_train (first 20): {only_train[:20]}\n"
            f"  only_in_part_bank (first 20): {only_part[:20]}\n"
            "Please rebuild/sync datasets so the font sets are exactly identical."
        )
        raise RuntimeError(msg)

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
            DataPreparation/PartBank/<font_name>/<Uxxxx>/part_NNN_UXXXX.png
        (legacy flat keys without the <Uxxxx> level are also supported)
        """
        from collections import defaultdict
        tmp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        tmp_char: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
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
            if ch:
                tmp_char[font_name][ch].append({
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
        by_font_char: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for font_name in sorted(tmp_char):
            char_rows: Dict[str, List[Dict[str, Any]]] = {}
            for ch in sorted(tmp_char[font_name]):
                rows = tmp_char[font_name][ch]
                rows.sort(key=lambda x: (int(x["index"]), str(x["lmdb_key"])))
                char_rows[ch] = rows
            by_font_char[font_name] = char_rows
        self.part_bank_by_font_char = by_font_char
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
    #  Part sampling -- fixed-size or variable-size range
    # ------------------------------------------------------------------ #
    def _sample_parts_for_font(
        self,
        font_name: str,
        ref_char: str,
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """Return all parts for one font+char only (no font-level fallback)."""
        char_map = self.part_bank_by_font_char.get(font_name, {})
        rows = char_map.get(ref_char, [])
        if not rows:
            return []
        # Deterministic order is already ensured at LMDB scan stage.
        out = list(rows)
        if self.part_pick_count <= 0 or self.part_pick_count >= len(out):
            return out
        # Deterministic per-sample RNG is passed in by caller.
        picked = sorted(rng.sample(range(len(out)), k=self.part_pick_count))
        return [out[i] for i in picked]

    def _part_array_from_bytes(self, b: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(b)).convert("L")
        if self.part_image_size > 0:
            img = img.resize((self.part_image_size, self.part_image_size), Image.BILINEAR)
        # Normalize to [-1, 1]: equivalent to T.ToTensor() + T.Normalize(0.5, 0.5)
        # ToTensor: /255 -> [0,1]; Normalize(0.5,0.5): (x-0.5)/0.5 = 2x-1 -> [-1,1]
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

    def _rng_for_index(self, sample_index: int, salt: int) -> random.Random:
        seed = (
            int(self._random_seed) * 1_000_003
            + int(sample_index) * 9_176
            + int(salt)
        ) & 0xFFFFFFFF
        return random.Random(seed)

    def _sample_style_char_index(
        self,
        font_name: str,
        target_index: int,
        rng: random.Random,
    ) -> int:
        """Pick a style reference index from the same font (prefer different char)."""
        candidates = self.valid_indices_by_font.get(font_name, [])
        if not candidates:
            return int(target_index)
        if len(candidates) == 1:
            return int(candidates[0])

        # Filter out target to guarantee a different char, then choose uniformly.
        others = [idx for idx in candidates if int(idx) != int(target_index)]
        if others:
            return int(rng.choice(others))
        return int(target_index)

    # ------------------------------------------------------------------ #
    #  Multi-worker LMDB safety
    # ------------------------------------------------------------------ #
    def _ensure_txns(self) -> None:
        """Re-open LMDB envs & txns if we detect we're in a forked worker process.

        After fork(), the parent's mmap-based LMDB handles are unsafe to use.
        We lazily re-open them on first __getitem__ call in each worker.
        """
        pid = os.getpid()
        if self._worker_pid == pid:
            return  # already initialised for this process
        # We're in a new forked worker — re-open everything.
        # Do NOT close the parent's envs (they belong to the parent process).
        self.content_env = lmdb.open(self._content_lmdb_path, readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(self._train_lmdb_path, readonly=True, lock=False, readahead=False)
        self._c_txn = self.content_env.begin(buffers=True)
        self._t_txn = self.train_env.begin(buffers=True)
        if self.use_part_bank and hasattr(self, '_part_bank_lmdb_path'):
            self.part_env = lmdb.open(self._part_bank_lmdb_path, readonly=True, lock=False, readahead=False)
            self._p_txn = self.part_env.begin(buffers=True)
        # Clear caches — they reference parent-process memoryviews.
        self._glyph_decode_cache.clear()
        self._part_tensor_cache.clear()
        # Re-seed per-worker RNG for diversity across workers.
        # Use the stored initial seed (deterministic) instead of internal RNG state.
        import torch as _torch
        worker_info = _torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.rng = random.Random(self._random_seed + worker_info.id)
        self._worker_pid = pid

    # ------------------------------------------------------------------ #
    #  __len__ / __getitem__
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        self._ensure_txns()
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
            "has_parts": 0.0,
        }

        ref_char = ch
        if self.use_style_image or self.use_part_bank:
            style_rng = self._rng_for_index(index, salt=17)
            style_idx = self._sample_style_char_index(font_name, real_idx, rng=style_rng)
            style_ch = self.char_list[style_idx]
            ref_char = style_ch
        if self.use_style_image:
            style_key = f"{font_name}@{style_ch}"
            style_bytes = self._t_txn.get(style_key.encode("utf-8"))
            # Fallback to target glyph if style glyph is missing unexpectedly.
            if style_bytes is None:
                style_ch = ch
                style_key = input_key
                style_bytes = input_bytes
            sample["style_img"] = self._bytes_to_img(bytes(style_bytes), cache_key=f"S@{style_key}")
            sample["style_char"] = style_ch

        if self.use_part_bank:
            if self._p_txn is None:
                raise RuntimeError("PartBank enabled but _p_txn is not initialized.")
            part_rows_a = self._sample_parts_for_font(
                font_name=font_name,
                ref_char=ref_char,
                rng=self._rng_for_index(index, salt=101),
            )
            part_rows_b = self._sample_parts_for_font(
                font_name=font_name,
                ref_char=ref_char,
                rng=self._rng_for_index(index, salt=202),
            )
            if not part_rows_a or not part_rows_b:
                raise KeyError(f"Missing PartBank parts for font: {font_name}")
            part_imgs_a = [self._load_part_tensor(self._p_txn, str(r["lmdb_key"])) for r in part_rows_a]
            part_imgs_b = [self._load_part_tensor(self._p_txn, str(r["lmdb_key"])) for r in part_rows_b]
            sample["parts"] = torch.stack(part_imgs_a, dim=0)
            sample["parts_b"] = torch.stack(part_imgs_b, dim=0)
            sample["part_mask"] = torch.ones((len(part_imgs_a),), dtype=torch.float32)
            sample["part_mask_b"] = torch.ones((len(part_imgs_b),), dtype=torch.float32)
            sample["has_parts"] = 1.0
            sample["part_char"] = ref_char

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
            try:
                self.content_env.close()
            except Exception:
                pass
        if hasattr(self, "train_env"):
            try:
                self.train_env.close()
            except Exception:
                pass
        if hasattr(self, "part_env") and self.part_env is not None:
            try:
                self.part_env.close()
            except Exception:
                pass
            self.part_env = None
        self._worker_pid = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    ds = FontImageDataset()
    print(f"Valid samples: {len(ds)}")
    sample = ds[0]
    print(sample["char"])
