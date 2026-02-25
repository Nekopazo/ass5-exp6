#!/usr/bin/env python3
"""
FontImageDataset
================
Dataset that returns a content glyph, the corresponding glyph of a training font
(*input image*), and a list of style reference glyphs sampled from a shared
reference pool.

Data sources
+------------
* **CharacterData/CharList.json**   : list of content characters **A**
* **CharacterData/ReferenceCharList.json** : list of global style reference chars
* **DataPreparation/FontList.json** : list of available *.ttf* fonts – the selected
  *font_index* denotes training font **B**
* **DataPreparation/LMDB/ContentFont.lmdb** : key `ContentFont@<char>`
* **DataPreparation/LMDB/TrainFont.lmdb**   : key `<FontName>@<char>`
* **DataPreparation/LMDB/PartBank.lmdb**    : key `<manifest part path>`

Returned sample
```
{
    "char":   str,                    # character code
    "content": Image/Tensor,          # glyph rendered in ContentFont
    "input":   Image/Tensor,          # same char rendered in font B (training target)
    "styles":  List[Image/Tensor]     # reference glyphs from font B
}
```
If *transform* is provided, it is applied to **every** image.
"""

from pathlib import Path
import json
import io
import random
import re
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter

import lmdb
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Fallback stub when torch is not installed.
    class Dataset:  # type: ignore
        pass


class FontImageDataset(Dataset):
    def __init__(
        self,
        project_root: Union[str, Path] = ".",
        font_index: int = 0,
        font_name: Optional[str] = None,
        font_mode: str = "fixed",
        max_fonts: int = 0,
        auto_select_font: bool = True,
        lmdb_font_scan_limit: int = 100000,
        num_style_refs: int = 3,
        style_sampling: str = "random",
        include_target_in_style: bool = False,
        use_part_bank: bool = False,
        part_bank_manifest: Optional[Union[str, Path]] = None,
        part_bank_lmdb: Optional[Union[str, Path]] = None,
        part_retrieval_ep_ckpt: Optional[Union[str, Path]] = None,
        part_retrieval_device: Optional[str] = None,
        part_set_size: int = 32,
        part_set_min_size: int = 32,
        part_set_sampling: str = "random",
        part_target_char_priority: bool = False,
        part_image_size: int = 64,
        part_image_cache_size: int = 50000,
        lmdb_decode_cache_size: int = 20000,
        use_style_plan_cache: bool = True,
        style_prefetch_limit: int = 20000,
        random_seed: int = 42,
        content_lmdb: Optional[Union[str, Path]] = None,
        train_lmdb: Optional[Union[str, Path]] = None,
        transform=None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.font_mode = str(font_mode).strip().lower()
        if self.font_mode not in {"fixed", "random"}:
            raise ValueError("font_mode must be either 'fixed' or 'random'")
        self.max_fonts = max(0, int(max_fonts))
        self.num_style_refs = max(1, int(num_style_refs))
        self.style_sampling = style_sampling
        self.rng = random.Random(random_seed)
        self.auto_select_font = auto_select_font
        self.lmdb_font_scan_limit = int(max(1000, lmdb_font_scan_limit))
        self.include_target_in_style = bool(include_target_in_style)
        self.use_part_bank = bool(use_part_bank)
        self.part_env = None
        self.part_retrieval_ep_ckpt = part_retrieval_ep_ckpt
        self.part_retrieval_device = str(part_retrieval_device).strip() if part_retrieval_device else "cpu"
        self.part_retrieval_conf_threshold = 0.85
        self.part_retrieval_margin_threshold = 0.25
        self.part_retrieval_temperature = 0.7
        self.part_retrieval_topk = 3
        self.part_set_size = max(1, int(part_set_size))
        self.part_set_min_size = max(1, int(part_set_min_size))
        if self.part_set_min_size > self.part_set_size:
            raise ValueError(
                "part_set_min_size must be <= part_set_size, "
                f"got min={self.part_set_min_size} max={self.part_set_size}"
            )
        self.part_set_sampling = str(part_set_sampling).strip().lower()
        if self.part_set_sampling not in {"deterministic", "random"}:
            raise ValueError("part_set_sampling must be either 'deterministic' or 'random'")
        self.part_target_char_priority = bool(part_target_char_priority)
        self.part_image_size = max(8, int(part_image_size))
        self.part_image_cache_size = max(0, int(part_image_cache_size))
        self.lmdb_decode_cache_size = max(0, int(lmdb_decode_cache_size))
        self.use_style_plan_cache = bool(use_style_plan_cache)
        self.style_prefetch_limit = max(0, int(style_prefetch_limit))
        self.part_bank_by_font: Dict[str, List[Dict[str, Any]]] = {}
        self._part_tensor_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._glyph_decode_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._style_plan_cache: Optional[List[List[str]]] = None
        self.part_retrieval_model = None
        self.part_retrieval_class_fonts: List[str] = []
        self.part_retrieval_candidate_labels = None

        # Paths
        char_list_path = self.root / "CharacterData" / "CharList.json"
        ref_char_list_path = self.root / "CharacterData" / "ReferenceCharList.json"
        font_list_path = self.root / "DataPreparation" / "FontList.json"

        # Load JSON files
        self.char_list: List[str] = json.load(char_list_path.open("r", encoding="utf-8"))
        if ref_char_list_path.exists():
            self.reference_chars: List[str] = json.load(ref_char_list_path.open("r", encoding="utf-8"))
        else:
            # Fallback for old datasets without explicit reference list
            self.reference_chars = self.char_list[:300]
        font_list: List[str] = json.load(font_list_path.open("r", encoding="utf-8"))
        self.font_list_stems: List[str] = [Path(x).stem for x in font_list]

        if font_index < 0 or font_index >= len(self.font_list_stems):
            raise IndexError(f"font_index {font_index} out of range of FontList.json")
        requested_font = font_name if font_name is not None else self.font_list_stems[font_index]
        self.font_name = requested_font

        # LMDB paths
        if content_lmdb is None:
            content_lmdb = self.root / "DataPreparation" / "LMDB" / "ContentFont.lmdb"
        if train_lmdb is None:
            train_lmdb = self.root / "DataPreparation" / "LMDB" / "TrainFont.lmdb"

        # Open LMDB in read-only mode (lock=False to allow multi-process readers)
        self.content_env = lmdb.open(str(content_lmdb), readonly=True, lock=False, readahead=False)
        self.train_env = lmdb.open(str(train_lmdb), readonly=True, lock=False, readahead=False)

        # Optional transform function
        self.transform = transform

        with self.content_env.begin() as c_txn, self.train_env.begin() as t_txn:
            lmdb_fonts = self._scan_lmdb_font_names(t_txn)
            if self.font_mode == "fixed":
                chosen_font, available_ref_chars, valid_indices = self._choose_font_and_build_indices(
                    requested_font=requested_font,
                    c_txn=c_txn,
                    t_txn=t_txn,
                    lmdb_fonts=lmdb_fonts,
                )
                self.font_name = chosen_font
                self.available_ref_chars = available_ref_chars
                self.valid_indices = valid_indices
                self.available_ref_chars_by_font = {chosen_font: available_ref_chars}
                self.valid_indices_by_font = {chosen_font: valid_indices}
                self.samples = [(chosen_font, idx) for idx in valid_indices]
            else:
                entries = self._build_multi_font_entries(
                    requested_font=requested_font,
                    c_txn=c_txn,
                    t_txn=t_txn,
                    lmdb_fonts=lmdb_fonts,
                )
                self.font_names = [x[0] for x in entries]
                self.available_ref_chars_by_font = {x[0]: x[1] for x in entries}
                self.valid_indices_by_font = {x[0]: x[2] for x in entries}
                self.samples = []
                for name, _, idxs in entries:
                    self.samples.extend((name, idx) for idx in idxs)
                self.font_name = requested_font
                self.available_ref_chars = self.available_ref_chars_by_font[self.font_names[0]]
                self.valid_indices = self.valid_indices_by_font[self.font_names[0]]

        self.available_style_chars_by_font = {
            font: self._build_style_char_pool(valid_indices)
            for font, valid_indices in self.valid_indices_by_font.items()
        }

        if self.use_part_bank:
            if "torch" not in globals():
                raise RuntimeError("use_part_bank=True requires torch to be installed.")
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
            self.part_bank_by_font = self._load_part_bank_manifest(part_bank_manifest)
            self._filter_samples_by_part_bank()
            self._init_part_retrieval_classifier(self.part_retrieval_ep_ckpt)

        self._build_style_plan_cache_if_needed()
        self._prefetch_style_decode_cache()

        if not self.samples:
            raise RuntimeError("No valid samples found – please check LMDB integrity and paths.")

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

    def _build_indices_for_font(self, font_name: str, c_txn, t_txn) -> Tuple[List[str], List[int]]:
        available_ref_chars: List[str] = []
        for rc in self.reference_chars:
            ref_key = f"{font_name}@{rc}".encode("utf-8")
            if t_txn.get(ref_key) is not None:
                available_ref_chars.append(rc)

        valid_indices: List[int] = []
        if not available_ref_chars:
            return available_ref_chars, valid_indices

        for idx, ch in enumerate(self.char_list):
            content_key = f"ContentFont@{ch}".encode("utf-8")
            if c_txn.get(content_key) is None:
                continue
            input_key = f"{font_name}@{ch}".encode("utf-8")
            if t_txn.get(input_key) is None:
                continue
            valid_indices.append(idx)
        return available_ref_chars, valid_indices

    def _ordered_font_candidates(self, requested_font: str, lmdb_fonts: List[str]) -> List[str]:
        candidates: List[str] = []
        seen = set()
        for n in [requested_font] + self.font_list_stems + lmdb_fonts:
            if n in seen:
                continue
            seen.add(n)
            candidates.append(n)
        return candidates

    def _build_style_char_pool(self, valid_indices: List[int]) -> List[str]:
        # Deduplicate while preserving order.
        seen = set()
        chars: List[str] = []
        for idx in valid_indices:
            ch = self.char_list[idx]
            if ch in seen:
                continue
            seen.add(ch)
            chars.append(ch)
        return chars

    @staticmethod
    def _parse_part_index(path_like: str) -> int:
        m = re.search(r"part_(\d+)", path_like)
        if m is None:
            return 1_000_000_000
        try:
            return int(m.group(1))
        except ValueError:
            return 1_000_000_000

    def _load_part_bank_manifest(self, part_bank_manifest: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
        path = Path(part_bank_manifest)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PartBank manifest not found: {path}")

        try:
            obj = json.load(path.open("r", encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse PartBank manifest: {path}") from e

        fonts = obj.get("fonts", {}) if isinstance(obj, dict) else {}
        if not isinstance(fonts, dict):
            raise RuntimeError(f"Invalid PartBank manifest format: {path}")

        out: Dict[str, List[Dict[str, Any]]] = {}
        for font_name, info in fonts.items():
            if not isinstance(font_name, str) or not isinstance(info, dict):
                continue
            part_rows: List[Dict[str, Any]] = []
            for row in info.get("parts", []):
                if not isinstance(row, dict):
                    continue
                rel = row.get("path")
                if not isinstance(rel, str) or not rel:
                    continue
                rel_posix = Path(rel).as_posix()
                lmdb_key = row.get("lmdb_key", rel_posix)
                if not isinstance(lmdb_key, str) or not lmdb_key:
                    lmdb_key = rel_posix
                ch = row.get("char")
                part_rows.append(
                    {
                        "lmdb_key": lmdb_key,
                        "char": ch if isinstance(ch, str) and len(ch) == 1 else "",
                        "x": int(row.get("x", 0)) if isinstance(row.get("x", 0), (int, float)) else 0,
                        "y": int(row.get("y", 0)) if isinstance(row.get("y", 0), (int, float)) else 0,
                        "response": float(row.get("response", 0.0)) if isinstance(row.get("response", 0.0), (int, float)) else 0.0,
                        "index": self._parse_part_index(str(rel_posix)),
                    }
                )

            if not part_rows:
                continue

            # Canonical deterministic order: file index first, then confidence.
            part_rows.sort(key=lambda x: (int(x["index"]), -float(x["response"]), str(x["lmdb_key"])))
            out[font_name] = part_rows

        return out

    def _filter_samples_by_part_bank(self) -> None:
        available_fonts = set(self.part_bank_by_font.keys())
        if self.font_mode == "fixed":
            if self.font_name not in available_fonts:
                raise RuntimeError(
                    f"Font '{self.font_name}' not found in PartBank manifest; "
                    "cannot use part-bank conditioning."
                )
            return

        before = len(self.samples)
        self.samples = [x for x in self.samples if x[0] in available_fonts]
        after = len(self.samples)
        if after <= 0:
            raise RuntimeError("No samples left after applying PartBank font filter.")
        if after < before:
            removed_fonts = sorted([x for x in self.valid_indices_by_font.keys() if x not in available_fonts])
            print(
                f"[FontImageDataset] PartBank filter removed {before - after} samples "
                f"across {len(removed_fonts)} fonts."
            )

    def _init_part_retrieval_classifier(self, ckpt_path_like: Optional[Union[str, Path]]) -> None:
        if ckpt_path_like is None:
            raise ValueError("part_retrieval_ep_ckpt is required when use_part_bank=True")
        if "torch" not in globals():
            raise RuntimeError("Part retrieval requires torch.")

        path = Path(ckpt_path_like)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"E_p classifier checkpoint not found: {path}")

        try:
            from models.style_encoders import FontClassifier
        except Exception as e:
            raise RuntimeError("Failed to import FontClassifier for part retrieval.") from e

        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"invalid classifier checkpoint: {path}")

        class_fonts = [str(x) for x in obj.get("font_names", [])]
        if not class_fonts:
            raise RuntimeError("classifier checkpoint missing 'font_names'")
        cfg = obj.get("config", {}) if isinstance(obj.get("config", {}), dict) else {}
        backbone = str(cfg.get("backbone", "resnet18"))
        state = obj.get("e_p", obj)

        model = FontClassifier(in_channels=3, num_fonts=len(class_fonts), backbone=backbone)
        model.load_state_dict(state, strict=False)
        try:
            model = model.to(self.part_retrieval_device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to move part retrieval model to device='{self.part_retrieval_device}'."
            ) from e
        model.eval()

        candidate_labels = [
            i for i, f in enumerate(class_fonts)
            if f in self.part_bank_by_font
        ]
        if not candidate_labels:
            raise RuntimeError("No overlap between classifier fonts and PartBank fonts.")

        self.part_retrieval_model = model
        self.part_retrieval_class_fonts = class_fonts
        self.part_retrieval_candidate_labels = torch.tensor(
            candidate_labels,
            dtype=torch.long,
            device=self.part_retrieval_device,
        )

    def _predict_part_font_from_styles(
        self,
        style_imgs: List[Any],
    ) -> Tuple[List[str], List[float], Dict[str, float | str]]:
        if self.part_retrieval_model is None or self.part_retrieval_candidate_labels is None:
            raise RuntimeError("Part retrieval model is not initialized.")
        if not style_imgs:
            raise RuntimeError("style_imgs is empty while part retrieval is enabled.")
        if not all(isinstance(x, torch.Tensor) for x in style_imgs):
            raise RuntimeError("part retrieval expects tensor style images; check transform pipeline.")

        with torch.no_grad():
            x = torch.stack(style_imgs, dim=0).to(self.part_retrieval_device, non_blocking=True)
            logits = self.part_retrieval_model(x)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            cand_probs = probs.index_select(0, self.part_retrieval_candidate_labels)
            k = min(int(self.part_retrieval_topk), int(cand_probs.numel()))
            topv, topi = torch.topk(cand_probs, k=k, dim=0)

        if topv.numel() <= 0:
            raise RuntimeError("part retrieval produced empty candidate scores.")

        top_fonts: List[str] = []
        top_probs: List[float] = []
        for i in range(int(topv.numel())):
            idx_local = int(topi[i].item())
            idx_global = int(self.part_retrieval_candidate_labels[idx_local].item())
            f = self.part_retrieval_class_fonts[idx_global]
            if f not in self.part_bank_by_font:
                continue
            top_fonts.append(f)
            top_probs.append(float(topv[i].item()))

        if not top_fonts:
            raise RuntimeError("part retrieval top-k fonts are not present in PartBank.")

        p1 = float(top_probs[0])
        p2 = float(top_probs[1]) if len(top_probs) >= 2 else 0.0
        margin = p1 - p2
        use_top1 = (p1 >= self.part_retrieval_conf_threshold) or (margin >= self.part_retrieval_margin_threshold)
        if use_top1:
            return [top_fonts[0]], [1.0], {"mode": "top1", "p1": p1, "p2": p2, "margin": margin}

        tau = float(max(1e-6, self.part_retrieval_temperature))
        arr = np.array(top_probs, dtype=np.float64)
        w = np.power(np.clip(arr, 1e-12, 1.0), 1.0 / tau)
        s = float(w.sum())
        if s <= 0.0:
            weights = [1.0 / float(len(top_fonts)) for _ in top_fonts]
        else:
            weights = [float(x / s) for x in w.tolist()]
        return top_fonts, weights, {"mode": "top3", "p1": p1, "p2": p2, "margin": margin}

    @staticmethod
    def _normalize_weights(ws: List[float]) -> List[float]:
        if not ws:
            return []
        s = float(sum(max(0.0, float(x)) for x in ws))
        if s <= 0.0:
            return [1.0 / float(len(ws)) for _ in ws]
        return [max(0.0, float(x)) / s for x in ws]

    def _split_counts_by_weights(self, total_k: int, weights: List[float]) -> List[int]:
        if total_k <= 0 or not weights:
            return [0 for _ in weights]
        w = self._normalize_weights(weights)
        raw = [float(total_k) * x for x in w]
        cnt = [int(round(x)) for x in raw]
        diff = int(total_k - sum(cnt))
        order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
        oi = 0
        while diff != 0 and order:
            idx = order[oi % len(order)]
            if diff > 0:
                cnt[idx] += 1
                diff -= 1
            else:
                if cnt[idx] > 0:
                    cnt[idx] -= 1
                    diff += 1
            oi += 1
            if oi > 10000:
                break
        return cnt

    def _part_array_from_bytes(self, b: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if self.part_image_size > 0:
            img = img.resize((self.part_image_size, self.part_image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        # [H, W, C] -> [C, H, W]
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    def _load_part_tensor_uncached(self, part_txn, lmdb_key: str):
        b = part_txn.get(lmdb_key.encode("utf-8"))
        if b is None:
            raise KeyError(f"missing PartBank lmdb key: {lmdb_key}")
        arr = self._part_array_from_bytes(b)
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

    def _sample_from_font_rows(self, rows: List[Dict[str, Any]], target_char: str, size: int) -> List[Dict[str, Any]]:
        if not rows or size <= 0:
            return []
        size = min(int(size), len(rows))
        size = max(0, size)
        size_hi = min(self.part_set_size, len(rows))
        if size_hi <= 0:
            return []

        same_char: List[Dict[str, Any]] = []
        if self.part_target_char_priority and target_char:
            same_char = [r for r in rows if r.get("char", "") == target_char]

        if self.part_set_sampling == "random":
            picked: List[Dict[str, Any]] = []
            if same_char:
                keep = min(len(same_char), size)
                picked.extend(self.rng.sample(same_char, keep) if len(same_char) > keep else same_char[:keep])
            if len(picked) < size:
                remain = [r for r in rows if r not in picked]
                need = size - len(picked)
                if len(remain) >= need:
                    picked.extend(self.rng.sample(remain, need))
                else:
                    picked.extend(remain)
            return picked[:size]

        # deterministic
        if same_char:
            seen_ids = {id(x) for x in same_char}
            merged = same_char + [r for r in rows if id(r) not in seen_ids]
            return merged[:size]
        return rows[:size]

    def _select_part_rows(
        self,
        font_name: str,
        target_char: str,
        retrieved_fonts: Optional[List[str]] = None,
        retrieved_weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        size = int(self.part_set_size)

        if not retrieved_fonts:
            retrieved_fonts = [font_name]
            retrieved_weights = [1.0]
        if not retrieved_weights or len(retrieved_weights) != len(retrieved_fonts):
            retrieved_weights = [1.0 / float(len(retrieved_fonts)) for _ in retrieved_fonts]

        valid_fonts: List[str] = []
        valid_weights: List[float] = []
        for f, w in zip(retrieved_fonts, retrieved_weights):
            if f in self.part_bank_by_font and len(self.part_bank_by_font[f]) > 0:
                valid_fonts.append(f)
                valid_weights.append(float(w))
        if not valid_fonts:
            rows = self.part_bank_by_font.get(font_name, [])
            if not rows:
                return []
            return self._sample_from_font_rows(rows, target_char, size)

        if len(valid_fonts) == 1:
            return self._sample_from_font_rows(self.part_bank_by_font[valid_fonts[0]], target_char, size)

        counts = self._split_counts_by_weights(size, valid_weights)
        out: List[Dict[str, Any]] = []
        for f, n in zip(valid_fonts, counts):
            if n <= 0:
                continue
            out.extend(self._sample_from_font_rows(self.part_bank_by_font[f], target_char, n))

        # Keep variable-length output when candidate fonts do not have enough unique parts.
        deduped: List[Dict[str, Any]] = []
        seen_keys = set()
        for row in out:
            k = str(row.get("lmdb_key", ""))
            if not k or k in seen_keys:
                continue
            seen_keys.add(k)
            deduped.append(row)

        if self.part_set_sampling == "random":
            self.rng.shuffle(deduped)
        return deduped[:size]

    def _build_multi_font_entries(self, requested_font: str, c_txn, t_txn, lmdb_fonts: List[str]):
        entries: List[Tuple[str, List[str], List[int]]] = []
        for name in self._ordered_font_candidates(requested_font, lmdb_fonts):
            refs, valid = self._build_indices_for_font(name, c_txn, t_txn)
            if refs and valid:
                entries.append((name, refs, valid))

        if not entries:
            raise RuntimeError(
                "Could not find any usable font in TrainFont.lmdb. "
                "Please verify LMDB keys format '<FontName>@<char>'."
            )

        if self.max_fonts > 0 and len(entries) > self.max_fonts:
            entries = sorted(entries, key=lambda x: (len(x[2]), len(x[1])), reverse=True)[: self.max_fonts]

        if requested_font not in {x[0] for x in entries}:
            top = max(entries, key=lambda x: (len(x[2]), len(x[1])))
            print(
                f"[FontImageDataset] Requested font '{requested_font}' unavailable for random mode. "
                f"Using {len(entries)} fonts; top '{top[0]}' has {len(top[2])} valid chars and {len(top[1])} refs."
            )
        else:
            print(f"[FontImageDataset] Random-font mode enabled with {len(entries)} fonts.")
        return entries

    def _choose_font_and_build_indices(self, requested_font: str, c_txn, t_txn, lmdb_fonts: List[str]):
        req_refs, req_valid = self._build_indices_for_font(requested_font, c_txn, t_txn)
        if req_refs and req_valid:
            return requested_font, req_refs, req_valid

        if not self.auto_select_font:
            raise RuntimeError(
                f"Requested font '{requested_font}' has no valid refs/samples in TrainFont.lmdb. "
                f"Set auto_select_font=True to fallback automatically."
            )

        candidates = self._ordered_font_candidates(requested_font, lmdb_fonts)

        best_name = None
        best_refs: List[str] = []
        best_valid: List[int] = []
        best_score = (-1, -1)
        for name in candidates:
            refs, valid = self._build_indices_for_font(name, c_txn, t_txn)
            score = (len(valid), len(refs))
            if score > best_score:
                best_score = score
                best_name = name
                best_refs = refs
                best_valid = valid

        if best_name is None or not best_refs or not best_valid:
            raise RuntimeError(
                "Could not find any usable font in TrainFont.lmdb. "
                "Please verify LMDB keys format '<FontName>@<char>'."
            )

        print(
            f"[FontImageDataset] Requested font '{requested_font}' unavailable. "
            f"Auto-selected '{best_name}' with {len(best_valid)} valid chars and {len(best_refs)} refs."
        )
        return best_name, best_refs, best_valid

    def __len__(self):
        return len(self.samples)

    def _decode_img(self, b: bytes):
        img = Image.open(io.BytesIO(b)).convert("RGB")
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

    def _build_style_plan_cache_if_needed(self) -> None:
        if not self.use_style_plan_cache:
            self._style_plan_cache = None
            return
        if not self.samples:
            self._style_plan_cache = []
            return
        plan: List[List[str]] = []
        old_state = self.rng.getstate()
        try:
            for font_name, real_idx in self.samples:
                ch = self.char_list[real_idx]
                plan.append(self._sample_style_chars_dynamic(ch, font_name))
        finally:
            self.rng.setstate(old_state)
        self._style_plan_cache = plan
        print(f"[FontImageDataset] style plan cache built for {len(plan)} samples.")

    def _prefetch_style_decode_cache(self) -> None:
        if self.lmdb_decode_cache_size <= 0:
            return
        if self.style_prefetch_limit <= 0:
            return
        if not self._style_plan_cache:
            return

        counter: Counter = Counter()
        for i, style_chars in enumerate(self._style_plan_cache):
            font_name, _ = self.samples[i]
            for sc in style_chars:
                counter[f"{font_name}@{sc}"] += 1
        if not counter:
            return

        n = min(self.style_prefetch_limit, self.lmdb_decode_cache_size, len(counter))
        top_keys = [k for k, _ in counter.most_common(n)]
        with self.train_env.begin() as t_txn:
            hit = 0
            for key in top_keys:
                b = t_txn.get(key.encode("utf-8"))
                if b is None:
                    continue
                self._bytes_to_img(b, cache_key=f"T@{key}")
                hit += 1
        print(f"[FontImageDataset] prefetched style decode cache: {hit}/{n} glyphs.")

    def _sample_style_chars_dynamic(self, ch: str, font_name: str) -> List[str]:
        available_ref_chars = self.available_ref_chars_by_font[font_name]
        style_char_pool = self.available_style_chars_by_font.get(font_name, [])
        picked: List[str] = []
        if self.include_target_in_style and ch in available_ref_chars:
            picked.append(ch)

        pool = [c for c in available_ref_chars if c != ch]
        if not pool:
            pool = [c for c in style_char_pool if c != ch]
        if not pool:
            pool = [c for c in available_ref_chars] or [c for c in style_char_pool]
        if not pool:
            # Should not happen as long as valid_indices is non-empty.
            pool = [ch]
        needed = self.num_style_refs - len(picked)

        if needed > 0:
            if self.style_sampling == "deterministic":
                picked.extend(pool[:needed])
            else:
                if len(pool) >= needed:
                    picked.extend(self.rng.sample(pool, needed))
                else:
                    picked.extend(pool)
                    while len(picked) < self.num_style_refs:
                        picked.append(self.rng.choice(pool))

        return picked[: self.num_style_refs]

    def _sample_style_chars(self, ch: str, font_name: str, sample_index: Optional[int] = None) -> List[str]:
        if self._style_plan_cache is not None and sample_index is not None:
            if 0 <= sample_index < len(self._style_plan_cache):
                row = self._style_plan_cache[sample_index]
                if row:
                    return row[: self.num_style_refs]
        return self._sample_style_chars_dynamic(ch, font_name)

    def sample_style_plan(self, index: int) -> Dict[str, Any]:
        """Sample style chars for one dataset item without reading LMDB image bytes."""
        font_name, real_idx = self.samples[index]
        ch = self.char_list[real_idx]
        style_chars = self._sample_style_chars(ch, font_name, sample_index=index)
        return {
            "index": index,
            "font": font_name,
            "char": ch,
            "style_chars": style_chars,
        }

    def __getitem__(self, index: int):
        font_name, real_idx = self.samples[index]
        ch = self.char_list[real_idx]

        content_key = f"ContentFont@{ch}"
        input_key = f"{font_name}@{ch}"
        style_chars = self._sample_style_chars(ch, font_name, sample_index=index)

        with self.content_env.begin() as c_txn, self.train_env.begin() as t_txn:
            content_bytes = c_txn.get(content_key.encode("utf-8"))
            input_bytes = t_txn.get(input_key.encode("utf-8"))
            if content_bytes is None or input_bytes is None:
                raise KeyError("Missing required glyph images")

            content_img = self._bytes_to_img(content_bytes, cache_key=f"C@{content_key}")
            input_img = self._bytes_to_img(input_bytes, cache_key=f"T@{input_key}")
            style_imgs: List[Any] = []
            for sc in style_chars:
                style_key = f"{font_name}@{sc}"
                sb = t_txn.get(style_key.encode("utf-8"))
                if sb is None:
                    raise KeyError(f"Missing style glyph: {sc}")
                style_imgs.append(self._bytes_to_img(sb, cache_key=f"T@{style_key}"))

        sample: Dict[str, Any] = {
            "font": font_name,
            "char": ch,
            "content": content_img,
            "input": input_img,
            "styles": style_imgs,
        }
        if self.use_part_bank:
            if self.part_env is None:
                raise RuntimeError("PartBank enabled but part_env is not initialized.")
            retrieved_fonts: Optional[List[str]] = None
            retrieved_weights: Optional[List[float]] = None
            retrieval_meta: Optional[Dict[str, float | str]] = None
            retrieved_fonts, retrieved_weights, retrieval_meta = self._predict_part_font_from_styles(
                style_imgs,
            )
            part_rows = self._select_part_rows(
                font_name,
                ch,
                retrieved_fonts=retrieved_fonts,
                retrieved_weights=retrieved_weights,
            )
            part_rows_b = self._select_part_rows(
                font_name,
                ch,
                retrieved_fonts=retrieved_fonts,
                retrieved_weights=retrieved_weights,
            )
            if not part_rows or not part_rows_b:
                raise KeyError(f"Missing PartBank parts for font: {font_name}")
            with self.part_env.begin() as p_txn:
                part_imgs = [self._load_part_tensor(p_txn, str(r["lmdb_key"])) for r in part_rows]
                part_imgs_b = [self._load_part_tensor(p_txn, str(r["lmdb_key"])) for r in part_rows_b]
            parts = torch.stack(part_imgs, dim=0)
            parts_b = torch.stack(part_imgs_b, dim=0)
            part_mask = torch.ones((parts.size(0),), dtype=torch.float32)
            part_mask_b = torch.ones((parts_b.size(0),), dtype=torch.float32)
            sample["parts"] = parts
            sample["part_mask"] = part_mask
            sample["parts_b"] = parts_b
            sample["part_mask_b"] = part_mask_b
            if retrieved_fonts:
                sample["retrieved_part_fonts"] = list(retrieved_fonts)
            if retrieved_weights:
                sample["retrieved_part_weights"] = [float(x) for x in retrieved_weights]
            if retrieval_meta:
                sample["retrieval_gate"] = retrieval_meta
        return sample

    def close(self):
        if hasattr(self, "content_env"):
            self.content_env.close()
        if hasattr(self, "train_env"):
            self.train_env.close()
        if hasattr(self, "part_env") and self.part_env is not None:
            self.part_env.close()
            self.part_env = None

    def __del__(self):
        self.close()


# Quick sanity test
if __name__ == "__main__":
    ds = FontImageDataset(num_style_refs=3)
    print(f"Valid samples: {len(ds)}")
    sample = ds[0]
    print(sample["char"], len(sample["styles"]))

    # Visualize loaded images (uncomment to view)
