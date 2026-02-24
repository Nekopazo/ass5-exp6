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
from typing import List, Dict, Any, Optional, Union, Tuple, Set
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
        component_guided_style: bool = False,
        style_overlap_topk: bool = False,
        decomposition_json: Optional[Union[str, Path]] = None,
        use_part_bank: bool = False,
        part_bank_manifest: Optional[Union[str, Path]] = None,
        part_retrieval_mode: str = "none",
        part_retrieval_ep_ckpt: Optional[Union[str, Path]] = None,
        part_set_size: int = 10,
        part_set_min_size: int = 2,
        part_set_sampling: str = "random",
        part_target_char_priority: bool = False,
        part_image_size: int = 64,
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
        self.component_guided_style = bool(component_guided_style)
        self.style_overlap_topk = bool(style_overlap_topk)
        self.use_part_bank = bool(use_part_bank)
        self.part_retrieval_mode = str(part_retrieval_mode).strip().lower()
        if self.part_retrieval_mode not in {"none", "font_softmax_top1"}:
            raise ValueError("part_retrieval_mode must be one of: none, font_softmax_top1")
        self.part_retrieval_ep_ckpt = part_retrieval_ep_ckpt
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
        self.part_bank_by_font: Dict[str, List[Dict[str, Any]]] = {}
        self.part_retrieval_model = None
        self.part_retrieval_class_fonts: List[str] = []
        self.part_retrieval_candidate_labels = None
        self.char_components: Dict[str, Set[str]] = {}
        if self.component_guided_style:
            if decomposition_json is None:
                decomposition_json = self.root / "CharacterData" / "decomposition.json"
            self.char_components = self._load_decomposition(decomposition_json)
            if not self.char_components:
                print("[FontImageDataset] decomposition unavailable, disable component-guided style sampling.")
                self.component_guided_style = False

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
            self.part_bank_by_font = self._load_part_bank_manifest(part_bank_manifest)
            self._filter_samples_by_part_bank()
            if self.part_retrieval_mode == "font_softmax_top1":
                self._init_part_retrieval_classifier(self.part_retrieval_ep_ckpt)

        if not self.samples:
            raise RuntimeError("No valid samples found – please check LMDB integrity and paths.")

    def _load_decomposition(self, decomposition_json: Union[str, Path]) -> Dict[str, Set[str]]:
        path = Path(decomposition_json)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not path.exists():
            return {}
        try:
            obj = json.load(path.open("r", encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}

        out: Dict[str, Set[str]] = {}
        for ch, raw in obj.items():
            if not isinstance(ch, str) or len(ch) != 1:
                continue
            tokens: List[str] = []
            if isinstance(raw, str) and raw:
                tokens.append(raw)
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item:
                        tokens.append(item)
            if not tokens:
                continue

            parts: Set[str] = set()
            for token in tokens:
                parts.add(token)
                if len(token) > 1:
                    parts.update(c for c in token if c.strip())
            out[ch] = parts
        return out

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
                p = Path(rel)
                if not p.is_absolute():
                    p = (self.root / p).resolve()
                if not p.exists():
                    continue
                ch = row.get("char")
                part_rows.append(
                    {
                        "path": p,
                        "char": ch if isinstance(ch, str) and len(ch) == 1 else "",
                        "x": int(row.get("x", 0)) if isinstance(row.get("x", 0), (int, float)) else 0,
                        "y": int(row.get("y", 0)) if isinstance(row.get("y", 0), (int, float)) else 0,
                        "response": float(row.get("response", 0.0)) if isinstance(row.get("response", 0.0), (int, float)) else 0.0,
                        "index": self._parse_part_index(str(p.name)),
                    }
                )

            if not part_rows:
                continue

            # Canonical deterministic order: file index first, then confidence.
            part_rows.sort(key=lambda x: (int(x["index"]), -float(x["response"]), str(x["path"])))
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
            raise ValueError("part_retrieval_ep_ckpt is required when part_retrieval_mode=font_softmax_top1")
        if "torch" not in globals():
            raise RuntimeError("font_softmax_top1 retrieval requires torch.")

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
        model.eval()

        candidate_labels = [
            i for i, f in enumerate(class_fonts)
            if f in self.part_bank_by_font
        ]
        if not candidate_labels:
            raise RuntimeError("No overlap between classifier fonts and PartBank fonts.")

        self.part_retrieval_model = model
        self.part_retrieval_class_fonts = class_fonts
        self.part_retrieval_candidate_labels = torch.tensor(candidate_labels, dtype=torch.long)

    def _predict_part_font_from_styles(self, style_imgs: List[Any], fallback_font: str) -> str:
        if self.part_retrieval_model is None or self.part_retrieval_candidate_labels is None:
            return fallback_font
        if not style_imgs:
            return fallback_font
        if not all(isinstance(x, torch.Tensor) for x in style_imgs):
            return fallback_font

        with torch.no_grad():
            x = torch.stack(style_imgs, dim=0)
            logits = self.part_retrieval_model(x)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            cand_probs = probs.index_select(0, self.part_retrieval_candidate_labels)
            idx_local = int(torch.argmax(cand_probs).item())
            idx_global = int(self.part_retrieval_candidate_labels[idx_local].item())
        pred_font = self.part_retrieval_class_fonts[idx_global]
        if pred_font not in self.part_bank_by_font:
            return fallback_font if fallback_font in self.part_bank_by_font else pred_font
        return pred_font

    def _load_part_tensor(self, path: Path):
        img = Image.open(path).convert("RGB")
        if self.part_image_size > 0:
            img = img.resize((self.part_image_size, self.part_image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def _select_part_rows(self, font_name: str, target_char: str, retrieved_font: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.part_retrieval_mode == "font_softmax_top1" and retrieved_font:
            font_name = retrieved_font
        rows = self.part_bank_by_font.get(font_name, [])
        if not rows:
            return []

        size_hi = min(self.part_set_size, len(rows))
        if size_hi <= 0:
            return []
        size_lo = min(self.part_set_min_size, size_hi)

        if self.part_set_sampling == "random":
            size = self.rng.randint(size_lo, size_hi)
        else:
            size = size_hi

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
                    while len(picked) < size:
                        picked.append(self.rng.choice(rows))
            return picked[:size]

        # deterministic
        if same_char:
            seen_ids = {id(x) for x in same_char}
            merged = same_char + [r for r in rows if id(r) not in seen_ids]
            return merged[:size]
        return rows[:size]

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

    def _bytes_to_img(self, b: bytes):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return self.transform(img) if self.transform else img

    def _component_overlap(self, ch1: str, ch2: str) -> int:
        if not self.component_guided_style:
            return 0
        p1 = self.char_components.get(ch1)
        p2 = self.char_components.get(ch2)
        if not p1 or not p2:
            return 0
        return len(p1.intersection(p2))

    def _weighted_sample_no_replacement(self, items: List[str], weights: List[float], k: int) -> List[str]:
        chosen: List[str] = []
        items = list(items)
        weights = [max(0.0, float(w)) for w in weights]
        k = min(k, len(items))
        for _ in range(k):
            total = sum(weights)
            if total <= 0.0:
                idx = self.rng.randrange(len(items))
            else:
                threshold = self.rng.random() * total
                acc = 0.0
                idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if acc >= threshold:
                        idx = i
                        break
            chosen.append(items.pop(idx))
            weights.pop(idx)
        return chosen

    def _component_guided_pick(self, ch: str, pool: List[str], needed: int) -> List[str]:
        scored = [(c, self._component_overlap(ch, c)) for c in pool]
        positive = [(c, s) for c, s in scored if s > 0]
        chosen: List[str] = []
        if positive:
            pos_items = [x[0] for x in positive]
            pos_weights = [float(x[1]) for x in positive]
            chosen.extend(self._weighted_sample_no_replacement(pos_items, pos_weights, needed))

        if len(chosen) >= needed:
            return chosen[:needed]

        rest_needed = needed - len(chosen)
        remaining_pool = [c for c in pool if c not in set(chosen)]
        if len(remaining_pool) >= rest_needed:
            chosen.extend(self.rng.sample(remaining_pool, rest_needed))
        else:
            chosen.extend(remaining_pool)
            while len(chosen) < needed:
                chosen.append(self.rng.choice(pool))
        return chosen

    def _sample_style_chars(self, ch: str, font_name: str) -> List[str]:
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
            if self.component_guided_style and self.style_overlap_topk:
                scored = sorted(
                    pool,
                    key=lambda c: (self._component_overlap(ch, c), c),
                    reverse=True,
                )
                picked.extend(scored[:needed])
            elif self.style_sampling == "deterministic":
                if self.component_guided_style:
                    scored = sorted(pool, key=lambda c: self._component_overlap(ch, c), reverse=True)
                    picked.extend(scored[:needed])
                else:
                    picked.extend(pool[:needed])
            else:
                if self.component_guided_style:
                    picked.extend(self._component_guided_pick(ch, pool, needed))
                else:
                    if len(pool) >= needed:
                        picked.extend(self.rng.sample(pool, needed))
                    else:
                        picked.extend(pool)
                        while len(picked) < self.num_style_refs:
                            picked.append(self.rng.choice(pool))

        return picked[: self.num_style_refs]

    def sample_style_plan(self, index: int) -> Dict[str, Any]:
        """Sample style chars for one dataset item without reading LMDB image bytes."""
        font_name, real_idx = self.samples[index]
        ch = self.char_list[real_idx]
        style_chars = self._sample_style_chars(ch, font_name)
        overlaps = [self._component_overlap(ch, sc) for sc in style_chars]
        return {
            "index": index,
            "font": font_name,
            "char": ch,
            "style_chars": style_chars,
            "overlaps": overlaps,
        }

    @staticmethod
    def _quantile(sorted_vals: List[int], q: float) -> float:
        if not sorted_vals:
            return 0.0
        q = max(0.0, min(1.0, float(q)))
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        pos = q * (len(sorted_vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        w = pos - lo
        return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)

    def component_overlap_stats(
        self,
        num_samples: int = 2000,
        random_seed: int = 42,
        top_char_k: int = 20,
        top_pair_k: int = 30,
    ) -> Dict[str, Any]:
        """Collect overlap statistics of sampled style chars.

        Notes:
        - This method does not decode images from LMDB; it only evaluates sampling logic.
        - RNG state of dataset-level sampler will be restored after analysis.
        """
        if not self.samples:
            return {
                "enabled": bool(self.component_guided_style),
                "message": "empty dataset",
            }

        n = max(1, int(num_samples))
        local_rng = random.Random(random_seed)
        if len(self.samples) >= n:
            picked_indices = local_rng.sample(range(len(self.samples)), n)
        else:
            picked_indices = [local_rng.randrange(len(self.samples)) for _ in range(n)]

        overlap_hist: Counter = Counter()
        overlap_values: List[int] = []
        per_font_sum: Dict[str, int] = {}
        per_font_pos: Dict[str, int] = {}
        per_font_total: Dict[str, int] = {}
        per_char_sum: Dict[str, int] = {}
        per_char_pos: Dict[str, int] = {}
        per_char_total: Dict[str, int] = {}
        pair_counter: Counter = Counter()

        old_state = self.rng.getstate()
        self.rng.seed(random_seed + 1919)
        try:
            for idx in picked_indices:
                plan = self.sample_style_plan(idx)
                font = plan["font"]
                ch = plan["char"]
                style_chars = plan["style_chars"]
                overlaps = plan["overlaps"]

                for sc, ov in zip(style_chars, overlaps):
                    ov_int = int(ov)
                    overlap_hist[ov_int] += 1
                    overlap_values.append(ov_int)
                    per_font_sum[font] = per_font_sum.get(font, 0) + ov_int
                    per_font_pos[font] = per_font_pos.get(font, 0) + int(ov_int > 0)
                    per_font_total[font] = per_font_total.get(font, 0) + 1
                    per_char_sum[ch] = per_char_sum.get(ch, 0) + ov_int
                    per_char_pos[ch] = per_char_pos.get(ch, 0) + int(ov_int > 0)
                    per_char_total[ch] = per_char_total.get(ch, 0) + 1
                    if ov_int > 0:
                        pair_counter[(ch, sc)] += 1
        finally:
            self.rng.setstate(old_state)

        if not overlap_values:
            return {
                "enabled": bool(self.component_guided_style),
                "message": "no overlap pairs collected",
                "num_samples": n,
            }

        vals_sorted = sorted(overlap_values)
        total_pairs = len(overlap_values)
        positive_pairs = sum(1 for v in overlap_values if v > 0)
        zero_pairs = total_pairs - positive_pairs

        font_rows = []
        for font, total in per_font_total.items():
            if total <= 0:
                continue
            font_rows.append(
                {
                    "font": font,
                    "pairs": int(total),
                    "positive_pairs": int(per_font_pos.get(font, 0)),
                    "positive_rate": float(per_font_pos.get(font, 0) / total),
                    "mean_overlap": float(per_font_sum.get(font, 0) / total),
                }
            )
        font_rows.sort(key=lambda x: x["mean_overlap"], reverse=True)

        char_rows = []
        for ch, total in per_char_total.items():
            if total <= 0:
                continue
            char_rows.append(
                {
                    "char": ch,
                    "pairs": int(total),
                    "positive_pairs": int(per_char_pos.get(ch, 0)),
                    "positive_rate": float(per_char_pos.get(ch, 0) / total),
                    "mean_overlap": float(per_char_sum.get(ch, 0) / total),
                }
            )
        char_rows.sort(key=lambda x: x["mean_overlap"], reverse=True)

        top_pairs = [
            {"target_char": a, "style_char": b, "count": int(c)}
            for (a, b), c in pair_counter.most_common(max(1, int(top_pair_k)))
        ]

        report = {
            "enabled": bool(self.component_guided_style),
            "num_dataset_samples": len(self.samples),
            "num_requested_samples": int(num_samples),
            "num_eval_samples": int(n),
            "style_k": int(self.num_style_refs),
            "total_pairs": int(total_pairs),
            "positive_pairs": int(positive_pairs),
            "zero_pairs": int(zero_pairs),
            "positive_rate": float(positive_pairs / total_pairs),
            "mean_overlap": float(sum(overlap_values) / total_pairs),
            "median_overlap": self._quantile(vals_sorted, 0.5),
            "p90_overlap": self._quantile(vals_sorted, 0.9),
            "p95_overlap": self._quantile(vals_sorted, 0.95),
            "max_overlap": int(vals_sorted[-1]),
            "overlap_hist": {str(k): int(v) for k, v in sorted(overlap_hist.items(), key=lambda x: x[0])},
            "top_fonts_by_mean_overlap": font_rows[: min(15, len(font_rows))],
            "top_chars_by_mean_overlap": char_rows[: min(max(1, int(top_char_k)), len(char_rows))],
            "top_positive_pairs": top_pairs,
        }
        return report

    def __getitem__(self, index: int):
        font_name, real_idx = self.samples[index]
        ch = self.char_list[real_idx]

        content_key = f"ContentFont@{ch}".encode("utf-8")
        input_key = f"{font_name}@{ch}".encode("utf-8")
        style_chars = self._sample_style_chars(ch, font_name)

        with self.content_env.begin() as c_txn, self.train_env.begin() as t_txn:
            content_bytes = c_txn.get(content_key)
            input_bytes = t_txn.get(input_key)
            if content_bytes is None or input_bytes is None:
                raise KeyError("Missing required glyph images")

            style_imgs = []
            for sc in style_chars:
                style_key = f"{font_name}@{sc}".encode("utf-8")
                sb = t_txn.get(style_key)
                if sb is None:
                    raise KeyError(f"Missing style glyph: {sc}")
                style_imgs.append(self._bytes_to_img(sb))

        sample: Dict[str, Any] = {
            "font": font_name,
            "char": ch,
            "char_index": int(real_idx),
            "content": self._bytes_to_img(content_bytes),
            "input": self._bytes_to_img(input_bytes),
            "styles": style_imgs,
        }
        if self.use_part_bank:
            retrieved_font = None
            if self.part_retrieval_mode == "font_softmax_top1":
                retrieved_font = self._predict_part_font_from_styles(style_imgs, fallback_font=font_name)
            part_rows = self._select_part_rows(font_name, ch, retrieved_font=retrieved_font)
            part_rows_b = self._select_part_rows(font_name, ch, retrieved_font=retrieved_font)
            if not part_rows or not part_rows_b:
                raise KeyError(f"Missing PartBank parts for font: {font_name}")
            part_imgs = [self._load_part_tensor(r["path"]) for r in part_rows]
            part_imgs_b = [self._load_part_tensor(r["path"]) for r in part_rows_b]
            parts = torch.stack(part_imgs, dim=0)
            parts_b = torch.stack(part_imgs_b, dim=0)
            part_mask = torch.ones((parts.size(0),), dtype=torch.float32)
            part_mask_b = torch.ones((parts_b.size(0),), dtype=torch.float32)
            sample["parts"] = parts
            sample["part_mask"] = part_mask
            sample["parts_b"] = parts_b
            sample["part_mask_b"] = part_mask_b
            if retrieved_font:
                sample["retrieved_part_font"] = retrieved_font
        return sample

    def close(self):
        if hasattr(self, "content_env"):
            self.content_env.close()
        if hasattr(self, "train_env"):
            self.train_env.close()

    def __del__(self):
        self.close()


# Quick sanity test
if __name__ == "__main__":
    ds = FontImageDataset(num_style_refs=3)
    print(f"Valid samples: {len(ds)}")
    sample = ds[0]
    print(sample["char"], len(sample["styles"]))

    # Visualize loaded images (uncomment to view)
