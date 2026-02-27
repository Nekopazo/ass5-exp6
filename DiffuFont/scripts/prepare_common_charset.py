#!/usr/bin/env python3
"""Prepare 2000 common Chinese chars and 300 reference chars.

Source:
- jieba small dict (frequency):
  https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

JIEBA_SMALL_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small"


def is_cjk_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF  # Ext A
        or 0x4E00 <= code <= 0x9FFF  # Unified Ideographs
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
    )


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def load_char_freq_from_jieba(text: str) -> dict[str, int]:
    freq: dict[str, int] = defaultdict(int)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        token = parts[0]
        if len(token) != 1 or not is_cjk_char(token):
            continue
        try:
            f = int(parts[1])
        except ValueError:
            f = 1
        freq[token] += f
    return dict(freq)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("CharacterData"))
    parser.add_argument("--char-count", type=int, default=2000)
    parser.add_argument("--ref-count", type=int, default=300)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jieba_text = fetch_text(JIEBA_SMALL_URL)
    freq = load_char_freq_from_jieba(jieba_text)
    if len(freq) < args.char_count:
        raise RuntimeError(f"Only found {len(freq)} single-char entries, need {args.char_count}")

    sorted_chars = [c for c, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
    chars = sorted_chars[: args.char_count]
    refs = chars[: args.ref_count]

    write_json(out_dir / "CharList.json", chars)
    write_json(out_dir / "ReferenceCharList.json", refs)

    (out_dir / "CharList.txt").write_text("".join(chars) + "\n", encoding="utf-8")
    (out_dir / "ReferenceCharList.txt").write_text("".join(refs) + "\n", encoding="utf-8")

    # Remove stale mapping file to keep the data contract mapping-free.
    mapping_path = out_dir / "mapping.json"
    removed_mapping = False
    if mapping_path.exists():
        mapping_path.unlink()
        removed_mapping = True

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "char_count": len(chars),
        "reference_count": len(refs),
        "sources": {
            "char_frequency": JIEBA_SMALL_URL,
        },
    }
    write_json(out_dir / "dataset_meta.json", metadata)

    print(f"Generated {len(chars)} chars and {len(refs)} references")
    if removed_mapping:
        print("Removed stale CharacterData/mapping.json")


if __name__ == "__main__":
    main()
