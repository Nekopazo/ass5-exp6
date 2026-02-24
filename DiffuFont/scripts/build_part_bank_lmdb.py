#!/usr/bin/env python3
"""Pack PartBank patch images into LMDB for training-time reads.

LMDB key default:
- row["lmdb_key"] if present
- otherwise row["path"] normalized as POSIX
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import lmdb


def resolve(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def row_key(row: Dict[str, Any]) -> str:
    k = row.get("lmdb_key")
    if isinstance(k, str) and k:
        return k
    p = row.get("path")
    if not isinstance(p, str) or not p:
        raise ValueError("manifest row missing 'path'")
    return Path(p).as_posix()


def row_path(root: Path, row: Dict[str, Any]) -> Path:
    p = row.get("path")
    if not isinstance(p, str) or not p:
        raise ValueError("manifest row missing 'path'")
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (root / pp).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("DataPreparation/PartBank/manifest.json"))
    parser.add_argument("--out-lmdb", type=Path, default=Path("DataPreparation/LMDB/PartBank.lmdb"))
    parser.add_argument("--map-size-gb", type=int, default=8)
    parser.add_argument("--commit-interval", type=int, default=1000)
    parser.add_argument(
        "--write-lmdb-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write resolved lmdb_key back to manifest rows.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = resolve(root, args.manifest)
    out_lmdb = resolve(root, args.out_lmdb)
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)

    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    fonts = obj.get("fonts", {}) if isinstance(obj, dict) else {}
    if not isinstance(fonts, dict):
        raise RuntimeError(f"invalid PartBank manifest: {manifest_path}")

    map_size = int(args.map_size_gb) * 1024 * 1024 * 1024
    if map_size <= 0:
        raise ValueError("--map-size-gb must be > 0")

    env = lmdb.open(str(out_lmdb), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)
    txn = env.begin(write=True)

    total_rows = 0
    written = 0
    missing = 0
    bytes_total = 0

    try:
        for font_name, info in fonts.items():
            if not isinstance(info, dict):
                continue
            rows = info.get("parts", [])
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                total_rows += 1
                key = row_key(row)
                path = row_path(root, row)
                if not path.exists():
                    missing += 1
                    continue
                b = path.read_bytes()
                txn.put(key.encode("utf-8"), b, overwrite=True)
                written += 1
                bytes_total += len(b)
                if args.write_lmdb_key:
                    row["lmdb_key"] = key
                if written % int(args.commit_interval) == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    print(f"[part-lmdb] written={written} rows", flush=True)
    finally:
        txn.commit()
        env.sync()
        env.close()

    if args.write_lmdb_key:
        manifest_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "out_lmdb": str(out_lmdb),
        "rows_total": int(total_rows),
        "rows_written": int(written),
        "rows_missing_files": int(missing),
        "bytes_written": int(bytes_total),
        "write_lmdb_key": bool(args.write_lmdb_key),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
