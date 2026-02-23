#!/usr/bin/env python3
"""Convert rendered glyph images into LMDB.

Key format:
    key = <image_stem>.encode("utf-8")

For generated glyph files named `<FontStem>@<char>.png`, this gives the expected
LMDB key format `<FontStem>@<char>`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb


IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def collect_images(img_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in img_roots:
        if not root.exists():
            print(f"[skip] image root does not exist: {root}")
            continue
        paths.extend([p for p in root.rglob("*") if p.suffix in IMG_EXTS])
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--img-roots",
        type=Path,
        nargs="+",
        default=[Path("DataPreparation/Generated/ContentFont")],
        help="One or more image root directories",
    )
    parser.add_argument("--lmdb-path", type=Path, default=Path("DataPreparation/LMDB/ContentFont.lmdb"))
    parser.add_argument("--map-size", type=int, default=2**30)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing keys")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    img_roots = [(project_root / p).resolve() for p in args.img_roots]
    lmdb_path = (project_root / args.lmdb_path).resolve()

    img_paths = collect_images(img_roots)
    if not img_paths:
        print("[warning] no images found, nothing to write")
        return

    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        map_size=args.map_size,
        subdir=True,
        meminit=False,
        map_async=True,
    )

    written = 0
    skipped = 0
    txn = env.begin(write=True)
    for idx, img_path in enumerate(img_paths, start=1):
        key = img_path.stem.encode("utf-8")
        if (not args.overwrite) and txn.get(key) is not None:
            skipped += 1
            continue

        with img_path.open("rb") as f:
            img_bytes = f.read()
        txn.put(key, img_bytes)
        written += 1

        if idx % args.batch_size == 0:
            txn.commit()
            print(f"processed={idx}/{len(img_paths)} written={written} skipped={skipped}")
            txn = env.begin(write=True)

    txn.commit()
    env.sync()
    env.close()

    print(f"done: total={len(img_paths)} written={written} skipped={skipped} lmdb={lmdb_path}")


if __name__ == "__main__":
    main()
