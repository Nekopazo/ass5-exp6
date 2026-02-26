#!/usr/bin/env python3
"""Build offline part-vector index (FAISS/Annoy) from PartBank with E_p."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import sys

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from models.style_encoders import PartPatchEncoder


def resolve_path(root: Path, p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def load_manifest(path: Path, root: Path) -> List[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fonts = obj.get("fonts", {})
    rows: List[Dict] = []
    for font_name, info in fonts.items():
        for row in info.get("parts", []):
            rel = row.get("path")
            if not rel:
                continue
            fp = resolve_path(root, Path(rel))
            if not fp.exists():
                continue
            rows.append(
                {
                    "font": font_name,
                    "path": str(fp),
                    "char": row.get("char", ""),
                    "x": int(row.get("x", 0)),
                    "y": int(row.get("y", 0)),
                    "response": float(row.get("response", 0.0)),
                }
            )
    return rows


def load_part_tensor(path: Path, image_size: int):
    tfm = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    img = Image.open(path).convert("RGB")
    return tfm(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("DataPreparation/PartBank/manifest.json"))
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/part_vector_index"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--backend", type=str, default="faiss,annoy", help="Comma-separated backends: faiss,annoy")
    parser.add_argument("--annoy-trees", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = resolve_path(root, args.manifest)
    ckpt_path = resolve_path(root, args.encoder_ckpt)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = load_manifest(manifest_path, root)
    if not rows:
        raise RuntimeError("No part rows found from manifest.")

    model = PartPatchEncoder(in_channels=3, embedding_dim=args.embedding_dim).to(device)
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    state = obj if isinstance(obj, dict) else {}
    if isinstance(state, dict) and "e_p" in state:
        state = state["e_p"]
    model.load_state_dict(state, strict=False)
    model.eval()

    vectors = np.zeros((len(rows), args.embedding_dim), dtype=np.float32)
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for st in range(0, len(rows), bs):
            ed = min(len(rows), st + bs)
            batch = torch.stack(
                [load_part_tensor(Path(rows[i]["path"]), args.image_size) for i in range(st, ed)],
                dim=0,
            ).to(device)
            z = model(batch, normalize=True).detach().cpu().numpy().astype(np.float32, copy=False)
            vectors[st:ed] = z

    np.save(out_dir / "part_vectors.npy", vectors)
    with (out_dir / "part_meta.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row = dict(row)
            row["index"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    backends = {x.strip().lower() for x in str(args.backend).split(",") if x.strip()}
    status = {"faiss": "skip", "annoy": "skip"}

    if "faiss" in backends:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(args.embedding_dim)
            index.add(vectors)
            faiss.write_index(index, str(out_dir / "part_index.faiss"))
            status["faiss"] = "ok"
        except Exception as e:
            status["faiss"] = f"fail:{type(e).__name__}"

    if "annoy" in backends:
        try:
            from annoy import AnnoyIndex  # type: ignore

            ann = AnnoyIndex(args.embedding_dim, metric="angular")
            for i in range(vectors.shape[0]):
                ann.add_item(i, vectors[i].tolist())
            ann.build(max(1, int(args.annoy_trees)))
            ann.save(str(out_dir / "part_index.ann"))
            status["annoy"] = "ok"
        except Exception as e:
            status["annoy"] = f"fail:{type(e).__name__}"

    summary = {
        "num_parts": int(len(rows)),
        "embedding_dim": int(args.embedding_dim),
        "vector_path": str(out_dir / "part_vectors.npy"),
        "meta_path": str(out_dir / "part_meta.jsonl"),
        "status": status,
    }
    (out_dir / "index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

