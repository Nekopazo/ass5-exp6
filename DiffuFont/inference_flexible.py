#!/usr/bin/env python3
"""Flexible inference script for DiffuFont.

Features:
- Supports conditioning modes: style_only / part_only (style-image branch), baseline.
- Supports manual loading of weights:
  - full checkpoint (--checkpoint)
  - split component weights (--main-weight / --vector-weight)
- Supports input from folders or LMDB:
  - content: folder or LMDB
  - style: folder or LMDB
- Saves outputs under output directory:
  - content/
  - style/ or part/
  - gen/
"""

from __future__ import annotations

import argparse
import io
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from models.model import DiffusionTrainer, FlowMatchingTrainer
from models.source_part_ref_unet import SourcePartRefUNet


def normalize_conditioning_mode(raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    if mode == "parts_vector_only":
        mode = "part_only"
    valid = {"baseline", "part_only", "style_only"}
    if mode not in valid:
        raise ValueError(f"conditioning mode must be one of {sorted(valid)}, got '{raw_mode}'")
    return mode


def mode_uses_style(mode: str) -> bool:
    return normalize_conditioning_mode(mode) in {"part_only", "style_only"}


def mode_uses_parts(mode: str) -> bool:
    _ = normalize_conditioning_mode(mode)
    return False


def parse_chars(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    return out


def parse_int_csv(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    vals = [int(x.strip()) for x in txt.split(",") if x.strip()]
    return tuple(vals) if vals else None


def set_global_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def safe_stem(name: str) -> str:
    out = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", str(name))
    return out.strip("_") or "sample"


def tensor_to_pil_gray(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().float()
    if x.dim() == 3:
        x = x[0]
    x = ((x.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().byte().numpy()
    return Image.fromarray(x, mode="L")


def build_part_preview(part_tensor: torch.Tensor, out_size: int, max_show: int = 8) -> torch.Tensor:
    """Build a single preview image from a part set tensor (P,1,H,W)."""
    p = int(part_tensor.size(0))
    if p <= 0:
        return torch.full((1, out_size, out_size), -1.0, dtype=torch.float32)
    show = min(max_show, p)
    cols = min(4, show)
    rows = (show + cols - 1) // cols
    tile_h = max(8, out_size // rows)
    tile_w = max(8, out_size // cols)
    canvas = torch.full((1, out_size, out_size), -1.0, dtype=torch.float32)

    for i in range(show):
        r = i // cols
        c = i % cols
        y0 = r * tile_h
        x0 = c * tile_w
        y1 = min(out_size, y0 + tile_h)
        x1 = min(out_size, x0 + tile_w)
        patch = part_tensor[i].unsqueeze(0)
        patch = torch.nn.functional.interpolate(
            patch, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False
        ).squeeze(0)
        canvas[:, y0:y1, x0:x1] = patch
    return canvas


def save_combined_grid(
    rows: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    mode: str,
    out_path: Path,
) -> None:
    if not rows:
        raise RuntimeError("No rows to render for combined grid.")

    cell = int(rows[0][1].shape[-1])
    n = len(rows)
    label_w = 180
    header_h = 34
    gap = 4
    cols = 4
    w = label_w + cols * cell + (cols - 1) * gap
    h = header_h + n * cell + max(0, n - 1) * gap
    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    headers = ["content", "reference", "original", "inference"]
    for c, text in enumerate(headers):
        x = label_w + c * (cell + gap) + 8
        draw.text((x, 8), text, fill=(0, 0, 0), font=font)

    for r, (sid, content, ref, original, gen) in enumerate(rows):
        y = header_h + r * (cell + gap)
        draw.text((8, y + cell // 2 - 8), sid[:20], fill=(0, 0, 0), font=font)
        imgs = [content, ref, original, gen]
        for c, t in enumerate(imgs):
            x = label_w + c * (cell + gap)
            p = tensor_to_pil_gray(t).convert("RGB")
            canvas.paste(p, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


def pil_gray_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tfm = T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    return tfm(img.convert("L"))


def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


class LMDBReader:
    def __init__(self, lmdb_path: Path) -> None:
        try:
            import lmdb  # type: ignore
        except Exception as e:
            raise RuntimeError("python package 'lmdb' is required for LMDB input.") from e
        self._lmdb = lmdb
        self.path = str(lmdb_path)
        self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        self.txn = self.env.begin(buffers=True)

    def get(self, key: str) -> Optional[bytes]:
        v = self.txn.get(key.encode("utf-8"))
        if v is None:
            return None
        return bytes(v)

    def keys_with_prefix(self, prefix: str) -> List[str]:
        out: List[str] = []
        p = prefix.encode("utf-8")
        cursor = self.txn.cursor()
        ok = cursor.set_range(p)
        if not ok:
            return out
        for raw_key, _ in cursor:
            # buffers=True may return memoryview keys; normalize to bytes first.
            key_bytes = bytes(raw_key) if isinstance(raw_key, memoryview) else raw_key
            if not key_bytes.startswith(p):
                break
            out.append(key_bytes.decode("utf-8", errors="ignore"))
        return out

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "part_encoder"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj  # plain state dict
    raise RuntimeError("Unsupported checkpoint format for state_dict extraction.")


def load_model_weights(
    model: SourcePartRefUNet,
    checkpoint: Optional[Path],
    main_weight: Optional[Path],
    vector_weight: Optional[Path],
    strict_load: bool,
) -> None:
    loaded_any = False
    if checkpoint is not None:
        obj = torch.load(str(checkpoint), map_location="cpu")
        sd = extract_state_dict(obj)
        missing, unexpected = model.load_state_dict(sd, strict=bool(strict_load))
        print(
            f"[load] checkpoint={checkpoint} strict={strict_load} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        loaded_any = True
    for comp_name, path in (("main", main_weight), ("vector", vector_weight)):
        if path is None:
            continue
        obj = torch.load(str(path), map_location="cpu")
        sd = extract_state_dict(obj)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] {comp_name}={path} strict=False missing={len(missing)} unexpected={len(unexpected)}")
        loaded_any = True
    if not loaded_any:
        raise ValueError("No weights loaded. Please set --checkpoint and/or --main-weight/--vector-weight.")


def build_trainer(
    model: SourcePartRefUNet,
    device: torch.device,
    trainer_type: str,
    diffusion_steps: int,
    conditioning_mode: str,
) -> DiffusionTrainer | FlowMatchingTrainer:
    trainer_cls = DiffusionTrainer if trainer_type == "diffusion" else FlowMatchingTrainer
    kwargs = {
        "lr": 1e-4,
        "T": int(diffusion_steps),
        "total_steps": 1,
        "conditioning_mode": conditioning_mode,
    }
    if trainer_type == "flow_matching":
        kwargs["lambda_fm"] = 1.0
    trainer = trainer_cls(model=model, device=device, **kwargs)
    trainer.model.eval()
    return trainer


def load_content_samples(
    source: str,
    content_folder: Optional[Path],
    content_lmdb: Optional[Path],
    content_chars: Sequence[str],
    image_size: int,
    max_samples: int,
) -> List[Tuple[str, str, torch.Tensor]]:
    samples: List[Tuple[str, str, torch.Tensor]] = []
    if source == "folder":
        if content_folder is None:
            raise ValueError("--content-folder is required when --content-source=folder")
        files = list_image_files(content_folder)
        if not files:
            raise RuntimeError(f"No images found in content folder: {content_folder}")
        if content_chars:
            char_to_path: Dict[str, Path] = {}
            for p in files:
                stem = p.stem
                char_to_path.setdefault(stem, p)
            for ch in content_chars:
                p = char_to_path.get(ch)
                if p is None:
                    raise KeyError(f"Content folder missing file for char='{ch}' (expected filename stem == char).")
                img = Image.open(p).convert("L")
                samples.append((ch, ch, pil_gray_to_tensor(img, image_size)))
        else:
            for p in files[:max_samples]:
                img = Image.open(p).convert("L")
                sid = p.stem
                samples.append((sid, sid, pil_gray_to_tensor(img, image_size)))
    else:
        if content_lmdb is None:
            raise ValueError("--content-lmdb is required when --content-source=lmdb")
        chars = list(content_chars)
        if not chars:
            raise ValueError("--content-chars is required when --content-source=lmdb")
        reader = LMDBReader(content_lmdb)
        try:
            for ch in chars[:max_samples]:
                key = f"ContentFont@{ch}"
                b = reader.get(key)
                if b is None:
                    raise KeyError(f"Missing LMDB key: {key}")
                img = Image.open(io.BytesIO(b)).convert("L")
                samples.append((ch, ch, pil_gray_to_tensor(img, image_size)))
        finally:
            reader.close()
    return samples


def resolve_style_tensor_for_index(
    idx: int,
    source: str,
    style_folder: Optional[Path],
    style_lmdb: Optional[Path],
    style_font: Optional[str],
    style_chars: Sequence[str],
    image_size: int,
    cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, str]:
    if source == "folder":
        if style_folder is None:
            raise ValueError("--style-folder is required when --style-source=folder")
        files = list_image_files(style_folder)
        if not files:
            raise RuntimeError(f"No images found in style folder: {style_folder}")
        if style_chars:
            ch = style_chars[idx % len(style_chars)]
            if ch not in cache:
                target = None
                for p in files:
                    if p.stem == ch:
                        target = p
                        break
                if target is None:
                    raise KeyError(f"Style folder missing file for char='{ch}' (filename stem should match char).")
                cache[ch] = pil_gray_to_tensor(Image.open(target).convert("L"), image_size)
            return cache[ch], ch
        p = files[idx % len(files)]
        k = str(p)
        if k not in cache:
            cache[k] = pil_gray_to_tensor(Image.open(p).convert("L"), image_size)
        return cache[k], p.stem

    if source == "lmdb":
        if style_lmdb is None:
            raise ValueError("--style-lmdb is required when --style-source=lmdb")
        if not style_font:
            raise ValueError("--style-font is required when --style-source=lmdb")
        if not style_chars:
            raise ValueError("--style-chars is required when --style-source=lmdb")
        ch = style_chars[idx % len(style_chars)]
        k = f"{style_font}@{ch}"
        if k not in cache:
            reader = LMDBReader(style_lmdb)
            try:
                b = reader.get(k)
            finally:
                reader.close()
            if b is None:
                raise KeyError(f"Missing LMDB key: {k}")
            cache[k] = pil_gray_to_tensor(Image.open(io.BytesIO(b)).convert("L"), image_size)
        return cache[k], ch

    raise ValueError(f"Unsupported style source: {source}")


def _extract_char_from_key(key: str) -> str:
    m = re.search(r"_U([0-9A-Fa-f]{4,6})", key)
    if m is None:
        return ""
    try:
        return chr(int(m.group(1), 16))
    except Exception:
        return ""


def load_part_set_from_folder(
    part_folder: Path,
    ref_char: Optional[str],
    part_image_size: int,
    max_parts: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    candidate_dir = part_folder
    if ref_char:
        ucode = f"U{ord(ref_char):04X}"
        for cand in (part_folder / ref_char, part_folder / ucode):
            if cand.exists() and cand.is_dir():
                candidate_dir = cand
                break
    files = list_image_files(candidate_dir)
    if not files:
        raise RuntimeError(f"No part images found in folder: {candidate_dir}")
    files = files[:max_parts]
    parts = [pil_gray_to_tensor(Image.open(p).convert("L"), part_image_size) for p in files]
    t = torch.stack(parts, dim=0)
    m = torch.ones((t.size(0),), dtype=torch.float32)
    return t, m, candidate_dir.name


def load_part_set_from_lmdb(
    part_lmdb: Path,
    part_font: str,
    ref_char: str,
    part_image_size: int,
    max_parts: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    reader = LMDBReader(part_lmdb)
    try:
        ucode = f"U{ord(ref_char):04X}"
        pref1 = f"DataPreparation/PartBank/{part_font}/{ucode}/"
        keys = reader.keys_with_prefix(pref1)
        if not keys:
            pref2 = f"DataPreparation/PartBank/{part_font}/"
            keys = [k for k in reader.keys_with_prefix(pref2) if _extract_char_from_key(k) == ref_char]
        if not keys:
            raise KeyError(f"No part keys found for font='{part_font}', char='{ref_char}' in {part_lmdb}")
        keys = sorted(keys)[:max_parts]
        parts: List[torch.Tensor] = []
        for k in keys:
            b = reader.get(k)
            if b is None:
                continue
            parts.append(pil_gray_to_tensor(Image.open(io.BytesIO(b)).convert("L"), part_image_size))
        if not parts:
            raise KeyError(f"Part keys exist but decode failed for font='{part_font}', char='{ref_char}'")
        t = torch.stack(parts, dim=0)
        m = torch.ones((t.size(0),), dtype=torch.float32)
        return t, m, ref_char
    finally:
        reader.close()


def load_train_reference_glyph(
    train_reader: LMDBReader,
    font_name: str,
    ch: str,
    image_size: int,
    cache: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    key = f"{font_name}@{ch}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    b = train_reader.get(key)
    if b is None:
        return None
    t = pil_gray_to_tensor(Image.open(io.BytesIO(b)).convert("L"), image_size)
    cache[key] = t
    return t


@torch.no_grad()
def sample_one(
    trainer: DiffusionTrainer | FlowMatchingTrainer,
    trainer_type: str,
    content: torch.Tensor,
    style_img: Optional[torch.Tensor],
    part_imgs: Optional[torch.Tensor],
    part_mask: Optional[torch.Tensor],
    inference_steps: int,
    condition_mode: str,
) -> torch.Tensor:
    x = content.unsqueeze(0).to(trainer.device)
    s = style_img.unsqueeze(0).to(trainer.device) if style_img is not None else None
    p = part_imgs.unsqueeze(0).to(trainer.device) if part_imgs is not None else None
    m = part_mask.unsqueeze(0).to(trainer.device) if part_mask is not None else None

    if trainer_type == "flow_matching":
        y = trainer.flow_sample(
            x,
            c=int(inference_steps),
            style_img=s,
            part_imgs=p,
            part_mask=m,
            condition_mode=condition_mode,
        )
    else:
        y = trainer.dpm_solver_sample(
            x,
            style_img=s,
            num_inference_steps=int(inference_steps),
            part_imgs=p,
            part_mask=m,
            condition_mode=condition_mode,
        )
    return y.squeeze(0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flexible inference for DiffuFont (style_only / part_only).")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trainer", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    parser.add_argument(
        "--conditioning-profile",
        type=str,
        default="style_only",
        choices=["baseline", "parts_vector_only", "part_only", "style_only"],
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--part-image-size", type=int, default=40)
    parser.add_argument("--style-start-channel", type=int, default=16)
    parser.add_argument("--style-token-dim", type=int, default=256)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-parts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Weight loading
    parser.add_argument("--checkpoint", type=Path, default=None, help="full checkpoint or raw state_dict")
    parser.add_argument("--main-weight", type=Path, default=None, help="optional split main_model weights")
    parser.add_argument("--vector-weight", type=Path, default=None, help="optional split vector/style/part weights")
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=False)

    # Content input
    parser.add_argument("--content-source", type=str, default="lmdb", choices=["folder", "lmdb"])
    parser.add_argument("--content-folder", type=Path, default=None)
    parser.add_argument("--content-lmdb", type=Path, default=None)
    parser.add_argument("--content-chars", type=str, default="")
    parser.add_argument(
        "--content-font",
        type=str,
        default=None,
        help="Font name used to load original glyph from TrainFont.lmdb as '<content_font>@<content_char>'.",
    )

    # Style input
    parser.add_argument("--style-source", type=str, default="lmdb", choices=["folder", "lmdb"])
    parser.add_argument("--style-folder", type=Path, default=None)
    parser.add_argument("--style-lmdb", type=Path, default=None)
    parser.add_argument("--style-font", type=str, default=None)
    parser.add_argument("--style-chars", type=str, default="")

    # Part input
    parser.add_argument("--part-source", type=str, default="lmdb", choices=["folder", "lmdb"])
    parser.add_argument("--part-folder", type=Path, default=None)
    parser.add_argument("--part-lmdb", type=Path, default=None)
    parser.add_argument("--part-font", type=str, default=None)
    parser.add_argument("--part-chars", type=str, default="")

    args = parser.parse_args()
    set_global_seed(int(args.seed))
    mode = normalize_conditioning_mode(args.conditioning_profile)
    use_style = mode_uses_style(mode)
    use_parts = mode_uses_parts(mode)
    project_root = args.data_root.resolve()
    content_lmdb = args.content_lmdb or (project_root / "DataPreparation" / "LMDB" / "ContentFont.lmdb")
    style_lmdb = args.style_lmdb or (project_root / "DataPreparation" / "LMDB" / "TrainFont.lmdb")
    part_lmdb = args.part_lmdb or (project_root / "DataPreparation" / "LMDB" / "PartBank.lmdb")
    train_lmdb = project_root / "DataPreparation" / "LMDB" / "TrainFont.lmdb"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = SourcePartRefUNet(
        in_channels=1,
        image_size=int(args.image_size),
        content_start_channel=64,
        style_start_channel=int(args.style_start_channel),
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=4,
        channel_attn=True,
        conditioning_profile=mode,
        style_token_dim=int(args.style_token_dim),
    )
    load_model_weights(
        model=model,
        checkpoint=args.checkpoint,
        main_weight=args.main_weight,
        vector_weight=args.vector_weight,
        strict_load=bool(args.strict_load),
    )

    trainer = build_trainer(
        model=model,
        device=device,
        trainer_type=args.trainer,
        diffusion_steps=int(args.diffusion_steps),
        conditioning_mode=mode,
    )

    content_chars = parse_chars(args.content_chars)
    style_chars = parse_chars(args.style_chars)
    part_chars = parse_chars(args.part_chars)

    content_samples = load_content_samples(
        source=args.content_source,
        content_folder=args.content_folder,
        content_lmdb=content_lmdb,
        content_chars=content_chars,
        image_size=model.unet_input_size,
        max_samples=int(args.max_samples),
    )
    if not content_samples:
        raise RuntimeError("No content samples loaded.")

    style_cache: Dict[str, torch.Tensor] = {}
    part_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
    train_ref_cache: Dict[str, torch.Tensor] = {}
    grid_rows: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    train_reader = LMDBReader(train_lmdb)

    try:
        for i, (sid_raw, content_char, content_tensor) in enumerate(content_samples):
            sid = safe_stem(sid_raw)

            style_tensor: Optional[torch.Tensor] = None
            part_tensor: Optional[torch.Tensor] = None
            part_mask: Optional[torch.Tensor] = None
            ref_vis: Optional[torch.Tensor] = None
            ref_font: Optional[str] = None
            ref_char: Optional[str] = None

            if use_style:
                style_tensor, style_ref = resolve_style_tensor_for_index(
                    idx=i,
                    source=args.style_source,
                    style_folder=args.style_folder,
                    style_lmdb=style_lmdb,
                    style_font=args.style_font,
                    style_chars=style_chars,
                    image_size=model.unet_input_size,
                    cache=style_cache,
                )
                ref_vis = style_tensor
                if args.style_source == "lmdb" and args.style_font:
                    ref_font = args.style_font
                    ref_char = style_ref

            if use_parts:
                part_ref_char = part_chars[i % len(part_chars)] if part_chars else content_char
                pkey = f"{args.part_source}:{part_ref_char}"
                if pkey not in part_cache:
                    if args.part_source == "folder":
                        if args.part_folder is None:
                            raise ValueError("--part-folder is required when --part-source=folder")
                        part_cache[pkey] = load_part_set_from_folder(
                            part_folder=args.part_folder,
                            ref_char=part_ref_char,
                            part_image_size=int(args.part_image_size),
                            max_parts=int(args.max_parts),
                        )
                    else:
                        if not args.part_font:
                            raise ValueError("--part-font is required when --part-source=lmdb")
                        part_cache[pkey] = load_part_set_from_lmdb(
                            part_lmdb=part_lmdb,
                            part_font=args.part_font,
                            ref_char=part_ref_char,
                            part_image_size=int(args.part_image_size),
                            max_parts=int(args.max_parts),
                        )
                part_tensor, part_mask, part_ref = part_cache[pkey]
                _ = part_ref
                ref_vis = build_part_preview(part_tensor, out_size=model.unet_input_size)
                if args.part_font:
                    ref_font = args.part_font
                    ref_char = part_ref_char

            # Prefer reference original glyph from TrainFont.lmdb when font+char is known.
            if ref_font and ref_char:
                ref_glyph = load_train_reference_glyph(
                    train_reader=train_reader,
                    font_name=ref_font,
                    ch=ref_char,
                    image_size=model.unet_input_size,
                    cache=train_ref_cache,
                )
                if ref_glyph is not None:
                    ref_vis = ref_glyph

            if ref_vis is None:
                ref_vis = torch.full_like(content_tensor, -1.0)

            original_vis = torch.full_like(content_tensor, -1.0)
            if args.content_font:
                original_glyph = load_train_reference_glyph(
                    train_reader=train_reader,
                    font_name=args.content_font,
                    ch=content_char,
                    image_size=model.unet_input_size,
                    cache=train_ref_cache,
                )
                if original_glyph is not None:
                    original_vis = original_glyph

            gen = sample_one(
                trainer=trainer,
                trainer_type=args.trainer,
                content=content_tensor,
                style_img=style_tensor,
                part_imgs=part_tensor,
                part_mask=part_mask,
                inference_steps=int(args.inference_steps),
                condition_mode=mode,
            )
            grid_rows.append((sid, content_tensor.cpu(), ref_vis.cpu(), original_vis.cpu(), gen.cpu()))
    finally:
        train_reader.close()

    out_img = output_dir / "inference_grid.png"
    save_combined_grid(grid_rows, mode=mode, out_path=out_img)
    print(f"[inference] done. mode={mode} seed={int(args.seed)} samples={len(content_samples)} output={out_img}")


if __name__ == "__main__":
    main()
