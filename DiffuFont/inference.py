#!/usr/bin/env python3
"""Inference entry for the content+style pixel-space flow path."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
import torch

from dataset import FontImageDataset
from models.model import FlowTrainer
from models.source_part_ref_dit import SourcePartRefDiT
from style_augment import build_base_glyph_transform


def load_trainer(checkpoint_path: Path, device: torch.device) -> FlowTrainer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("stage") != "flow":
        raise RuntimeError(f"Checkpoint is not a flow checkpoint: {checkpoint_path}")
    if "model_config" not in checkpoint:
        raise RuntimeError("Checkpoint is missing 'model_config'.")
    trainer_config = checkpoint.get("trainer_config", {})
    model = SourcePartRefDiT(**checkpoint["model_config"])
    trainer = FlowTrainer(
        model,
        device,
        total_steps=1,
        flow_sample_steps=int(trainer_config.get("flow_sample_steps", 24)),
        ema_decay=float(trainer_config.get("ema_decay", 0.9999)),
        aux_loss_t_logistic_steepness=float(trainer_config.get("aux_loss_t_logistic_steepness", 8.0)),
        perceptual_loss_t_midpoint=float(trainer_config.get("perceptual_loss_t_midpoint", 0.35)),
    )
    trainer.load(checkpoint_path)
    trainer.model.eval()
    if trainer.ema_model is not None:
        trainer.ema_model.eval()
    return trainer


def sample_style_refs(sample: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    style_img = sample["style_img"]
    style_ref_mask = sample["style_ref_mask"]
    min_refs = int(sample.get("style_ref_count_min", style_img.size(0)))
    max_refs = min(int(sample.get("style_ref_count_max", style_img.size(0))), int(style_img.size(0)))
    if max_refs < min_refs:
        raise RuntimeError(f"Invalid style ref bounds: min_refs={min_refs} max_refs={max_refs}")
    return style_img[:max_refs], style_ref_mask[:max_refs]


def tensor_to_pil(tensor: torch.Tensor, size: int = 128) -> Image.Image:
    x = tensor.detach().cpu().float()
    if x.dim() == 3:
        x = x[0]
    x = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte().numpy()
    return Image.fromarray(x, mode="L").resize((size, size), Image.Resampling.LANCZOS)


def find_sample_index(dataset: FontImageDataset, font_name: str, char: str) -> int:
    for idx, (font, char_index) in enumerate(dataset.samples):
        if font == font_name and dataset.char_list[char_index] == char:
            return idx
    raise KeyError(f"Missing sample for font='{font_name}' char='{char}'")


def resolve_chars_for_fonts(
    dataset: FontImageDataset,
    *,
    font_names: List[str],
    requested_chars: List[str],
    num_chars: int,
) -> List[str]:
    common_char_indices = None
    for font_name in font_names:
        font_char_indices = set(dataset.sample_index_by_font_char[font_name].keys())
        common_char_indices = font_char_indices if common_char_indices is None else (common_char_indices & font_char_indices)
    common_char_indices = set() if common_char_indices is None else common_char_indices
    common_chars = [dataset.char_list[idx] for idx in sorted(common_char_indices)]
    if not common_chars:
        raise RuntimeError(f"No shared chars found across selected fonts: {font_names}")

    if requested_chars:
        missing = [char for char in requested_chars if char not in common_chars]
        if missing:
            raise KeyError(
                "Requested chars are not available for every selected font: "
                f"missing={missing} fonts={font_names}"
            )
        return requested_chars

    return random.sample(common_chars, k=min(int(num_chars), len(common_chars)))


@torch.no_grad()
def run_inference(
    trainer: FlowTrainer,
    dataset: FontImageDataset,
    *,
    font_names: List[str],
    chars: List[str],
    inference_steps: int,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for font_name in font_names:
        font_rows: Dict[str, Dict[str, torch.Tensor]] = {}
        for char in chars:
            sample = dataset[find_sample_index(dataset, font_name, char)]
            content = sample["content"].unsqueeze(0)
            style_refs, style_ref_mask = sample_style_refs(sample)
            style = style_refs.unsqueeze(0)
            style_ref_mask = style_ref_mask.unsqueeze(0)
            generation = trainer.flow_sample(
                content,
                content_index=torch.tensor([0], dtype=torch.long),
                style_img=style,
                style_index=torch.tensor([0], dtype=torch.long),
                style_ref_mask=style_ref_mask,
                num_inference_steps=int(inference_steps),
            )
            font_rows[char] = {
                "content": sample["content"],
                "style": style_refs[0],
                "target": sample["target"],
                "generation": generation.squeeze(0).cpu(),
            }
        results[font_name] = font_rows
    return results


def build_grid(
    results: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    *,
    font_names: List[str],
    chars: List[str],
    cell_size: int,
) -> Image.Image:
    label_w = 180
    header_h = 44
    per_char_cols = 4
    width = label_w + len(chars) * per_char_cols * cell_size
    height = header_h + len(font_names) * cell_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for char_idx, char in enumerate(chars):
        x0 = label_w + char_idx * per_char_cols * cell_size
        for offset, label in enumerate(("content", "style", "gt", "gen")):
            draw.text((x0 + offset * cell_size + 4, 8), f"{char}\n{label}", fill=(0, 0, 0), font=font)

    for row_idx, font_name in enumerate(font_names):
        y0 = header_h + row_idx * cell_size
        draw.text((6, y0 + cell_size // 2 - 8), font_name[:18], fill=(0, 0, 0), font=font)
        for char_idx, char in enumerate(chars):
            row = results[font_name][char]
            images = (row["content"], row["style"], row["target"], row["generation"])
            x0 = label_w + char_idx * per_char_cols * cell_size
            for offset, tensor in enumerate(images):
                canvas.paste(tensor_to_pil(tensor, size=cell_size).convert("RGB"), (x0 + offset * cell_size, y0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/inference_grid.png"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fonts", type=int, default=0)
    parser.add_argument("--style-ref-count", type=int, default=0)
    parser.add_argument("--style-ref-count-min", type=int, default=6)
    parser.add_argument("--style-ref-count-max", type=int, default=8)
    parser.add_argument("--num-fonts", type=int, default=4)
    parser.add_argument("--num-chars", type=int, default=6)
    parser.add_argument("--font-names", type=str, default="")
    parser.add_argument("--chars", type=str, default="")
    parser.add_argument("--inference-steps", type=int, default=24)
    parser.add_argument("--cell-size", type=int, default=128)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    trainer = load_trainer(args.checkpoint, device)
    glyph_transform = build_base_glyph_transform(image_size=int(trainer.model.image_size))
    style_ref_count = None if int(args.style_ref_count) <= 0 else int(args.style_ref_count)
    dataset = FontImageDataset(
        project_root=args.data_root,
        max_fonts=int(args.max_fonts),
        style_ref_count=style_ref_count,
        style_ref_count_min=int(args.style_ref_count_min),
        style_ref_count_max=int(args.style_ref_count_max),
        random_seed=int(args.seed),
        transform=glyph_transform,
        style_transform=glyph_transform,
    )

    all_fonts = sorted(dataset.font_names)
    if args.font_names.strip():
        font_names = [name.strip() for name in args.font_names.split(",") if name.strip()]
    else:
        font_names = random.sample(all_fonts, k=min(int(args.num_fonts), len(all_fonts)))
    missing_fonts = [name for name in font_names if name not in dataset.font_id_by_name]
    if missing_fonts:
        raise KeyError(f"Missing fonts in dataset: {missing_fonts}")

    requested_chars = [char.strip() for char in args.chars.split(",") if char.strip()] if args.chars.strip() else []
    chars = resolve_chars_for_fonts(
        dataset,
        font_names=font_names,
        requested_chars=requested_chars,
        num_chars=int(args.num_chars),
    )

    results = run_inference(
        trainer,
        dataset,
        font_names=font_names,
        chars=chars,
        inference_steps=int(args.inference_steps),
    )
    grid = build_grid(results, font_names=font_names, chars=chars, cell_size=int(args.cell_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"[inference] saved {args.output}")


if __name__ == "__main__":
    main()
