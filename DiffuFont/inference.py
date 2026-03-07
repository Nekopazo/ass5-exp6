#!/usr/bin/env python3
"""
Multi-font comparison inference for DiffuFont.

Generates a grid image comparing model outputs across multiple fonts.
Each row = one font, each column = one target character.
Columns are: content glyph | ground-truth | generation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont as PILImageFont

from dataset import FontImageDataset
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


def mode_uses_parts(mode: str) -> bool:
    _ = normalize_conditioning_mode(mode)
    return False


def mode_uses_style(mode: str) -> bool:
    return normalize_conditioning_mode(mode) in {"part_only", "style_only"}


def load_model_and_trainer(
    ckpt_path: str,
    device: torch.device,
    trainer_type: str = "diffusion",
    conditioning_profile: str = "part_only",
    attn_scales: Optional[tuple[int, ...]] = None,
    image_size: int = 256,
    style_start_channel: int = 16,
    style_token_dim: int = 256,
    diffusion_steps: int = 1000,
) -> DiffusionTrainer | FlowMatchingTrainer:
    """Build model and trainer, then load checkpoint weights."""
    mode = normalize_conditioning_mode(conditioning_profile)
    model = SourcePartRefUNet(
        in_channels=1,
        image_size=image_size,
        content_start_channel=64,
        style_start_channel=style_start_channel,
        unet_channels=(64, 128, 256, 512),
        content_encoder_downsample_size=4,
        channel_attn=True,
        conditioning_profile=mode,
        attn_scales=attn_scales,
        style_token_dim=style_token_dim,
    )
    trainer_cls = DiffusionTrainer if trainer_type == "diffusion" else FlowMatchingTrainer
    trainer_kwargs = {
        "lr": 1e-4,
        "T": diffusion_steps,
        "total_steps": 1,
        "conditioning_mode": mode,
    }
    if trainer_type == "flow_matching":
        trainer_kwargs["lambda_fm"] = 1.0
    trainer = trainer_cls(model, device, **trainer_kwargs)
    trainer.load(ckpt_path)
    trainer.model.eval()
    print(f"[inference] loaded checkpoint from {ckpt_path} (step={trainer.global_step})")
    return trainer


def _parse_int_csv(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return tuple(int(t.strip()) for t in text.split(",") if t.strip())


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [-1,1] tensor (C,H,W) to PIL Image."""
    t = (t.clamp(-1, 1) + 1) / 2  # -> [0,1]
    t = t.cpu()
    if t.dim() == 3:
        import torchvision.transforms.functional as TF
        return TF.to_pil_image(t)
    return Image.fromarray((t.squeeze().numpy() * 255).astype("uint8"))


@torch.no_grad()
def run_inference(
    trainer: DiffusionTrainer | FlowMatchingTrainer,
    dataset: FontImageDataset,
    font_names: List[str],
    char_list: List[str],
    num_inference_steps: int = 20,
) -> Dict[str, Dict[str, dict]]:
    """
    Returns: {font_name: {char: {"content": Tensor, "gt": Tensor, "gen": Tensor}}}
    """
    transform = T.Compose([T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
    device = trainer.device
    results: Dict[str, Dict[str, dict]] = {}

    for font in font_names:
        results[font] = {}
        for ch in char_list:
            # Find the sample index for this font+char
            idx = None
            for i, (fn, ci) in enumerate(dataset.samples):
                if fn == font and dataset.char_list[ci] == ch:
                    idx = i
                    break
            if idx is None:
                print(f"[inference] WARNING: font={font} char={ch} not found in dataset, skipping.")
                continue

            sample = dataset[idx]
            content = sample["content"].unsqueeze(0).to(device)
            gt = sample["input"]
            style_img = sample["style_img"].unsqueeze(0).to(device) if "style_img" in sample else None

            part_imgs = None
            part_mask = None
            if "parts" in sample:
                part_imgs = sample["parts"].unsqueeze(0).to(device)
                part_mask = sample["part_mask"].unsqueeze(0).to(device)

            if isinstance(trainer, FlowMatchingTrainer):
                gen = trainer.flow_sample(
                    content, c=num_inference_steps,
                    style_img=style_img,
                    part_imgs=part_imgs, part_mask=part_mask,
                )
            else:
                gen = trainer.dpm_solver_sample(
                    content,
                    style_img=style_img,
                    num_inference_steps=num_inference_steps,
                    part_imgs=part_imgs,
                    part_mask=part_mask,
                )

            results[font][ch] = {
                "content": sample["content"].cpu(),
                "gt": gt.cpu(),
                "gen": gen.squeeze(0).cpu(),
            }

    return results


def build_comparison_grid(
    results: Dict[str, Dict[str, dict]],
    font_names: List[str],
    char_list: List[str],
    cell_size: int = 128,
) -> Image.Image:
    """Build a comparison grid: rows = fonts, per-char columns = [content | GT | gen]."""
    n_fonts = len(font_names)
    n_chars = len(char_list)
    cols = n_chars * 3  # content, GT, gen
    label_w = 180  # left column for font names
    header_h = 40  # top row for char labels
    grid_w = label_w + cols * cell_size
    grid_h = header_h + n_fonts * cell_size

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        fnt = PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        fnt = PILImageFont.load_default()

    # Header: char labels with sub-labels
    for ci, ch in enumerate(char_list):
        x0 = label_w + ci * 3 * cell_size
        for si, sub in enumerate(["cont", "GT", "gen"]):
            cx = x0 + si * cell_size + cell_size // 2
            draw.text((cx - 15, 10), f"{ch}\n{sub}", fill=(0, 0, 0), font=fnt)

    # Body
    for fi, font in enumerate(font_names):
        y0 = header_h + fi * cell_size
        # Font label
        short_name = font if len(font) <= 20 else font[:17] + "..."
        draw.text((5, y0 + cell_size // 2 - 10), short_name, fill=(0, 0, 0), font=fnt)

        for ci, ch in enumerate(char_list):
            entry = results.get(font, {}).get(ch)
            if entry is None:
                continue
            x0 = label_w + ci * 3 * cell_size
            for si, key in enumerate(["content", "gt", "gen"]):
                img = _tensor_to_pil(entry[key]).resize((cell_size, cell_size), Image.LANCZOS)
                canvas.paste(img, (x0 + si * cell_size, y0))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Multi-font comparison inference for DiffuFont")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trainer checkpoint (.pt)")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=str, default="inference_comparison.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trainer", type=str, default="diffusion", choices=["diffusion", "flow_matching"])
    parser.add_argument("--conditioning-profile", type=str, default="part_only",
                        choices=["baseline", "parts_vector_only", "part_only", "style_only"])
    parser.add_argument("--attn-scales", type=str, default="16,32,64")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=20, help="Sampling steps")
    parser.add_argument("--num-fonts", type=int, default=4, help="Number of fonts to compare")
    parser.add_argument("--num-chars", type=int, default=6, help="Number of characters per font")
    parser.add_argument("--font-names", type=str, default=None,
                        help="Comma-separated font names to use (overrides --num-fonts)")
    parser.add_argument("--chars", type=str, default=None,
                        help="Comma-separated characters to generate (overrides --num-chars)")
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style-start-channel", type=int, default=16)
    parser.add_argument("--style-token-dim", type=int, default=256)

    # PartBank
    parser.add_argument("--part-bank-manifest", type=str, default="DataPreparation/PartBank/manifest.json")
    parser.add_argument("--part-bank-lmdb", type=str, default="DataPreparation/LMDB/PartBank.lmdb")
    parser.add_argument("--part-set-max", type=int, default=8)
    parser.add_argument("--part-image-size", type=int, default=40)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    attn_scales = _parse_int_csv(args.attn_scales)

    profile = normalize_conditioning_mode(args.conditioning_profile)
    use_part_bank = mode_uses_parts(profile)
    use_style_image = mode_uses_style(profile)

    # Load dataset (for data access)
    transform = T.Compose([T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
    dataset = FontImageDataset(
        project_root=args.data_root,
        use_style_image=use_style_image,
        use_part_bank=use_part_bank,
        part_bank_manifest=args.part_bank_manifest,
        part_bank_lmdb=args.part_bank_lmdb,
        part_set_max=args.part_set_max,
        part_image_size=args.part_image_size,
        transform=transform,
    )

    # Determine fonts and chars
    all_fonts = sorted(set(fn for fn, _ in dataset.samples))
    if args.font_names:
        font_names = [f.strip() for f in args.font_names.split(",")]
        missing = [f for f in font_names if f not in all_fonts]
        if missing:
            print(f"[inference] WARNING: fonts not found in dataset: {missing}")
            font_names = [f for f in font_names if f in all_fonts]
    else:
        n = min(args.num_fonts, len(all_fonts))
        font_names = random.sample(all_fonts, n)

    if args.chars:
        char_list = [c.strip() for c in args.chars.split(",")]
    else:
        n = min(args.num_chars, len(dataset.char_list))
        char_list = random.sample(dataset.char_list, n)

    print(f"[inference] fonts ({len(font_names)}): {font_names}")
    print(f"[inference] chars ({len(char_list)}): {char_list}")

    # Load model
    trainer = load_model_and_trainer(
        ckpt_path=args.checkpoint,
        device=device,
        trainer_type=args.trainer,
        conditioning_profile=args.conditioning_profile,
        attn_scales=attn_scales,
        image_size=args.image_size,
        style_start_channel=int(args.style_start_channel),
        style_token_dim=args.style_token_dim,
        diffusion_steps=args.diffusion_steps,
    )

    # Run inference
    results = run_inference(
        trainer=trainer,
        dataset=dataset,
        font_names=font_names,
        char_list=char_list,
        num_inference_steps=args.inference_steps,
    )

    # Build and save comparison grid
    grid_img = build_comparison_grid(results, font_names, char_list, cell_size=args.cell_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(str(out_path))
    print(f"[inference] saved comparison grid to {out_path}")

    # Also save individual per-font strips
    ind_dir = out_path.parent / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)
    for font in font_names:
        font_results = results.get(font, {})
        if not font_results:
            continue
        gen_tensors = [font_results[ch]["gen"] for ch in char_list if ch in font_results]
        if gen_tensors:
            strip = make_grid(torch.stack(gen_tensors), nrow=len(gen_tensors), padding=2, normalize=True, value_range=(-1, 1))
            save_image(strip, ind_dir / f"{font}_generated.png")

    print(f"[inference] individual font strips saved to {ind_dir}/")


if __name__ == "__main__":
    main()
