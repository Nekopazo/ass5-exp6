#!/usr/bin/env python3
"""Inspect the grayscale CNN font perceptor architecture and tensor shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.font_perceptor import FontPerceptor


def shape_of(value: torch.Tensor) -> str:
    return str(tuple(int(dim) for dim in value.shape))


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(int(param.numel()) for param in module.parameters())
    trainable = sum(int(param.numel()) for param in module.parameters() if param.requires_grad)
    return total, trainable


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def build_model_from_args(args: argparse.Namespace) -> tuple[FontPerceptor, dict]:
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model_config = dict(checkpoint.get("model_config", {}))
        if not model_config:
            raise RuntimeError(f"Checkpoint is missing model_config: {args.checkpoint}")
        model = FontPerceptor(**model_config)
        if args.load_weights:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        return model, model_config
    model = FontPerceptor()
    return model, model.export_config()


def trace_model(model: FontPerceptor, batch_size: int, image_size: int, device: torch.device) -> None:
    print_header("Resolved Config")
    print(json.dumps(model.export_config(), ensure_ascii=False, indent=2, sort_keys=True))

    total_params, trainable_params = count_params(model)
    print_header("Model Summary")
    print(f"type: {model.__class__.__name__}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")
    print(f"feature_stages: {model.feature_stage_names}")

    print()
    print("top_level_modules:")
    for name in [
        "stem",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "global_proj",
        "font_head",
        "char_head",
    ]:
        module = getattr(model, name)
        total, trainable = count_params(module)
        print(f"  - {name}: {module.__class__.__name__} total={total:,} trainable={trainable:,}")

    x = torch.randn(batch_size, model.in_channels, image_size, image_size, device=device)
    print_header("Forward With Shapes")
    print(f"input: {shape_of(x)}")

    x = model.stem(x)
    print(f"stem: {shape_of(x)}")
    stage1 = model.stage1(x)
    print(f"stage1: {shape_of(stage1)}")
    stage2 = model.stage2(stage1)
    print(f"stage2: {shape_of(stage2)}")
    stage3 = model.stage3(stage2)
    print(f"stage3: {shape_of(stage3)}")
    stage4 = model.stage4(stage3)
    print(f"stage4: {shape_of(stage4)}")

    pooled = model.global_pool(stage4).flatten(1)
    print(f"global_pool_flatten: {shape_of(pooled)}")
    global_feat = model.global_proj(pooled)
    print(f"global_feat: {shape_of(global_feat)}")
    font_logits = model.font_head(global_feat)
    print(f"font_logits: {shape_of(font_logits)}")
    char_logits = model.char_head(global_feat)
    print(f"char_logits: {shape_of(char_logits)}")

    print()
    print("feature_maps:")
    for idx, feature_map in enumerate([stage1, stage2, stage3, stage4], start=1):
        print(f"  - stage{idx}: {shape_of(feature_map)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--load-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = build_model_from_args(args)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        trace_model(model, int(args.batch_size), int(args.image_size), device)


if __name__ == "__main__":
    main()
