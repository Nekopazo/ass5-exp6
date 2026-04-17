#!/usr/bin/env python3
"""Inspect the current pixel-space flow model architecture and tensor shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.diffusion_transformer_backbone import timestep_embedding
from models.source_part_ref_dit import SourcePartRefDiT


def shape_of(value: torch.Tensor | None) -> str:
    if value is None:
        return "None"
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


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def build_model_from_args(args: argparse.Namespace) -> tuple[SourcePartRefDiT, dict]:
    checkpoint_config: dict = {}
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_config = dict(checkpoint.get("model_config", {}))
        if not checkpoint_config:
            raise RuntimeError(f"Checkpoint is missing model_config: {args.checkpoint}")
        model = SourcePartRefDiT(**checkpoint_config)
        if args.load_weights:
            model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model = SourcePartRefDiT()
        checkpoint_config = model.export_config()
    return model, checkpoint_config


def print_model_summary(model: SourcePartRefDiT) -> None:
    total_params, trainable_params = count_params(model)
    print_header("Model Summary")
    print_lines(
        [
            f"type: {model.__class__.__name__}",
            f"total_params: {total_params:,}",
            f"trainable_params: {trainable_params:,}",
            f"image_size: {model.image_size}",
            f"patch_size: {model.patch_size}",
            f"patch_grid_size: {model.patch_grid_size}",
            f"num_patches: {model.num_patches}",
            f"encoder_hidden_dim: {model.encoder_hidden_dim}",
            f"style_token_hidden_dim: {model.style_token_hidden_dim}",
            f"dit_hidden_dim: {model.dit_hidden_dim}",
            f"dit_depth: {model.dit_depth}",
            f"dit_heads: {model.dit_heads}",
            f"ffn_activation: {model.ffn_activation}",
            f"norm_variant: {model.norm_variant}",
            f"content_style_fusion_heads: {model.content_style_fusion_heads}",
            f"content_injection_layers: {list(model.content_injection_layers)}",
        ]
    )

    print()
    print("top_level_modules:")
    top_level_names = [
        "content_encoder",
        "content_proj",
        "style_encoder",
        "style_token_proj",
        "content_style_attn",
        "content_style_fusion_norm",
        "content_style_fusion_proj",
        "backbone",
        "refiner",
    ]
    for name in top_level_names:
        module = getattr(model, name)
        total, trainable = count_params(module)
        print(f"  - {name}: {module.__class__.__name__} total={total:,} trainable={trainable:,}")

    print()
    print("backbone_layer_plan:")
    for idx, block in enumerate(model.backbone.blocks):
        print(f"  - block_{idx:02d}: content_injection={int(block.use_content_injection)}")


def trace_content_path(model: SourcePartRefDiT, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = model.content_encoder
    print_header("Content Path")
    print(f"content_input: {shape_of(content)}")
    x = encoder.stem(content)
    print(f"content.stem: {shape_of(x)}")
    x = encoder.stem_block(x)
    print(f"content.stem_block: {shape_of(x)}")
    for idx, (downsample, resblock) in enumerate(zip(encoder.stage_downsamples, encoder.stage_blocks)):
        x = downsample(x)
        print(f"content.downsample_{idx}: {shape_of(x)}")
        x = resblock(x)
        print(f"content.resblock_{idx}: {shape_of(x)}")
    if x.shape[-2:] != (encoder.output_grid_size, encoder.output_grid_size):
        x = F.interpolate(
            x,
            size=(encoder.output_grid_size, encoder.output_grid_size),
            mode="bilinear",
            align_corners=False,
        )
        print(f"content.interpolate_to_grid: {shape_of(x)}")
    content_features = x
    content_tokens = content_features.flatten(2).transpose(1, 2).contiguous()
    print(f"content.tokens_before_proj: {shape_of(content_tokens)}")
    content_tokens = model.content_proj(content_tokens)
    print(f"content.tokens_after_proj: {shape_of(content_tokens)}")
    return content_features, content_tokens


def trace_style_path(
    model: SourcePartRefDiT,
    style_img: torch.Tensor,
    style_ref_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = model.style_encoder
    print_header("Style Path")
    print(f"style_input: {shape_of(style_img)}")
    batch, refs, channels, height, width = style_img.shape
    flat_style = style_img.view(batch * refs, channels, height, width)
    print(f"style.flatten_refs: {shape_of(flat_style)}")
    x = encoder.stem(flat_style)
    print(f"style.stem: {shape_of(x)}")
    x = encoder.stem_block(x)
    print(f"style.stem_block: {shape_of(x)}")
    for idx, (downsample, resblock) in enumerate(zip(encoder.stage_downsamples, encoder.stage_blocks)):
        x = downsample(x)
        print(f"style.downsample_{idx}: {shape_of(x)}")
        x = resblock(x)
        print(f"style.resblock_{idx}: {shape_of(x)}")
    style_tokens = x.flatten(2).transpose(1, 2).contiguous()
    print(f"style.per_ref_spatial_tokens: {shape_of(style_tokens)}")
    tokens_per_ref = int(style_tokens.size(1))
    style_tokens = style_tokens.view(batch, refs * tokens_per_ref, style_tokens.size(-1))
    print(f"style.all_ref_spatial_tokens: {shape_of(style_tokens)}")
    ref_valid_mask = style_ref_mask.to(device=style_tokens.device, dtype=torch.bool)
    token_valid_mask = (
        ref_valid_mask[:, :, None]
        .expand(batch, refs, tokens_per_ref)
        .reshape(batch, refs * tokens_per_ref)
    )
    print(f"style.all_ref_spatial_token_mask: {shape_of(token_valid_mask)}")
    style_token_bank = model.style_token_proj(style_tokens)
    print(f"style.token_bank: {shape_of(style_token_bank)}")
    return style_token_bank, token_valid_mask


def trace_backbone_path(
    model: SourcePartRefDiT,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    content_tokens: torch.Tensor,
) -> torch.Tensor:
    backbone = model.backbone
    print_header("Backbone Path")
    print(f"xt_input: {shape_of(xt)}")
    x = backbone.patch_embed(xt).flatten(2).transpose(1, 2).contiguous()
    print(f"backbone.patch_embed_tokens: {shape_of(x)}")
    x = x + backbone.pos_embed.to(device=x.device, dtype=x.dtype)
    print(f"backbone.tokens_plus_pos: {shape_of(x)}")
    content_tokens = content_tokens.to(device=x.device, dtype=x.dtype)
    print(f"backbone.content_tokens: {shape_of(content_tokens)}")
    time_cond = timestep_embedding(timesteps, backbone.hidden_dim).to(dtype=x.dtype)
    print(f"backbone.timestep_embedding: {shape_of(time_cond)}")
    time_cond = backbone.time_mlp(time_cond)
    print(f"backbone.time_mlp: {shape_of(time_cond)}")
    time_cond = backbone.time_cond_norm(time_cond)
    print(f"backbone.time_cond_norm: {shape_of(time_cond)}")
    for idx, block in enumerate(backbone.blocks):
        x = block(
            x,
            time_cond=time_cond,
            content_tokens=content_tokens if block.use_content_injection else None,
        )
        print(
            f"backbone.block_{idx:02d}: out={shape_of(x)} "
            f"content_injection={int(block.use_content_injection)}"
        )
    x = backbone.final_norm(x)
    print(f"backbone.final_norm: {shape_of(x)}")
    return x


def trace_refiner_path(
    model: SourcePartRefDiT,
    patch_tokens: torch.Tensor,
    noisy_image: torch.Tensor,
) -> torch.Tensor:
    refiner = model.refiner
    print_header("Refiner Path")
    print(f"noisy_image: {shape_of(noisy_image)}")
    print(f"patch_tokens: {shape_of(patch_tokens)}")
    print(f"refiner_mode: {model.refiner_mode}")
    if model.refiner_mode == "image":
        skips: list[torch.Tensor] = []
        x = refiner.input_proj(noisy_image)
        skips.append(x)
        print(f"refiner.input_proj: {shape_of(x)}")
        x = refiner.input_pool(x)
        print(f"refiner.input_pool: {shape_of(x)}")
        for idx, (downsample, pool) in enumerate(zip(refiner.downsample_blocks, refiner.downsample_pools)):
            x = downsample(x)
            skips.append(x)
            print(f"refiner.downsample_block_{idx}: {shape_of(x)}")
            x = pool(x)
            print(f"refiner.max_pool_{idx}: {shape_of(x)}")
        token_map = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.size(0),
            patch_tokens.size(-1),
            refiner.token_grid_size,
            refiner.token_grid_size,
        )
        print(f"refiner.token_map: {shape_of(token_map)}")
        x = torch.cat([x, token_map], dim=1)
        print(f"refiner.concat_context: {shape_of(x)}")
        x = refiner.bottleneck(x)
        print(f"refiner.bottleneck: {shape_of(x)}")
        for idx, (upsample, dec_block, skip) in enumerate(zip(refiner.upsample_layers, refiner.dec_blocks, reversed(skips))):
            x = upsample(x)
            print(f"refiner.upsample_{idx}: {shape_of(x)}")
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
                print(f"refiner.align_skip_{idx}: {shape_of(x)}")
            x = torch.cat([x, skip], dim=1)
            print(f"refiner.concat_skip_{idx}: {shape_of(x)}")
            x = dec_block(x)
            print(f"refiner.dec_block_{idx}: {shape_of(x)}")
        x = refiner.out_proj(x)
        print(f"refiner.out_proj: {shape_of(x)}")
        return x
    noisy_patches = (
        noisy_image.unfold(2, refiner.patch_size, refiner.patch_size)
        .unfold(3, refiner.patch_size, refiner.patch_size)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(
            noisy_image.size(0),
            refiner.num_patches,
            refiner.in_channels,
            refiner.patch_size,
            refiner.patch_size,
        )
    )
    print(f"refiner.noisy_patches: {shape_of(noisy_patches)}")
    skips: list[torch.Tensor] = []
    x = noisy_patches.view(
        noisy_image.size(0) * refiner.num_patches,
        refiner.in_channels,
        refiner.patch_size,
        refiner.patch_size,
    )
    print(f"refiner.flat_patches: {shape_of(x)}")
    x = refiner.input_proj(x)
    skips.append(x)
    print(f"refiner.input_proj: {shape_of(x)}")
    x = refiner.input_pool(x)
    print(f"refiner.input_pool: {shape_of(x)}")
    for idx, (downsample, pool) in enumerate(zip(refiner.downsample_blocks, refiner.downsample_pools)):
        x = downsample(x)
        skips.append(x)
        print(f"refiner.downsample_block_{idx}: {shape_of(x)}")
        x = pool(x)
        print(f"refiner.max_pool_{idx}: {shape_of(x)}")
    token_map = patch_tokens.contiguous().view(
        patch_tokens.size(0) * patch_tokens.size(1),
        patch_tokens.size(-1),
        1,
        1,
    )
    print(f"refiner.token_map: {shape_of(token_map)}")
    x = torch.cat([x, token_map], dim=1)
    print(f"refiner.concat_context: {shape_of(x)}")
    x = refiner.bottleneck(x)
    print(f"refiner.bottleneck: {shape_of(x)}")
    for idx, (upsample, dec_block, skip) in enumerate(zip(refiner.upsample_layers, refiner.dec_blocks, reversed(skips))):
        x = upsample(x)
        print(f"refiner.upsample_{idx}: {shape_of(x)}")
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            print(f"refiner.align_skip_{idx}: {shape_of(x)}")
        x = torch.cat([x, skip], dim=1)
        print(f"refiner.concat_skip_{idx}: {shape_of(x)}")
        x = dec_block(x)
        print(f"refiner.dec_block_{idx}: {shape_of(x)}")
    x = refiner.out_proj(x)
    print(f"refiner.out_proj_patches: {shape_of(x)}")
    x = x.view(
        patch_tokens.size(0),
        refiner.token_grid_size,
        refiner.token_grid_size,
        refiner.in_channels,
        refiner.patch_size,
        refiner.patch_size,
    )
    print(f"refiner.pred_patches: {shape_of(x)}")
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(
        patch_tokens.size(0),
        refiner.in_channels,
        refiner.image_size,
        refiner.image_size,
    )
    print(f"refiner.unpatchify: {shape_of(x)}")
    return x


def print_training_flow(
    model: SourcePartRefDiT,
    batch_size: int,
    style_refs: int,
    device: torch.device,
) -> None:
    print_header("Training Flow With Shapes")
    image_size = model.image_size
    content = torch.randn(batch_size, 1, image_size, image_size, device=device)
    target = torch.randn(batch_size, 1, image_size, image_size, device=device)
    style_img = torch.randn(batch_size, style_refs, 1, image_size, image_size, device=device)
    style_ref_mask = torch.ones(batch_size, style_refs, device=device, dtype=torch.bool)
    timesteps = torch.linspace(0.1, 0.9, batch_size, device=device)
    x0 = torch.randn_like(target)
    t_view = timesteps.view(-1, 1, 1, 1).to(dtype=target.dtype)
    xt = (1.0 - t_view) * x0 + t_view * target
    target_flow = target - x0

    print(f"content: {shape_of(content)}")
    print(f"target(x1): {shape_of(target)}")
    print(f"style_img: {shape_of(style_img)}")
    print(f"style_ref_mask: {shape_of(style_ref_mask)}")
    print(f"x0_noise: {shape_of(x0)}")
    print(f"timesteps: {shape_of(timesteps)}")
    print(f"xt=(1-t)*x0+t*x1: {shape_of(xt)}")
    print(f"target_flow=x1-x0: {shape_of(target_flow)}")

    _, content_tokens = trace_content_path(model, content)
    style_token_bank, token_valid_mask = trace_style_path(model, style_img, style_ref_mask)
    content_tokens = model.fuse_content_style_tokens(
        content_tokens,
        style_token_bank,
        token_valid_mask=token_valid_mask,
    )
    print(f"content.tokens_after_style_fusion: {shape_of(content_tokens)}")
    patch_tokens = trace_backbone_path(
        model,
        xt,
        timesteps,
        content_tokens,
    )
    pred_flow = trace_refiner_path(model, patch_tokens, xt)
    pred_target = xt + (1.0 - t_view) * pred_flow
    print_header("Outputs")
    print(f"pred_flow: {shape_of(pred_flow)}")
    print(f"pred_target=xt+(1-t)*pred_flow: {shape_of(pred_target)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--load-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--style-refs", type=int, default=6)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = build_model_from_args(args)
    model = model.to(device)
    model.eval()

    print_header("Resolved Config")
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    print_model_summary(model)
    print_training_flow(model, int(args.batch_size), int(args.style_refs), device)


if __name__ == "__main__":
    main()
