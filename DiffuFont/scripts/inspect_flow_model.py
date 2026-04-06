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
            f"style_hidden_dim: {model.style_hidden_dim}",
            f"style_pool_heads: {model.style_pool_heads}",
            f"dit_hidden_dim: {model.dit_hidden_dim}",
            f"dit_depth: {model.dit_depth}",
            f"dit_heads: {model.dit_heads}",
            f"content_injection_layers: {list(model.content_injection_layers)}",
            f"style_injection_layers: {list(model.style_injection_layers)}",
        ]
    )

    print()
    print("top_level_modules:")
    top_level_names = [
        "content_encoder",
        "content_proj",
        "style_encoder",
        "style_proj",
        "backbone",
        "detailer",
    ]
    for name in top_level_names:
        module = getattr(model, name)
        total, trainable = count_params(module)
        print(f"  - {name}: {module.__class__.__name__} total={total:,} trainable={trainable:,}")
        if name == "style_encoder":
            style_pool_params = sum(int(param.numel()) for param in model.style_pool.parameters())
            print(
                "  - style_pool: "
                f"attention_pool+masked_mean total={style_pool_params:,} trainable={style_pool_params:,}"
            )

    print()
    print("backbone_layer_plan:")
    for idx, block in enumerate(model.backbone.blocks):
        print(
            f"  - block_{idx:02d}: content_injection={int(block.use_content_injection)} "
            f"style_injection={int(block.use_style_injection)}"
        )


def trace_content_path(model: SourcePartRefDiT, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = model.content_encoder
    print_header("Content Path")
    print(f"content_input: {shape_of(content)}")
    x = encoder.stem(content)
    print(f"content.stem: {shape_of(x)}")
    x = encoder.stem_resblock(x)
    print(f"content.stem_resblock: {shape_of(x)}")
    for idx, (downsample, resblock) in enumerate(zip(encoder.downsample_layers, encoder.resblocks)):
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
) -> torch.Tensor:
    encoder = model.style_encoder
    print_header("Style Path")
    print(f"style_input: {shape_of(style_img)}")
    batch, refs, channels, height, width = style_img.shape
    flat_style = style_img.view(batch * refs, channels, height, width)
    print(f"style.flatten_refs: {shape_of(flat_style)}")
    x = flat_style
    for idx, (downsample, resblock) in enumerate(zip(encoder.downsample_layers, encoder.resblocks)):
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
    style_query = model.style_pool.query.expand(batch, -1, -1)
    print(f"style.pool_query: {shape_of(style_query)}")
    style_query = model.style_pool.query_norm(style_query)
    print(f"style.pool_query_norm: {shape_of(style_query)}")
    style_keys = model.style_pool.token_norm(style_tokens)
    print(f"style.pool_token_norm: {shape_of(style_keys)}")
    key_padding_mask = None
    need_weights = False
    if bool((~token_valid_mask).any().item()):
        key_padding_mask = ~token_valid_mask
        need_weights = True
    style_vectors, _ = model.style_pool.attn(
        style_query,
        style_keys,
        style_tokens,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
    )
    print(f"style.attention_pool: {shape_of(style_vectors)}")
    pooled_style = style_vectors.squeeze(1)
    print(f"style.pooled_style: {shape_of(pooled_style)}")
    style_global = model.style_proj(pooled_style)
    print(f"style.global: {shape_of(style_global)}")
    return style_global


def trace_backbone_path(
    model: SourcePartRefDiT,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    content_tokens: torch.Tensor,
    style_global: torch.Tensor,
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
    style_global = style_global.to(device=x.device, dtype=x.dtype)
    print(f"backbone.style_global: {shape_of(style_global)}")
    time_cond = timestep_embedding(timesteps, backbone.hidden_dim).to(dtype=x.dtype)
    print(f"backbone.timestep_embedding: {shape_of(time_cond)}")
    time_cond = backbone.time_mlp(time_cond)
    print(f"backbone.time_mlp: {shape_of(time_cond)}")
    for idx, block in enumerate(backbone.blocks):
        x = block(
            x,
            time_cond=time_cond,
            content_tokens=content_tokens if block.use_content_injection else None,
            style_global=style_global if block.use_style_injection else None,
        )
        print(
            f"backbone.block_{idx:02d}: out={shape_of(x)} "
            f"content_injection={int(block.use_content_injection)} "
            f"style_injection={int(block.use_style_injection)}"
        )
    x = backbone.final_norm(x)
    print(f"backbone.final_norm: {shape_of(x)}")
    return x


def trace_detailer_path(
    model: SourcePartRefDiT,
    patch_tokens: torch.Tensor,
    noisy_patches: torch.Tensor,
) -> torch.Tensor:
    detailer = model.detailer
    print_header("Detailer Path")
    print(f"noisy_patches: {shape_of(noisy_patches)}")
    print(f"patch_tokens: {shape_of(patch_tokens)}")
    batch, patch_count, channels, patch_h, patch_w = noisy_patches.shape
    flat_patches = noisy_patches.reshape(batch * patch_count, channels, patch_h, patch_w)
    flat_tokens = patch_tokens.reshape(batch * patch_count, patch_tokens.size(-1))
    print(f"detailer.flat_patches: {shape_of(flat_patches)}")
    print(f"detailer.flat_tokens: {shape_of(flat_tokens)}")

    skips: list[torch.Tensor] = []
    x = detailer.input_proj(flat_patches)
    skips.append(x)
    print(f"detailer.input_proj: {shape_of(x)}")
    for idx, downsample in enumerate(detailer.downsample_blocks):
        x = downsample(x)
        print(f"detailer.downsample_block_{idx}: {shape_of(x)}")
        if idx < len(detailer.downsample_blocks) - 1:
            skips.append(x)
    context = detailer.context_proj(flat_tokens).view(flat_tokens.size(0), -1, 1, 1)
    print(f"detailer.context_proj: {shape_of(context)}")
    x = torch.cat([x, context], dim=1)
    print(f"detailer.concat_context: {shape_of(x)}")
    x = detailer.bottleneck(x)
    print(f"detailer.bottleneck: {shape_of(x)}")
    for idx, (upsample, dec_block, skip) in enumerate(zip(detailer.upsample_layers, detailer.dec_blocks, reversed(skips))):
        x = upsample(x)
        print(f"detailer.upsample_{idx}: {shape_of(x)}")
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            print(f"detailer.align_skip_{idx}: {shape_of(x)}")
        x = torch.cat([x, skip], dim=1)
        print(f"detailer.concat_skip_{idx}: {shape_of(x)}")
        x = dec_block(x)
        print(f"detailer.dec_block_{idx}: {shape_of(x)}")
    x = detailer.out_proj(x)
    print(f"detailer.out_proj: {shape_of(x)}")
    pred_patches = x.view(batch, patch_count, detailer.in_channels, detailer.patch_size, detailer.patch_size)
    print(f"detailer.pred_patches: {shape_of(pred_patches)}")
    return pred_patches


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
    style_global = trace_style_path(model, style_img, style_ref_mask)
    patch_tokens = trace_backbone_path(
        model,
        xt,
        timesteps,
        content_tokens,
        style_global,
    )
    noisy_patches = model._patchify(xt)
    print_header("Patchify")
    print(f"model._patchify(xt): {shape_of(noisy_patches)}")
    pred_patches = trace_detailer_path(model, patch_tokens, noisy_patches)
    pred_flow = model._unpatchify(pred_patches)
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
