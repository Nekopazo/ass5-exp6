#!/usr/bin/env python3
"""Minimal smoke test for the 3-token style routing path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.source_part_ref_unet import SourcePartRefUNet


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = SourcePartRefUNet(
        in_channels=1,
        image_size=256,
        content_start_channel=64,
        style_start_channel=8,
        unet_channels=(32, 64, 128, 256),
        conditioning_profile="style_only",
        attn_scales=(16, 32, 64),
        style_token_dim=256,
        style_token_count=3,
    ).cpu()
    model.eval()

    assert all(not block.enable_style_attn for block in model.unet.down_blocks)
    assert model.unet.up_block_style_keys == ["up_16", "up_32", "up_64", None]

    batch = 1
    refs = 2
    x_t = torch.randn(batch, 1, 128, 128)
    content = torch.randn(batch, 1, 128, 128)
    style = torch.randn(batch, refs, 1, 128, 128)
    mask = torch.tensor([[1, 1]], dtype=torch.float32)
    timesteps = torch.tensor([3], dtype=torch.long)

    with torch.no_grad():
        tokens, proxy = model.encode_style_tokens_with_proxy(style, mask)
        contexts = model._resolve_style_hidden_states("style_only", style, mask)
        out = model(
            x_t,
            timesteps,
            content,
            style_img=style,
            condition_mode="style_only",
            style_ref_mask=mask,
        )

    assert tokens.shape == (batch, 3, 256)
    assert out.shape == x_t.shape
    assert contexts is not None
    assert set(contexts.keys()) == {"mid", "up_16", "up_32", "up_64"}
    for value in contexts.values():
        assert value.shape == (batch, 1, 256)
    for key in ("pred_low", "pred_mid", "pred_high", "target_low", "target_mid", "target_high"):
        assert key in proxy
        assert proxy[key].shape[0] == batch

    print("smoke test passed")


if __name__ == "__main__":
    main()
