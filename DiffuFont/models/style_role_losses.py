#!/usr/bin/env python3
"""Role-specialization losses for multi-token style conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def aggregate_token_attention(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate per-reference token attention into per-token probability maps."""
    if token_attn.dim() == 5:
        bsz, token_count, ref_count, height, width = token_attn.shape
        if style_ref_mask is None:
            weights = torch.ones((bsz, ref_count), device=token_attn.device, dtype=token_attn.dtype)
        else:
            if style_ref_mask.shape != (bsz, ref_count):
                raise ValueError(
                    f"style_ref_mask must be {(bsz, ref_count)}, got {tuple(style_ref_mask.shape)}"
                )
            weights = style_ref_mask.to(device=token_attn.device, dtype=token_attn.dtype).clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        maps = (token_attn * weights.view(bsz, 1, ref_count, 1, 1)).sum(dim=2)
    elif token_attn.dim() == 4:
        bsz, token_count, height, width = token_attn.shape
        maps = token_attn
    else:
        raise ValueError(f"token_attn must be 4D or 5D, got {tuple(token_attn.shape)}")

    probs = maps.reshape(bsz, token_count, height * width).to(dtype=torch.float32)
    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def _aggregate_style_ink(
    style_img: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if style_img.dim() == 4:
        style_img = style_img.unsqueeze(1)
    if style_img.dim() != 5:
        raise ValueError(f"style_img must be 4D or 5D, got {tuple(style_img.shape)}")
    bsz, ref_count, channels, _, _ = style_img.shape
    if channels != 1:
        raise ValueError(f"style_img must be grayscale, got channels={channels}")

    refs = 0.5 * (1.0 - style_img.to(dtype=torch.float32))
    refs = refs.clamp_(0.0, 1.0)
    if style_ref_mask is None:
        return refs.mean(dim=1)
    if style_ref_mask.shape != (bsz, ref_count):
        raise ValueError(f"style_ref_mask must be {(bsz, ref_count)}, got {tuple(style_ref_mask.shape)}")
    weights = style_ref_mask.to(device=refs.device, dtype=refs.dtype).clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (refs * weights.view(bsz, ref_count, 1, 1, 1)).sum(dim=1)


def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return x
    if (k % 2) == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    radius = (k - 1) * 0.5
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - radius
    kernel = torch.exp(-(coords * coords) / max(2.0 * float(sigma) * float(sigma), 1e-6))
    kernel = kernel / kernel.sum().clamp_min(1e-12)

    channels = int(x.size(1))
    kernel_x = kernel.view(1, 1, 1, k).expand(channels, 1, 1, k)
    kernel_y = kernel.view(1, 1, k, 1).expand(channels, 1, k, 1)
    x = F.conv2d(x, kernel_x, padding=(0, k // 2), groups=channels)
    x = F.conv2d(x, kernel_y, padding=(k // 2, 0), groups=channels)
    return x


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    device = x.device
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    dx = F.conv2d(x, gx, padding=1)
    dy = F.conv2d(x, gy, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def build_attention_role_targets(
    style_img: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    out_hw: tuple[int, int] = (32, 32),
    low_uniform_mix: float = 0.35,
    high_edge_mix: float = 0.50,
) -> torch.Tensor:
    """Build low/mid/high spatial role targets from averaged glyph ink maps."""
    ink = _aggregate_style_ink(style_img, style_ref_mask=style_ref_mask)
    ink = F.interpolate(ink, size=tuple(int(v) for v in out_hw), mode="bilinear", align_corners=False)

    low = _gaussian_blur(ink, kernel_size=11, sigma=3.0)
    if float(low_uniform_mix) > 0.0:
        low = low + float(low_uniform_mix) * low.mean(dim=(-2, -1), keepdim=True)

    mid = _gaussian_blur(ink, kernel_size=5, sigma=1.2)
    high_detail = torch.abs(ink - mid)
    high_edge = _sobel_magnitude(ink)
    high = high_detail + float(high_edge_mix) * high_edge + 0.10 * ink

    targets = torch.cat([low, mid, high], dim=1).clamp_min_(0.0)
    flat = targets.flatten(2)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return flat.view_as(targets)


def attention_role_alignment_loss(
    token_attn: torch.Tensor,
    style_img: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    low_uniform_mix: float = 0.35,
    high_edge_mix: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align low/mid/high attention with broad-body/body-edge role targets."""
    probs = aggregate_token_attention(token_attn, style_ref_mask=style_ref_mask)
    if int(probs.size(1)) != 3:
        raise ValueError(f"attention_role_alignment_loss expects 3 tokens, got {tuple(probs.shape)}")

    height = int(token_attn.size(-2))
    width = int(token_attn.size(-1))
    targets = build_attention_role_targets(
        style_img,
        style_ref_mask=style_ref_mask,
        out_hw=(height, width),
        low_uniform_mix=float(low_uniform_mix),
        high_edge_mix=float(high_edge_mix),
    )
    tgt = targets.flatten(2).clamp_min(1e-12)
    src = probs.clamp_min(1e-12)
    sym_kl = 0.5 * ((src * (src.log() - tgt.log())).sum(dim=-1) + (tgt * (tgt.log() - src.log())).sum(dim=-1))
    return sym_kl.mean(), sym_kl.mean(dim=0), targets


def attention_overlap_margin_loss(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    margin: float = 0.80,
) -> torch.Tensor:
    """Softly penalize token attention maps that are too similar."""
    probs = aggregate_token_attention(token_attn, style_ref_mask=style_ref_mask)
    vecs = F.normalize(probs, dim=-1)
    sim = torch.matmul(vecs, vecs.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(int(sim.size(0)), -1)
    return F.relu(off_diag - float(margin)).pow(2).mean()


def attention_mean_overlap(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean off-diagonal cosine overlap of aggregated token attention maps."""
    probs = aggregate_token_attention(token_attn, style_ref_mask=style_ref_mask)
    vecs = F.normalize(probs, dim=-1)
    sim = torch.matmul(vecs, vecs.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_select(~eye).view(int(sim.size(0)), -1).mean()


def attention_entropy_order_loss(
    token_attn: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    gap: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encourage low/mid/high tokens to follow coarse-to-fine entropy order."""
    probs = aggregate_token_attention(token_attn, style_ref_mask=style_ref_mask)
    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    ent = ent / math.log(float(probs.size(-1)))
    if int(ent.size(1)) != 3:
        raise ValueError(f"attention_entropy_order_loss expects 3 tokens, got {tuple(ent.shape)}")
    h_low = ent[:, 0]
    h_mid = ent[:, 1]
    h_high = ent[:, 2]
    loss = F.relu(h_mid - h_low + float(gap)) + F.relu(h_high - h_mid + float(gap))
    return loss.mean(), ent.mean(dim=0)
