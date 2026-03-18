#!/usr/bin/env python3
"""Smoke test for the refactored content+style latent diffusion path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import DiffusionTrainer, StylePretrainTrainer, VAETrainer
from models.source_part_ref_dit import SourcePartRefDiT


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SourcePartRefDiT(
        image_size=128,
        latent_channels=4,
        latent_size=16,
        encoder_patch_size=8,
        encoder_hidden_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        dit_hidden_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_mlp_ratio=2.0,
    )

    batch = {
        "content": torch.randn(2, 1, 128, 128),
        "target": torch.randn(2, 1, 128, 128),
        "style_img": torch.randn(2, 8, 1, 128, 128),
        "style_ref_mask": torch.ones(2, 8),
        "style_img_pos": torch.randn(2, 8, 1, 128, 128),
        "style_ref_mask_pos": torch.ones(2, 8),
    }

    model = model.to(device)
    with torch.no_grad():
        recon, z, mu, logvar = model.vae_forward(batch["target"].to(device), sample_posterior=False)
        assert recon.shape == (2, 1, 128, 128)
        assert z.shape == (2, 4, 16, 16)
        assert mu.shape == (2, 4, 16, 16)
        assert logvar.shape == (2, 4, 16, 16)

        noise_pred = model(
            z,
            torch.tensor([1, 2], device=device),
            batch["content"].to(device),
            style_img=batch["style_img"].to(device),
            style_ref_mask=batch["style_ref_mask"].to(device),
        )
        assert noise_pred.shape == z.shape

    vae_trainer = VAETrainer(
        model,
        device,
        lr=1e-4,
        total_steps=1,
        lambda_rec=1.0,
        lambda_perc=0.0,
        lambda_kl=1e-4,
        log_every_steps=1,
        save_every_steps=None,
    )
    vae_metrics = vae_trainer.train_step(batch)
    assert "loss" in vae_metrics and vae_metrics["loss"] > 0.0

    style_model = SourcePartRefDiT(
        image_size=128,
        latent_channels=4,
        latent_size=16,
        encoder_patch_size=8,
        encoder_hidden_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        dit_hidden_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_mlp_ratio=2.0,
    )
    style_trainer = StylePretrainTrainer(
        style_model,
        device,
        lr=1e-4,
        total_steps=1,
        contrastive_temperature=0.1,
        log_every_steps=1,
        save_every_steps=None,
    )
    style_metrics = style_trainer.train_step(batch)
    assert "loss_ctr" in style_metrics and style_metrics["loss_ctr"] > 0.0

    diffusion_model = SourcePartRefDiT(
        image_size=128,
        latent_channels=4,
        latent_size=16,
        encoder_patch_size=8,
        encoder_hidden_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        dit_hidden_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_mlp_ratio=2.0,
    )
    diffusion_trainer = DiffusionTrainer(
        diffusion_model,
        device,
        lr=1e-4,
        timesteps=20,
        total_steps=1,
        lambda_diff=1.0,
        lambda_rec=0.0,
        lambda_perc=0.0,
        lambda_ctr=0.0,
        style_lr_scale=0.1,
        freeze_vae=False,
        log_every_steps=1,
        save_every_steps=None,
    )
    diffusion_batch = {
        "content": batch["content"],
        "target": batch["target"],
        "style_img": batch["style_img"],
        "style_ref_mask": batch["style_ref_mask"],
    }
    diff_metrics = diffusion_trainer.train_step(diffusion_batch)
    assert "loss_diff" in diff_metrics and diff_metrics["loss_diff"] > 0.0
    assert "loss_ctr" in diff_metrics and diff_metrics["loss_ctr"] == 0.0

    sample = diffusion_trainer.ddim_sample(
        batch["content"][:1],
        style_img=batch["style_img"][:1],
        style_ref_mask=batch["style_ref_mask"][:1],
        num_inference_steps=4,
    )
    assert sample.shape == (1, 1, 128, 128)

    print(f"smoke test passed on {device}")


if __name__ == "__main__":
    main()
