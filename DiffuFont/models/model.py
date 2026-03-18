#!/usr/bin/env python3
"""Training utilities for the content+style latent DiT path."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()


def _gradient_maps(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return grad_x, grad_y


def glyph_perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scales = (1, 2, 4)
    loss = pred.new_zeros(())
    for scale in scales:
        if scale == 1:
            pred_s = pred
            target_s = target
        else:
            pred_s = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_s = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        gx_pred, gy_pred = _gradient_maps(pred_s)
        gx_tgt, gy_tgt = _gradient_maps(target_s)
        loss = loss + F.l1_loss(pred_s, target_s)
        loss = loss + 0.5 * (F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt))
    return loss / float(len(scales))


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    targets = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits.float(), targets)


def _metrics_to_floats(metrics: Dict[str, torch.Tensor | float | int]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            output[key] = float(value.detach().item())
        else:
            output[key] = float(value)
    return output


class NoiseScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: torch.device) -> "NoiseScheduler":
        for name in ("betas", "alphas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        sqrt_alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        noisy = sqrt_alpha * clean + sqrt_one_minus_alpha * noise
        return noisy, noise

    def predict_start_from_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return (noisy - sqrt_one_minus_alpha * noise) / sqrt_alpha.clamp_min(1e-6)


class _BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float,
        total_steps: int,
        log_every_steps: int,
        save_every_steps: Optional[int],
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        track_best_on_val: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.total_steps = max(1, int(total_steps))
        self.log_every_steps = max(1, int(log_every_steps))
        self.save_every_steps = None if save_every_steps is None else max(1, int(save_every_steps))
        self.val_every_steps = (
            self.log_every_steps if val_every_steps is None else max(1, int(val_every_steps))
        )
        self.val_max_batches = None if val_max_batches is None else max(1, int(val_max_batches))
        self.track_best_on_val = bool(track_best_on_val)
        self.global_step = 0
        self.current_epoch = 0
        self.sample_every_steps: Optional[int] = None
        self.sample_batch: Optional[Dict[str, torch.Tensor]] = None
        self.sample_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.step_log_file: Optional[Path] = None
        self.val_log_file: Optional[Path] = None
        self.best_val_loss: Optional[float] = None

        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found.")
        self.optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.05)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _write_step_log(self, row: Dict[str, float | int | str]) -> None:
        if self.step_log_file is None:
            return
        with self.step_log_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_val_log(self, row: Dict[str, float | int | str]) -> None:
        if self.val_log_file is None:
            return
        with self.val_log_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _optimizer_lr_metrics(self) -> Dict[str, float]:
        return {
            f"lr_group_{idx}": float(group["lr"])
            for idx, group in enumerate(self.optimizer.param_groups)
        }

    def _memory_metrics(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {}
        torch.cuda.synchronize(self.device)
        return {
            "cuda_max_mem_allocated_gb": float(torch.cuda.max_memory_allocated(self.device) / (1024**3)),
            "cuda_mem_allocated_gb": float(torch.cuda.memory_allocated(self.device) / (1024**3)),
            "cuda_mem_reserved_gb": float(torch.cuda.memory_reserved(self.device) / (1024**3)),
        }

    def evaluate(self, dataloader) -> Dict[str, float]:
        start = time.time()
        sums: Dict[str, float] = {}
        count = 0
        for batch_idx, batch in enumerate(dataloader):
            if self.val_max_batches is not None and batch_idx >= self.val_max_batches:
                break
            metrics = self.eval_step(batch)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        if count == 0:
            raise RuntimeError("Validation dataloader produced no batches.")
        averaged = {key: value / float(count) for key, value in sums.items()}
        averaged["val_elapsed_sec"] = time.time() - start
        averaged["val_batches"] = float(count)
        return averaged

    def fit(
        self,
        dataloader,
        *,
        epochs: int,
        save_dir: str | Path,
        val_dataloader=None,
    ) -> None:
        save_root = Path(save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = save_root
        self.step_log_file = save_root / "train_step_metrics.jsonl"
        self.val_log_file = save_root / "val_step_metrics.jsonl"
        if self.global_step == 0:
            self.step_log_file.write_text("", encoding="utf-8")
            self.val_log_file.write_text("", encoding="utf-8")

        for epoch in range(1, int(epochs) + 1):
            self.current_epoch = epoch
            for batch in dataloader:
                if self.global_step >= self.total_steps:
                    print(f"[fit] reached total_steps={self.total_steps}, stopping at epoch={epoch}")
                    return
                step_start = time.time()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                metrics = self.train_step(batch)
                step_time = time.time() - step_start
                self.global_step += 1
                if self.global_step % self.log_every_steps == 0:
                    train_row = {
                        "step": int(self.global_step),
                        "epoch": int(self.current_epoch),
                        "step_time_sec": float(step_time),
                        **self._optimizer_lr_metrics(),
                        **self._memory_metrics(),
                        **{k: float(v) for k, v in metrics.items()},
                    }
                    self._write_step_log(train_row)
                    metric_str = " ".join(
                        f"{key}={value:.4f}"
                        for key, value in train_row.items()
                        if key not in {"step", "epoch"}
                    )
                    print(f"[step] step={self.global_step} epoch={self.current_epoch} {metric_str}", flush=True)

                if val_dataloader is not None and self.global_step % self.val_every_steps == 0:
                    val_metrics = self.evaluate(val_dataloader)
                    val_row: Dict[str, float | int] = {
                        "step": int(self.global_step),
                        "epoch": int(self.current_epoch),
                        **val_metrics,
                    }
                    is_best = False
                    if self.track_best_on_val and "loss" in val_metrics:
                        current_val_loss = float(val_metrics["loss"])
                        if self.best_val_loss is None or current_val_loss < self.best_val_loss:
                            self.best_val_loss = current_val_loss
                            is_best = True
                            if self.checkpoint_dir is not None:
                                best_path = self.checkpoint_dir / "best.pt"
                                self.save(best_path)
                                (self.checkpoint_dir / "best_val_metrics.json").write_text(
                                    json.dumps(
                                        {
                                            "step": int(self.global_step),
                                            "epoch": int(self.current_epoch),
                                            **{k: float(v) for k, v in val_metrics.items()},
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                        sort_keys=True,
                                    ),
                                    encoding="utf-8",
                                )
                    val_row["is_best"] = int(is_best)
                    self._write_val_log(val_row)
                    val_metric_str = " ".join(
                        f"{key}={value:.4f}"
                        for key, value in val_row.items()
                        if key not in {"step", "epoch", "is_best"}
                    )
                    best_suffix = " is_best=1" if is_best else ""
                    print(f"[val] step={self.global_step} epoch={self.current_epoch} {val_metric_str}{best_suffix}", flush=True)

                if (
                    self.sample_every_steps is not None
                    and self.sample_batch is not None
                    and self.sample_dir is not None
                    and self.global_step % self.sample_every_steps == 0
                ):
                    self.sample_and_save(self.sample_batch, self.sample_dir)

                if (
                    self.save_every_steps is not None
                    and self.checkpoint_dir is not None
                    and self.global_step % self.save_every_steps == 0
                ):
                    self.save(self.checkpoint_dir / f"ckpt_step_{self.global_step}.pt")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        raise NotImplementedError


class VAETrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        lambda_rec: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_kl: float = 1e-4,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
    ) -> None:
        super().__init__(
            model,
            device,
            lr=lr,
            total_steps=total_steps,
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            track_best_on_val=True,
        )
        self.lambda_rec = float(lambda_rec)
        self.lambda_perc = float(lambda_perc)
        self.lambda_kl = float(lambda_kl)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = batch["target"].to(self.device)
        with self._autocast_context():
            recon, _, mu, logvar = self.model.vae_forward(target, sample_posterior=True)
            loss_rec = F.l1_loss(recon, target)
            loss_perc = glyph_perceptual_loss(recon, target) if self.lambda_perc > 0.0 else target.new_zeros(())
            loss_kl = kl_divergence_loss(mu, logvar)
            loss = self.lambda_rec * loss_rec + self.lambda_perc * loss_perc + self.lambda_kl * loss_kl
        return {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_perc": loss_perc,
            "loss_kl": loss_kl,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        metrics["loss"].backward()
        self.optimizer.step()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return _metrics_to_floats(self._compute_losses(batch))

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "vae",
                "vae_state": self.model.vae.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_vae_checkpoint(path)
        if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.global_step = int(checkpoint.get("step", 0))
            self.current_epoch = int(checkpoint.get("epoch", 0))

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = batch["target"][:8].to(self.device)
        self.model.eval()
        recon, _, _, _ = self.model.vae_forward(target, sample_posterior=False)
        vis = torch.cat([(target + 1.0) * 0.5, (recon + 1.0) * 0.5], dim=0)
        save_image(vis, out_dir / f"vae_step_{self.global_step}.png", nrow=target.size(0))


class StylePretrainTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        contrastive_temperature: float = 0.1,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
    ) -> None:
        super().__init__(
            model,
            device,
            lr=lr,
            total_steps=total_steps,
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            track_best_on_val=True,
        )
        self.contrastive_temperature = float(contrastive_temperature)
        style_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and self.model._is_style_state_key(name)
        ]
        if not style_params:
            raise RuntimeError("No style parameters found for style pretraining.")
        self.optimizer = torch.optim.AdamW(style_params, lr=float(lr), weight_decay=0.05)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        style = batch["style_img"].to(self.device)
        style_pos = batch["style_img_pos"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        style_ref_mask_pos = batch.get("style_ref_mask_pos")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)
        if style_ref_mask_pos is not None:
            style_ref_mask_pos = style_ref_mask_pos.to(self.device)

        with self._autocast_context():
            _, _, _, anchor_style_embed = self.model.encode_style(
                style_img=style,
                style_ref_mask=style_ref_mask,
                return_contrastive=True,
            )
            _, _, _, positive_style_embed = self.model.encode_style(
                style_img=style_pos,
                style_ref_mask=style_ref_mask_pos,
                return_contrastive=True,
            )
            if anchor_style_embed is None or positive_style_embed is None:
                raise RuntimeError("Style contrastive embeddings were not produced.")
            loss_ctr = info_nce_loss(anchor_style_embed, positive_style_embed, self.contrastive_temperature)
        return {
            "loss": loss_ctr,
            "loss_ctr": loss_ctr,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        metrics["loss"].backward()
        self.optimizer.step()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return _metrics_to_floats(self._compute_losses(batch))

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "style",
                "style_state": self.model.style_state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_style_checkpoint(path)
        if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.global_step = int(checkpoint.get("step", 0))
            self.current_epoch = int(checkpoint.get("epoch", 0))

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        return


class DiffusionTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        timesteps: int = 1000,
        total_steps: int = 100_000,
        lambda_diff: float = 1.0,
        lambda_rec: float = 0.0,
        lambda_perc: float = 0.0,
        lambda_kl: float = 0.0,
        lambda_ctr: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_warmup_steps: int = 5000,
        style_lr_scale: float = 0.1,
        style_unfreeze_step: int = 0,
        style_unfreeze_last_encoder_blocks: int = 1,
        freeze_vae: bool = True,
        freeze_style: bool = True,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
    ) -> None:
        if freeze_vae:
            model.freeze_vae()
        if freeze_style:
            model.freeze_style()
        super().__init__(
            model,
            device,
            lr=lr,
            total_steps=total_steps,
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            track_best_on_val=False,
        )
        self.scheduler = NoiseScheduler(int(timesteps)).to(device)
        self.lambda_diff = float(lambda_diff)
        self.lambda_rec = float(lambda_rec)
        self.lambda_perc = float(lambda_perc)
        self.lambda_kl = float(lambda_kl)
        self.lambda_ctr = float(lambda_ctr)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_warmup_steps = max(0, int(contrastive_warmup_steps))
        self.base_lr = float(lr)
        self.freeze_vae = bool(freeze_vae)
        self.freeze_style = bool(freeze_style)
        self.style_lr_scale = max(0.0, float(style_lr_scale))
        self.style_unfreeze_step = max(0, int(style_unfreeze_step))
        self.style_unfreeze_last_encoder_blocks = max(0, int(style_unfreeze_last_encoder_blocks))
        self.style_finetune_active = not self.freeze_style
        self.style_grad_enabled = not self.freeze_style
        self.style_finetune_param_names: list[str] = []
        style_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.model._is_style_state_key(name):
                style_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": float(lr)})
        if style_params:
            param_groups.append({"params": style_params, "lr": self.base_lr * self.style_lr_scale})
        if not param_groups:
            raise RuntimeError("No trainable parameters found for diffusion training.")
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.base_lr, weight_decay=0.05)

    def _maybe_activate_partial_style_finetune(self, *, step_for_state: Optional[int] = None) -> None:
        if not self.freeze_style or self.style_finetune_active:
            return
        current_step = self.global_step if step_for_state is None else int(step_for_state)
        if current_step < self.style_unfreeze_step:
            return
        named_params = self.model.enable_partial_style_finetune(
            train_last_encoder_blocks=self.style_unfreeze_last_encoder_blocks,
        )
        if not named_params:
            return
        self.optimizer.add_param_group(
            {
                "params": [param for _, param in named_params],
                "lr": self.base_lr * self.style_lr_scale,
            }
        )
        self.style_finetune_active = True
        self.style_grad_enabled = True
        self.style_finetune_param_names = [name for name, _ in named_params]
        preview_names = ", ".join(self.style_finetune_param_names[:4])
        if len(self.style_finetune_param_names) > 4:
            preview_names = f"{preview_names}, ..."
        print(
            "[diffusion] enabled partial style finetuning "
            f"at step={current_step} lr={self.base_lr * self.style_lr_scale:.2e} "
            f"params={len(self.style_finetune_param_names)} [{preview_names}]",
            flush=True,
        )

    def _current_contrastive_weight(self) -> float:
        if self.lambda_ctr <= 0.0:
            return 0.0
        if self.contrastive_warmup_steps <= 0:
            return self.lambda_ctr
        if self.global_step + 1 >= self.contrastive_warmup_steps:
            return self.lambda_ctr
        alpha = float(self.global_step + 1) / float(self.contrastive_warmup_steps)
        return alpha * self.lambda_ctr

    def _encode_latent(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_vae:
            with torch.no_grad():
                z = self.model.encode_to_latent(target, sample_posterior=False)
            zeros = torch.zeros_like(z)
            return z, zeros, zeros
        z, mu, logvar = self.model.encode_to_latent(target, sample_posterior=True, return_stats=True)
        return z, mu, logvar

    def _encode_style_conditions(
        self,
        style: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.style_grad_enabled:
            with torch.no_grad():
                return self.model.encode_style(
                    style_img=style,
                    style_ref_mask=style_ref_mask,
                    return_contrastive=(self.lambda_ctr > 0.0),
                )
        return self.model.encode_style(
            style_img=style,
            style_ref_mask=style_ref_mask,
            return_contrastive=(self.lambda_ctr > 0.0),
        )

    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | float]:
        target = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style = batch["style_img"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        with self._autocast_context():
            z0, mu, logvar = self._encode_latent(target)
            timesteps = torch.randint(0, self.scheduler.timesteps, (z0.size(0),), device=self.device)
            zt, noise = self.scheduler.add_noise(z0, timesteps)
            content_tokens = self.model.encode_content(content)
            style_tokens, style_global, style_token_mask, anchor_style_embed = self._encode_style_conditions(
                style,
                style_ref_mask,
            )
            noise_pred = self.model.predict_noise(
                zt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_tokens,
                style_global=style_global,
                style_token_mask=style_token_mask,
            )
            loss_diff = F.mse_loss(noise_pred, noise)

            loss_rec = target.new_zeros(())
            loss_perc = target.new_zeros(())
            if self.lambda_rec > 0.0 or self.lambda_perc > 0.0:
                z0_pred = self.scheduler.predict_start_from_noise(zt, timesteps, noise_pred).clamp(-4.0, 4.0)
                recon = self.model.decode_from_latent(z0_pred)
                if self.lambda_rec > 0.0:
                    loss_rec = F.l1_loss(recon, target)
                if self.lambda_perc > 0.0:
                    loss_perc = glyph_perceptual_loss(recon, target)

            loss_kl = kl_divergence_loss(mu, logvar) if self.lambda_kl > 0.0 and not self.freeze_vae else target.new_zeros(())
            loss_ctr = target.new_zeros(())
            active_lambda_ctr = self._current_contrastive_weight()
            if active_lambda_ctr > 0.0:
                style_pos = batch.get("style_img_pos")
                style_ref_mask_pos = batch.get("style_ref_mask_pos")
                if style_pos is None:
                    raise RuntimeError("Diffusion contrastive loss requires style_img_pos in the batch.")
                style_pos = style_pos.to(self.device)
                if style_ref_mask_pos is not None:
                    style_ref_mask_pos = style_ref_mask_pos.to(self.device)
                _, _, _, positive_style_embed = self._encode_style_conditions(
                    style_pos,
                    style_ref_mask_pos,
                )
                if anchor_style_embed is None or positive_style_embed is None:
                    raise RuntimeError("Diffusion contrastive loss requires style embeddings.")
                loss_ctr = info_nce_loss(anchor_style_embed, positive_style_embed, self.contrastive_temperature)
            loss = (
                self.lambda_diff * loss_diff
                + self.lambda_rec * loss_rec
                + self.lambda_perc * loss_perc
                + self.lambda_kl * loss_kl
                + active_lambda_ctr * loss_ctr
            )
        return {
            "loss": loss,
            "loss_diff": loss_diff,
            "loss_rec": loss_rec,
            "loss_perc": loss_perc,
            "loss_kl": loss_kl,
            "loss_ctr": loss_ctr,
            "lambda_ctr": active_lambda_ctr,
            "style_finetune_active": float(self.style_finetune_active),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._maybe_activate_partial_style_finetune()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        metrics["loss"].backward()
        self.optimizer.step()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._maybe_activate_partial_style_finetune()
        self.model.eval()
        with torch.no_grad():
            return _metrics_to_floats(self._compute_losses(batch))

    @torch.no_grad()
    def ddim_sample(
        self,
        content: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        self.model.eval()
        content = content.to(self.device)
        style_img = style_img.to(self.device)
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        batch_size = content.size(0)
        sample = torch.randn(
            batch_size,
            self.model.latent_channels,
            self.model.latent_size,
            self.model.latent_size,
            device=self.device,
        )
        step_count = max(1, int(num_inference_steps))
        step_indices = torch.linspace(
            self.scheduler.timesteps - 1,
            0,
            steps=step_count,
            device=self.device,
        ).long()

        for idx, timestep in enumerate(step_indices):
            t = torch.full((batch_size,), int(timestep.item()), device=self.device, dtype=torch.long)
            noise_pred = self.model(
                sample,
                t,
                content,
                style_img=style_img,
                style_ref_mask=style_ref_mask,
            )
            x0_pred = self.scheduler.predict_start_from_noise(sample, t, noise_pred).clamp(-4.0, 4.0)

            if idx == len(step_indices) - 1:
                sample = x0_pred
                continue

            prev_timestep = step_indices[idx + 1]
            alpha_prev = self.scheduler.alpha_bars[prev_timestep].view(1, 1, 1, 1)
            sample = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * noise_pred

        return self.model.decode_from_latent(sample).clamp(-1.0, 1.0)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "diffusion",
                "model_state": self.model.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state" not in checkpoint:
            raise RuntimeError("Diffusion checkpoint is missing 'model_state'.")
        load_report = self.model.load_state_dict_compat(checkpoint["model_state"])
        if load_report["resized_keys"]:
            print("[diffusion] resized checkpoint tensors: " + ", ".join(load_report["resized_keys"]), flush=True)
        if load_report["skipped_unknown_keys"] or load_report["skipped_incompatible_keys"]:
            raise RuntimeError(
                "Diffusion checkpoint state mismatch before load: "
                f"unknown={load_report['skipped_unknown_keys']} "
                f"incompatible={load_report['skipped_incompatible_keys']}"
            )
        allowed_missing_keys = {
            name
            for name, _ in self.model.named_parameters()
            if name.endswith("style_residual_gate")
        }
        disallowed_missing = [key for key in load_report["missing_keys"] if key not in allowed_missing_keys]
        if disallowed_missing or load_report["unexpected_keys"]:
            raise RuntimeError(
                "Diffusion checkpoint state mismatch: "
                f"missing={disallowed_missing} unexpected={load_report['unexpected_keys']}"
            )
        resume_step = int(checkpoint.get("step", 0))
        self._maybe_activate_partial_style_finetune(step_for_state=resume_step)
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[diffusion] skipped optimizer state load: {exc}", flush=True)
        self.global_step = resume_step
        self.current_epoch = int(checkpoint.get("epoch", 0))

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"][:8].to(self.device)
        target = batch["target"][:8].to(self.device)
        style = batch["style_img"][:8].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask[:8].to(self.device)
        sample = self.ddim_sample(
            content,
            style_img=style,
            style_ref_mask=style_ref_mask,
            num_inference_steps=20,
        )
        vis = torch.cat([(content + 1.0) * 0.5, (target + 1.0) * 0.5, (sample + 1.0) * 0.5], dim=0)
        save_image(vis, out_dir / f"sample_step_{self.global_step}.png", nrow=content.size(0))
