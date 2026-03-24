#!/usr/bin/env python3
"""Training utilities for the content+style latent DiT path."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).mean(dim=1)


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


def glyph_perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    scales = (1, 2, 4)
    losses = []
    for scale in scales:
        if scale == 1:
            pred_s = pred
            target_s = target
        else:
            pred_s = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_s = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        gx_pred, gy_pred = _gradient_maps(pred_s)
        gx_tgt, gy_tgt = _gradient_maps(target_s)
        scale_loss = _per_sample_mean((pred_s - target_s).abs())
        scale_loss = scale_loss + 0.5 * (
            _per_sample_mean((gx_pred - gx_tgt).abs()) + _per_sample_mean((gy_pred - gy_tgt).abs())
        )
        losses.append(scale_loss)
    loss = torch.stack(losses, dim=0).mean(dim=0)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


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


def latent_regularization_losses(
    mu: torch.Tensor,
    *,
    target_std: float,
) -> Dict[str, torch.Tensor]:
    mu_f = mu.float()
    flat = mu_f.permute(1, 0, 2, 3).reshape(mu_f.size(1), -1)
    channel_mean = flat.mean(dim=1)
    channel_std = flat.std(dim=1, unbiased=False)
    loss_mean = channel_mean.pow(2).mean()
    loss_std = (channel_std - float(target_std)).pow(2).mean()

    if flat.size(0) > 1 and flat.size(1) > 1:
        centered = flat - channel_mean.unsqueeze(1)
        cov = centered @ centered.transpose(0, 1) / float(flat.size(1))
        var = cov.diag().clamp_min(1e-8)
        denom = torch.sqrt(torch.outer(var, var)).clamp_min(1e-8)
        corr = cov / denom
        offdiag_mask = ~torch.eye(corr.size(0), device=corr.device, dtype=torch.bool)
        loss_corr = corr[offdiag_mask].pow(2).mean()
    else:
        loss_corr = mu_f.new_zeros(())

    return {
        "loss_latent_mean": loss_mean,
        "loss_latent_std": loss_std,
        "loss_latent_corr": loss_corr,
    }


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

        stop_training = False
        for epoch in range(1, int(epochs) + 1):
            self.current_epoch = epoch
            self.on_epoch_start()
            for batch in dataloader:
                if self.global_step >= self.total_steps:
                    stop_training = True
                    break
                step_start = time.time()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                metrics = self.train_step(batch)
                step_time = time.time() - step_start
                self.global_step += 1
                self.on_after_train_step(batch, metrics)
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
                if self.global_step >= self.total_steps:
                    stop_training = True
                    break
            self.on_epoch_end()
            if stop_training:
                print(f"[fit] reached total_steps={self.total_steps}, stopping at epoch={epoch}")
                return

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

    def on_epoch_start(self) -> None:
        return None

    def on_after_train_step(self, batch: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> None:
        return None

    def on_epoch_end(self) -> None:
        return None


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
        lambda_kl: float = 2e-4,
        kl_warmup_steps: int = 10_000,
        latent_mean_weight: float = 1e-3,
        latent_std_weight: float = 1e-3,
        latent_corr_weight: float = 5e-4,
        latent_std_target: float = 1.0,
        difficulty_sampler=None,
        difficulty_warmup_steps: int = 2000,
        difficulty_ema_decay: float = 0.95,
        difficulty_alpha: float = 1.0,
        difficulty_min_weight: float = 0.5,
        difficulty_max_weight: float = 3.0,
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
        self.kl_warmup_steps = max(0, int(kl_warmup_steps))
        self.latent_mean_weight = max(0.0, float(latent_mean_weight))
        self.latent_std_weight = max(0.0, float(latent_std_weight))
        self.latent_corr_weight = max(0.0, float(latent_corr_weight))
        self.latent_std_target = float(latent_std_target)
        self.difficulty_sampler = difficulty_sampler
        self.difficulty_warmup_steps = max(0, int(difficulty_warmup_steps))
        self.difficulty_ema_decay = float(difficulty_ema_decay)
        self.difficulty_alpha = max(0.0, float(difficulty_alpha))
        self.difficulty_min_weight = max(0.0, float(difficulty_min_weight))
        self.difficulty_max_weight = max(self.difficulty_min_weight, float(difficulty_max_weight))
        self.font_difficulty_ema: Dict[str, float] = {}
        self.font_observation_count: Dict[str, int] = {}
        self.latent_stats_count = 0
        self.latent_stats_sum: Optional[torch.Tensor] = None
        self.latent_stats_sq_sum: Optional[torch.Tensor] = None

    def _effective_lambda_kl(self) -> float:
        if self.kl_warmup_steps <= 0:
            return self.lambda_kl
        scale = min(1.0, float(self.global_step + 1) / float(self.kl_warmup_steps))
        return self.lambda_kl * scale

    def _update_latent_norm_stats(self, mu: torch.Tensor) -> None:
        mu_f = mu.detach().float()
        flat = mu_f.permute(1, 0, 2, 3).reshape(mu_f.size(1), -1).double()
        batch_sum = flat.sum(dim=1)
        batch_sq_sum = flat.square().sum(dim=1)
        batch_count = int(flat.size(1))
        if self.latent_stats_sum is None or self.latent_stats_sq_sum is None:
            self.latent_stats_sum = torch.zeros_like(batch_sum)
            self.latent_stats_sq_sum = torch.zeros_like(batch_sq_sum)
        self.latent_stats_sum += batch_sum
        self.latent_stats_sq_sum += batch_sq_sum
        self.latent_stats_count += batch_count
        count = max(1, self.latent_stats_count)
        mean = self.latent_stats_sum / float(count)
        var = self.latent_stats_sq_sum / float(count) - mean.square()
        std = torch.sqrt(var.clamp_min(1e-6))
        self.model.set_latent_norm_stats(
            mean.view(1, -1, 1, 1).to(device=self.device, dtype=torch.float32),
            std.view(1, -1, 1, 1).to(device=self.device, dtype=torch.float32),
        )

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        target = batch["target"].to(self.device)
        with self._autocast_context():
            recon, _, mu, logvar = self.model.vae_forward(target, sample_posterior=True)
            loss_rec_per_sample = _per_sample_mean((recon - target).abs())
            loss_rec = loss_rec_per_sample.mean()
            if self.lambda_perc > 0.0:
                loss_perc_per_sample = glyph_perceptual_loss(recon, target, reduction="none")
                loss_perc = loss_perc_per_sample.mean()
            else:
                loss_perc_per_sample = target.new_zeros((target.size(0),))
                loss_perc = target.new_zeros(())
            loss_kl = kl_divergence_loss(mu, logvar)
            latent_reg = latent_regularization_losses(mu, target_std=self.latent_std_target)
            effective_lambda_kl = self._effective_lambda_kl()
            loss = self.lambda_rec * loss_rec + self.lambda_perc * loss_perc + effective_lambda_kl * loss_kl
            loss = loss + self.latent_mean_weight * latent_reg["loss_latent_mean"]
            loss = loss + self.latent_std_weight * latent_reg["loss_latent_std"]
            loss = loss + self.latent_corr_weight * latent_reg["loss_latent_corr"]
        metrics = {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_perc": loss_perc,
            "loss_kl": loss_kl,
            "lambda_kl_eff": target.new_tensor(effective_lambda_kl),
            **latent_reg,
            "latent_norm_mean_abs": mu.mean(dim=(0, 2, 3)).abs().mean(),
            "latent_norm_std_mean": mu.permute(1, 0, 2, 3).reshape(mu.size(1), -1).std(dim=1, unbiased=False).mean(),
        }
        if return_aux:
            metrics["difficulty_per_sample"] = (
                self.lambda_rec * loss_rec_per_sample.detach()
                + self.lambda_perc * loss_perc_per_sample.detach()
            )
            metrics["latent_mu_detached"] = mu.detach()
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch, return_aux=True)
        difficulty_per_sample = metrics.pop("difficulty_per_sample", None)
        latent_mu = metrics.pop("latent_mu_detached", None)
        metrics["loss"].backward()
        self.optimizer.step()
        if latent_mu is not None:
            self._update_latent_norm_stats(latent_mu)
        if difficulty_per_sample is not None:
            self._update_font_difficulty_ema(batch["font"], difficulty_per_sample)
            metrics["difficulty_mean"] = difficulty_per_sample.mean()
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
                "latent_norm_state": self.model.latent_norm_state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "latent_norm_accum_state": {
                    "count": int(self.latent_stats_count),
                    "sum": None if self.latent_stats_sum is None else self.latent_stats_sum.cpu(),
                    "sq_sum": None if self.latent_stats_sq_sum is None else self.latent_stats_sq_sum.cpu(),
                },
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
            accum_state = checkpoint.get("latent_norm_accum_state")
            if isinstance(accum_state, dict):
                self.latent_stats_count = int(accum_state.get("count", 0))
                latent_sum = accum_state.get("sum")
                latent_sq_sum = accum_state.get("sq_sum")
                self.latent_stats_sum = None if latent_sum is None else torch.as_tensor(latent_sum, dtype=torch.float64)
                self.latent_stats_sq_sum = (
                    None if latent_sq_sum is None else torch.as_tensor(latent_sq_sum, dtype=torch.float64)
                )

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = batch["target"][:8].to(self.device)
        self.model.eval()
        recon, _, _, _ = self.model.vae_forward(target, sample_posterior=False)
        vis = torch.cat([(target + 1.0) * 0.5, (recon + 1.0) * 0.5], dim=0)
        save_image(vis, out_dir / f"vae_step_{self.global_step}.png", nrow=target.size(0))

    def _update_font_difficulty_ema(self, fonts, difficulty_per_sample: torch.Tensor) -> None:
        difficulty_values = difficulty_per_sample.detach().float().cpu().tolist()
        batch_sums: Dict[str, float] = {}
        batch_counts: Dict[str, int] = {}
        for font_name, score in zip(fonts, difficulty_values):
            batch_sums[font_name] = batch_sums.get(font_name, 0.0) + float(score)
            batch_counts[font_name] = batch_counts.get(font_name, 0) + 1
        ema_keep = self.difficulty_ema_decay
        ema_add = 1.0 - ema_keep
        for font_name, score_sum in batch_sums.items():
            batch_mean = score_sum / float(batch_counts[font_name])
            prev = self.font_difficulty_ema.get(font_name)
            self.font_difficulty_ema[font_name] = batch_mean if prev is None else ema_keep * prev + ema_add * batch_mean
            self.font_observation_count[font_name] = self.font_observation_count.get(font_name, 0) + batch_counts[font_name]

    def on_epoch_end(self) -> None:
        sampler_summary = self._refresh_difficulty_sampler()
        if sampler_summary is None:
            return
        top_fonts = ", ".join(
            f"{item['font']}:{float(item['weight']):.2f}"
            for item in sampler_summary["top_fonts"]
        )
        bottom_fonts = ", ".join(
            f"{item['font']}:{float(item['weight']):.2f}"
            for item in sampler_summary["bottom_fonts"]
        )
        print(
            "[train] refreshed VAE difficulty sampler "
            f"step={self.global_step} epoch={self.current_epoch} "
            f"observed_fonts={int(sampler_summary['observed_font_count'])} "
            f"baseline_difficulty={float(sampler_summary['baseline_difficulty']):.4f} "
            f"weight_range=[{float(sampler_summary['font_weight_min']):.2f},"
            f"{float(sampler_summary['font_weight_max']):.2f}] "
            f"weight_bins(["
            f"{float(sampler_summary['font_weight_bucket_min']):.2f},"
            f"{float(sampler_summary['font_weight_bucket_edge_1']):.2f})="
            f"{int(sampler_summary['font_weight_bucket_count_low'])},"
            f"[{float(sampler_summary['font_weight_bucket_edge_1']):.2f},"
            f"{float(sampler_summary['font_weight_bucket_edge_2']):.2f})="
            f"{int(sampler_summary['font_weight_bucket_count_mid'])},"
            f"[{float(sampler_summary['font_weight_bucket_edge_2']):.2f},"
            f"{float(sampler_summary['font_weight_bucket_max']):.2f}]="
            f"{int(sampler_summary['font_weight_bucket_count_high'])}) "
            f"top={top_fonts} bottom={bottom_fonts}",
            flush=True,
        )

    def _refresh_difficulty_sampler(self) -> Optional[Dict[str, object]]:
        if self.difficulty_sampler is None:
            return None
        if self.global_step < self.difficulty_warmup_steps:
            return None
        if not self.font_difficulty_ema:
            return None

        observed_values = list(self.font_difficulty_ema.values())
        baseline = sum(observed_values) / float(len(observed_values))
        baseline = max(baseline, 1e-8)
        font_weights: Dict[str, float] = {}
        for font_name in self.difficulty_sampler.dataset.font_names:
            difficulty = self.font_difficulty_ema.get(font_name, baseline)
            normalized = max(difficulty / baseline, 1e-8)
            weight = normalized ** self.difficulty_alpha
            weight = min(self.difficulty_max_weight, max(self.difficulty_min_weight, weight))
            font_weights[font_name] = float(weight)

        sampler_summary = self.difficulty_sampler.set_font_weights(font_weights)
        sampler_summary["observed_font_count"] = int(len(self.font_difficulty_ema))
        sampler_summary["baseline_difficulty"] = float(baseline)
        sampler_summary["warmup_steps"] = int(self.difficulty_warmup_steps)
        sampler_summary["ema_decay"] = float(self.difficulty_ema_decay)
        sampler_summary["alpha"] = float(self.difficulty_alpha)
        sampler_summary["min_weight"] = float(self.difficulty_min_weight)
        sampler_summary["max_weight"] = float(self.difficulty_max_weight)
        return sampler_summary


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
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.model.dit_hidden_dim, self.model.dit_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.model.dit_hidden_dim, self.model.contrastive_proj_dim),
        ).to(device)
        style_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and self.model._is_global_style_state_key(name)
        ]
        head_params = list(self.contrastive_head.parameters())
        if not style_params:
            raise RuntimeError("No global style parameters found for style pretraining.")
        self.optimizer = torch.optim.AdamW(style_params + head_params, lr=float(lr), weight_decay=0.05)

    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        style = batch["style_img"].to(self.device)
        style_pos = batch["style_img_pos"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        style_ref_mask_pos = batch.get("style_ref_mask_pos")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)
        if style_ref_mask_pos is not None:
            style_ref_mask_pos = style_ref_mask_pos.to(self.device)

        combined_style = torch.cat([style, style_pos], dim=0)
        combined_ref_mask = None
        if style_ref_mask is not None and style_ref_mask_pos is not None:
            combined_ref_mask = torch.cat([style_ref_mask, style_ref_mask_pos], dim=0)

        with self._autocast_context():
            combined_style_pack = self.model.encode_style(
                style_img=combined_style,
                style_ref_mask=combined_ref_mask,
                need_style_tokens=False,
                need_style_global=True,
                detach_global_style_encoder=False,
                detach_global_style=False,
            )
            combined_style_global = combined_style_pack["style_global"]
            if combined_style_global is None:
                raise RuntimeError("Global style embeddings were not produced.")
            anchor_style_global, positive_style_global = combined_style_global.chunk(2, dim=0)
            if anchor_style_global is None or positive_style_global is None:
                raise RuntimeError("Global style embeddings were not produced.")
            anchor_style_embed = F.normalize(self.contrastive_head(anchor_style_global), dim=-1)
            positive_style_embed = F.normalize(self.contrastive_head(positive_style_global), dim=-1)
            loss_ctr = info_nce_loss(anchor_style_embed, positive_style_embed, self.contrastive_temperature)
        return {
            "loss": loss_ctr,
            "loss_ctr": loss_ctr,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.contrastive_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        metrics["loss"].backward()
        self.optimizer.step()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        self.contrastive_head.eval()
        with torch.no_grad():
            return _metrics_to_floats(self._compute_losses(batch))

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "style",
                "global_style_state": self.model.global_style_state_dict(),
                "contrastive_head_state": self.contrastive_head.state_dict(),
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
        if "contrastive_head_state" not in checkpoint:
            raise RuntimeError("Style checkpoint is missing 'contrastive_head_state'.")
        self.contrastive_head.load_state_dict(checkpoint["contrastive_head_state"], strict=True)
        if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.global_step = int(checkpoint.get("step", 0))
            self.current_epoch = int(checkpoint.get("epoch", 0))

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        return


class FlowTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        lambda_flow: float = 1.0,
        lambda_img_l1: float = 0.2,
        lambda_img_perc: float = 0.02,
        freeze_vae: bool = True,
        freeze_style_global: bool = True,
        style_attn_probe_enabled: bool = True,
        style_attn_probe_every_steps: int = 1000,
        flow_sample_steps: int = 24,
        lr_warmup_steps: int = 0,
        lr_min_scale: float = 1.0,
        difficulty_sampler=None,
        difficulty_warmup_steps: int = 10_000,
        difficulty_ema_decay: float = 0.99,
        difficulty_alpha: float = 0.5,
        difficulty_min_weight: float = 0.7,
        difficulty_max_weight: float = 1.5,
        difficulty_refresh_every_steps: int = 1000,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
    ) -> None:
        if freeze_vae:
            model.freeze_vae()
        if freeze_style_global:
            model.freeze_style_global()
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
        self.lambda_flow = float(lambda_flow)
        self.lambda_img_l1 = max(0.0, float(lambda_img_l1))
        self.lambda_img_perc = max(0.0, float(lambda_img_perc))
        self.base_lr = float(lr)
        self.freeze_vae = bool(freeze_vae)
        self.freeze_style_global = bool(freeze_style_global)
        self.style_attn_probe_enabled = bool(style_attn_probe_enabled)
        self.style_attn_probe_every_steps = max(1, int(style_attn_probe_every_steps))
        self.style_attn_probe_batch: Optional[Dict[str, torch.Tensor]] = None
        self.style_attn_probe_font_source: Optional[str] = None
        self.style_attn_probe_log_file: Optional[Path] = None
        self.flow_sample_steps = max(1, int(flow_sample_steps))
        self.lr_warmup_steps = max(0, int(lr_warmup_steps))
        self.lr_min_scale = float(min(max(0.0, float(lr_min_scale)), 1.0))
        self.difficulty_sampler = difficulty_sampler
        self.difficulty_warmup_steps = max(0, int(difficulty_warmup_steps))
        self.difficulty_ema_decay = float(difficulty_ema_decay)
        self.difficulty_alpha = max(0.0, float(difficulty_alpha))
        self.difficulty_min_weight = max(0.0, float(difficulty_min_weight))
        self.difficulty_max_weight = max(self.difficulty_min_weight, float(difficulty_max_weight))
        self.difficulty_refresh_every_steps = max(1, int(difficulty_refresh_every_steps))
        self.font_difficulty_ema: Dict[str, float] = {}
        self.font_observation_count: Dict[str, int] = {}
        self.group_base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self._set_learning_rate_for_step(0)

    def _lr_scale_for_step(self, step: int) -> float:
        step = max(0, int(step))
        if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            return float(step + 1) / float(self.lr_warmup_steps)
        if self.total_steps <= self.lr_warmup_steps:
            return 1.0
        progress = float(step - self.lr_warmup_steps) / float(max(1, self.total_steps - self.lr_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min_scale + (1.0 - self.lr_min_scale) * cosine

    def _set_learning_rate_for_step(self, step: int) -> None:
        lr_scale = self._lr_scale_for_step(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.group_base_lrs):
            group["lr"] = float(base_lr) * lr_scale

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
    ) -> dict[str, Optional[torch.Tensor]]:
        return self.model.encode_style(
            style_img=style,
            style_ref_mask=style_ref_mask,
            need_style_tokens=True,
            need_style_global=True,
            detach_global_style_encoder=self.freeze_style_global,
            detach_global_style=self.freeze_style_global,
        )

    def _write_style_attn_probe_log(self, row: Dict[str, object]) -> None:
        if self.checkpoint_dir is None:
            return
        if self.style_attn_probe_log_file is None:
            self.style_attn_probe_log_file = self.checkpoint_dir / "style_attn_probe_metrics.jsonl"
        with self.style_attn_probe_log_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _summarize_style_attn_probe(
        self,
        *,
        style_attn_weights: list[dict[str, torch.Tensor] | None],
        style_token_mask: Optional[torch.Tensor],
        fonts: list[str],
        refs: int,
        tokens_per_ref: int,
    ) -> Dict[str, object]:
        if style_token_mask is None:
            style_token_mask = torch.ones(
                (len(fonts), refs, tokens_per_ref),
                device=self.device,
                dtype=torch.bool,
            )
        style_token_mask = style_token_mask.detach().to(dtype=torch.bool, device=self.device).cpu()
        flat_style_token_mask = style_token_mask.reshape(style_token_mask.size(0), -1)
        valid_token_counts = flat_style_token_mask.sum(dim=1).clamp(min=1)
        block_rows: list[Dict[str, object]] = []
        brief_parts: list[str] = []
        for block_idx, attn_info in enumerate(style_attn_weights):
            if attn_info is None:
                continue
            ref_weights = attn_info["ref_weights"].detach().float().cpu()
            token_weights = attn_info["token_weights"].detach().float().cpu()
            valid_ref_mask = attn_info["valid_ref_mask"].detach().to(dtype=torch.bool).cpu()

            ref_attn_mean = ref_weights.mean(dim=1)
            ref_entropy = -(ref_weights.clamp_min(1e-8) * ref_weights.clamp_min(1e-8).log()).sum(dim=-1)
            valid_ref_counts = valid_ref_mask.sum(dim=1).clamp(min=1)
            ref_entropy_norm = ref_entropy / valid_ref_counts.float().log().clamp_min(1.0).unsqueeze(1)

            weighted_token_attn = token_weights * ref_weights.permute(0, 2, 1).unsqueeze(-1)
            combined_token_mass = weighted_token_attn.mean(dim=2).reshape(token_weights.size(0), refs * tokens_per_ref)
            per_position_token_mass = weighted_token_attn.permute(0, 2, 1, 3).reshape(
                token_weights.size(0),
                token_weights.size(2),
                refs * tokens_per_ref,
            )

            per_font_rows: list[Dict[str, object]] = []
            coverage_values: list[float] = []
            for sample_idx, font_name in enumerate(fonts):
                valid_mask = flat_style_token_mask[sample_idx]
                valid_token_count = max(1, int(valid_mask.sum().item()))
                masked_per_position_token_mass = per_position_token_mass[sample_idx].clone()
                masked_per_position_token_mass[:, ~valid_mask] = -1.0
                per_position_top_tokens = masked_per_position_token_mass.argmax(dim=-1)
                top1_unique = torch.unique(per_position_top_tokens)
                coverage = float(top1_unique.numel()) / float(valid_token_count)
                coverage_values.append(coverage)

                masked_token_mass = combined_token_mass[sample_idx].clone()
                masked_token_mass[~valid_mask] = -1.0
                token_topk = min(3, valid_token_count)
                top_token_values, top_token_indices = torch.topk(masked_token_mass, k=token_topk)

                valid_ref_count = max(1, int(valid_ref_mask[sample_idx].sum().item()))
                masked_ref_mass = ref_attn_mean[sample_idx].clone()
                masked_ref_mass[~valid_ref_mask[sample_idx]] = -1.0
                ref_values, ref_indices = torch.topk(masked_ref_mass, k=min(3, valid_ref_count))
                per_font_rows.append(
                    {
                        "font": font_name,
                        "ref_entropy_norm": float(ref_entropy_norm[sample_idx].mean().item()),
                        "token_coverage": float(coverage),
                        "top_refs": [int(idx) for idx in ref_indices.tolist()],
                        "top_ref_mass": [float(value) for value in ref_values.tolist()],
                        "top_tokens": [int(idx) for idx in top_token_indices.tolist()],
                        "top_token_mass": [float(value) for value in top_token_values.tolist()],
                    }
                )

            mean_ref_mass = ref_attn_mean.mean(dim=0)
            top_ref = int(mean_ref_mass.argmax().item())
            top_ref_mass = float(mean_ref_mass[top_ref].item())
            mean_entropy_norm = float(ref_entropy_norm.mean().item())
            mean_token_coverage = float(sum(coverage_values) / float(max(1, len(coverage_values))))
            brief_parts.append(
                f"b{block_idx}:r{top_ref}@{top_ref_mass:.2f}/h{mean_entropy_norm:.2f}/c{mean_token_coverage:.2f}"
            )
            block_rows.append(
                {
                    "block_idx": int(block_idx),
                    "mean_entropy_norm": mean_entropy_norm,
                    "mean_token_coverage": mean_token_coverage,
                    "mean_ref_mass": [float(value) for value in mean_ref_mass.tolist()],
                    "per_font": per_font_rows,
                }
            )

        return {
            "step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "font_count": int(len(fonts)),
            "fonts": list(fonts),
            "font_source": self.style_attn_probe_font_source,
            "probe_timestep": 0.5,
            "probe_latent_mode": "zt=0.5*z1",
            "refs": int(refs),
            "tokens_per_ref": int(tokens_per_ref),
            "blocks": block_rows,
            "brief_blocks": brief_parts,
        }

    @torch.no_grad()
    def _run_style_attn_probe(self) -> Optional[Dict[str, object]]:
        if not self.style_attn_probe_enabled or self.style_attn_probe_batch is None:
            return None
        batch = self.style_attn_probe_batch
        fonts = list(batch.get("font", []))
        if not fonts:
            return None

        self.model.eval()
        content = batch["content"].to(self.device)
        target = batch["target"].to(self.device)
        style = batch["style_img"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        with self._autocast_context():
            z1 = self.model.encode_to_latent(target, sample_posterior=False)
            timesteps = torch.full((target.size(0),), 0.5, device=self.device, dtype=torch.float32)
            zt = z1 * 0.5
            content_features = self.model.encode_content_features(content)
            content_tokens = self.model.content_proj(content_features)
            style_pack = self.model.encode_style(
                style_img=style,
                style_ref_mask=style_ref_mask,
                need_style_tokens=True,
                need_style_global=True,
                detach_global_style_encoder=self.freeze_style_global,
                detach_global_style=self.freeze_style_global,
            )
            _, style_attn_weights = self.model.predict_flow(
                zt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_pack["style_tokens"],
                style_global=style_pack["style_global"],
                style_token_mask=style_pack["style_token_mask"],
                return_style_attn_weights=True,
            )

        refs = int(style.size(1))
        tokens_per_ref = int(self.model.style_tokens_per_ref)
        return self._summarize_style_attn_probe(
            style_attn_weights=style_attn_weights,
            style_token_mask=style_pack["style_token_mask"],
            fonts=fonts,
            refs=refs,
            tokens_per_ref=tokens_per_ref,
        )

    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | float]:
        target = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style = batch["style_img"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        with self._autocast_context():
            z1, _, _ = self._encode_latent(target)
            z0 = torch.randn_like(z1)
            timesteps = torch.rand(z1.size(0), device=self.device)
            t_view = timesteps.view(-1, 1, 1, 1).to(dtype=z1.dtype)
            zt = (1.0 - t_view) * z0 + t_view * z1
            target_flow = z1 - z0
            content_features = self.model.encode_content_features(content)
            content_tokens = self.model.content_proj(content_features)
            style_pack = self._encode_style_conditions(
                style,
                style_ref_mask,
            )
            pred_flow = self.model.predict_flow(
                zt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_pack["style_tokens"],
                style_global=style_pack["style_global"],
                style_token_mask=style_pack["style_token_mask"],
            )
            loss_flow = F.mse_loss(pred_flow, target_flow)
            z1_hat = zt + (1.0 - t_view) * pred_flow
            decoded = self.model.decode_from_latent(z1_hat)
            loss_img_l1_per_sample = _per_sample_mean((decoded - target).abs())
            loss_img_l1 = (
                loss_img_l1_per_sample.mean()
                if self.lambda_img_l1 > 0.0
                else target.new_zeros(())
            )
            loss_img_perc_per_sample = (
                glyph_perceptual_loss(decoded, target, reduction="none")
                if self.lambda_img_perc > 0.0
                else target.new_zeros((target.size(0),))
            )
            loss_img_perc = (
                loss_img_perc_per_sample.mean()
                if self.lambda_img_perc > 0.0
                else target.new_zeros(())
            )
            loss = self.lambda_flow * loss_flow
            loss = loss + self.lambda_img_l1 * loss_img_l1 + self.lambda_img_perc * loss_img_perc
        metrics = {
            "loss": loss,
            "loss_flow": loss_flow,
            "loss_img_l1": loss_img_l1,
            "loss_img_perc": loss_img_perc,
            "t_mean": timesteps.mean(),
            "style_global_frozen": float(self.freeze_style_global),
        }
        metrics["difficulty_per_sample"] = loss_img_l1_per_sample.detach() + loss_img_perc_per_sample.detach()
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        difficulty_per_sample = metrics.pop("difficulty_per_sample", None)
        metrics["loss"].backward()
        self.optimizer.step()
        self._set_learning_rate_for_step(self.global_step + 1)
        if difficulty_per_sample is not None:
            self._update_font_difficulty_ema(batch["font"], difficulty_per_sample)
            metrics["difficulty_mean"] = difficulty_per_sample.mean()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            metrics = self._compute_losses(batch)
            metrics.pop("difficulty_per_sample", None)
            return _metrics_to_floats(metrics)

    def _update_font_difficulty_ema(self, fonts, difficulty_per_sample: torch.Tensor) -> None:
        difficulty_values = difficulty_per_sample.detach().float().cpu().tolist()
        batch_sums: Dict[str, float] = {}
        batch_counts: Dict[str, int] = {}
        for font_name, score in zip(fonts, difficulty_values):
            batch_sums[font_name] = batch_sums.get(font_name, 0.0) + float(score)
            batch_counts[font_name] = batch_counts.get(font_name, 0) + 1
        ema_keep = self.difficulty_ema_decay
        ema_add = 1.0 - ema_keep
        for font_name, score_sum in batch_sums.items():
            batch_mean = score_sum / float(batch_counts[font_name])
            prev = self.font_difficulty_ema.get(font_name)
            self.font_difficulty_ema[font_name] = batch_mean if prev is None else ema_keep * prev + ema_add * batch_mean
            self.font_observation_count[font_name] = self.font_observation_count.get(font_name, 0) + batch_counts[font_name]

    def on_after_train_step(self, batch: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> None:
        if self.difficulty_sampler is not None:
            if self.global_step >= self.difficulty_warmup_steps and self.global_step % self.difficulty_refresh_every_steps == 0:
                sampler_summary = self._refresh_difficulty_sampler()
                if sampler_summary is not None:
                    top_fonts = ", ".join(
                        f"{item['font']}:{float(item['weight']):.2f}"
                        for item in sampler_summary["top_fonts"]
                    )
                    bottom_fonts = ", ".join(
                        f"{item['font']}:{float(item['weight']):.2f}"
                        for item in sampler_summary["bottom_fonts"]
                    )
                    print(
                        "[train] refreshed flow difficulty sampler "
                        f"step={self.global_step} epoch={self.current_epoch} "
                        f"observed_fonts={int(sampler_summary['observed_font_count'])} "
                        f"baseline_difficulty={float(sampler_summary['baseline_difficulty']):.4f} "
                        f"weight_range=[{float(sampler_summary['font_weight_min']):.2f},"
                        f"{float(sampler_summary['font_weight_max']):.2f}] "
                        f"weight_bins(["
                        f"{float(sampler_summary['font_weight_bucket_min']):.2f},"
                        f"{float(sampler_summary['font_weight_bucket_edge_1']):.2f})="
                        f"{int(sampler_summary['font_weight_bucket_count_low'])},"
                        f"[{float(sampler_summary['font_weight_bucket_edge_1']):.2f},"
                        f"{float(sampler_summary['font_weight_bucket_edge_2']):.2f})="
                        f"{int(sampler_summary['font_weight_bucket_count_mid'])},"
                        f"[{float(sampler_summary['font_weight_bucket_edge_2']):.2f},"
                        f"{float(sampler_summary['font_weight_bucket_max']):.2f}]="
                        f"{int(sampler_summary['font_weight_bucket_count_high'])}) "
                        f"top={top_fonts} bottom={bottom_fonts}",
                        flush=True,
                    )

        if (
            self.style_attn_probe_enabled
            and self.style_attn_probe_batch is not None
            and self.global_step % self.style_attn_probe_every_steps == 0
        ):
            probe_row = self._run_style_attn_probe()
            if probe_row is not None:
                self._write_style_attn_probe_log(probe_row)
                brief = " ".join(str(part) for part in probe_row["brief_blocks"])
                print(
                    "[style_attn_probe] "
                    f"step={self.global_step} epoch={self.current_epoch} "
                    f"fonts={int(probe_row['font_count'])} {brief}",
                    flush=True,
                )

    def _refresh_difficulty_sampler(self) -> Optional[Dict[str, object]]:
        if self.difficulty_sampler is None:
            return None
        if self.global_step < self.difficulty_warmup_steps:
            return None
        if not self.font_difficulty_ema:
            return None

        observed_values = list(self.font_difficulty_ema.values())
        baseline = sum(observed_values) / float(len(observed_values))
        baseline = max(baseline, 1e-8)
        font_weights: Dict[str, float] = {}
        for font_name in self.difficulty_sampler.dataset.font_names:
            difficulty = self.font_difficulty_ema.get(font_name, baseline)
            normalized = max(difficulty / baseline, 1e-8)
            weight = normalized ** self.difficulty_alpha
            weight = min(self.difficulty_max_weight, max(self.difficulty_min_weight, weight))
            font_weights[font_name] = float(weight)

        sampler_summary = self.difficulty_sampler.set_font_weights(font_weights)
        sampler_summary["observed_font_count"] = int(len(self.font_difficulty_ema))
        sampler_summary["baseline_difficulty"] = float(baseline)
        sampler_summary["warmup_steps"] = int(self.difficulty_warmup_steps)
        sampler_summary["ema_decay"] = float(self.difficulty_ema_decay)
        sampler_summary["alpha"] = float(self.difficulty_alpha)
        sampler_summary["min_weight"] = float(self.difficulty_min_weight)
        sampler_summary["max_weight"] = float(self.difficulty_max_weight)
        sampler_summary["refresh_every_steps"] = int(self.difficulty_refresh_every_steps)
        return sampler_summary

    @torch.no_grad()
    def flow_sample(
        self,
        content: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
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
        step_count = self.flow_sample_steps if num_inference_steps is None else max(1, int(num_inference_steps))
        dt = 1.0 / float(step_count)
        with self._autocast_context():
            content_features = self.model.encode_content_features(content)
            content_tokens = self.model.content_proj(content_features)
            style_pack = self.model.encode_style(
                style_img=style_img,
                style_ref_mask=style_ref_mask,
                need_style_tokens=True,
                need_style_global=True,
                detach_global_style_encoder=self.freeze_style_global,
                detach_global_style=self.freeze_style_global,
            )
            for step_idx in range(step_count):
                t = torch.full(
                    (batch_size,),
                    float(step_idx) / float(step_count),
                    device=self.device,
                    dtype=torch.float32,
                )
                pred_flow = self.model.predict_flow(
                    sample,
                    t,
                    content_tokens=content_tokens,
                    style_tokens=style_pack["style_tokens"],
                    style_global=style_pack["style_global"],
                    style_token_mask=style_pack["style_token_mask"],
                )
                sample = sample + dt * pred_flow

            decoded = self.model.decode_from_latent(sample).clamp(-1.0, 1.0)
        return decoded.float()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
            "stage": "flow",
            "model_state": self.model.state_dict(),
            "latent_norm_state": self.model.latent_norm_state_dict(),
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
            raise RuntimeError("Flow checkpoint is missing 'model_state'.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.load_latent_norm_state(checkpoint.get("latent_norm_state"))
        resume_step = int(checkpoint.get("step", 0))
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[flow] skipped optimizer state load: {exc}", flush=True)
        self.global_step = resume_step
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self._set_learning_rate_for_step(self.global_step)

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"][:8].to(self.device)
        target = batch["target"][:8].to(self.device)
        style = batch["style_img"][:8].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask[:8].to(self.device)
        sample = self.flow_sample(
            content,
            style_img=style,
            style_ref_mask=style_ref_mask,
            num_inference_steps=self.flow_sample_steps,
        )
        vis = torch.cat([(content + 1.0) * 0.5, (target + 1.0) * 0.5, (sample + 1.0) * 0.5], dim=0)
        save_image(vis, out_dir / f"sample_step_{self.global_step}.png", nrow=content.size(0))
