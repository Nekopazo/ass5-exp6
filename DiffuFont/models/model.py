#!/usr/bin/env python3
"""
Training utilities for DiffuFont – Diffusion & Flow-Matching trainers.

Key changes from prior version
-------------------------------
* LR scheduler → OneCycleLR (per-step, built-in warmup).
* CFG dropout now null-ifies **all** conditions (content **and** part_imgs).
* InfoNCE contrastive loss on dual part-set views (lambda_nce).
* FlowMatchingTrainer overrides DDPM sampling methods to prevent misuse.
* No style reference image (RSI) — model takes only content + parts.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import DPMSolverMultistepScheduler


# ====================================================================== #
#  Noise schedule (DDPM forward process)
# ====================================================================== #


class NoiseScheduler:
    """Beta / alpha_bar schedules and forward-process noise addition."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        self._register_schedule(beta_start, beta_end)

    def _register_schedule(self, beta_start: float, beta_end: float):
        betas = torch.linspace(beta_start, beta_end, self.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._buf = {}
        for name, t in [
            ("betas", betas),
            ("alpha_bars", alpha_bars),
            ("sqrt_alpha_bars", torch.sqrt(alpha_bars)),
            ("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars)),
        ]:
            setattr(self, name, t)

    def to(self, device):
        for name in ["betas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        device = x0.device
        t = t.long()
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        eps = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_1m * eps
        return x_t, eps


# ====================================================================== #
#  InfoNCE helper
# ====================================================================== #


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    font_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE given two views and per-sample font-id labels.

    z_a, z_b : (B, D), L2-normalised.
    font_ids  : (B,), integer font label.
    positive  : same font_id.  negative : different font_id.
    """
    B = z_a.size(0)
    # Stack both views: (2B, D)
    z = torch.cat([z_a, z_b], dim=0)  # 2B x D
    sim = z @ z.T / temperature       # 2B x 2B

    # Build positive mask.  For sample i (view A), positive = {i+B} (view B).
    # For sample i+B (view B), positive = {i} (view A).
    # Also, same-font different-sample pairs are **additional positives**.
    ids = torch.cat([font_ids, font_ids], dim=0)  # 2B
    pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1))  # 2B x 2B
    # Self-loop is NOT positive.
    pos_mask.fill_diagonal_(False)

    # Exclude self from log-softmax denominator.
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    logits = sim.masked_fill(self_mask, float("-inf"))

    # For each row, the loss = -log( sum(exp(pos)) / sum(exp(all excl self)) ).
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Zero out self-loop positions to avoid -inf * 0 = NaN.
    log_prob = log_prob.masked_fill(self_mask, 0.0)
    # Average over positive entries per row.
    n_pos = pos_mask.sum(dim=1).clamp_min(1).float()
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
    return loss.mean()


# ====================================================================== #
#  DiffusionTrainer
# ====================================================================== #


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 2e-4,
        lambda_mse: float = 1.0,
        lambda_nce: float = 0.05,
        cfg_drop_prob: float = 0.1,
        T: int = 1000,
        total_steps: int = 100_000,
        sample_every_steps: int | None = None,
        precision: str = "fp32",
        save_every_steps: int | None = None,
        log_every_steps: int | None = None,
        detailed_log: bool = True,
    ):
        self.device = device
        self.model = model.to(device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if int(T) <= 0:
            raise ValueError(f"Diffusion timestep T must be > 0, got {T}")
        self.diffusion_steps = int(T)
        self.total_steps = max(1, int(total_steps))

        # OneCycleLR — per-step, built-in warmup + cosine annealing.
        self.lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=lr,
            total_steps=self.total_steps,
            pct_start=0.05,          # 5 % warmup (~5k steps for 100k total)
            anneal_strategy="cos",
            div_factor=25.0,         # initial_lr = max_lr / 25
            final_div_factor=1e3,    # final_lr  = initial_lr / 1e3 ≈ 8e-9
        )

        self.scheduler = NoiseScheduler(self.diffusion_steps).to(device)
        self.lambda_mse = float(lambda_mse)
        self.lambda_nce = float(lambda_nce)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.global_step = 0
        self.local_step = 0
        self.current_epoch = 0
        self._step_data_time = 0.0
        self._step_train_time = 0.0
        self._step_data_time_ema = 0.0
        self._step_train_time_ema = 0.0
        self._time_ema_decay = 0.9
        self.save_every_steps = save_every_steps
        self.checkpoint_dir: Path | None = None
        self.step_log_file: Path | None = None
        self.log_every_steps = int(log_every_steps) if log_every_steps and int(log_every_steps) > 0 else None
        self.detailed_log = bool(detailed_log)

        # Sampling config
        self.sample_every_steps = sample_every_steps
        self.sample_batch: Dict[str, torch.Tensor] | None = None
        self.sample_dir: Path | None = None
        self.sample_solver: str = "dpm"
        self.sample_guidance_scale: float = 7.5
        self.sample_use_cfg: bool = True
        self.sample_inference_steps: int = 20
        self.save_split_components: bool = False

        # Precision / AMP
        precision_key = str(precision).strip().lower()
        self.precision = precision_key
        self.use_amp = False
        self.amp_dtype: torch.dtype | None = None
        if precision_key in {"bf16", "bfloat16"}:
            # Avoid torch.cuda.is_bf16_supported() — it can trigger NVML internally.
            # On Ampere+ (sm_80+) bf16 is always available; fall back to a simple test.
            _bf16_ok = False
            if self.device.type == "cuda":
                try:
                    _cap = torch.cuda.get_device_capability(self.device)
                    _bf16_ok = _cap[0] >= 8
                except Exception:
                    pass
            if _bf16_ok:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError("bf16 requested but unavailable on current device.")
        elif precision_key in {"fp16", "float16", "half"}:
            if self.device.type == "cuda":
                self.use_amp = True
                self.amp_dtype = torch.float16
            else:
                raise ValueError("fp16 requested on non-cuda device.")
        elif precision_key in {"fp32", "float32", "32"}:
            pass
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        self.use_grad_scaler = bool(self.use_amp and self.amp_dtype == torch.float16 and self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

    # ------------------------------------------------------------------ #
    #  Step log
    # ------------------------------------------------------------------ #
    def _write_step_log(self, row: Dict[str, float | int]) -> None:
        if self.step_log_file is None:
            return
        self.step_log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.step_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    #  InfoNCE on dual part-set views
    # ------------------------------------------------------------------ #
    def _compute_nce(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute InfoNCE if parts_b present and model supports contrastive head."""
        if self.lambda_nce <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if "parts_b" not in batch or "font_ids" not in batch:
            return torch.tensor(0.0, device=self.device)
        if not hasattr(self.model, "encode_contrastive_z"):
            return torch.tensor(0.0, device=self.device)

        parts_a = batch["parts"].to(self.device)
        mask_a = batch["part_mask"].to(self.device)
        parts_b = batch["parts_b"].to(self.device)
        mask_b = batch["part_mask_b"].to(self.device)
        font_ids = batch["font_ids"].to(self.device)

        z_a = self.model.encode_contrastive_z(parts_a, mask_a)
        z_b = self.model.encode_contrastive_z(parts_b, mask_b)
        return info_nce_loss(z_a, z_b, font_ids, temperature=0.07)

    # ------------------------------------------------------------------ #
    #  train_step
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Dict[str, torch.Tensor], grad_clip: float | None = 1.0):
        _train_t0 = time.perf_counter()
        self.model.train()
        x0 = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        B = x0.size(0)
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # CFG dropout — null ALL conditions (content, parts).
        cfg_mask = (torch.rand((B,), device=self.device) < self.cfg_drop_prob)
        if bool(cfg_mask.any().item()):
            content = content.clone()
            content[cfg_mask] = 1.0
            if part_imgs is not None:
                part_imgs = part_imgs.clone()
                part_imgs[cfg_mask] = 0.0
                if part_mask is not None:
                    part_mask = part_mask.clone()
                    part_mask[cfg_mask] = 0.0

        # Convert pixel → latent BEFORE adding noise (no grad needed for PixelUnshuffle).
        with torch.no_grad():
            x0_latent = self.model.encode_to_latent(x0)     # (B,3,256,256) → (B,12,128,128)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            x_t_latent, eps_latent = self.scheduler.add_noise(x0_latent, t)
            eps_hat_latent = self.model(
                x_t_latent, t, content,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )
            loss_mse = F.mse_loss(eps_hat_latent, eps_latent)
            loss_nce = self._compute_nce(batch)

            loss = (
                self.lambda_mse * loss_mse
                + self.lambda_nce * loss_nce
            )

        self.opt.zero_grad(set_to_none=True)
        grad_norm: float | None = None
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
            if grad_clip is not None:
                self.grad_scaler.unscale_(self.opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip).item())
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip).item())
            self.opt.step()

        # OneCycleLR steps per batch.
        if self.global_step < self.total_steps:
            self.lr_schedule.step()

        self.global_step += 1
        self.local_step += 1
        self._step_train_time = float(time.perf_counter() - _train_t0)
        decay = float(self._time_ema_decay)
        if self.global_step <= 1:
            self._step_data_time_ema = float(self._step_data_time)
            self._step_train_time_ema = float(self._step_train_time)
        else:
            self._step_data_time_ema = decay * self._step_data_time_ema + (1.0 - decay) * float(self._step_data_time)
            self._step_train_time_ema = decay * self._step_train_time_ema + (1.0 - decay) * float(self._step_train_time)

        # Auto sampling
        if (
            self.sample_every_steps
            and self.sample_batch is not None
            and self.sample_dir is not None
            and self.global_step % self.sample_every_steps == 0
        ):
            self.sample_and_save(self.sample_batch, self.sample_dir)

        # Auto checkpoint
        if (
            self.save_every_steps
            and self.checkpoint_dir is not None
            and self.global_step % self.save_every_steps == 0
        ):
            self.save(self.checkpoint_dir / f"ckpt_step_{self.global_step}.pt")

        # Logging
        if self.log_every_steps and self.global_step % self.log_every_steps == 0:
            log_row: Dict[str, float | int] = {
                "global_step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "epoch_step": int(self.local_step),
                "loss": float(loss.item()),
                "loss_mse": float(loss_mse.item()),
                "loss_nce": float(loss_nce.item()),
                "cfg_dropped": int(cfg_mask.sum().item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(self._step_train_time),
                "data_time_ema": float(self._step_data_time_ema),
                "train_time_ema": float(self._step_train_time_ema),
            }
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            self._write_step_log(log_row)

            msg = (
                f"[step] gstep={self.global_step} ep={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} mse={loss_mse.item():.4f} "
                f"nce={loss_nce.item():.4f} "
                f"cfg_drop={int(cfg_mask.sum().item())}/{int(B)} lr={self.lr_schedule.get_last_lr()[0]:.6e} "
                f"data_t={self._step_data_time:.3f}s train_t={self._step_train_time:.3f}s"
            )
            if grad_norm is not None:
                msg += f" gnorm={grad_norm:.4f}"
            print(msg, flush=True)

        return {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_nce": loss_nce.item(),
            "cfg_dropped": int(cfg_mask.sum().item()),
            "lr": self.lr_schedule.get_last_lr()[0],
            "data_time": float(self._step_data_time),
            "train_time": float(self._step_train_time),
        }

    # ------------------------------------------------------------------ #
    #  DPM-Solver++ sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def dpm_solver_sample(
        self,
        content_img: torch.Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_cfg: bool = True,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample in **latent** space, return **pixel**-space result."""
        self.model.eval()
        bsz = content_img.shape[0]
        device = self.device

        content_img = content_img.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)

        # Latent dimensions
        latent_ch = self.model.unet_in_channels   # 12
        latent_h = self.model.unet_input_size      # 128
        latent_w = self.model.unet_input_size      # 128

        dpm = DPMSolverMultistepScheduler(
            num_train_timesteps=int(self.diffusion_steps),
            beta_start=float(self.scheduler.betas[0].item()),
            beta_end=float(self.scheduler.betas[-1].item()),
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
        )
        dpm.set_timesteps(int(num_inference_steps), device=device)

        # Start from pure noise in latent space
        x_t = torch.randn((bsz, latent_ch, latent_h, latent_w), device=device, dtype=content_img.dtype)

        if use_cfg:
            uncond_content = torch.ones_like(content_img)
        else:
            uncond_content = None

        for t in dpm.timesteps:
            t_cond = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
            eps_cond = self.model(
                x_t, t_cond, content_img,
                part_imgs=part_imgs, part_mask=part_mask,
            )
            if use_cfg:
                eps_uncond = self.model(
                    x_t, t_cond, uncond_content,
                    part_imgs=None, part_mask=None,
                )
                eps_hat = eps_uncond + float(guidance_scale) * (eps_cond - eps_uncond)
            else:
                eps_hat = eps_cond
            x_t = dpm.step(eps_hat, t, x_t).prev_sample

        # Decode latent → pixel
        return self.model.decode_from_latent(x_t).clamp(-1, 1)

    # ------------------------------------------------------------------ #
    #  Checkpoint
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path):
        model_state = self.model.state_dict()
        chkpt = {
            "model_state": model_state,
            "opt_state": self.opt.state_dict(),
            "step": self.global_step,
            "epoch": self.current_epoch,
            "local_step": self.local_step,
            "lr_schedule_state": self.lr_schedule.state_dict(),
            "diffusion_steps": self.diffusion_steps,
            "total_steps": self.total_steps,
            "precision": self.precision,
        }
        if self.use_grad_scaler:
            chkpt["grad_scaler_state"] = self.grad_scaler.state_dict()
        save_path = Path(path)
        torch.save(chkpt, save_path)
        self._save_split_components(save_path, model_state)

    def load(self, path: str | Path):
        chkpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chkpt["model_state"])
        self.opt.load_state_dict(chkpt["opt_state"])
        self.global_step = chkpt.get("step", 0)
        self.current_epoch = chkpt.get("epoch", 0)
        self.local_step = chkpt.get("local_step", 0)
        if "lr_schedule_state" in chkpt:
            self.lr_schedule.load_state_dict(chkpt["lr_schedule_state"])
        if self.use_grad_scaler and "grad_scaler_state" in chkpt:
            self.grad_scaler.load_state_dict(chkpt["grad_scaler_state"])

    def _save_split_components(self, path: Path, model_state: Dict[str, torch.Tensor]) -> None:
        if not self.save_split_components:
            return
        comp_dir = path.parent / "components"
        comp_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem

        vector_prefixes = ("part_patch_encoder.", "part_set_norm.", "style_token_mlp.", "contrastive_head.")
        main_state = {k: v for k, v in model_state.items() if not k.startswith(vector_prefixes)}
        vector_state = {k: v for k, v in model_state.items() if k.startswith(vector_prefixes)}

        torch.save(
            {"component": "main_model", "step": self.global_step, "epoch": self.current_epoch, "state_dict": main_state},
            comp_dir / f"{stem}.main_model.pt",
        )
        torch.save(
            {"component": "trainable_vector_cnn", "step": self.global_step, "epoch": self.current_epoch, "state_dict": vector_state},
            comp_dir / f"{stem}.trainable_vector_cnn.pt",
        )

    # ------------------------------------------------------------------ #
    #  Epoch / fit
    # ------------------------------------------------------------------ #
    def train_epoch(self, dataloader, epoch: int, grad_clip: float | None = 1.0):
        # Use running averages instead of accumulating all log dicts
        sums: Dict[str, float] = {}
        count = 0
        self.current_epoch = epoch
        self.local_step = 0
        it = iter(dataloader)
        while True:
            t_data_start = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            self._step_data_time = float(time.perf_counter() - t_data_start)
            out = self.train_step(batch, grad_clip=grad_clip)
            for k, v in out.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1
        if count == 0:
            return {}
        return {k: v / count for k, v in sums.items()}

    def fit(
        self,
        dataloader,
        epochs: int,
        save_every: int | None = None,
        save_dir: str | Path | None = None,
    ):
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = save_dir
            self.step_log_file = save_dir / "train_step_metrics.jsonl"
            if self.global_step == 0:
                self.step_log_file.write_text("", encoding="utf-8")
                self._write_step_log({
                    "meta": 1,
                    "diffusion_steps": int(self.diffusion_steps),
                    "total_steps": int(self.total_steps),
                    "log_every_steps": int(self.log_every_steps or 0),
                    "save_every_steps": int(self.save_every_steps or 0),
                    "sample_every_steps": int(self.sample_every_steps or 0),
                })
        for epoch in range(1, epochs + 1):
            log = self.train_epoch(dataloader, epoch)
            print(f"\nEpoch {epoch}:", {k: round(v, 4) for k, v in log.items()})
            if save_every and epoch % save_every == 0 and save_dir is not None:
                self.save(save_dir / f"ckpt_{epoch}.pt")

    # ------------------------------------------------------------------ #
    #  Convenience sample & save
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Limit to at most 8 images for speed
        max_vis = 8
        content = batch["content"][:max_vis].to(self.device)
        target = batch["target"][:max_vis].to(self.device)
        part_imgs = batch["parts"][:max_vis].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"][:max_vis].to(self.device) if "part_mask" in batch else None

        # Early training: auto-reduce guidance to avoid noise amplification
        gs = float(self.sample_guidance_scale)
        use_cfg = bool(self.sample_use_cfg)
        if self.global_step < 5000:
            gs = 1.0
            use_cfg = False

        sample = self.dpm_solver_sample(
            content,
            num_inference_steps=int(self.sample_inference_steps),
            guidance_scale=gs,
            use_cfg=use_cfg,
            part_imgs=part_imgs,
            part_mask=part_mask,
        )
        # Save comparison grid: row0=content, row1=GT, row2=generated
        vis = torch.cat([(content + 1) / 2, (target + 1) / 2, (sample + 1) / 2], dim=0)
        filename = f"sample_ep{self.current_epoch}_gstep{self.global_step}_estep{self.local_step}.png"
        save_image(vis, out_dir / filename, nrow=content.size(0))
        # Free CUDA cache fragmented by the inference-mode scheduler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ====================================================================== # #
#  FlowMatchingTrainer
# ====================================================================== #


class FlowMatchingTrainer(DiffusionTrainer):
    """Flow-Matching trainer: linear interpolation path + ODE sampling."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 2e-4,
        lambda_fm: float = 1.0,
        lambda_nce: float = 0.05,
        cfg_drop_prob: float = 0.1,
        T: int = 1000,
        total_steps: int = 100_000,
        sample_every_steps: int | None = None,
        precision: str = "fp32",
        save_every_steps: int | None = None,
        log_every_steps: int | None = None,
        detailed_log: bool = True,
    ):
        super().__init__(
            model=model, device=device, lr=lr,
            lambda_mse=1.0, lambda_nce=lambda_nce,
            cfg_drop_prob=cfg_drop_prob, T=T, total_steps=total_steps,
            sample_every_steps=sample_every_steps, precision=precision,
            save_every_steps=save_every_steps, log_every_steps=log_every_steps,
            detailed_log=detailed_log,
        )
        self.lambda_fm = float(lambda_fm)

    # ---- Block DDPM-specific samplers (wrong for velocity prediction) ---- #
    def dpm_solver_sample(self, *args, **kwargs):
        raise RuntimeError(
            "dpm_solver_sample() is a noise-prediction sampler and must not be "
            "used with FlowMatchingTrainer. Use flow_sample() instead."
        )

    def ddim_sample(self, *args, **kwargs):
        raise RuntimeError(
            "ddim_sample() is a noise-prediction sampler and must not be "
            "used with FlowMatchingTrainer. Use flow_sample() instead."
        )

    # ---- train_step (velocity target) ---- #
    def train_step(self, batch: Dict[str, torch.Tensor], grad_clip: float | None = 1.0):
        _train_t0 = time.perf_counter()
        self.model.train()
        x0 = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        b = x0.size(0)

        # Convert pixel → latent
        with torch.no_grad():
            x0_latent = self.model.encode_to_latent(x0)  # (B,3,256,256) → (B,12,128,128)

        # Linear path in latent space: x_t = (1-t) x0_l + t x1_l,  v* = x1_l - x0_l.
        t = torch.rand((b,), device=self.device, dtype=x0_latent.dtype)
        x1_latent = torch.randn_like(x0_latent)
        x_t_latent = (1.0 - t.view(-1, 1, 1, 1)) * x0_latent + t.view(-1, 1, 1, 1) * x1_latent
        v_target = x1_latent - x0_latent
        t_idx = (t * float(self.diffusion_steps - 1)).round().long()

        # CFG dropout — null ALL conditions.
        cfg_mask = (torch.rand((b,), device=self.device) < self.cfg_drop_prob)
        if bool(cfg_mask.any().item()):
            content = content.clone()
            content[cfg_mask] = 1.0
            if part_imgs is not None:
                part_imgs = part_imgs.clone()
                part_imgs[cfg_mask] = 0.0
                if part_mask is not None:
                    part_mask = part_mask.clone()
                    part_mask[cfg_mask] = 0.0

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            v_hat = self.model(
                x_t_latent, t_idx, content,
                part_imgs=part_imgs, part_mask=part_mask,
            )
            loss_fm = F.mse_loss(v_hat, v_target)
            loss_nce = self._compute_nce(batch)

            loss = (
                self.lambda_fm * loss_fm
                + self.lambda_nce * loss_nce
            )

        self.opt.zero_grad(set_to_none=True)
        grad_norm: float | None = None
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
            if grad_clip is not None:
                self.grad_scaler.unscale_(self.opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip).item())
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip).item())
            self.opt.step()

        if self.global_step < self.total_steps:
            self.lr_schedule.step()

        self.global_step += 1
        self.local_step += 1
        self._step_train_time = float(time.perf_counter() - _train_t0)
        decay = float(self._time_ema_decay)
        if self.global_step <= 1:
            self._step_data_time_ema = float(self._step_data_time)
            self._step_train_time_ema = float(self._step_train_time)
        else:
            self._step_data_time_ema = decay * self._step_data_time_ema + (1.0 - decay) * float(self._step_data_time)
            self._step_train_time_ema = decay * self._step_train_time_ema + (1.0 - decay) * float(self._step_train_time)

        # Auto sampling
        if (
            self.sample_every_steps
            and self.sample_batch is not None
            and self.sample_dir is not None
            and self.global_step % self.sample_every_steps == 0
        ):
            self.sample_and_save(self.sample_batch, self.sample_dir)

        if (
            self.save_every_steps
            and self.checkpoint_dir is not None
            and self.global_step % self.save_every_steps == 0
        ):
            self.save(self.checkpoint_dir / f"ckpt_step_{self.global_step}.pt")

        if self.log_every_steps and self.global_step % self.log_every_steps == 0:
            log_row: Dict[str, float | int] = {
                "global_step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "epoch_step": int(self.local_step),
                "loss": float(loss.item()),
                "loss_fm": float(loss_fm.item()),
                "loss_nce": float(loss_nce.item()),
                "cfg_dropped": int(cfg_mask.sum().item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(self._step_train_time),
            }
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            self._write_step_log(log_row)

            msg = (
                f"[fm-step] gstep={self.global_step} ep={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} fm={loss_fm.item():.4f} "
                f"nce={loss_nce.item():.4f} "
                f"cfg_drop={int(cfg_mask.sum().item())}/{int(b)} lr={self.lr_schedule.get_last_lr()[0]:.6e} "
                f"data_t={self._step_data_time:.3f}s train_t={self._step_train_time:.3f}s"
            )
            if grad_norm is not None:
                msg += f" gnorm={grad_norm:.4f}"
            print(msg, flush=True)

        return {
            "loss": loss.item(),
            "loss_fm": loss_fm.item(),
            "loss_nce": loss_nce.item(),
            "cfg_dropped": int(cfg_mask.sum().item()),
            "lr": self.lr_schedule.get_last_lr()[0],
            "data_time": float(self._step_data_time),
            "train_time": float(self._step_train_time),
        }

    # ---- ODE flow sampling ---- #
    @torch.no_grad()
    def flow_sample(
        self,
        content_img: torch.Tensor,
        c: int = 50,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        b, ch, h, w = content_img.shape
        device = self.device

        content_img = content_img.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)

        use_cfg = bool(self.sample_use_cfg)
        guidance_scale = float(self.sample_guidance_scale)
        if use_cfg:
            uncond_content = torch.ones_like(content_img)
        else:
            uncond_content = None

        # Latent dimensions
        latent_ch = self.model.unet_in_channels
        latent_h = self.model.unet_input_size
        latent_w = self.model.unet_input_size

        x_t = torch.randn(b, latent_ch, latent_h, latent_w, device=device)
        n_steps = max(2, int(c))
        dt = 1.0 / float(n_steps - 1)

        for i in range(n_steps - 1):
            t_cur = 1.0 - i * dt
            t_tensor = torch.full((b,), t_cur, device=device, dtype=x_t.dtype)
            t_idx = (t_tensor * float(self.diffusion_steps - 1)).round().long()
            v_cond = self.model(
                x_t, t_idx, content_img,
                part_imgs=part_imgs, part_mask=part_mask,
            )
            if use_cfg:
                v_uncond = self.model(
                    x_t, t_idx, uncond_content,
                    part_imgs=None, part_mask=None,
                )
                v_hat = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_hat = v_cond
            x_t = x_t - dt * v_hat
        # Decode latent → pixel
        return self.model.decode_from_latent(x_t).clamp(-1, 1)

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        out_dir.mkdir(parents=True, exist_ok=True)
        max_vis = 8
        content = batch["content"][:max_vis].to(self.device)
        target = batch["target"][:max_vis].to(self.device)
        part_imgs = batch["parts"][:max_vis].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"][:max_vis].to(self.device) if "part_mask" in batch else None
        flow_steps = int(self.sample_inference_steps) if int(self.sample_inference_steps) > 1 else int(steps)

        # Early training: disable CFG to avoid noise amplification
        orig_cfg = self.sample_use_cfg
        orig_gs = self.sample_guidance_scale
        if self.global_step < 5000:
            self.sample_use_cfg = False
            self.sample_guidance_scale = 1.0

        sample = self.flow_sample(
            content, c=flow_steps,
            part_imgs=part_imgs, part_mask=part_mask,
        )

        self.sample_use_cfg = orig_cfg
        self.sample_guidance_scale = orig_gs

        vis = torch.cat([(content + 1) / 2, (target + 1) / 2, (sample + 1) / 2], dim=0)
        filename = f"sample_ep{self.current_epoch}_gstep{self.global_step}_estep{self.local_step}.png"
        save_image(vis, out_dir / filename, nrow=content.size(0))
        # Free CUDA cache fragmented by the inference-mode scheduler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
