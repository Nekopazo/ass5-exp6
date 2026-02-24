#!/usr/bin/env python3
"""
Training utilities for FontDiffusionUNet.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import save_image
from diffusers import DPMSolverMultistepScheduler


class NoiseScheduler:
    """Creates beta / alpha_bar schedules and provides utilities for the forward (noise adding) process."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        self.register_schedule(beta_start, beta_end)

    def register_schedule(self, beta_start: float, beta_end: float):
        betas = torch.linspace(beta_start, beta_end, self.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def to(self, device):
        for name in ["betas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """Return noisy sample `x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε`."""
        device = x0.device
        t = t.long()
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1).to(device)
        eps = torch.randn_like(x0)
        x_t = sqrt_alpha_bar * x0 + sqrt_1m * eps
        return x_t, eps


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss based on the first few layers of a pretrained VGG19 network."""

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]  # till relu3_1
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Training tensors are in [-1, 1]; convert to [0, 1] before ImageNet normalization.
        input_01 = ((input + 1.0) * 0.5).clamp(0.0, 1.0)
        target_01 = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
        input_norm = self.norm(input_01)
        target_norm = self.norm(target_01)
        feat_in = self.vgg(input_norm)
        feat_tg = self.vgg(target_norm)
        return F.l1_loss(feat_in, feat_tg)


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 2e-4,
        lambda_mse: float = 1.0,
        lambda_off: float = 0.5,
        lambda_cp: float = 0.01,
        lambda_cons: float = 0.0,
        cfg_drop_prob: float = 0.1,
        T: int = 1000,
        lr_tmax: int | None = None,
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
        if lr_tmax is None:
            lr_tmax = self.diffusion_steps
        if int(lr_tmax) <= 0:
            raise ValueError(f"LR scheduler T_max must be > 0, got {lr_tmax}")
        self.lr_tmax = int(lr_tmax)
        # Keep diffusion steps and LR schedule horizon decoupled.
        self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.lr_tmax)
        self.scheduler = NoiseScheduler(self.diffusion_steps).to(device)
        self.lambda_mse = lambda_mse
        self.lambda_off = lambda_off
        self.lambda_cp = lambda_cp
        self.lambda_cons = float(lambda_cons)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.perc_loss_fn = VGGPerceptualLoss().to(device) if float(self.lambda_cp) > 0.0 else None
        self.global_step = 0
        self.local_step = 0  # batch idx within current epoch
        self.current_epoch = 0
        self.save_every_steps = save_every_steps
        self.checkpoint_dir: Path | None = None
        self.step_log_file: Path | None = None
        self.log_every_steps = int(log_every_steps) if log_every_steps and int(log_every_steps) > 0 else None
        self.detailed_log = bool(detailed_log)

        # sampling config
        self.sample_every_steps = sample_every_steps  # if None → no automatic batch sampling
        self.sample_batch: Dict[str, torch.Tensor] | None = None
        self.sample_dir: Path | None = None
        self.sample_solver: str = "dpm"
        self.sample_guidance_scale: float = 7.5
        self.sample_use_cfg: bool = True
        self.sample_inference_steps: int = 20
        self.save_split_components: bool = False
        self.save_frozen_retrieval_copy: bool = True
        self.frozen_retrieval_ckpt_path: str | None = None

        precision_key = str(precision).strip().lower()
        self.precision = precision_key
        self.use_amp = False
        self.amp_dtype: torch.dtype | None = None
        if precision_key in {"bf16", "bfloat16"}:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
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

    def _offset_loss(self) -> torch.Tensor:
        if hasattr(self.model, "last_offset_loss"):
            v = getattr(self.model, "last_offset_loss")
            if isinstance(v, torch.Tensor):
                return v.to(self.device)
        return torch.tensor(0.0, device=self.device)

    def _write_step_log(self, row: Dict[str, float | int]) -> None:
        if self.step_log_file is None:
            return
        self.step_log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.step_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def train_step(self, batch: Dict[str, torch.Tensor], grad_clip: float | None = 1.0):
        self.model.train()
        self._clear_offset()
        x0 = batch["target"].to(self.device)  # clean target image
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        part_imgs_b = batch["parts_b"].to(self.device) if "parts_b" in batch else None
        part_mask_b = batch["part_mask_b"].to(self.device) if "part_mask_b" in batch else None
        B = x0.size(0)
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # Classifier-free training dropout:
        # condition is defined by (content, style) only.
        cfg_mask = (torch.rand((B,), device=self.device) < self.cfg_drop_prob)
        if bool(cfg_mask.any().item()):
            content = content.clone()
            style = style.clone()
            content[cfg_mask] = 1.0
            style[cfg_mask] = 1.0

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            # forward diffusion
            x_t, eps = self.scheduler.add_noise(x0, t)

            # prediction (noise target, aligned with original FontDiffuser training objective)
            eps_hat = self.model(
                x_t,
                t,
                content,
                style,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )

            # losses
            loss_mse = F.mse_loss(eps_hat, eps)
            # Keep source behavior scale for offset term.
            loss_off = self._offset_loss() / 2.0
            sqrt_ab = self.scheduler.sqrt_alpha_bars[t].view(-1, 1, 1, 1).to(self.device)
            sqrt_1m = self.scheduler.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1).to(self.device)
            x0_hat = (x_t - sqrt_1m * eps_hat) / torch.clamp(sqrt_ab, min=1e-8)
            if self.lambda_cp > 0.0:
                if self.perc_loss_fn is None:
                    raise ValueError("lambda_cp > 0 requires perceptual loss network.")
                loss_cp = self.perc_loss_fn(x0_hat, x0)
            else:
                loss_cp = torch.tensor(0.0, device=self.device, dtype=loss_mse.dtype)
            loss_cons = torch.tensor(0.0, device=self.device, dtype=loss_mse.dtype)
            if self.lambda_cons > 0.0:
                if part_imgs is None or part_imgs_b is None:
                    raise ValueError("lambda_cons > 0 requires parts and parts_b in each batch.")
                if not hasattr(self.model, "encode_part_vector"):
                    raise ValueError("lambda_cons > 0 requires model.encode_part_vector().")
                if not bool(getattr(self.model, "enable_parts_vector_condition", False)):
                    raise ValueError("lambda_cons > 0 requires part-vector conditioning enabled.")
                tok_a = self.model.encode_part_vector(part_imgs, part_mask)  # type: ignore[attr-defined]
                tok_b = self.model.encode_part_vector(part_imgs_b, part_mask_b)  # type: ignore[attr-defined]
                loss_cons = F.mse_loss(tok_a, tok_b)

            loss = (
                self.lambda_mse * loss_mse
                + self.lambda_off * loss_off
                + self.lambda_cp * loss_cp
                + self.lambda_cons * loss_cons
            )

        # backward
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
        self.lr_schedule.step()

        self.global_step += 1
        self.local_step += 1

        # automatic sampling triggered by batch counter
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
                "loss_mse": float(loss_mse.item()),
                "loss_off": float(loss_off.item()),
                "loss_cp": float(loss_cp.item()),
                "loss_cons": float(loss_cons.item()),
                "cfg_dropped": int(cfg_mask.sum().item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
            }
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            self._write_step_log(log_row)

            msg = (
                f"[debug-step] gstep={self.global_step} epoch={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} mse={loss_mse.item():.4f} off={loss_off.item():.4f} "
                f"cp={loss_cp.item():.4f} cons={loss_cons.item():.4f} "
                f"cfg_drop={int(cfg_mask.sum().item())}/{int(B)} lr={self.lr_schedule.get_last_lr()[0]:.6e}"
            )
            if grad_norm is not None:
                msg += f" grad_norm={grad_norm:.4f}"
            print(msg, flush=True)
            if self.detailed_log:
                x0_det = x0.detach()
                xt_det = x_t.detach()
                xh_det = x0_hat.detach()
                print(
                    "[debug-stats] "
                    f"x_t(mean={xt_det.mean().item():.4f},std={xt_det.std().item():.4f},"
                    f"min={xt_det.min().item():.4f},max={xt_det.max().item():.4f}) "
                    f"x0_hat(mean={xh_det.mean().item():.4f},std={xh_det.std().item():.4f},"
                    f"min={xh_det.min().item():.4f},max={xh_det.max().item():.4f}) "
                    f"target(mean={x0_det.mean().item():.4f},std={x0_det.std().item():.4f},"
                    f"min={x0_det.min().item():.4f},max={x0_det.max().item():.4f})",
                    flush=True,
                )

        return {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_off": loss_off.item(),
            "loss_cp": loss_cp.item(),
            "loss_cons": loss_cons.item(),
            "cfg_dropped": int(cfg_mask.sum().item()),
            "lr": self.lr_schedule.get_last_lr()[0],
        }

    # --------------------- sampling (DDIM简化) --------------------- #
    @torch.no_grad()
    def dpm_solver_sample(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_cfg: bool = True,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        bsz, ch, h, w = content_img.shape
        device = self.device

        content_img = content_img.to(device)
        style_img = style_img.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)

        dpm = DPMSolverMultistepScheduler(
            num_train_timesteps=int(self.diffusion_steps),
            beta_start=float(self.scheduler.betas[0].item()),
            beta_end=float(self.scheduler.betas[-1].item()),
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
        )
        dpm.set_timesteps(int(num_inference_steps), device=device)

        x_t = torch.randn((bsz, ch, h, w), device=device, dtype=content_img.dtype)

        if use_cfg:
            uncond_content = torch.ones_like(content_img)
            uncond_style = torch.ones_like(style_img)
            # Strict uncond branch: no parts condition.
            uncond_part_imgs = None
            uncond_part_mask = None
        else:
            uncond_content = None
            uncond_style = None
            uncond_part_imgs = None
            uncond_part_mask = None

        for t in dpm.timesteps:
            t_cond = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
            eps_cond = self.model(
                x_t,
                t_cond,
                content_img,
                style_img,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )
            if use_cfg:
                eps_uncond = self.model(
                    x_t,
                    t_cond,
                    uncond_content,
                    uncond_style,
                    part_imgs=uncond_part_imgs,
                    part_mask=uncond_part_mask,
                )
                eps_hat = eps_uncond + float(guidance_scale) * (eps_cond - eps_uncond)
            else:
                eps_hat = eps_cond
            x_t = dpm.step(eps_hat, t, x_t).prev_sample

        return x_t.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        c: int = 10,
        eta: float = 0.0,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DDIM 采样实现，默认使用 η=0 的确定性路径。

        Args:
            content_img: (B,C,H,W)
            style_img:   (B,C*k,H,W)
            c:           子序列线性间隔因子
            eta:         噪声比例，0 为确定性 DDIM
        """
        self.model.eval()

        B, C, H, W = content_img.shape
        device = self.device

        # 构造子序列 tau
        steps = max(2, self.scheduler.T // c)
        tau = torch.linspace(self.scheduler.T - 1, 0, steps).long()  # descending

        x_t = torch.randn(B, C, H, W, device=device)
        content_img = content_img.to(device)
        style_img = style_img.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)

        for i in range(len(tau) - 1):
            t_i = tau[i]
            t_prev = tau[i + 1]  # tau_{i-1}

            t_cond = torch.full((B,), t_i, device=device, dtype=torch.long)
            eps_hat = self.model(
                x_t,
                t_cond,
                content_img,
                style_img,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )

            alphabar_i = self.scheduler.alpha_bars[t_i].to(device)
            alphabar_prev = self.scheduler.alpha_bars[t_prev].to(device)
            sqrt_ab_i = torch.sqrt(alphabar_i)
            sqrt_1m_ab_i = torch.sqrt(torch.clamp(1 - alphabar_i, min=1e-12))
            pred_eps = eps_hat
            x0_hat = (x_t - sqrt_1m_ab_i * pred_eps) / torch.clamp(sqrt_ab_i, min=1e-8)

            sigma = torch.tensor(0.0, device=device, dtype=x_t.dtype)
            noise = torch.zeros_like(x_t)
            if eta > 0 and i < len(tau) - 1:
                sigma = eta * torch.sqrt(
                    (1 - alphabar_prev) / (1 - alphabar_i) * (1 - alphabar_i / alphabar_prev)
                )
                noise = torch.randn_like(x_t)

            coeff_eps = torch.sqrt(torch.clamp(1 - alphabar_prev - sigma * sigma, min=0.0))
            x_t = torch.sqrt(alphabar_prev) * x0_hat + coeff_eps * pred_eps + sigma * noise

        return x_t.clamp(-1, 1)

    # --------------------- checkpoint --------------------- #
    def _save_split_components(self, path: Path, model_state: Dict[str, torch.Tensor]) -> None:
        if not self.save_split_components:
            return
        comp_dir = path.parent / "components"
        comp_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem

        vector_prefixes = (
            "part_patch_encoder.",
        )
        main_state = {k: v for k, v in model_state.items() if not k.startswith(vector_prefixes)}
        vector_state = {k: v for k, v in model_state.items() if k.startswith(vector_prefixes)}

        torch.save(
            {
                "component": "main_model",
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "state_dict": main_state,
            },
            comp_dir / f"{stem}.main_model.pt",
        )
        torch.save(
            {
                "component": "trainable_vector_cnn",
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "state_dict": vector_state,
            },
            comp_dir / f"{stem}.trainable_vector_cnn.pt",
        )

        if self.save_frozen_retrieval_copy and self.frozen_retrieval_ckpt_path:
            src = Path(self.frozen_retrieval_ckpt_path)
            if src.exists():
                shutil.copy2(src, comp_dir / f"{stem}.frozen_retrieval_cnn.pt")

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
            "lr_tmax": self.lr_tmax,
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

    # ------------------ internal helpers ------------------ #
    def _clear_offset(self):
        if hasattr(self.model, "last_offset_loss"):
            self.model.last_offset_loss = torch.tensor(0.0, device=self.device)

    # --------------------- epoch / fit --------------------- #
    def train_epoch(self, dataloader, epoch: int, grad_clip: float | None = 1.0):
        logs: List[Dict] = []
        self.current_epoch = epoch
        self.local_step = 0
        for batch in dataloader:
            out = self.train_step(batch, grad_clip=grad_clip)
            logs.append(out)

        keys = logs[0].keys()
        return {k: sum(d[k] for d in logs) / len(logs) for k in keys}

    def fit(
        self,
        dataloader,
        epochs: int,
        save_every: int | None = None,
        save_dir: str | Path | None = None,
        sample_every: int | None = None,
        sample_callback=None,
    ):
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = save_dir
            self.step_log_file = save_dir / "train_step_metrics.jsonl"
            if self.global_step == 0:
                self.step_log_file.write_text("", encoding="utf-8")
                self._write_step_log(
                    {
                        "meta": 1,
                        "diffusion_steps": int(self.diffusion_steps),
                        "lr_tmax": int(self.lr_tmax),
                        "log_every_steps": int(self.log_every_steps or 0),
                        "save_every_steps": int(self.save_every_steps or 0),
                        "sample_every_steps": int(self.sample_every_steps or 0),
                    }
                )
        for epoch in range(1, epochs + 1):
            log = self.train_epoch(dataloader, epoch)
            print(f"\nEpoch {epoch}:", {k: round(v,4) for k,v in log.items()})

            if save_every and epoch % save_every == 0 and save_dir is not None:
                self.save(save_dir / f"ckpt_{epoch}.pt")

            if sample_every and epoch % sample_every == 0 and sample_callback is not None:
                sample_callback(self)

    # --------------------- convenience sample & save --------------------- #
    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        """使用固定 batch 采样并保存到 out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        if str(self.sample_solver).lower() == "ddim":
            if self.sample_use_cfg:
                raise ValueError("DDIM sampling path does not support CFG; set sample_use_cfg=False.")
            ddim_steps = max(2, int(self.sample_inference_steps))
            c = max(1, int(self.scheduler.T) // ddim_steps)
            sample = self.ddim_sample(
                content,
                style,
                c=c,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )
        else:
            sample = self.dpm_solver_sample(
                content,
                style,
                num_inference_steps=int(self.sample_inference_steps),
                guidance_scale=float(self.sample_guidance_scale),
                use_cfg=bool(self.sample_use_cfg),
                part_imgs=part_imgs,
                part_mask=part_mask,
            )
        filename = f"sample_ep{self.current_epoch}_gstep{self.global_step}_estep{self.local_step}.png"
        save_image((sample + 1) / 2, out_dir / filename)


class FlowMatchingTrainer(DiffusionTrainer):
    """Flow Matching trainer using linear interpolation path and ODE sampling."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 2e-4,
        lambda_fm: float = 1.0,
        lambda_off: float = 0.5,
        lambda_cp: float = 0.01,
        lambda_cons: float = 0.0,
        cfg_drop_prob: float = 0.1,
        T: int = 1000,
        lr_tmax: int | None = None,
        sample_every_steps: int | None = None,
        precision: str = "fp32",
        save_every_steps: int | None = None,
        log_every_steps: int | None = None,
        detailed_log: bool = True,
    ):
        super().__init__(
            model=model,
            device=device,
            lr=lr,
            lambda_mse=1.0,
            lambda_off=lambda_off,
            lambda_cp=lambda_cp,
            lambda_cons=lambda_cons,
            cfg_drop_prob=cfg_drop_prob,
            T=T,
            lr_tmax=lr_tmax,
            sample_every_steps=sample_every_steps,
            precision=precision,
            save_every_steps=save_every_steps,
            log_every_steps=log_every_steps,
            detailed_log=detailed_log,
        )
        self.lambda_fm = float(lambda_fm)

    def train_step(self, batch: Dict[str, torch.Tensor], grad_clip: float | None = 1.0):
        self.model.train()
        self._clear_offset()
        x0 = batch["target"].to(self.device)  # data sample
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        part_imgs_b = batch["parts_b"].to(self.device) if "parts_b" in batch else None
        part_mask_b = batch["part_mask_b"].to(self.device) if "part_mask_b" in batch else None
        b = x0.size(0)

        # Linear probability path: x_t = (1-t) * x0 + t * x1, x1 ~ N(0, I), v* = x1 - x0.
        t = torch.rand((b,), device=self.device, dtype=x0.dtype)
        x1 = torch.randn_like(x0)
        x_t = (1.0 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        v_target = x1 - x0
        t_idx = (t * float(self.diffusion_steps - 1)).round().long()

        cfg_mask = (torch.rand((b,), device=self.device) < self.cfg_drop_prob)
        if bool(cfg_mask.any().item()):
            content = content.clone()
            style = style.clone()
            content[cfg_mask] = 1.0
            style[cfg_mask] = 1.0

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            v_hat = self.model(
                x_t,
                t_idx,
                content,
                style,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )

            loss_fm = F.mse_loss(v_hat, v_target)
            # Keep source behavior scale for offset term.
            loss_off = self._offset_loss() / 2.0
            # Reconstruct x0 estimate from velocity for optional perceptual regularization.
            x0_hat = x_t - t.view(-1, 1, 1, 1) * v_hat
            if self.lambda_cp > 0.0:
                if self.perc_loss_fn is None:
                    raise ValueError("lambda_cp > 0 requires perceptual loss network.")
                loss_cp = self.perc_loss_fn(x0_hat, x0)
            else:
                loss_cp = torch.tensor(0.0, device=self.device, dtype=loss_fm.dtype)
            loss_cons = torch.tensor(0.0, device=self.device, dtype=loss_fm.dtype)
            if self.lambda_cons > 0.0:
                if part_imgs is None or part_imgs_b is None:
                    raise ValueError("lambda_cons > 0 requires parts and parts_b in each batch.")
                if not hasattr(self.model, "encode_part_vector"):
                    raise ValueError("lambda_cons > 0 requires model.encode_part_vector().")
                if not bool(getattr(self.model, "enable_parts_vector_condition", False)):
                    raise ValueError("lambda_cons > 0 requires part-vector conditioning enabled.")
                tok_a = self.model.encode_part_vector(part_imgs, part_mask)  # type: ignore[attr-defined]
                tok_b = self.model.encode_part_vector(part_imgs_b, part_mask_b)  # type: ignore[attr-defined]
                loss_cons = F.mse_loss(tok_a, tok_b)

            loss = (
                self.lambda_fm * loss_fm
                + self.lambda_off * loss_off
                + self.lambda_cp * loss_cp
                + self.lambda_cons * loss_cons
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
        self.lr_schedule.step()

        self.global_step += 1
        self.local_step += 1

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
                "loss_off": float(loss_off.item()),
                "loss_cp": float(loss_cp.item()),
                "loss_cons": float(loss_cons.item()),
                "cfg_dropped": int(cfg_mask.sum().item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
            }
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            self._write_step_log(log_row)

            msg = (
                f"[fm-step] gstep={self.global_step} epoch={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} fm={loss_fm.item():.4f} off={loss_off.item():.4f} "
                f"cp={loss_cp.item():.4f} cons={loss_cons.item():.4f} "
                f"cfg_drop={int(cfg_mask.sum().item())}/{int(b)} lr={self.lr_schedule.get_last_lr()[0]:.6e}"
            )
            if grad_norm is not None:
                msg += f" grad_norm={grad_norm:.4f}"
            print(msg, flush=True)

        return {
            "loss": loss.item(),
            "loss_fm": loss_fm.item(),
            "loss_off": loss_off.item(),
            "loss_cp": loss_cp.item(),
            "loss_cons": loss_cons.item(),
            "cfg_dropped": int(cfg_mask.sum().item()),
            "lr": self.lr_schedule.get_last_lr()[0],
        }

    @torch.no_grad()
    def flow_sample(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        c: int = 50,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        b, ch, h, w = content_img.shape
        device = self.device

        content_img = content_img.to(device)
        style_img = style_img.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)
        use_cfg = bool(self.sample_use_cfg)
        guidance_scale = float(self.sample_guidance_scale)
        if use_cfg:
            uncond_content = torch.ones_like(content_img)
            uncond_style = torch.ones_like(style_img)
            # Strict uncond branch: no parts condition.
            uncond_part_imgs = None
            uncond_part_mask = None
        else:
            uncond_content = None
            uncond_style = None
            uncond_part_imgs = None
            uncond_part_mask = None

        # Start from Gaussian endpoint x_1.
        x_t = torch.randn(b, ch, h, w, device=device)
        n_steps = max(2, int(c))
        dt = 1.0 / float(n_steps - 1)

        # Integrate backward from t=1 to t=0.
        for i in range(n_steps - 1):
            t_cur = 1.0 - i * dt
            t_tensor = torch.full((b,), t_cur, device=device, dtype=x_t.dtype)
            t_idx = (t_tensor * float(self.diffusion_steps - 1)).round().long()
            v_cond = self.model(
                x_t,
                t_idx,
                content_img,
                style_img,
                part_imgs=part_imgs,
                part_mask=part_mask,
            )
            if use_cfg:
                v_uncond = self.model(
                    x_t,
                    t_idx,
                    uncond_content,
                    uncond_style,
                    part_imgs=uncond_part_imgs,
                    part_mask=uncond_part_mask,
                )
                v_hat = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_hat = v_cond
            x_t = x_t - dt * v_hat
        return x_t.clamp(-1, 1)

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"].to(self.device)
        style = batch["style"].to(self.device)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        flow_steps = int(self.sample_inference_steps) if int(self.sample_inference_steps) > 1 else int(steps)
        sample = self.flow_sample(
            content,
            style,
            c=flow_steps,
            part_imgs=part_imgs,
            part_mask=part_mask,
        )
        filename = f"sample_ep{self.current_epoch}_gstep{self.global_step}_estep{self.local_step}.png"
        save_image((sample + 1) / 2, out_dir / filename)
