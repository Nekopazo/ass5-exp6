#!/usr/bin/env python3
"""
Training utilities for DiffuFont – Diffusion & Flow-Matching trainers.

Key changes from prior version
-------------------------------
* LR scheduler → linear warmup + cosine decay (per-step).
* InfoNCE contrastive loss on dual part-set views (lambda_nce).
* FlowMatchingTrainer overrides DDPM sampling methods to prevent misuse.
* CFG removed — model always sees full conditions during training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import DPMSolverMultistepScheduler

from .style_role_losses import (
    attention_entropy_order_loss,
    attention_mean_overlap,
    attention_overlap_margin_loss,
    attention_role_alignment_loss,
)


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


def slotwise_info_nce_loss(
    tok_a: torch.Tensor,
    tok_b: torch.Tensor,
    font_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if tok_a.shape != tok_b.shape:
        raise ValueError(f"slotwise_info_nce shape mismatch: {tuple(tok_a.shape)} vs {tuple(tok_b.shape)}")
    if tok_a.dim() != 3:
        raise ValueError(f"slotwise_info_nce expects 3D tokens, got {tuple(tok_a.shape)}")
    losses = []
    for idx in range(int(tok_a.size(1))):
        z_a = F.normalize(tok_a[:, idx], dim=-1)
        z_b = F.normalize(tok_b[:, idx], dim=-1)
        losses.append(info_nce_loss(z_a, z_b, font_ids=font_ids, temperature=float(temperature)))
    return torch.stack(losses).mean()


def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Off-diagonal token cosine^2; lower means better token specialization."""
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_fill(eye, 0.0)
    denom = max(1, int(sim.size(-1) * (sim.size(-1) - 1)))
    return (off_diag.pow(2).sum(dim=(1, 2)) / float(denom)).mean()


def proxy_band_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def slot_consistency_loss(tok1: torch.Tensor, tok2: torch.Tensor) -> torch.Tensor:
    if tok1.shape != tok2.shape:
        raise ValueError(f"slot consistency shape mismatch: {tuple(tok1.shape)} vs {tuple(tok2.shape)}")
    t1 = F.normalize(tok1, dim=-1)
    t2 = F.normalize(tok2, dim=-1)
    return (1.0 - (t1 * t2).sum(dim=-1)).mean()


def token_collapse_score(tokens: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal token cosine; high means token collapse."""
    t = F.normalize(tokens, dim=-1)
    sim = torch.matmul(t, t.transpose(1, 2))
    eye = torch.eye(int(sim.size(-1)), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(int(sim.size(0)), -1)
    return off_diag.mean()


def mean_tensor_loss(terms: list[torch.Tensor]) -> torch.Tensor:
    if not terms:
        raise ValueError("mean_tensor_loss requires at least one tensor")
    if len(terms) == 1:
        return terms[0]
    return torch.stack(terms).mean()


def cosine_same_diff(z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = F.cosine_similarity(z1, z2, dim=-1).mean()
    sim = z1 @ z2.t()
    b = int(sim.size(0))
    if b <= 1:
        diff = torch.zeros((), device=sim.device, dtype=sim.dtype)
    else:
        eye = torch.eye(b, device=sim.device, dtype=torch.bool)
        diff = sim.masked_select(~eye).mean()
    return same, diff


def probability_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(float(eps))
    return -(p * p.log()).sum(dim=-1)


def kl_to_uniform(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(float(eps))
    num_bins = max(1, int(p.size(-1)))
    return (p * (p.log() + math.log(float(num_bins)))).sum(dim=-1)


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
        lambda_nce: float = 0.0,
        lambda_slot_nce: float = 0.0,
        lambda_cons: float = 0.0,
        lambda_div: float = 0.0,
        lambda_proxy_low: float = 0.0,
        lambda_proxy_mid: float = 0.0,
        lambda_proxy_high: float = 0.0,
        lambda_attn_sep: float = 0.0,
        lambda_attn_order: float = 0.0,
        lambda_attn_role: float = 0.0,
        lambda_route_sparse: float = 0.0,
        lambda_route_balance: float = 0.0,
        lambda_route_div: float = 0.0,
        lambda_route_gate: float = 0.0,
        nce_temperature: float = 0.07,
        aux_loss_warmup_steps: int = 0,
        attn_overlap_margin: float = 0.80,
        attn_entropy_gap: float = 0.03,
        T: int = 1000,
        total_steps: int = 100_000,
        lr_warmup_steps: int = 0,
        lr_min_scale: float = 1e-3,
        sample_every_steps: int | None = None,
        save_every_steps: int | None = None,
        log_every_steps: int | None = None,
        detailed_log: bool = True,
        grad_accum_steps: int = 1,
        conditioning_mode: str = "part_only",
        part_drop_prob: float = 0.0,
        style_ref_drop_prob: float = 0.0,
        style_ref_drop_min_keep: int = 1,
        style_site_drop_prob: float = 0.0,
        style_site_drop_min_keep: int = 1,
        freeze_part_encoder_steps: int = 0,
        freeze_style_backbone_steps: int = 0,
        style_backbone_lr_scale: float = 0.1,
        cfg_drop_prob: float = 0.0,  # deprecated, kept for backward compat (ignored)
    ):
        self.device = device
        self.model = model.to(device)
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        self.part_drop_prob = float(max(0.0, min(1.0, part_drop_prob)))
        self.style_ref_drop_prob = float(max(0.0, min(1.0, style_ref_drop_prob)))
        self.style_ref_drop_min_keep = max(1, int(style_ref_drop_min_keep))
        self.style_site_drop_prob = float(max(0.0, min(1.0, style_site_drop_prob)))
        self.style_site_drop_min_keep = max(1, int(style_site_drop_min_keep))
        if hasattr(self.model, "set_style_site_dropout"):
            self.model.set_style_site_dropout(
                prob=self.style_site_drop_prob,
                min_keep=self.style_site_drop_min_keep,
            )

        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self._accum_step = 0
        self.freeze_part_encoder_steps = max(0, int(freeze_part_encoder_steps))
        self._part_encoder_is_frozen = False
        self.freeze_style_backbone_steps = max(0, int(freeze_style_backbone_steps))
        self.style_backbone_lr_scale = float(max(0.0, min(1.0, float(style_backbone_lr_scale))))
        self._style_backbone_low_is_frozen = False
        self._style_backbone_high_is_frozen = False

        low_params = list(self._iter_style_backbone_low_params())
        low_param_ids = {id(p) for p in low_params}
        high_params = list(self._iter_style_backbone_high_params())
        style_backbone_params = list(low_params) + [p for p in high_params if id(p) not in low_param_ids]
        style_backbone_param_ids = {id(p) for p in style_backbone_params}
        main_params = [p for p in self.model.parameters() if id(p) not in style_backbone_param_ids]
        if style_backbone_params:
            self.opt = torch.optim.AdamW(
                [
                    {"params": main_params, "lr": lr},
                    {"params": style_backbone_params, "lr": lr * self.style_backbone_lr_scale},
                ],
                weight_decay=1e-4,
            )
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if int(T) <= 0:
            raise ValueError(f"Diffusion timestep T must be > 0, got {T}")
        self.diffusion_steps = int(T)
        self.total_steps = max(1, int(total_steps))
        self.lr_warmup_steps = max(0, min(int(lr_warmup_steps), self.total_steps - 1))
        self.lr_min_scale = float(max(0.0, min(1.0, float(lr_min_scale))))

        # Per-step linear warmup followed by cosine decay.
        def _lr_lambda(step: int) -> float:
            step_i = max(0, min(int(step), self.total_steps))
            if self.lr_warmup_steps > 0 and step_i < self.lr_warmup_steps:
                return float(step_i + 1) / float(self.lr_warmup_steps)
            if self.total_steps <= self.lr_warmup_steps:
                return 1.0
            progress = float(step_i - self.lr_warmup_steps) / float(self.total_steps - self.lr_warmup_steps)
            progress = max(0.0, min(1.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lr_min_scale + (1.0 - self.lr_min_scale) * cosine

        self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=_lr_lambda)

        self.scheduler = NoiseScheduler(self.diffusion_steps).to(device)
        self.lambda_mse = float(lambda_mse)
        self.lambda_nce = float(lambda_nce)
        self.lambda_slot_nce = float(lambda_slot_nce)
        self.lambda_cons = float(lambda_cons)
        self.lambda_div = float(lambda_div)
        self.lambda_proxy_low = float(lambda_proxy_low)
        self.lambda_proxy_mid = float(lambda_proxy_mid)
        self.lambda_proxy_high = float(lambda_proxy_high)
        self.lambda_attn_sep = float(lambda_attn_sep)
        self.lambda_attn_order = float(lambda_attn_order)
        self.lambda_attn_role = float(lambda_attn_role)
        self.lambda_route_sparse = float(lambda_route_sparse)
        self.lambda_route_balance = float(lambda_route_balance)
        self.lambda_route_div = float(lambda_route_div)
        self.lambda_route_gate = float(lambda_route_gate)
        self.nce_temperature = float(nce_temperature)
        self.aux_loss_warmup_steps = max(0, int(aux_loss_warmup_steps))
        self.attn_overlap_margin = float(attn_overlap_margin)
        self.attn_entropy_gap = float(attn_entropy_gap)
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
        self.sample_inference_steps: int = 20
        self.save_split_components: bool = False

        # Fixed mixed precision path: always use bf16 autocast on CUDA.
        if self.device.type != "cuda":
            raise ValueError("DiffuFont training now requires CUDA because precision is fixed to bf16.")
        self.precision = "bf16"

        if self.freeze_part_encoder_steps > 0:
            frozen_params = self._set_part_encoder_requires_grad(False)
            if frozen_params > 0:
                self._part_encoder_is_frozen = True
                print(
                    f"[trainer] froze part encoder params={frozen_params} "
                    f"until global_step>={self.freeze_part_encoder_steps}",
                    flush=True,
                )
        if self.freeze_style_backbone_steps > 0:
            frozen_low = self._set_style_backbone_low_requires_grad(False)
            frozen_high = self._set_style_backbone_high_requires_grad(False)
            self._style_backbone_low_is_frozen = frozen_low > 0
            self._style_backbone_high_is_frozen = frozen_high > 0
            if frozen_low > 0 or frozen_high > 0:
                print(
                    "[trainer] froze style backbone "
                    f"low_params={frozen_low} high_params={frozen_high} "
                    f"until global_step>={self.freeze_style_backbone_steps} (fully unfreezes)",
                    flush=True,
                )

    def _set_part_encoder_requires_grad(self, requires_grad: bool) -> int:
        """Set requires_grad for part encoder submodules if present."""
        module_names = ("part_patch_encoder", "part_feat_to_token")
        changed = 0
        for name in module_names:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(requires_grad)
                changed += 1
        return changed

    def _maybe_unfreeze_part_encoder(self) -> None:
        if not self._part_encoder_is_frozen:
            return
        if self.global_step < self.freeze_part_encoder_steps:
            return
        unfrozen_params = self._set_part_encoder_requires_grad(True)
        self._part_encoder_is_frozen = False
        print(
            f"[trainer] unfroze part encoder at global_step={self.global_step} "
            f"(params={unfrozen_params})",
            flush=True,
        )

    def _iter_style_backbone_low_params(self):
        if hasattr(self.model, "iter_style_backbone_low_parameters"):
            yield from self.model.iter_style_backbone_low_parameters()

    def _iter_style_backbone_high_params(self):
        if hasattr(self.model, "iter_style_backbone_high_parameters"):
            yield from self.model.iter_style_backbone_high_parameters()

    def _set_style_backbone_low_requires_grad(self, requires_grad: bool) -> int:
        changed = 0
        for p in self._iter_style_backbone_low_params():
            p.requires_grad_(requires_grad)
            changed += 1
        return changed

    def _set_style_backbone_high_requires_grad(self, requires_grad: bool) -> int:
        changed = 0
        for p in self._iter_style_backbone_high_params():
            p.requires_grad_(requires_grad)
            changed += 1
        return changed

    def _set_style_backbone_modules_train_mode(self, *, low_training: bool, high_training: bool) -> None:
        if hasattr(self.model, "iter_style_backbone_low_modules"):
            for module in self.model.iter_style_backbone_low_modules():
                module.train(bool(low_training))
        if hasattr(self.model, "iter_style_backbone_high_modules"):
            for module in self.model.iter_style_backbone_high_modules():
                module.train(bool(high_training))

    def _maybe_unfreeze_style_backbone(self) -> None:
        if self._style_backbone_low_is_frozen or self._style_backbone_high_is_frozen:
            low_training = not self._style_backbone_low_is_frozen
            high_training = not self._style_backbone_high_is_frozen
            self._set_style_backbone_modules_train_mode(low_training=low_training, high_training=high_training)

        if (
            (self._style_backbone_low_is_frozen or self._style_backbone_high_is_frozen)
            and self.global_step >= self.freeze_style_backbone_steps
        ):
            unfrozen_low = self._set_style_backbone_low_requires_grad(True) if self._style_backbone_low_is_frozen else 0
            unfrozen_high = self._set_style_backbone_high_requires_grad(True) if self._style_backbone_high_is_frozen else 0
            self._style_backbone_low_is_frozen = False
            self._style_backbone_high_is_frozen = False
            self._set_style_backbone_modules_train_mode(low_training=True, high_training=True)
            print(
                f"[trainer] unfroze full style backbone at global_step={self.global_step} "
                f"(low_params={unfrozen_low}, high_params={unfrozen_high}, "
                f"style_backbone_lr_scale={self.style_backbone_lr_scale:g})",
                flush=True,
            )

    def _style_backbone_status(self) -> Dict[str, float]:
        return {
            "style_bb_low_frozen": float(self._style_backbone_low_is_frozen),
            "style_bb_high_frozen": float(self._style_backbone_high_is_frozen),
        }

    def _style_backbone_phase(self) -> str:
        if self._style_backbone_low_is_frozen or self._style_backbone_high_is_frozen:
            return "freeze_all"
        return "fully_trainable"

    # ------------------------------------------------------------------ #
    #  Gradient accumulation helpers
    # ------------------------------------------------------------------ #
    def _do_optimizer_update(
        self, loss: torch.Tensor, grad_clip: float | None
    ) -> tuple[bool, float | None, Dict[str, float]]:
        """Scale loss, backward, and conditionally step optimizer.

        Returns (did_update, grad_norm).
        did_update is True only when the accumulation cycle completes and
        the optimizer actually steps.
        """
        scaled_loss = loss / self.grad_accum_steps

        # zero_grad at the start of each accumulation cycle
        if self._accum_step == 0:
            _t_zero0 = time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            time_zero_grad = float(time.perf_counter() - _t_zero0)
        else:
            time_zero_grad = 0.0

        _t_backward0 = time.perf_counter()
        scaled_loss.backward()
        time_backward = float(time.perf_counter() - _t_backward0)

        self._accum_step += 1
        if self._accum_step < self.grad_accum_steps:
            # Not yet time to step
            return False, None, {
                "time_backward": time_backward,
                "time_grad_clip": 0.0,
                "time_optimizer_step": 0.0,
                "time_zero_grad": time_zero_grad,
            }

        # Complete cycle: perform optimizer step
        self._accum_step = 0
        grad_norm: float | None = None
        timing = {
            "time_backward": time_backward,
            "time_grad_clip": 0.0,
            "time_optimizer_step": 0.0,
            "time_zero_grad": time_zero_grad,
        }
        if grad_clip is not None:
            _t_clip0 = time.perf_counter()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip).item()
            )
            timing["time_grad_clip"] = float(time.perf_counter() - _t_clip0)
        _t_step0 = time.perf_counter()
        self.opt.step()
        timing["time_optimizer_step"] = float(time.perf_counter() - _t_step0)
        return True, grad_norm, timing

    def _flush_accumulated_grads(self, grad_clip: float | None = 1.0) -> None:
        """Flush any remaining accumulated gradients at epoch end.

        Called by train_epoch when the number of batches is not divisible
        by grad_accum_steps, so the last micro-batch group never completed.
        Only performs the optimizer step — does NOT increment global_step or
        lr_schedule, because no new training data was consumed.
        """
        if self._accum_step == 0:
            return
        self._accum_step = 0
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    #  Step log
    # ------------------------------------------------------------------ #
    def _write_step_log(self, row: Dict[str, float | int]) -> None:
        if self.step_log_file is None:
            return
        self.step_log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.step_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _set_style_site_force_keep(self, keep_sites: tuple[str, ...] | None) -> None:
        if hasattr(self.model, "set_style_site_force_keep"):
            self.model.set_style_site_force_keep(keep_sites)

    @property
    def effective_aux_loss_scale(self) -> float:
        """Shared linear warmup scale for all auxiliary losses except the main reconstruction loss."""
        if self.aux_loss_warmup_steps <= 0:
            return 1.0
        return min(1.0, self.global_step / self.aux_loss_warmup_steps)

    @staticmethod
    def _mode_uses_parts(mode: str) -> bool:
        _ = str(mode).strip().lower()
        return False

    @staticmethod
    def _mode_uses_style(mode: str) -> bool:
        m = str(mode).strip().lower()
        return m in {"part_only", "style_only", "parts_vector_only"}

    def _ensure_required_conditions(
        self,
        mode: str,
        style_img: torch.Tensor | None,
        stage: str,
    ) -> None:
        if self._mode_uses_style(mode) and style_img is None:
            raise ValueError(
                f"{stage}: conditioning_mode='{mode}' requires style_img, but style_img is missing."
            )

    def _select_style_view(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        style_img_v1 = batch["style_img"].to(self.device) if "style_img" in batch else None
        style_ref_mask_v1 = (
            batch["style_ref_mask"].to(self.device) if "style_ref_mask" in batch else None
        )
        style_img_v2 = batch["style_img_view2"].to(self.device) if "style_img_view2" in batch else None
        style_ref_mask_v2 = (
            batch["style_ref_mask_view2"].to(self.device) if "style_ref_mask_view2" in batch else None
        )
        if style_img_v1 is None and style_img_v2 is None:
            return None, None
        if style_img_v1 is not None:
            return style_img_v1, style_ref_mask_v1
        return style_img_v2, style_ref_mask_v2

    def _apply_style_ref_dropout(self, style_ref_mask: torch.Tensor | None) -> torch.Tensor | None:
        if style_ref_mask is None:
            return None
        if self.style_ref_drop_prob <= 0.0:
            return style_ref_mask
        if style_ref_mask.dim() != 2:
            return style_ref_mask

        valid = style_ref_mask > 0
        if not bool(valid.any().item()):
            return style_ref_mask

        keep = valid & (torch.rand_like(style_ref_mask) >= self.style_ref_drop_prob)
        valid_count = valid.sum(dim=1)
        keep_count = keep.sum(dim=1)
        min_keep = torch.full_like(valid_count, self.style_ref_drop_min_keep)
        target_keep = torch.minimum(valid_count, min_keep).clamp_min(1)

        rows_need = torch.nonzero(keep_count < target_keep, as_tuple=False).flatten()
        for row in rows_need.tolist():
            need = int(target_keep[row].item() - keep_count[row].item())
            if need <= 0:
                continue
            cand = torch.nonzero(valid[row] & ~keep[row], as_tuple=False).flatten()
            if cand.numel() <= 0:
                continue
            perm = torch.randperm(cand.numel(), device=style_ref_mask.device)
            picked = cand.index_select(0, perm[:need])
            keep[row, picked] = True

        return keep.to(dtype=style_ref_mask.dtype)

    def _compute_route_losses(
        self,
        route_views: list[dict[str, dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        out: Dict[str, torch.Tensor] = {
            "loss_route_sparse": zero,
            "loss_route_balance": zero,
            "loss_route_div": zero,
            "loss_route_gate": zero,
            "route_gate_mid": zero,
            "route_gate_up_16": zero,
            "route_gate_up_32": zero,
            "route_mid_to_mid": zero,
            "route_mid_to_up_16": zero,
            "route_mid_to_up_32": zero,
            "route_up_16_to_mid": zero,
            "route_up_16_to_up_16": zero,
            "route_up_16_to_up_32": zero,
            "route_up_32_to_mid": zero,
            "route_up_32_to_up_16": zero,
            "route_up_32_to_up_32": zero,
            "route_cos_mid_up_16": zero,
            "route_cos_mid_up_32": zero,
            "route_cos_up_16_up_32": zero,
        }
        if not route_views:
            return out

        site_names = ("mid", "up_16", "up_32")
        alpha_by_site: dict[str, list[torch.Tensor]] = {site: [] for site in site_names}
        gate_by_site: dict[str, list[torch.Tensor]] = {site: [] for site in site_names}

        for route_view in route_views:
            for site in site_names:
                site_stats = route_view.get(site)
                if not site_stats:
                    continue
                alpha = site_stats.get("alpha")
                gate = site_stats.get("gate")
                if alpha is not None:
                    alpha_by_site[site].append(alpha)
                if gate is not None:
                    gate_by_site[site].append(gate)

        entropy_terms: list[torch.Tensor] = []
        balance_terms: list[torch.Tensor] = []
        gate_terms: list[torch.Tensor] = []

        for site in site_names:
            if alpha_by_site[site]:
                alpha_cat = torch.cat(alpha_by_site[site], dim=0)
                denom = math.log(max(2, int(alpha_cat.size(-1))))
                entropy_terms.append(probability_entropy(alpha_cat).mean() / denom)
                balance_terms.append(kl_to_uniform(alpha_cat.mean(dim=0, keepdim=True)).mean())
                metric_site = site
                out[f"route_{metric_site}_to_mid"] = alpha_cat[:, 0].mean()
                out[f"route_{metric_site}_to_up_16"] = alpha_cat[:, 1].mean()
                out[f"route_{metric_site}_to_up_32"] = alpha_cat[:, 2].mean()
            if gate_by_site[site]:
                gate_cat = torch.cat(gate_by_site[site], dim=0)
                gate_terms.append(gate_cat.mean())
                out[f"route_gate_{site}"] = gate_cat.mean()

        if entropy_terms:
            out["loss_route_sparse"] = torch.stack(entropy_terms).mean()
        if balance_terms:
            out["loss_route_balance"] = torch.stack(balance_terms).mean()
        if gate_terms:
            out["loss_route_gate"] = torch.stack(gate_terms).mean()

        pair_terms: list[torch.Tensor] = []
        pair_metric_map = {
            ("mid", "up_16"): "route_cos_mid_up_16",
            ("mid", "up_32"): "route_cos_mid_up_32",
            ("up_16", "up_32"): "route_cos_up_16_up_32",
        }
        pair_counts = {metric_name: 0 for metric_name in pair_metric_map.values()}
        for route_view in route_views:
            for (site_a, site_b), metric_name in pair_metric_map.items():
                stats_a = route_view.get(site_a)
                stats_b = route_view.get(site_b)
                if not stats_a or not stats_b:
                    continue
                alpha_a = stats_a.get("alpha")
                alpha_b = stats_b.get("alpha")
                if alpha_a is None or alpha_b is None:
                    continue
                cos_val = F.cosine_similarity(alpha_a, alpha_b, dim=-1).mean()
                pair_terms.append(cos_val)
                out[metric_name] = out[metric_name] + cos_val
                pair_counts[metric_name] += 1
        for metric_name in pair_metric_map.values():
            if pair_counts[metric_name] > 0:
                out[metric_name] = out[metric_name] / pair_counts[metric_name]
        if pair_terms:
            out["loss_route_div"] = torch.stack(pair_terms).mean()

        return out

    # ------------------------------------------------------------------ #
    #  Style regularizers on dual reference views
    # ------------------------------------------------------------------ #
    def _compute_style_losses(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        out: Dict[str, torch.Tensor] = {
            "loss_nce": zero,
            "loss_slot_nce": zero,
            "loss_cons": zero,
            "loss_div": zero,
            "loss_proxy_low": zero,
            "loss_proxy_mid": zero,
            "loss_proxy_high": zero,
            "loss_attn_sep": zero,
            "loss_attn_order": zero,
            "loss_attn_role": zero,
            "loss_attn_role_low": zero,
            "loss_attn_role_mid": zero,
            "loss_attn_role_high": zero,
            "attn_overlap": zero,
            "attn_entropy_low": zero,
            "attn_entropy_mid": zero,
            "attn_entropy_high": zero,
            "cos_same": zero,
            "cos_diff": zero,
            "token_collapse": zero,
            "loss_route_sparse": zero,
            "loss_route_balance": zero,
            "loss_route_div": zero,
            "loss_route_gate": zero,
            "route_gate_mid": zero,
            "route_gate_up_16": zero,
            "route_gate_up_32": zero,
            "route_mid_to_mid": zero,
            "route_mid_to_up_16": zero,
            "route_mid_to_up_32": zero,
            "route_up_16_to_mid": zero,
            "route_up_16_to_up_16": zero,
            "route_up_16_to_up_32": zero,
            "route_up_32_to_mid": zero,
            "route_up_32_to_up_16": zero,
            "route_up_32_to_up_32": zero,
            "route_cos_mid_up_16": zero,
            "route_cos_mid_up_32": zero,
            "route_cos_up_16_up_32": zero,
        }
        if not self._mode_uses_style(self.conditioning_mode):
            return out
        if "style_img" not in batch:
            return out

        style_img_v1 = batch["style_img"].to(self.device)
        style_ref_mask_v1 = batch.get("style_ref_mask", None)
        if style_ref_mask_v1 is not None:
            style_ref_mask_v1 = style_ref_mask_v1.to(self.device)

        style_img_v2 = batch.get("style_img_view2", None)
        style_ref_mask_v2 = batch.get("style_ref_mask_view2", None)
        if style_img_v2 is None:
            style_img_v2 = style_img_v1
        else:
            style_img_v2 = style_img_v2.to(self.device)
        if style_ref_mask_v2 is None:
            style_ref_mask_v2 = style_ref_mask_v1
        elif style_ref_mask_v2 is not None:
            style_ref_mask_v2 = style_ref_mask_v2.to(self.device)

        style_ref_mask_v1 = self._apply_style_ref_dropout(style_ref_mask_v1)
        style_ref_mask_v2 = self._apply_style_ref_dropout(style_ref_mask_v2)
        content_img = batch.get("content", None)
        if content_img is not None:
            content_img = content_img.to(self.device)

        route_views: list[dict[str, dict[str, torch.Tensor]]] = []
        if content_img is not None and hasattr(self.model, "compute_style_route_diagnostics"):
            route_v1 = self.model.compute_style_route_diagnostics(
                content_img=content_img,
                style_img=style_img_v1,
                style_ref_mask=style_ref_mask_v1,
                detach=False,
            )
            if route_v1:
                route_views.append(route_v1)
            route_v2 = self.model.compute_style_route_diagnostics(
                content_img=content_img,
                style_img=style_img_v2,
                style_ref_mask=style_ref_mask_v2,
                detach=False,
            )
            if route_v2:
                route_views.append(route_v2)
        out.update(self._compute_route_losses(route_views))

        attn1 = None
        attn2 = None
        site_tok1 = None
        site_tok2 = None
        mem_mid_1 = None
        mem_mid_2 = None
        mem_up16_1 = None
        mem_up16_2 = None
        mem_up32_1 = None
        mem_up32_2 = None
        if hasattr(self.model, "encode_style_pack_full"):
            pack1, proxy1, attn1 = self.model.encode_style_pack_full(
                style_img_v1,
                style_ref_mask=style_ref_mask_v1,
            )
            pack2, proxy2, attn2 = self.model.encode_style_pack_full(
                style_img_v2,
                style_ref_mask=style_ref_mask_v2,
            )
            tok1 = pack1["tokens"]
            tok2 = pack2["tokens"]
            mem_mid_1 = pack1.get("memory_mid_8", None)
            mem_mid_2 = pack2.get("memory_mid_8", None)
            mem_up16_1 = pack1["memory_up_16"]
            mem_up16_2 = pack2["memory_up_16"]
            mem_up32_1 = pack1["memory_up_32"]
            mem_up32_2 = pack2["memory_up_32"]
            if hasattr(self.model, "stack_style_site_contexts_from_pack"):
                site_tok1 = self.model.stack_style_site_contexts_from_pack(pack1)
                site_tok2 = self.model.stack_style_site_contexts_from_pack(pack2)
        elif hasattr(self.model, "encode_style_tokens_full"):
            tok1, proxy1, attn1 = self.model.encode_style_tokens_full(
                style_img_v1,
                style_ref_mask=style_ref_mask_v1,
            )
            tok2, proxy2, attn2 = self.model.encode_style_tokens_full(
                style_img_v2,
                style_ref_mask=style_ref_mask_v2,
            )
        elif hasattr(self.model, "encode_style_tokens_with_proxy"):
            tok1, proxy1 = self.model.encode_style_tokens_with_proxy(
                style_img_v1,
                style_ref_mask=style_ref_mask_v1,
            )
            tok2, proxy2 = self.model.encode_style_tokens_with_proxy(
                style_img_v2,
                style_ref_mask=style_ref_mask_v2,
            )
        else:
            tok1 = self.model.encode_style_tokens(
                style_img_v1,
                style_ref_mask=style_ref_mask_v1,
            )
            tok2 = self.model.encode_style_tokens(
                style_img_v2,
                style_ref_mask=style_ref_mask_v2,
            )
        if "proxy1" in locals() and "proxy2" in locals():
            out["loss_proxy_low"] = 0.5 * (
                proxy_band_loss(proxy1["pred_low"], proxy1["target_low"])
                + proxy_band_loss(proxy2["pred_low"], proxy2["target_low"])
            )
            out["loss_proxy_mid"] = 0.5 * (
                proxy_band_loss(proxy1["pred_mid"], proxy1["target_mid"])
                + proxy_band_loss(proxy2["pred_mid"], proxy2["target_mid"])
            )
            out["loss_proxy_high"] = 0.5 * (
                proxy_band_loss(proxy1["pred_high"], proxy1["target_high"])
                + proxy_band_loss(proxy2["pred_high"], proxy2["target_high"])
            )
        z1_src = tok1.mean(dim=1)
        z2_src = tok2.mean(dim=1)
        if mem_mid_1 is not None and mem_mid_2 is not None and site_tok1 is not None and site_tok2 is not None:
            z1_src = 0.5 * (mem_mid_1.mean(dim=1) + site_tok1[:, 1:].mean(dim=1))
            z2_src = 0.5 * (mem_mid_2.mean(dim=1) + site_tok2[:, 1:].mean(dim=1))
        elif site_tok1 is not None and site_tok2 is not None:
            z1_src = 0.5 * (z1_src + site_tok1.mean(dim=1))
            z2_src = 0.5 * (z2_src + site_tok2.mean(dim=1))
        z1 = F.normalize(z1_src, dim=-1)
        z2 = F.normalize(z2_src, dim=-1)

        font_ids = batch.get("font_ids", None)
        if font_ids is None:
            b = int(z1.size(0))
            font_ids = torch.arange(b, device=self.device, dtype=torch.long)
        else:
            font_ids = font_ids.to(self.device, dtype=torch.long)

        out["loss_nce"] = info_nce_loss(z1, z2, font_ids=font_ids, temperature=self.nce_temperature)
        out["cos_same"], out["cos_diff"] = cosine_same_diff(z1, z2)
        if (
            site_tok1 is not None
            and site_tok2 is not None
            and mem_mid_1 is not None
            and mem_mid_2 is not None
            and mem_up16_1 is not None
            and mem_up16_2 is not None
            and mem_up32_1 is not None
            and mem_up32_2 is not None
        ):
            out["loss_slot_nce"] = mean_tensor_loss(
                [
                    slotwise_info_nce_loss(site_tok1, site_tok2, font_ids=font_ids, temperature=self.nce_temperature),
                    slotwise_info_nce_loss(mem_mid_1, mem_mid_2, font_ids=font_ids, temperature=self.nce_temperature),
                    slotwise_info_nce_loss(mem_up16_1, mem_up16_2, font_ids=font_ids, temperature=self.nce_temperature),
                    slotwise_info_nce_loss(mem_up32_1, mem_up32_2, font_ids=font_ids, temperature=self.nce_temperature),
                ]
            )
            out["loss_cons"] = mean_tensor_loss(
                [
                    slot_consistency_loss(site_tok1, site_tok2),
                    slot_consistency_loss(mem_mid_1, mem_mid_2),
                    slot_consistency_loss(mem_up16_1, mem_up16_2),
                    slot_consistency_loss(mem_up32_1, mem_up32_2),
                ]
            )
            out["loss_div"] = mean_tensor_loss(
                [
                    token_diversity_loss(site_tok1),
                    token_diversity_loss(site_tok2),
                    token_diversity_loss(mem_mid_1),
                    token_diversity_loss(mem_mid_2),
                    token_diversity_loss(mem_up16_1),
                    token_diversity_loss(mem_up16_2),
                    token_diversity_loss(mem_up32_1),
                    token_diversity_loss(mem_up32_2),
                ]
            )
            out["token_collapse"] = mean_tensor_loss(
                [
                    token_collapse_score(site_tok1),
                    token_collapse_score(site_tok2),
                    token_collapse_score(mem_mid_1),
                    token_collapse_score(mem_mid_2),
                    token_collapse_score(mem_up16_1),
                    token_collapse_score(mem_up16_2),
                    token_collapse_score(mem_up32_1),
                    token_collapse_score(mem_up32_2),
                ]
            )
        else:
            out["loss_slot_nce"] = slotwise_info_nce_loss(tok1, tok2, font_ids=font_ids, temperature=self.nce_temperature)
            if site_tok1 is not None and site_tok2 is not None:
                out["loss_slot_nce"] = 0.5 * (
                    out["loss_slot_nce"]
                    + slotwise_info_nce_loss(site_tok1, site_tok2, font_ids=font_ids, temperature=self.nce_temperature)
                )
            out["loss_cons"] = slot_consistency_loss(tok1, tok2)
            if site_tok1 is not None and site_tok2 is not None:
                out["loss_cons"] = 0.5 * (out["loss_cons"] + slot_consistency_loss(site_tok1, site_tok2))
            out["loss_div"] = 0.5 * (token_diversity_loss(tok1) + token_diversity_loss(tok2))
            out["token_collapse"] = 0.5 * (token_collapse_score(tok1) + token_collapse_score(tok2))
        if attn1 is not None and attn2 is not None:
            out["loss_attn_sep"] = 0.5 * (
                attention_overlap_margin_loss(
                    attn1,
                    style_ref_mask=style_ref_mask_v1,
                    margin=self.attn_overlap_margin,
                )
                + attention_overlap_margin_loss(
                    attn2,
                    style_ref_mask=style_ref_mask_v2,
                    margin=self.attn_overlap_margin,
                )
            )
            loss_attn_order_1, ent1 = attention_entropy_order_loss(
                attn1,
                style_ref_mask=style_ref_mask_v1,
                gap=self.attn_entropy_gap,
            )
            loss_attn_order_2, ent2 = attention_entropy_order_loss(
                attn2,
                style_ref_mask=style_ref_mask_v2,
                gap=self.attn_entropy_gap,
            )
            out["loss_attn_order"] = 0.5 * (loss_attn_order_1 + loss_attn_order_2)
            loss_attn_role_1, role_vec_1, _ = attention_role_alignment_loss(
                attn1,
                style_img_v1,
                style_ref_mask=style_ref_mask_v1,
            )
            loss_attn_role_2, role_vec_2, _ = attention_role_alignment_loss(
                attn2,
                style_img_v2,
                style_ref_mask=style_ref_mask_v2,
            )
            role_vec = 0.5 * (role_vec_1 + role_vec_2)
            out["loss_attn_role"] = 0.5 * (loss_attn_role_1 + loss_attn_role_2)
            out["loss_attn_role_low"] = role_vec[0]
            out["loss_attn_role_mid"] = role_vec[1]
            out["loss_attn_role_high"] = role_vec[2]
            out["attn_overlap"] = 0.5 * (
                attention_mean_overlap(attn1, style_ref_mask=style_ref_mask_v1)
                + attention_mean_overlap(attn2, style_ref_mask=style_ref_mask_v2)
            )
            ent = 0.5 * (ent1 + ent2)
            out["attn_entropy_low"] = ent[0]
            out["attn_entropy_mid"] = ent[1]
            out["attn_entropy_high"] = ent[2]
        return out

    # ------------------------------------------------------------------ #
    #  train_step
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Dict[str, torch.Tensor], grad_clip: float | None = 1.0):
        _train_t0 = time.perf_counter()
        timing_main_forward = 0.0
        timing_style_losses = 0.0
        timing_counterfactual = 0.0
        timing_opt_step = 0.0
        self.model.train()
        self._maybe_unfreeze_part_encoder()
        self._maybe_unfreeze_style_backbone()
        x0 = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style_img, style_ref_mask = self._select_style_view(batch)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        self._ensure_required_conditions(self.conditioning_mode, style_img, stage="train_step")
        if self._mode_uses_style(self.conditioning_mode):
            style_ref_mask = self._apply_style_ref_dropout(style_ref_mask)
        B = x0.size(0)
        t = torch.randint(0, self.scheduler.T, (B,), device=self.device)

        # Optional part drop path (inactive when part branch is disabled).
        part_drop_mask = None
        if (
            self.part_drop_prob > 0.0
            and part_imgs is not None
            and self._mode_uses_parts(self.conditioning_mode)
        ):
            part_drop_mask = (torch.rand((B,), device=self.device) < self.part_drop_prob)
            if bool(part_drop_mask.any().item()):
                part_imgs = part_imgs.clone()
                part_imgs[part_drop_mask] = 1.0
                if part_mask is not None:
                    part_mask = part_mask.clone()
                    part_mask[part_drop_mask] = 0.0

        # Convert pixel → latent BEFORE adding noise (no grad needed for bilinear resize).
        with torch.no_grad():
            x0_latent = self.model.encode_to_latent(x0)     # (B,1,128,128) → (B,1,128,128) no-op (dataset already resized)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_t_latent, eps_latent = self.scheduler.add_noise(x0_latent, t)
            _t_main0 = time.perf_counter()
            eps_hat_latent = self.model(
                x_t_latent, t, content,
                style_img=style_img,
                part_imgs=part_imgs,
                part_mask=part_mask,
                condition_mode=self.conditioning_mode,
                style_ref_mask=style_ref_mask,
            )
            loss_mse = F.mse_loss(eps_hat_latent, eps_latent)
            timing_main_forward += float(time.perf_counter() - _t_main0)
            _t_style0 = time.perf_counter()
            style_losses = self._compute_style_losses(batch)
            timing_style_losses += float(time.perf_counter() - _t_style0)
            loss_nce = style_losses["loss_nce"]
            loss_slot_nce = style_losses["loss_slot_nce"]
            loss_cons = style_losses["loss_cons"]
            loss_div = style_losses["loss_div"]
            loss_proxy_low = style_losses["loss_proxy_low"]
            loss_proxy_mid = style_losses["loss_proxy_mid"]
            loss_proxy_high = style_losses["loss_proxy_high"]
            loss_attn_sep = style_losses["loss_attn_sep"]
            loss_attn_order = style_losses["loss_attn_order"]
            loss_attn_role = style_losses["loss_attn_role"]
            loss_attn_role_low = style_losses["loss_attn_role_low"]
            loss_attn_role_mid = style_losses["loss_attn_role_mid"]
            loss_attn_role_high = style_losses["loss_attn_role_high"]
            attn_overlap = style_losses["attn_overlap"]
            attn_entropy_low = style_losses["attn_entropy_low"]
            attn_entropy_mid = style_losses["attn_entropy_mid"]
            attn_entropy_high = style_losses["attn_entropy_high"]
            cos_same = style_losses["cos_same"]
            cos_diff = style_losses["cos_diff"]
            token_collapse = style_losses["token_collapse"]
            loss_route_sparse = style_losses["loss_route_sparse"]
            loss_route_balance = style_losses["loss_route_balance"]
            loss_route_div = style_losses["loss_route_div"]
            loss_route_gate = style_losses["loss_route_gate"]
            route_gate_mid = style_losses["route_gate_mid"]
            route_gate_up_16 = style_losses["route_gate_up_16"]
            route_gate_up_32 = style_losses["route_gate_up_32"]
            route_mid_to_mid = style_losses["route_mid_to_mid"]
            route_mid_to_up_16 = style_losses["route_mid_to_up_16"]
            route_mid_to_up_32 = style_losses["route_mid_to_up_32"]
            route_up_16_to_mid = style_losses["route_up_16_to_mid"]
            route_up_16_to_up_16 = style_losses["route_up_16_to_up_16"]
            route_up_16_to_up_32 = style_losses["route_up_16_to_up_32"]
            route_up_32_to_mid = style_losses["route_up_32_to_mid"]
            route_up_32_to_up_16 = style_losses["route_up_32_to_up_16"]
            route_up_32_to_up_32 = style_losses["route_up_32_to_up_32"]
            route_cos_mid_up_16 = style_losses["route_cos_mid_up_16"]
            route_cos_mid_up_32 = style_losses["route_cos_mid_up_32"]
            route_cos_up_16_up_32 = style_losses["route_cos_up_16_up_32"]
            route_metrics = {
                "loss_route_sparse": loss_route_sparse.item(),
                "loss_route_balance": loss_route_balance.item(),
                "loss_route_div": loss_route_div.item(),
                "loss_route_gate": loss_route_gate.item(),
                "route_gate_mid": route_gate_mid.item(),
                "route_gate_up_16": route_gate_up_16.item(),
                "route_gate_up_32": route_gate_up_32.item(),
                "route_mid_to_mid": route_mid_to_mid.item(),
                "route_mid_to_up_16": route_mid_to_up_16.item(),
                "route_mid_to_up_32": route_mid_to_up_32.item(),
                "route_up_16_to_mid": route_up_16_to_mid.item(),
                "route_up_16_to_up_16": route_up_16_to_up_16.item(),
                "route_up_16_to_up_32": route_up_16_to_up_32.item(),
                "route_up_32_to_mid": route_up_32_to_mid.item(),
                "route_up_32_to_up_16": route_up_32_to_up_16.item(),
                "route_up_32_to_up_32": route_up_32_to_up_32.item(),
                "route_cos_mid_up_16": route_cos_mid_up_16.item(),
                "route_cos_mid_up_32": route_cos_mid_up_32.item(),
                "route_cos_up_16_up_32": route_cos_up_16_up_32.item(),
            }
            eff_aux = self.effective_aux_loss_scale
            eff_lnce = eff_aux * self.lambda_nce
            eff_lslot = eff_aux * self.lambda_slot_nce
            eff_lcons = eff_aux * self.lambda_cons
            eff_ldiv = eff_aux * self.lambda_div
            eff_lproxy_low = eff_aux * self.lambda_proxy_low
            eff_lproxy_mid = eff_aux * self.lambda_proxy_mid
            eff_lproxy_high = eff_aux * self.lambda_proxy_high
            eff_lattn_sep = eff_aux * self.lambda_attn_sep
            eff_lattn_order = eff_aux * self.lambda_attn_order
            eff_lattn_role = eff_aux * self.lambda_attn_role
            eff_lroute_sparse = eff_aux * self.lambda_route_sparse
            eff_lroute_balance = eff_aux * self.lambda_route_balance
            eff_lroute_div = eff_aux * self.lambda_route_div
            eff_lroute_gate = eff_aux * self.lambda_route_gate
            loss = (
                self.lambda_mse * loss_mse
                + eff_lnce * loss_nce
                + eff_lslot * loss_slot_nce
                + eff_lcons * loss_cons
                + eff_ldiv * loss_div
                + eff_lproxy_low * loss_proxy_low
                + eff_lproxy_mid * loss_proxy_mid
                + eff_lproxy_high * loss_proxy_high
                + eff_lattn_sep * loss_attn_sep
                + eff_lattn_order * loss_attn_order
                + eff_lattn_role * loss_attn_role
                + eff_lroute_sparse * loss_route_sparse
                + eff_lroute_balance * loss_route_balance
                + eff_lroute_div * loss_route_div
                + eff_lroute_gate * loss_route_gate
            )

        counterfactual_stats: Dict[str, float] = {}
        should_log_counterfactual = bool(
            self.log_every_steps and ((self.global_step + 1) % self.log_every_steps == 0)
        )
        if should_log_counterfactual and style_img is not None and self._mode_uses_style(self.conditioning_mode):
            _t_cf0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if part_imgs is not None and self._mode_uses_parts(self.conditioning_mode):
                    eps_hat_part_only = self.model(
                        x_t_latent, t, content,
                        style_img=None,
                        part_imgs=part_imgs,
                        part_mask=part_mask,
                        condition_mode="part_only",
                    )
                    cf_loss_part_only = F.mse_loss(eps_hat_part_only, eps_latent)
                else:
                    cf_loss_part_only = loss_mse.detach()
                eps_hat_style_only = self.model(
                    x_t_latent, t, content,
                    style_img=style_img,
                    part_imgs=None,
                    part_mask=None,
                    condition_mode="style_only",
                    style_ref_mask=style_ref_mask,
                )
                cf_loss_style_only = F.mse_loss(eps_hat_style_only, eps_latent)
                solo_site_losses: Dict[str, float] = {}
                if self.conditioning_mode == "style_only" and hasattr(self.model, "set_style_site_force_keep"):
                    try:
                        for site_name, stat_key in (
                            ("mid", "cf_loss_style_mid_only"),
                            ("up_16", "cf_loss_style_up16_only"),
                            ("up_32", "cf_loss_style_up32_only"),
                        ):
                            self._set_style_site_force_keep((site_name,))
                            eps_hat_solo = self.model(
                                x_t_latent, t, content,
                                style_img=style_img,
                                part_imgs=None,
                                part_mask=None,
                                condition_mode="style_only",
                                style_ref_mask=style_ref_mask,
                            )
                            solo_site_losses[stat_key] = float(F.mse_loss(eps_hat_solo, eps_latent).item())
                    finally:
                        self._set_style_site_force_keep(None)
            timing_counterfactual += float(time.perf_counter() - _t_cf0)
            base = float(loss_mse.detach().item())
            lp = float(cf_loss_part_only.item())
            ls = float(cf_loss_style_only.item())
            counterfactual_stats = {
                "cf_loss_both": base,
                "cf_loss_part_only": lp,
                "cf_loss_style_only": ls,
                # Remove-style effect: compare both vs part_only.
                "cf_delta_drop_style": lp - base,
                # Remove-part effect: compare both vs style_only.
                "cf_delta_drop_part": ls - base,
            }
            counterfactual_stats.update(solo_site_losses)
            if solo_site_losses:
                counterfactual_stats["cf_delta_style_mid_only"] = solo_site_losses["cf_loss_style_mid_only"] - base
                counterfactual_stats["cf_delta_style_up16_only"] = solo_site_losses["cf_loss_style_up16_only"] - base
                counterfactual_stats["cf_delta_style_up32_only"] = solo_site_losses["cf_loss_style_up32_only"] - base

        did_update, grad_norm, opt_timing = self._do_optimizer_update(loss, grad_clip)
        timing_opt_step += float(
            opt_timing["time_backward"]
            + opt_timing["time_grad_clip"]
            + opt_timing["time_optimizer_step"]
            + opt_timing["time_zero_grad"]
        )
        # For micro-steps (not yet an optimizer update), return early with metrics
        if not did_update:
            out = {
                "loss": loss.item(),
                "loss_mse": loss_mse.item(),
                "loss_nce": loss_nce.item(),
                "loss_slot_nce": loss_slot_nce.item(),
                "loss_cons": loss_cons.item(),
                "loss_div": loss_div.item(),
                "loss_proxy_low": loss_proxy_low.item(),
                "loss_proxy_mid": loss_proxy_mid.item(),
                "loss_proxy_high": loss_proxy_high.item(),
                "loss_attn_sep": loss_attn_sep.item(),
                "loss_attn_order": loss_attn_order.item(),
                "loss_attn_role": loss_attn_role.item(),
                "loss_attn_role_low": loss_attn_role_low.item(),
                "loss_attn_role_mid": loss_attn_role_mid.item(),
                "loss_attn_role_high": loss_attn_role_high.item(),
                "attn_overlap": attn_overlap.item(),
                "attn_entropy_low": attn_entropy_low.item(),
                "attn_entropy_mid": attn_entropy_mid.item(),
                "attn_entropy_high": attn_entropy_high.item(),
                "cos_same": cos_same.item(),
                "cos_diff": cos_diff.item(),
                "token_collapse": token_collapse.item(),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(time.perf_counter() - _train_t0),
                "time_main_forward": float(timing_main_forward),
                "time_style_losses": float(timing_style_losses),
                "time_counterfactual": float(timing_counterfactual),
                "time_opt_step": float(timing_opt_step),
                "time_backward": float(opt_timing["time_backward"]),
                "time_grad_clip": float(opt_timing["time_grad_clip"]),
                "time_optimizer_step": float(opt_timing["time_optimizer_step"]),
                "time_zero_grad": float(opt_timing["time_zero_grad"]),
            }
            out.update(route_metrics)
            out.update(self._style_backbone_status())
            out.update(counterfactual_stats)
            return out

        # LR scheduler steps once per optimizer update.
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
                "loss_slot_nce": float(loss_slot_nce.item()),
                "loss_cons": float(loss_cons.item()),
                "loss_div": float(loss_div.item()),
                "loss_proxy_low": float(loss_proxy_low.item()),
                "loss_proxy_mid": float(loss_proxy_mid.item()),
                "loss_proxy_high": float(loss_proxy_high.item()),
                "loss_attn_sep": float(loss_attn_sep.item()),
                "loss_attn_order": float(loss_attn_order.item()),
                "loss_attn_role": float(loss_attn_role.item()),
                "loss_attn_role_low": float(loss_attn_role_low.item()),
                "loss_attn_role_mid": float(loss_attn_role_mid.item()),
                "loss_attn_role_high": float(loss_attn_role_high.item()),
                "attn_overlap": float(attn_overlap.item()),
                "attn_entropy_low": float(attn_entropy_low.item()),
                "attn_entropy_mid": float(attn_entropy_mid.item()),
                "attn_entropy_high": float(attn_entropy_high.item()),
                "cos_same": float(cos_same.item()),
                "cos_diff": float(cos_diff.item()),
                "token_collapse": float(token_collapse.item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(self._step_train_time),
                "data_time_ema": float(self._step_data_time_ema),
                "train_time_ema": float(self._step_train_time_ema),
                "time_main_forward": float(timing_main_forward),
                "time_style_losses": float(timing_style_losses),
                "time_counterfactual": float(timing_counterfactual),
                "time_opt_step": float(timing_opt_step),
                "time_backward": float(opt_timing["time_backward"]),
                "time_grad_clip": float(opt_timing["time_grad_clip"]),
                "time_optimizer_step": float(opt_timing["time_optimizer_step"]),
                "time_zero_grad": float(opt_timing["time_zero_grad"]),
            }
            log_row.update(route_metrics)
            log_row.update(self._style_backbone_status())
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            log_row.update(counterfactual_stats)
            self._write_step_log(log_row)

            msg = (
                f"[step] gstep={self.global_step} ep={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} mse={loss_mse.item():.4f} "
                f"nce={loss_nce.item():.4f} slot_nce={loss_slot_nce.item():.4f} cons={loss_cons.item():.4f} "
                f"div={loss_div.item():.4f} "
                f"proxy=({loss_proxy_low.item():.4f},{loss_proxy_mid.item():.4f},{loss_proxy_high.item():.4f}) "
                f"attn_sep={loss_attn_sep.item():.4f} attn_ord={loss_attn_order.item():.4f} "
                f"attn_role={loss_attn_role.item():.4f} "
                f"attn_ov={attn_overlap.item():.4f} "
                f"H=({attn_entropy_low.item():.3f},{attn_entropy_mid.item():.3f},{attn_entropy_high.item():.3f}) "
                f"route=({loss_route_sparse.item():.4f},{loss_route_balance.item():.4f},{loss_route_div.item():.4f},{loss_route_gate.item():.4f}) "
                f"route_mid=({route_mid_to_mid.item():.2f},{route_mid_to_up_16.item():.2f},{route_mid_to_up_32.item():.2f}) "
                f"route_u16=({route_up_16_to_mid.item():.2f},{route_up_16_to_up_16.item():.2f},{route_up_16_to_up_32.item():.2f}) "
                f"route_u32=({route_up_32_to_mid.item():.2f},{route_up_32_to_up_16.item():.2f},{route_up_32_to_up_32.item():.2f}) "
                f"route_gate=({route_gate_mid.item():.2f},{route_gate_up_16.item():.2f},{route_gate_up_32.item():.2f}) "
                f"route_cos=({route_cos_mid_up_16.item():.3f},{route_cos_mid_up_32.item():.3f},{route_cos_up_16_up_32.item():.3f}) "
                f"cos_same={cos_same.item():.4f} "
                f"cos_diff={cos_diff.item():.4f} tok_coll={token_collapse.item():.4f} "
                f"λnce={eff_lnce:.4f} λslot={eff_lslot:.4f} λroute=({eff_lroute_sparse:.4f},{eff_lroute_balance:.4f},{eff_lroute_div:.4f},{eff_lroute_gate:.4f}) λaux={eff_aux:.4f} "
                f"lr={self.lr_schedule.get_last_lr()[0]:.6e} "
                f"data_t={self._step_data_time:.3f}s train_t={self._step_train_time:.3f}s "
                f"split_t=(main:{timing_main_forward:.3f},style:{timing_style_losses:.3f},cf:{timing_counterfactual:.3f},opt:{timing_opt_step:.3f},"
                f"bw:{opt_timing['time_backward']:.3f},clip:{opt_timing['time_grad_clip']:.3f},step:{opt_timing['time_optimizer_step']:.3f},zero:{opt_timing['time_zero_grad']:.3f}) "
                f"style_bb={self._style_backbone_phase()}"
            )
            if grad_norm is not None:
                msg += f" gnorm={grad_norm:.4f}"
            if counterfactual_stats:
                msg += (
                    f" cf_drop_part={counterfactual_stats['cf_delta_drop_part']:.4f}"
                    f" cf_drop_style={counterfactual_stats['cf_delta_drop_style']:.4f}"
                )
                if "cf_delta_style_mid_only" in counterfactual_stats:
                    msg += (
                        f" cf_solo=(mid:{counterfactual_stats['cf_delta_style_mid_only']:.4f},"
                        f"u16:{counterfactual_stats['cf_delta_style_up16_only']:.4f},"
                        f"u32:{counterfactual_stats['cf_delta_style_up32_only']:.4f})"
                    )
            print(msg, flush=True)

        out = {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_nce": loss_nce.item(),
            "loss_slot_nce": loss_slot_nce.item(),
            "loss_cons": loss_cons.item(),
            "loss_div": loss_div.item(),
            "loss_proxy_low": loss_proxy_low.item(),
            "loss_proxy_mid": loss_proxy_mid.item(),
            "loss_proxy_high": loss_proxy_high.item(),
            "loss_attn_sep": loss_attn_sep.item(),
            "loss_attn_order": loss_attn_order.item(),
            "loss_attn_role": loss_attn_role.item(),
            "loss_attn_role_low": loss_attn_role_low.item(),
            "loss_attn_role_mid": loss_attn_role_mid.item(),
            "loss_attn_role_high": loss_attn_role_high.item(),
            "attn_overlap": attn_overlap.item(),
            "attn_entropy_low": attn_entropy_low.item(),
            "attn_entropy_mid": attn_entropy_mid.item(),
            "attn_entropy_high": attn_entropy_high.item(),
            "cos_same": cos_same.item(),
            "cos_diff": cos_diff.item(),
            "token_collapse": token_collapse.item(),
            "lr": self.lr_schedule.get_last_lr()[0],
            "data_time": float(self._step_data_time),
            "train_time": float(self._step_train_time),
        }
        out.update(route_metrics)
        out.update(self._style_backbone_status())
        out.update(counterfactual_stats)
        return out

    # ------------------------------------------------------------------ #
    #  DPM-Solver++ sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def dpm_solver_sample(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor | None = None,
        style_ref_mask: torch.Tensor | None = None,
        num_inference_steps: int = 20,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
        condition_mode: str | None = None,
    ) -> torch.Tensor:
        """Sample in **latent** space, return **pixel**-space result (no CFG)."""
        self.model.eval()
        bsz = content_img.shape[0]
        device = self.device

        content_img = content_img.to(device)
        if style_img is not None:
            style_img = style_img.to(device)
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)
        mode = self.conditioning_mode if condition_mode is None else str(condition_mode).strip().lower()
        self._ensure_required_conditions(mode, style_img, stage="dpm_solver_sample")

        # Latent dimensions
        latent_ch = self.model.unet_in_channels   # 1 (grayscale)
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

        # Keep sampling dtype aligned with training precision to avoid large fp32 cache growth.
        sample_dtype = torch.bfloat16
        x_t = torch.randn((bsz, latent_ch, latent_h, latent_w), device=device, dtype=sample_dtype)

        for t in dpm.timesteps:
            t_cond = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_hat = self.model(
                    x_t, t_cond, content_img,
                    style_img=style_img,
                    part_imgs=part_imgs, part_mask=part_mask,
                    condition_mode=mode,
                    style_ref_mask=style_ref_mask,
                )
            x_t = dpm.step(eps_hat, t, x_t).prev_sample

        return x_t.clamp(-1, 1)

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
        save_path = Path(path)
        torch.save(chkpt, save_path)
        self._save_split_components(save_path, model_state)

    def load(self, path: str | Path):
        chkpt = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(chkpt["model_state"], strict=False)
        if missing or unexpected:
            print(
                f"[trainer.load] non-strict model load: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        self.opt.load_state_dict(chkpt["opt_state"])
        self.global_step = chkpt.get("step", 0)
        self.current_epoch = chkpt.get("epoch", 0)
        self.local_step = chkpt.get("local_step", 0)
        if "lr_schedule_state" in chkpt:
            try:
                self.lr_schedule.load_state_dict(chkpt["lr_schedule_state"])
            except Exception as e:
                print(
                    "[trainer.load] warning: failed to restore lr scheduler state "
                    f"({type(e).__name__}: {e}); rebuilding scheduler from current config.",
                    flush=True,
                )

    def _save_split_components(self, path: Path, model_state: Dict[str, torch.Tensor]) -> None:
        if not self.save_split_components:
            return
        comp_dir = path.parent / "components"
        comp_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem

        # Auto-detect style/part encoder keys: everything that is NOT content_encoder or unet.
        _main_prefixes = ("content_encoder.", "unet.")
        main_state = {k: v for k, v in model_state.items() if k.startswith(_main_prefixes)}
        vector_state = {k: v for k, v in model_state.items() if not k.startswith(_main_prefixes)}

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
            if self.global_step >= self.total_steps:
                break
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
            if self.global_step >= self.total_steps:
                break
        # Flush any remaining accumulated gradients at epoch end
        if self.global_step < self.total_steps:
            self._flush_accumulated_grads(grad_clip)
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
                    "freeze_style_backbone_steps": int(self.freeze_style_backbone_steps),
                    "style_backbone_lr_scale": float(self.style_backbone_lr_scale),
                })
        # On resume, skip already-completed epochs.
        start_epoch = max(1, self.current_epoch + 1) if self.global_step > 0 else 1
        for epoch in range(start_epoch, epochs + 1):
            log = self.train_epoch(dataloader, epoch)
            print(f"\nEpoch {epoch}:", {k: round(v, 4) for k, v in log.items()})
            if save_every and epoch % save_every == 0 and save_dir is not None:
                self.save(save_dir / f"ckpt_{epoch}.pt")
            if self.global_step >= self.total_steps:
                print(
                    f"[fit] reached total_steps={self.total_steps}; stop training at epoch={epoch}.",
                    flush=True,
                )
                break

    # ------------------------------------------------------------------ #
    #  Convenience sample & save
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path, steps: int = 50):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Limit to at most 8 images for speed; prefer a balanced seen/unseen mix.
        max_vis = 8
        total_n = int(batch["content"].size(0))
        take_n = min(max_vis, total_n)

        split_flag = batch.get("viz_split_flag")
        selected_idx: List[int] = list(range(take_n))
        selected_flags: List[int] = []
        if torch.is_tensor(split_flag):
            all_flags = split_flag.detach().cpu().long().tolist()
            seen_idx = [i for i, v in enumerate(all_flags[:total_n]) if int(v) == 1]
            unseen_idx = [i for i, v in enumerate(all_flags[:total_n]) if int(v) == 0]
            half = take_n // 2
            picked = seen_idx[:half] + unseen_idx[:half]
            if len(picked) < take_n:
                used = set(picked)
                for i in range(total_n):
                    if i in used:
                        continue
                    picked.append(i)
                    if len(picked) >= take_n:
                        break
            selected_idx = picked[:take_n]
            selected_flags = [int(all_flags[i]) for i in selected_idx]

        idx_t = torch.tensor(selected_idx, dtype=torch.long)
        content = batch["content"].index_select(0, idx_t).to(self.device)
        target = batch["target"].index_select(0, idx_t).to(self.device)
        style_img = (
            batch["style_img"].index_select(0, idx_t).to(self.device)
            if "style_img" in batch else None
        )
        style_ref_mask = (
            batch["style_ref_mask"].index_select(0, idx_t).to(self.device)
            if "style_ref_mask" in batch else None
        )
        part_imgs = (
            batch["parts"].index_select(0, idx_t).to(self.device)
            if "parts" in batch else None
        )
        part_mask = (
            batch["part_mask"].index_select(0, idx_t).to(self.device)
            if "part_mask" in batch else None
        )

        sample = self._generate_vis_samples(
            content, style_img=style_img, part_imgs=part_imgs,
            part_mask=part_mask, style_ref_mask=style_ref_mask,
        )
        # Save comparison grid: row0=content, row1=GT, row2=generated
        vis = torch.cat([(content + 1) / 2, (target + 1) / 2, (sample + 1) / 2], dim=0)
        split_suffix = ""
        split_lines: List[str] = []
        if selected_flags:
            flags = selected_flags
            seen_n = int(sum(1 for x in flags if int(x) == 1))
            unseen_n = int(sum(1 for x in flags if int(x) == 0))
            split_suffix = f"_seen{seen_n}_unseen{unseen_n}"
            split_lines = [
                f"col={i:02d} split={'train_seen' if int(v) == 1 else 'val_unseen'}"
                for i, v in enumerate(flags)
            ]
        filename = (
            f"sample_ep{self.current_epoch}_gstep{self.global_step}_estep{self.local_step}"
            f"{split_suffix}.png"
        )
        out_img = out_dir / filename
        save_image(vis, out_img, nrow=content.size(0))
        if split_lines:
            sidecar = out_img.with_suffix(".txt")
            sidecar.write_text(
                "\n".join(
                    [
                        "Columns are left-to-right; rows are content / gt / generated.",
                        *split_lines,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        # Free CUDA cache fragmented by the inference-mode scheduler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _generate_vis_samples(
        self, content, *, style_img, part_imgs, part_mask, style_ref_mask,
    ) -> torch.Tensor:
        """Generate samples for visualization. Subclasses override this."""
        return self.dpm_solver_sample(
            content,
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            num_inference_steps=int(self.sample_inference_steps),
            part_imgs=part_imgs,
            part_mask=part_mask,
            condition_mode=self.conditioning_mode,
        )


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
        lambda_nce: float = 0.0,
        lambda_slot_nce: float = 0.0,
        lambda_cons: float = 0.0,
        lambda_div: float = 0.0,
        lambda_proxy_low: float = 0.0,
        lambda_proxy_mid: float = 0.0,
        lambda_proxy_high: float = 0.0,
        lambda_attn_sep: float = 0.0,
        lambda_attn_order: float = 0.0,
        lambda_attn_role: float = 0.0,
        lambda_route_sparse: float = 0.0,
        lambda_route_balance: float = 0.0,
        lambda_route_div: float = 0.0,
        lambda_route_gate: float = 0.0,
        nce_temperature: float = 0.07,
        aux_loss_warmup_steps: int = 0,
        attn_overlap_margin: float = 0.80,
        attn_entropy_gap: float = 0.03,
        T: int = 1000,
        total_steps: int = 100_000,
        lr_warmup_steps: int = 0,
        lr_min_scale: float = 1e-3,
        sample_every_steps: int | None = None,
        save_every_steps: int | None = None,
        log_every_steps: int | None = None,
        detailed_log: bool = True,
        grad_accum_steps: int = 1,
        conditioning_mode: str = "part_only",
        part_drop_prob: float = 0.0,
        style_ref_drop_prob: float = 0.0,
        style_ref_drop_min_keep: int = 1,
        style_site_drop_prob: float = 0.0,
        style_site_drop_min_keep: int = 1,
        freeze_part_encoder_steps: int = 0,
        freeze_style_backbone_steps: int = 0,
        style_backbone_lr_scale: float = 0.1,
        cfg_drop_prob: float = 0.0,  # deprecated, kept for backward compat (ignored)
    ):
        super().__init__(
            model=model, device=device, lr=lr,
            lambda_mse=1.0,
            lambda_nce=lambda_nce,
            lambda_slot_nce=lambda_slot_nce,
            lambda_cons=lambda_cons,
            lambda_div=lambda_div,
            lambda_proxy_low=lambda_proxy_low,
            lambda_proxy_mid=lambda_proxy_mid,
            lambda_proxy_high=lambda_proxy_high,
            lambda_attn_sep=lambda_attn_sep,
            lambda_attn_order=lambda_attn_order,
            lambda_attn_role=lambda_attn_role,
            lambda_route_sparse=lambda_route_sparse,
            lambda_route_balance=lambda_route_balance,
            lambda_route_div=lambda_route_div,
            lambda_route_gate=lambda_route_gate,
            nce_temperature=nce_temperature,
            aux_loss_warmup_steps=aux_loss_warmup_steps,
            attn_overlap_margin=attn_overlap_margin,
            attn_entropy_gap=attn_entropy_gap,
            T=T, total_steps=total_steps,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_scale=lr_min_scale,
            sample_every_steps=sample_every_steps,
            save_every_steps=save_every_steps, log_every_steps=log_every_steps,
            detailed_log=detailed_log,
            grad_accum_steps=grad_accum_steps,
            conditioning_mode=conditioning_mode,
            part_drop_prob=part_drop_prob,
            style_ref_drop_prob=style_ref_drop_prob,
            style_ref_drop_min_keep=style_ref_drop_min_keep,
            style_site_drop_prob=style_site_drop_prob,
            style_site_drop_min_keep=style_site_drop_min_keep,
            freeze_part_encoder_steps=freeze_part_encoder_steps,
            freeze_style_backbone_steps=freeze_style_backbone_steps,
            style_backbone_lr_scale=style_backbone_lr_scale,
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
        timing_main_forward = 0.0
        timing_style_losses = 0.0
        timing_counterfactual = 0.0
        timing_opt_step = 0.0
        self.model.train()
        self._maybe_unfreeze_part_encoder()
        self._maybe_unfreeze_style_backbone()
        x0 = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style_img, style_ref_mask = self._select_style_view(batch)
        part_imgs = batch["parts"].to(self.device) if "parts" in batch else None
        part_mask = batch["part_mask"].to(self.device) if "part_mask" in batch else None
        self._ensure_required_conditions(self.conditioning_mode, style_img, stage="flow_train_step")
        if self._mode_uses_style(self.conditioning_mode):
            style_ref_mask = self._apply_style_ref_dropout(style_ref_mask)
        b = x0.size(0)

        # Convert pixel → latent
        with torch.no_grad():
            x0_latent = self.model.encode_to_latent(x0)  # (B,1,128,128) → (B,1,128,128) no-op

        # Linear path in latent space: x_t = (1-t) x0_l + t x1_l,  v* = x1_l - x0_l.
        t = torch.rand((b,), device=self.device, dtype=x0_latent.dtype)
        x1_latent = torch.randn_like(x0_latent)
        x_t_latent = (1.0 - t.view(-1, 1, 1, 1)) * x0_latent + t.view(-1, 1, 1, 1) * x1_latent
        v_target = x1_latent - x0_latent
        t_idx = (t * float(self.diffusion_steps - 1)).round().long()

        # Optional part drop path (inactive when part branch is disabled).
        part_drop_mask = None
        if (
            self.part_drop_prob > 0.0
            and part_imgs is not None
            and self._mode_uses_parts(self.conditioning_mode)
        ):
            part_drop_mask = (torch.rand((b,), device=self.device) < self.part_drop_prob)
            if bool(part_drop_mask.any().item()):
                part_imgs = part_imgs.clone()
                part_imgs[part_drop_mask] = 1.0
                if part_mask is not None:
                    part_mask = part_mask.clone()
                    part_mask[part_drop_mask] = 0.0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _t_main0 = time.perf_counter()
            v_hat = self.model(
                x_t_latent, t_idx, content,
                style_img=style_img,
                part_imgs=part_imgs, part_mask=part_mask,
                condition_mode=self.conditioning_mode,
                style_ref_mask=style_ref_mask,
            )
            loss_fm = F.mse_loss(v_hat, v_target)
            timing_main_forward += float(time.perf_counter() - _t_main0)
            _t_style0 = time.perf_counter()
            style_losses = self._compute_style_losses(batch)
            timing_style_losses += float(time.perf_counter() - _t_style0)
            loss_nce = style_losses["loss_nce"]
            loss_slot_nce = style_losses["loss_slot_nce"]
            loss_cons = style_losses["loss_cons"]
            loss_div = style_losses["loss_div"]
            loss_proxy_low = style_losses["loss_proxy_low"]
            loss_proxy_mid = style_losses["loss_proxy_mid"]
            loss_proxy_high = style_losses["loss_proxy_high"]
            loss_attn_sep = style_losses["loss_attn_sep"]
            loss_attn_order = style_losses["loss_attn_order"]
            loss_attn_role = style_losses["loss_attn_role"]
            loss_attn_role_low = style_losses["loss_attn_role_low"]
            loss_attn_role_mid = style_losses["loss_attn_role_mid"]
            loss_attn_role_high = style_losses["loss_attn_role_high"]
            attn_overlap = style_losses["attn_overlap"]
            attn_entropy_low = style_losses["attn_entropy_low"]
            attn_entropy_mid = style_losses["attn_entropy_mid"]
            attn_entropy_high = style_losses["attn_entropy_high"]
            cos_same = style_losses["cos_same"]
            cos_diff = style_losses["cos_diff"]
            token_collapse = style_losses["token_collapse"]
            loss_route_sparse = style_losses["loss_route_sparse"]
            loss_route_balance = style_losses["loss_route_balance"]
            loss_route_div = style_losses["loss_route_div"]
            loss_route_gate = style_losses["loss_route_gate"]
            route_gate_mid = style_losses["route_gate_mid"]
            route_gate_up_16 = style_losses["route_gate_up_16"]
            route_gate_up_32 = style_losses["route_gate_up_32"]
            route_mid_to_mid = style_losses["route_mid_to_mid"]
            route_mid_to_up_16 = style_losses["route_mid_to_up_16"]
            route_mid_to_up_32 = style_losses["route_mid_to_up_32"]
            route_up_16_to_mid = style_losses["route_up_16_to_mid"]
            route_up_16_to_up_16 = style_losses["route_up_16_to_up_16"]
            route_up_16_to_up_32 = style_losses["route_up_16_to_up_32"]
            route_up_32_to_mid = style_losses["route_up_32_to_mid"]
            route_up_32_to_up_16 = style_losses["route_up_32_to_up_16"]
            route_up_32_to_up_32 = style_losses["route_up_32_to_up_32"]
            route_cos_mid_up_16 = style_losses["route_cos_mid_up_16"]
            route_cos_mid_up_32 = style_losses["route_cos_mid_up_32"]
            route_cos_up_16_up_32 = style_losses["route_cos_up_16_up_32"]
            route_metrics = {
                "loss_route_sparse": loss_route_sparse.item(),
                "loss_route_balance": loss_route_balance.item(),
                "loss_route_div": loss_route_div.item(),
                "loss_route_gate": loss_route_gate.item(),
                "route_gate_mid": route_gate_mid.item(),
                "route_gate_up_16": route_gate_up_16.item(),
                "route_gate_up_32": route_gate_up_32.item(),
                "route_mid_to_mid": route_mid_to_mid.item(),
                "route_mid_to_up_16": route_mid_to_up_16.item(),
                "route_mid_to_up_32": route_mid_to_up_32.item(),
                "route_up_16_to_mid": route_up_16_to_mid.item(),
                "route_up_16_to_up_16": route_up_16_to_up_16.item(),
                "route_up_16_to_up_32": route_up_16_to_up_32.item(),
                "route_up_32_to_mid": route_up_32_to_mid.item(),
                "route_up_32_to_up_16": route_up_32_to_up_16.item(),
                "route_up_32_to_up_32": route_up_32_to_up_32.item(),
                "route_cos_mid_up_16": route_cos_mid_up_16.item(),
                "route_cos_mid_up_32": route_cos_mid_up_32.item(),
                "route_cos_up_16_up_32": route_cos_up_16_up_32.item(),
            }
            eff_aux = self.effective_aux_loss_scale
            eff_lnce = eff_aux * self.lambda_nce
            eff_lslot = eff_aux * self.lambda_slot_nce
            eff_lcons = eff_aux * self.lambda_cons
            eff_ldiv = eff_aux * self.lambda_div
            eff_lproxy_low = eff_aux * self.lambda_proxy_low
            eff_lproxy_mid = eff_aux * self.lambda_proxy_mid
            eff_lproxy_high = eff_aux * self.lambda_proxy_high
            eff_lattn_sep = eff_aux * self.lambda_attn_sep
            eff_lattn_order = eff_aux * self.lambda_attn_order
            eff_lattn_role = eff_aux * self.lambda_attn_role
            eff_lroute_sparse = eff_aux * self.lambda_route_sparse
            eff_lroute_balance = eff_aux * self.lambda_route_balance
            eff_lroute_div = eff_aux * self.lambda_route_div
            eff_lroute_gate = eff_aux * self.lambda_route_gate
            loss = (
                self.lambda_fm * loss_fm
                + eff_lnce * loss_nce
                + eff_lslot * loss_slot_nce
                + eff_lcons * loss_cons
                + eff_ldiv * loss_div
                + eff_lproxy_low * loss_proxy_low
                + eff_lproxy_mid * loss_proxy_mid
                + eff_lproxy_high * loss_proxy_high
                + eff_lattn_sep * loss_attn_sep
                + eff_lattn_order * loss_attn_order
                + eff_lattn_role * loss_attn_role
                + eff_lroute_sparse * loss_route_sparse
                + eff_lroute_balance * loss_route_balance
                + eff_lroute_div * loss_route_div
                + eff_lroute_gate * loss_route_gate
            )

        counterfactual_stats: Dict[str, float] = {}
        should_log_counterfactual = bool(
            self.log_every_steps and ((self.global_step + 1) % self.log_every_steps == 0)
        )
        if should_log_counterfactual and style_img is not None and self._mode_uses_style(self.conditioning_mode):
            _t_cf0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if part_imgs is not None and self._mode_uses_parts(self.conditioning_mode):
                    v_hat_part_only = self.model(
                        x_t_latent, t_idx, content,
                        style_img=None,
                        part_imgs=part_imgs,
                        part_mask=part_mask,
                        condition_mode="part_only",
                    )
                    cf_loss_part_only = F.mse_loss(v_hat_part_only, v_target)
                else:
                    cf_loss_part_only = loss_fm.detach()
                v_hat_style_only = self.model(
                    x_t_latent, t_idx, content,
                    style_img=style_img,
                    part_imgs=None,
                    part_mask=None,
                    condition_mode="style_only",
                    style_ref_mask=style_ref_mask,
                )
                cf_loss_style_only = F.mse_loss(v_hat_style_only, v_target)
                solo_site_losses: Dict[str, float] = {}
                if self.conditioning_mode == "style_only" and hasattr(self.model, "set_style_site_force_keep"):
                    try:
                        for site_name, stat_key in (
                            ("mid", "cf_loss_style_mid_only"),
                            ("up_16", "cf_loss_style_up16_only"),
                            ("up_32", "cf_loss_style_up32_only"),
                        ):
                            self._set_style_site_force_keep((site_name,))
                            v_hat_solo = self.model(
                                x_t_latent, t_idx, content,
                                style_img=style_img,
                                part_imgs=None,
                                part_mask=None,
                                condition_mode="style_only",
                                style_ref_mask=style_ref_mask,
                            )
                            solo_site_losses[stat_key] = float(F.mse_loss(v_hat_solo, v_target).item())
                    finally:
                        self._set_style_site_force_keep(None)
            timing_counterfactual += float(time.perf_counter() - _t_cf0)
            base = float(loss_fm.detach().item())
            lp = float(cf_loss_part_only.item())
            ls = float(cf_loss_style_only.item())
            counterfactual_stats = {
                "cf_loss_both": base,
                "cf_loss_part_only": lp,
                "cf_loss_style_only": ls,
                "cf_delta_drop_style": lp - base,
                "cf_delta_drop_part": ls - base,
            }
            counterfactual_stats.update(solo_site_losses)
            if solo_site_losses:
                counterfactual_stats["cf_delta_style_mid_only"] = solo_site_losses["cf_loss_style_mid_only"] - base
                counterfactual_stats["cf_delta_style_up16_only"] = solo_site_losses["cf_loss_style_up16_only"] - base
                counterfactual_stats["cf_delta_style_up32_only"] = solo_site_losses["cf_loss_style_up32_only"] - base

        did_update, grad_norm, opt_timing = self._do_optimizer_update(loss, grad_clip)
        timing_opt_step += float(
            opt_timing["time_backward"]
            + opt_timing["time_grad_clip"]
            + opt_timing["time_optimizer_step"]
            + opt_timing["time_zero_grad"]
        )
        if not did_update:
            out = {
                "loss": loss.item(),
                "loss_fm": loss_fm.item(),
                "loss_nce": loss_nce.item(),
                "loss_slot_nce": loss_slot_nce.item(),
                "loss_cons": loss_cons.item(),
                "loss_div": loss_div.item(),
                "loss_proxy_low": loss_proxy_low.item(),
                "loss_proxy_mid": loss_proxy_mid.item(),
                "loss_proxy_high": loss_proxy_high.item(),
                "loss_attn_sep": loss_attn_sep.item(),
                "loss_attn_order": loss_attn_order.item(),
                "loss_attn_role": loss_attn_role.item(),
                "loss_attn_role_low": loss_attn_role_low.item(),
                "loss_attn_role_mid": loss_attn_role_mid.item(),
                "loss_attn_role_high": loss_attn_role_high.item(),
                "attn_overlap": attn_overlap.item(),
                "attn_entropy_low": attn_entropy_low.item(),
                "attn_entropy_mid": attn_entropy_mid.item(),
                "attn_entropy_high": attn_entropy_high.item(),
                "cos_same": cos_same.item(),
                "cos_diff": cos_diff.item(),
                "token_collapse": token_collapse.item(),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(time.perf_counter() - _train_t0),
                "time_main_forward": float(timing_main_forward),
                "time_style_losses": float(timing_style_losses),
                "time_counterfactual": float(timing_counterfactual),
                "time_opt_step": float(timing_opt_step),
                "time_backward": float(opt_timing["time_backward"]),
                "time_grad_clip": float(opt_timing["time_grad_clip"]),
                "time_optimizer_step": float(opt_timing["time_optimizer_step"]),
                "time_zero_grad": float(opt_timing["time_zero_grad"]),
            }
            out.update(route_metrics)
            out.update(self._style_backbone_status())
            out.update(counterfactual_stats)
            return out

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
                "loss_slot_nce": float(loss_slot_nce.item()),
                "loss_cons": float(loss_cons.item()),
                "loss_div": float(loss_div.item()),
                "loss_proxy_low": float(loss_proxy_low.item()),
                "loss_proxy_mid": float(loss_proxy_mid.item()),
                "loss_proxy_high": float(loss_proxy_high.item()),
                "loss_attn_sep": float(loss_attn_sep.item()),
                "loss_attn_order": float(loss_attn_order.item()),
                "loss_attn_role": float(loss_attn_role.item()),
                "loss_attn_role_low": float(loss_attn_role_low.item()),
                "loss_attn_role_mid": float(loss_attn_role_mid.item()),
                "loss_attn_role_high": float(loss_attn_role_high.item()),
                "attn_overlap": float(attn_overlap.item()),
                "attn_entropy_low": float(attn_entropy_low.item()),
                "attn_entropy_mid": float(attn_entropy_mid.item()),
                "attn_entropy_high": float(attn_entropy_high.item()),
                "cos_same": float(cos_same.item()),
                "cos_diff": float(cos_diff.item()),
                "token_collapse": float(token_collapse.item()),
                "lr": float(self.lr_schedule.get_last_lr()[0]),
                "data_time": float(self._step_data_time),
                "train_time": float(self._step_train_time),
                "time_main_forward": float(timing_main_forward),
                "time_style_losses": float(timing_style_losses),
                "time_counterfactual": float(timing_counterfactual),
                "time_opt_step": float(timing_opt_step),
                "time_backward": float(opt_timing["time_backward"]),
                "time_grad_clip": float(opt_timing["time_grad_clip"]),
                "time_optimizer_step": float(opt_timing["time_optimizer_step"]),
                "time_zero_grad": float(opt_timing["time_zero_grad"]),
            }
            log_row.update(route_metrics)
            log_row.update(self._style_backbone_status())
            if grad_norm is not None:
                log_row["grad_norm"] = float(grad_norm)
            log_row.update(counterfactual_stats)
            self._write_step_log(log_row)

            msg = (
                f"[fm-step] gstep={self.global_step} ep={self.current_epoch} estep={self.local_step} "
                f"loss={loss.item():.4f} fm={loss_fm.item():.4f} "
                f"nce={loss_nce.item():.4f} slot_nce={loss_slot_nce.item():.4f} cons={loss_cons.item():.4f} "
                f"div={loss_div.item():.4f} "
                f"proxy=({loss_proxy_low.item():.4f},{loss_proxy_mid.item():.4f},{loss_proxy_high.item():.4f}) "
                f"attn_sep={loss_attn_sep.item():.4f} attn_ord={loss_attn_order.item():.4f} "
                f"attn_role={loss_attn_role.item():.4f} "
                f"attn_ov={attn_overlap.item():.4f} "
                f"H=({attn_entropy_low.item():.3f},{attn_entropy_mid.item():.3f},{attn_entropy_high.item():.3f}) "
                f"route=({loss_route_sparse.item():.4f},{loss_route_balance.item():.4f},{loss_route_div.item():.4f},{loss_route_gate.item():.4f}) "
                f"route_mid=({route_mid_to_mid.item():.2f},{route_mid_to_up_16.item():.2f},{route_mid_to_up_32.item():.2f}) "
                f"route_u16=({route_up_16_to_mid.item():.2f},{route_up_16_to_up_16.item():.2f},{route_up_16_to_up_32.item():.2f}) "
                f"route_u32=({route_up_32_to_mid.item():.2f},{route_up_32_to_up_16.item():.2f},{route_up_32_to_up_32.item():.2f}) "
                f"route_gate=({route_gate_mid.item():.2f},{route_gate_up_16.item():.2f},{route_gate_up_32.item():.2f}) "
                f"route_cos=({route_cos_mid_up_16.item():.3f},{route_cos_mid_up_32.item():.3f},{route_cos_up_16_up_32.item():.3f}) "
                f"cos_same={cos_same.item():.4f} "
                f"cos_diff={cos_diff.item():.4f} tok_coll={token_collapse.item():.4f} "
                f"λnce={eff_lnce:.4f} λslot={eff_lslot:.4f} λroute=({eff_lroute_sparse:.4f},{eff_lroute_balance:.4f},{eff_lroute_div:.4f},{eff_lroute_gate:.4f}) λaux={eff_aux:.4f} "
                f"lr={self.lr_schedule.get_last_lr()[0]:.6e} "
                f"data_t={self._step_data_time:.3f}s train_t={self._step_train_time:.3f}s "
                f"split_t=(main:{timing_main_forward:.3f},style:{timing_style_losses:.3f},cf:{timing_counterfactual:.3f},opt:{timing_opt_step:.3f},"
                f"bw:{opt_timing['time_backward']:.3f},clip:{opt_timing['time_grad_clip']:.3f},step:{opt_timing['time_optimizer_step']:.3f},zero:{opt_timing['time_zero_grad']:.3f}) "
                f"style_bb={self._style_backbone_phase()}"
            )
            if grad_norm is not None:
                msg += f" gnorm={grad_norm:.4f}"
            if counterfactual_stats:
                msg += (
                    f" cf_drop_part={counterfactual_stats['cf_delta_drop_part']:.4f}"
                    f" cf_drop_style={counterfactual_stats['cf_delta_drop_style']:.4f}"
                )
                if "cf_delta_style_mid_only" in counterfactual_stats:
                    msg += (
                        f" cf_solo=(mid:{counterfactual_stats['cf_delta_style_mid_only']:.4f},"
                        f"u16:{counterfactual_stats['cf_delta_style_up16_only']:.4f},"
                        f"u32:{counterfactual_stats['cf_delta_style_up32_only']:.4f})"
                    )
            print(msg, flush=True)

        out = {
            "loss": loss.item(),
            "loss_fm": loss_fm.item(),
            "loss_nce": loss_nce.item(),
            "loss_slot_nce": loss_slot_nce.item(),
            "loss_cons": loss_cons.item(),
            "loss_div": loss_div.item(),
            "loss_proxy_low": loss_proxy_low.item(),
            "loss_proxy_mid": loss_proxy_mid.item(),
            "loss_proxy_high": loss_proxy_high.item(),
            "loss_attn_sep": loss_attn_sep.item(),
            "loss_attn_order": loss_attn_order.item(),
            "loss_attn_role": loss_attn_role.item(),
            "loss_attn_role_low": loss_attn_role_low.item(),
            "loss_attn_role_mid": loss_attn_role_mid.item(),
            "loss_attn_role_high": loss_attn_role_high.item(),
            "attn_overlap": attn_overlap.item(),
            "attn_entropy_low": attn_entropy_low.item(),
            "attn_entropy_mid": attn_entropy_mid.item(),
            "attn_entropy_high": attn_entropy_high.item(),
            "cos_same": cos_same.item(),
            "cos_diff": cos_diff.item(),
            "token_collapse": token_collapse.item(),
            "lr": self.lr_schedule.get_last_lr()[0],
            "data_time": float(self._step_data_time),
            "train_time": float(self._step_train_time),
        }
        out.update(route_metrics)
        out.update(self._style_backbone_status())
        out.update(counterfactual_stats)
        return out

    # ---- ODE flow sampling ---- #
    @torch.no_grad()
    def flow_sample(
        self,
        content_img: torch.Tensor,
        c: int = 50,
        style_img: torch.Tensor | None = None,
        style_ref_mask: torch.Tensor | None = None,
        part_imgs: torch.Tensor | None = None,
        part_mask: torch.Tensor | None = None,
        condition_mode: str | None = None,
    ) -> torch.Tensor:
        """ODE flow sampling (no CFG)."""
        self.model.eval()
        b, ch, h, w = content_img.shape
        device = self.device

        content_img = content_img.to(device)
        if style_img is not None:
            style_img = style_img.to(device)
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(device)
        if part_imgs is not None:
            part_imgs = part_imgs.to(device)
        if part_mask is not None:
            part_mask = part_mask.to(device)
        mode = self.conditioning_mode if condition_mode is None else str(condition_mode).strip().lower()
        self._ensure_required_conditions(mode, style_img, stage="flow_sample")

        # Latent dimensions
        latent_ch = self.model.unet_in_channels
        latent_h = self.model.unet_input_size
        latent_w = self.model.unet_input_size

        x_t = torch.randn(b, latent_ch, latent_h, latent_w, device=device)
        n_steps = max(2, int(c))
        dt = 1.0 / float(n_steps)

        for i in range(n_steps):
            t_cur = 1.0 - i * dt
            t_tensor = torch.full((b,), t_cur, device=device, dtype=x_t.dtype)
            t_idx = (t_tensor * float(self.diffusion_steps - 1)).round().long()
            v_hat = self.model(
                x_t, t_idx, content_img,
                style_img=style_img,
                part_imgs=part_imgs, part_mask=part_mask,
                condition_mode=mode,
                style_ref_mask=style_ref_mask,
            )
            x_t = x_t - dt * v_hat
        return x_t.clamp(-1, 1)

    def _generate_vis_samples(
        self, content, *, style_img, part_imgs, part_mask, style_ref_mask,
    ) -> torch.Tensor:
        """Override: use flow_sample instead of dpm_solver_sample."""
        flow_steps = int(self.sample_inference_steps) if int(self.sample_inference_steps) > 1 else 50
        return self.flow_sample(
            content, c=flow_steps,
            style_img=style_img,
            style_ref_mask=style_ref_mask,
            part_imgs=part_imgs, part_mask=part_mask,
            condition_mode=self.conditioning_mode,
        )
