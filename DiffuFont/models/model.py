#!/usr/bin/env python3
"""Training utilities for the content+style pixel-space DiP path."""

from __future__ import annotations

import copy
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


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).mean(dim=1)


def _metrics_to_floats(metrics: Dict[str, torch.Tensor | float | int]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            output[key] = float(value.detach().item())
        else:
            output[key] = float(value)
    return output


def _hold_then_cosine_lr_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_scale: float,
    decay_fraction: float = 0.2,
) -> float:
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_scale = float(min(max(0.0, float(min_scale)), 1.0))
    decay_fraction = min(max(float(decay_fraction), 0.0), 1.0)

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    decay_start = int(total_steps * (1.0 - decay_fraction))
    decay_start = max(warmup_steps, min(total_steps, decay_start))
    if step < decay_start:
        return 1.0

    decay_steps = max(1, total_steps - decay_start)
    progress = min(max(float(step - decay_start) / float(decay_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


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
        grad_clip_norm: Optional[float] = 1.0,
        track_best_on_val: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.total_steps = max(1, int(total_steps))
        self.log_every_steps = max(1, int(log_every_steps))
        self.save_every_steps = None if save_every_steps is None else max(1, int(save_every_steps))
        self.val_every_steps = self.log_every_steps if val_every_steps is None else max(1, int(val_every_steps))
        self.val_max_batches = None if val_max_batches is None else max(1, int(val_max_batches))
        self.grad_clip_norm = None if grad_clip_norm is None or float(grad_clip_norm) <= 0.0 else float(grad_clip_norm)
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
        self.trainable_params = params
        self.optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.05)

    def _apply_grad_clip(self) -> Dict[str, float]:
        if self.grad_clip_norm is None:
            return {}
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip_norm)
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.detach().item())
        else:
            grad_norm = float(grad_norm)
        return {"grad_norm": grad_norm}

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


class FlowTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        lambda_flow: float = 1.0,
        style_lr_scale: float = 1.0,
        style_lr_warmup_steps: int = 5000,
        freeze_style: bool = False,
        flow_sample_steps: int = 24,
        ema_decay: float = 0.9999,
        lr_warmup_steps: int = 0,
        lr_min_scale: float = 0.1,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
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
            grad_clip_norm=grad_clip_norm,
            track_best_on_val=True,
        )
        self.lambda_flow = float(lambda_flow)
        self.base_lr = float(lr)
        self.freeze_style = bool(freeze_style)
        self.style_lr_scale = max(0.0, float(style_lr_scale))
        self.style_lr_warmup_steps = max(0, int(style_lr_warmup_steps))
        self.flow_sample_steps = max(1, int(flow_sample_steps))
        self.ema_model: Optional[nn.Module] = None
        self.ema_enabled = False
        self._set_ema_decay(float(ema_decay))
        self.lr_warmup_steps = max(0, int(lr_warmup_steps))
        self.lr_min_scale = float(min(max(0.0, float(lr_min_scale)), 1.0))
        self.style_finetune_active = not self.freeze_style
        self.style_grad_enabled = not self.freeze_style
        style_params = []
        other_params = []
        group_is_style = []
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
            group_is_style.append(False)
        if style_params:
            param_groups.append({"params": style_params, "lr": self.base_lr * self.style_lr_scale})
            group_is_style.append(True)
        if not param_groups:
            raise RuntimeError("No trainable parameters found for flow training.")
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.base_lr, weight_decay=0.05)
        self.group_base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.group_is_style = group_is_style
        self._ensure_ema_model()
        self._sync_ema_from_model()
        self._set_learning_rate_for_step(0)

    def _set_ema_decay(self, ema_decay: float) -> None:
        ema_decay = float(ema_decay)
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.ema_decay = ema_decay
        self.ema_enabled = bool(self.ema_decay > 0.0)

    def _ensure_ema_model(self) -> None:
        if not self.ema_enabled:
            self.ema_model = None
            return
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _sync_ema_from_model(self) -> None:
        if not self.ema_enabled or self.ema_model is None:
            return
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if not self.ema_enabled or self.ema_model is None:
            return
        ema_state = self.ema_model.state_dict()
        model_state = self.model.state_dict()
        one_minus_decay = 1.0 - self.ema_decay
        for key, model_tensor in model_state.items():
            ema_tensor = ema_state[key]
            model_tensor = model_tensor.detach()
            if torch.is_floating_point(ema_tensor):
                ema_tensor.mul_(self.ema_decay).add_(model_tensor, alpha=one_minus_decay)
            else:
                ema_tensor.copy_(model_tensor)

    def _inference_model(self) -> nn.Module:
        if self.ema_enabled and self.ema_model is not None:
            return self.ema_model
        return self.model

    def _lr_scale_for_step(self, step: int) -> float:
        return _hold_then_cosine_lr_scale(
            step=step,
            total_steps=self.total_steps,
            warmup_steps=self.lr_warmup_steps,
            min_scale=self.lr_min_scale,
            decay_fraction=0.2,
        )

    def _set_learning_rate_for_step(self, step: int) -> None:
        lr_scale = self._lr_scale_for_step(step)
        style_lr_scale = self._style_lr_scale_for_step(step)
        for group, base_lr, is_style in zip(self.optimizer.param_groups, self.group_base_lrs, self.group_is_style):
            group_lr = float(base_lr) * lr_scale
            if is_style:
                group_lr *= style_lr_scale
            group["lr"] = group_lr

    def _style_lr_scale_for_step(self, step: int) -> float:
        if self.freeze_style:
            return 0.0
        if self.style_lr_warmup_steps <= 0:
            return 1.0
        step = max(0, int(step))
        return min(float(step) / float(self.style_lr_warmup_steps), 1.0)

    def _encode_style_conditions(
        self,
        model: nn.Module,
        style: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor],
        *,
        detach_style_encoder: bool,
    ) -> dict[str, Optional[torch.Tensor]]:
        return model.encode_style(
            style_img=style,
            style_ref_mask=style_ref_mask,
            return_contrastive=False,
            detach_style_encoder=bool(detach_style_encoder),
        )

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        model: Optional[nn.Module] = None,
        detach_style_encoder: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor | float]:
        model = self.model if model is None else model
        if detach_style_encoder is None:
            detach_style_encoder = (not self.style_grad_enabled)
        target = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style = batch["style_img"].to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        with self._autocast_context():
            x1 = target
            x0 = torch.randn_like(x1)
            timesteps = torch.rand(x1.size(0), device=self.device)
            t_view = timesteps.view(-1, 1, 1, 1).to(dtype=x1.dtype)
            xt = (1.0 - t_view) * x0 + t_view * x1
            target_flow = x1 - x0
            content_features = model.encode_content_features(content)
            content_tokens = model.content_proj(content_features)
            style_pack = self._encode_style_conditions(
                model,
                style,
                style_ref_mask,
                detach_style_encoder=bool(detach_style_encoder),
            )
            pred_flow = model.predict_flow(
                xt,
                timesteps,
                content_tokens=content_tokens,
                style_tokens=style_pack["style_tokens"],
                style_global=style_pack["style_global"],
                style_token_mask=style_pack["style_token_mask"],
            )
            loss_flow_per_sample = (pred_flow.float() - target_flow.float()).pow(2).reshape(target.size(0), -1).mean(dim=1)
            loss_flow = loss_flow_per_sample.mean()
            pred_target = xt + (1.0 - t_view) * pred_flow
            pred_target_l1 = _per_sample_mean((pred_target.float() - target.float()).abs()).mean()
            loss = self.lambda_flow * loss_flow
        metrics = {
            "loss": loss,
            "loss_flow": loss_flow,
            "pred_target_l1": pred_target_l1,
            "t_mean": timesteps.mean(),
            "style_finetune_active": float(self.style_finetune_active),
            "style_lr_mult": float(self._style_lr_scale_for_step(self.global_step)),
        }
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch)
        metrics["loss"].backward()
        metrics.update(self._apply_grad_clip())
        self.optimizer.step()
        self._update_ema()
        self._set_learning_rate_for_step(self.global_step + 1)
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        if self.ema_model is not None:
            self.ema_model.eval()
        with torch.no_grad():
            return _metrics_to_floats(
                self._compute_losses(
                    batch,
                    model=self._inference_model(),
                    detach_style_encoder=True,
                )
            )

    @torch.no_grad()
    def flow_sample(
        self,
        content: torch.Tensor,
        *,
        style_img: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        sample_model = self._inference_model() if use_ema else self.model
        sample_model.eval()
        content = content.to(self.device)
        style_img = style_img.to(self.device)
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        batch_size = content.size(0)
        sample = torch.randn(
            batch_size,
            self.model.in_channels,
            self.model.image_size,
            self.model.image_size,
            device=self.device,
        )
        step_count = self.flow_sample_steps if num_inference_steps is None else max(1, int(num_inference_steps))
        dt = 1.0 / float(step_count)
        with self._autocast_context():
            content_features = sample_model.encode_content_features(content)
            content_tokens = sample_model.content_proj(content_features)
            style_pack = sample_model.encode_style(
                style_img=style_img,
                style_ref_mask=style_ref_mask,
                return_contrastive=False,
                detach_style_encoder=True,
            )
            for step_idx in range(step_count):
                t = torch.full(
                    (batch_size,),
                    float(step_idx) / float(step_count),
                    device=self.device,
                    dtype=torch.float32,
                )
                pred_flow = sample_model.predict_flow(
                    sample,
                    t,
                    content_tokens=content_tokens,
                    style_tokens=style_pack["style_tokens"],
                    style_global=style_pack["style_global"],
                    style_token_mask=style_pack["style_token_mask"],
                )
                sample = sample + dt * pred_flow

        return sample.clamp(-1.0, 1.0).float()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "flow",
                "model_state": self.model.state_dict(),
                "ema_model_state": None if self.ema_model is None else self.ema_model.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "trainer_config": {
                    "flow_sample_steps": int(self.flow_sample_steps),
                    "ema_decay": float(self.ema_decay),
                    "lr_warmup_steps": int(self.lr_warmup_steps),
                    "lr_min_scale": float(self.lr_min_scale),
                    "grad_clip_norm": None if self.grad_clip_norm is None else float(self.grad_clip_norm),
                },
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
        resume_step = int(checkpoint.get("step", 0))
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[flow] skipped optimizer state load: {exc}", flush=True)
        trainer_config = checkpoint.get("trainer_config", {})
        if isinstance(trainer_config, dict):
            if "flow_sample_steps" in trainer_config:
                self.flow_sample_steps = max(1, int(trainer_config["flow_sample_steps"]))
            if "ema_decay" in trainer_config:
                ema_decay = float(trainer_config["ema_decay"])
                if 0.0 <= ema_decay < 1.0:
                    self._set_ema_decay(ema_decay)
        else:
            self._set_ema_decay(self.ema_decay)
        self._ensure_ema_model()
        ema_state = checkpoint.get("ema_model_state")
        if self.ema_enabled and isinstance(ema_state, dict) and self.ema_model is not None:
            self.ema_model.load_state_dict(ema_state, strict=True)
        else:
            self._sync_ema_from_model()
        if self.ema_model is not None:
            self.ema_model.eval()
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
