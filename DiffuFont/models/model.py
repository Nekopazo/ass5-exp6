#!/usr/bin/env python3
"""Training utilities for the content+style DiT x-pred path."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from .font_perceptor import FontPerceptor


def _metrics_to_floats(metrics: Dict[str, torch.Tensor | float | int]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            output[key] = float(value.detach().item())
        else:
            output[key] = float(value)
    return output


def _parameter_is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    lowered = str(name).lower()
    if not param.requires_grad:
        return False
    if param.ndim <= 1:
        return True
    return lowered.endswith(".bias") or "bn" in lowered


def _build_adamw_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
) -> tuple[list[dict], list[float], list[torch.nn.Parameter]]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = _parameter_is_no_decay(name, param)
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict] = []
    base_lrs: list[float] = []

    def _append_group(params: list[torch.nn.Parameter], *, lr: float, group_weight_decay: float) -> None:
        if not params:
            return
        param_groups.append(
            {
                "params": params,
                "lr": float(lr),
                "weight_decay": float(group_weight_decay),
            }
        )
        base_lrs.append(float(lr))

    _append_group(decay_params, lr=float(base_lr), group_weight_decay=float(weight_decay))
    _append_group(no_decay_params, lr=float(base_lr), group_weight_decay=0.0)
    trainable_params = decay_params + no_decay_params
    return param_groups, base_lrs, trainable_params


def _apply_weight_decay_to_optimizer(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        current = float(group.get("weight_decay", 0.0))
        group["weight_decay"] = 0.0 if current == 0.0 else float(weight_decay)


def _apply_adam_betas_to_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    beta1: float,
    beta2: float,
) -> None:
    for group in optimizer.param_groups:
        group["betas"] = (float(beta1), float(beta2))


class _ReuseCountMeanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expanded: torch.Tensor, condition_index: torch.Tensor) -> torch.Tensor:
        index = condition_index.to(device=expanded.device, dtype=torch.long)
        counts = torch.bincount(index, minlength=int(expanded.size(0))).to(device=expanded.device)
        ctx.save_for_backward(index, counts)
        return expanded

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        index, counts = ctx.saved_tensors
        scale = counts.index_select(0, index).clamp_min_(1).to(dtype=grad_output.dtype)
        while scale.ndim < grad_output.ndim:
            scale = scale.unsqueeze(-1)
        return grad_output / scale, None


def _hold_then_linear_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_scale: float,
    decay_start_step: Optional[int] = None,
) -> float:
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_scale = float(min(max(0.0, float(min_scale)), 1.0))

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if decay_start_step is None:
        return 1.0

    decay_start = max(0, int(decay_start_step))
    decay_start = max(warmup_steps, min(total_steps, decay_start))
    if step < decay_start:
        return 1.0

    final_step = max(decay_start, total_steps - 1)
    decay_steps = max(1, final_step - decay_start)
    progress = min(max(float(step - decay_start) / float(decay_steps), 0.0), 1.0)
    return 1.0 + (min_scale - 1.0) * progress


def _hold_then_linear_value(
    *,
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
    hold_start_step: Optional[int] = None,
) -> float:
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    start_value = float(start_value)
    end_value = float(end_value)

    if hold_start_step is None:
        return start_value

    start_step = max(0, min(int(hold_start_step), total_steps))
    if step < start_step:
        return start_value

    final_step = max(start_step, total_steps - 1)
    decay_steps = max(1, final_step - start_step)
    progress = min(max(float(step - start_step) / float(decay_steps), 0.0), 1.0)
    return start_value + (end_value - start_value) * progress


def _cosine_decay_scale(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    min_scale: float = 0.0,
) -> float:
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_scale = float(min(max(0.0, float(min_scale)), 1.0))

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if total_steps <= 1:
        return 1.0

    cosine_start_step = min(warmup_steps, total_steps - 1)
    cosine_steps = max(1, total_steps - cosine_start_step - 1)
    progress = min(max(float(step - cosine_start_step) / float(cosine_steps), 0.0), 1.0)
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine_value


def _cosine_interpolate(
    *,
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    total_steps = max(1, int(total_steps))
    start_value = float(start_value)
    end_value = float(end_value)
    if total_steps <= 1:
        return start_value
    clamped_step = min(max(int(step), 1), total_steps)
    progress = float(clamped_step - 1) / float(total_steps - 1)
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_value + (start_value - end_value) * cosine_value


class _BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float,
        total_steps: int,
        lr_warmup_steps: int = 0,
        lr_decay_start_step: Optional[int] = None,
        lr_schedule: str = "constant",
        lr_min_scale: float = 0.1,
        log_every_steps: int,
        save_every_steps: Optional[int],
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_min_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
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
        if self.grad_clip_norm is None:
            self.grad_clip_min_norm = None
        elif grad_clip_min_norm is None:
            self.grad_clip_min_norm = self.grad_clip_norm
        else:
            self.grad_clip_min_norm = max(0.0, min(float(grad_clip_min_norm), self.grad_clip_norm))
        self.track_best_on_val = bool(track_best_on_val)
        self.global_step = 0
        self.current_epoch = 0
        self.sample_every_steps: Optional[int] = None
        self.sample_batch: Optional[Dict[str, torch.Tensor]] = None
        self.sample_batch_builder: Optional[Callable[[], Dict[str, torch.Tensor]]] = None
        self.sample_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.step_log_file: Optional[Path] = None
        self.val_log_file: Optional[Path] = None
        self.best_val_loss: Optional[float] = None

        self.weight_decay = max(0.0, float(weight_decay))
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        param_groups, base_lrs, trainable_params = _build_adamw_param_groups(
            self.model,
            base_lr=float(lr),
            weight_decay=self.weight_decay,
        )
        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")
        self.trainable_params = trainable_params
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.adam_beta1, self.adam_beta2),
        )
        self.base_lrs = base_lrs
        self.lr_warmup_steps = max(0, int(lr_warmup_steps))
        self.lr_decay_start_step = None if lr_decay_start_step is None else max(0, int(lr_decay_start_step))
        self.lr_schedule = str(lr_schedule)
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {lr_schedule!r}")
        self.lr_min_scale = float(min(max(0.0, float(lr_min_scale)), 1.0))

    def _lr_scale_for_step(self, step: int) -> float:
        if self.lr_schedule == "constant":
            if self.lr_warmup_steps <= 0:
                return 1.0
            clamped_step = min(max(int(step), 1), self.total_steps)
            if clamped_step <= self.lr_warmup_steps:
                return float(clamped_step) / float(max(1, self.lr_warmup_steps))
            return 1.0
        if self.lr_schedule != "cosine":
            raise ValueError(f"unsupported lr_schedule: {self.lr_schedule!r}")
        return _hold_then_linear_scale(
            step=step,
            total_steps=self.total_steps,
            warmup_steps=self.lr_warmup_steps,
            min_scale=self.lr_min_scale,
            decay_start_step=self.lr_decay_start_step,
        )

    def _set_learning_rate_for_step(self, step: int) -> None:
        lr_scale = self._lr_scale_for_step(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr) * float(lr_scale)

    def _grad_clip_value_for_step(self, step: int) -> Optional[float]:
        if self.grad_clip_norm is None:
            return None
        return _hold_then_linear_value(
            step=step,
            total_steps=self.total_steps,
            start_value=self.grad_clip_norm,
            end_value=self.grad_clip_min_norm if self.grad_clip_min_norm is not None else self.grad_clip_norm,
            hold_start_step=self.lr_decay_start_step,
        )

    def _apply_grad_clip(self, *, step: int) -> Dict[str, float]:
        if self.grad_clip_norm is None:
            return {}
        clip_value = self._grad_clip_value_for_step(step)
        if clip_value is None:
            return {}
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, clip_value)
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.detach().item())
        else:
            grad_norm = float(grad_norm)
        return {"grad_norm": grad_norm, "grad_clip_norm_threshold": float(clip_value)}

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
                    extra_val_metrics = self.additional_val_metrics()
                    if extra_val_metrics:
                        val_metrics.update(extra_val_metrics)
                    val_row: Dict[str, float | int] = {
                        "step": int(self.global_step),
                        "epoch": int(self.current_epoch),
                        **val_metrics,
                    }
                    is_best = False
                    best_metric_key = "loss"
                    if self.track_best_on_val and best_metric_key in val_metrics:
                        current_val_loss = float(val_metrics[best_metric_key])
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
                                            "best_metric_key": best_metric_key,
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
                    and (self.sample_batch is not None or self.sample_batch_builder is not None)
                    and self.sample_dir is not None
                    and self.global_step % self.sample_every_steps == 0
                ):
                    if self.sample_batch is None and self.sample_batch_builder is not None:
                        self.sample_batch = self.sample_batch_builder()
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

    def additional_val_metrics(self) -> Dict[str, float]:
        return {}

    def on_epoch_end(self) -> None:
        return None


class FontPerceptorTrainer(_BaseTrainer):
    def __init__(
        self,
        model: FontPerceptor,
        device: torch.device,
        *,
        lr: float = 2e-4,
        total_steps: int = 50_000,
        font_loss_lambda_start: float = 0.2,
        font_loss_lambda_end: float = 0.05,
        font_label_smoothing: float = 0.05,
        char_label_smoothing: float = 0.0,
        qualify_min_char_acc: float = 0.70,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_min_norm: Optional[float] = None,
    ) -> None:
        super().__init__(
            model,
            device,
            lr=lr,
            total_steps=total_steps,
            lr_warmup_steps=0,
            lr_decay_start_step=None,
            lr_schedule="cosine",
            lr_min_scale=0.0,
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            grad_clip_norm=grad_clip_norm,
            grad_clip_min_norm=grad_clip_min_norm,
            weight_decay=1e-4,
            track_best_on_val=False,
        )
        self.font_loss_lambda_start = max(0.0, float(font_loss_lambda_start))
        self.font_loss_lambda_end = max(0.0, float(font_loss_lambda_end))
        if self.font_loss_lambda_end > self.font_loss_lambda_start:
            raise ValueError(
                "font_loss_lambda_end must be <= font_loss_lambda_start, "
                f"got start={self.font_loss_lambda_start} end={self.font_loss_lambda_end}"
            )
        self.font_label_smoothing = min(max(float(font_label_smoothing), 0.0), 0.2)
        self.char_label_smoothing = min(max(float(char_label_smoothing), 0.0), 0.2)
        self.qualify_min_char_acc = float(qualify_min_char_acc)
        self.best_train_loss: Optional[float] = None

    def _lr_scale_for_step(self, step: int) -> float:
        return _cosine_decay_scale(
            step=step,
            total_steps=self.total_steps,
            warmup_steps=self.lr_warmup_steps,
            min_scale=self.lr_min_scale,
        )

    def _font_loss_lambda_for_step(self, step: int) -> float:
        return _cosine_interpolate(
            step=step,
            total_steps=self.total_steps,
            start_value=self.font_loss_lambda_start,
            end_value=self.font_loss_lambda_end,
        )

    def _compute_losses(self, batch: Dict[str, torch.Tensor], *, step: Optional[int] = None) -> Dict[str, torch.Tensor | float]:
        if step is None:
            step = self.global_step + 1
        images = batch["target"].to(self.device)
        font_ids = batch["font_id"].to(self.device, dtype=torch.long)
        char_ids = batch["char_id"].to(self.device, dtype=torch.long)
        font_loss_lambda = self._font_loss_lambda_for_step(int(step))
        with self._autocast_context():
            outputs = self.model(images)
            font_logits = outputs["font_logits"]
            char_logits = outputs["char_logits"]
            loss_font_ce = F.cross_entropy(
                font_logits,
                font_ids,
                label_smoothing=self.font_label_smoothing,
            )
            loss_char_ce = F.cross_entropy(
                char_logits,
                char_ids,
                label_smoothing=self.char_label_smoothing,
            )
            loss = loss_char_ce + font_loss_lambda * loss_font_ce

        font_acc = (font_logits.argmax(dim=1) == font_ids).float().mean()
        char_acc = (char_logits.argmax(dim=1) == char_ids).float().mean()
        qualified_now = float(char_acc.detach().item()) >= self.qualify_min_char_acc
        metrics: Dict[str, torch.Tensor | float] = {
            "loss": loss,
            "loss_font_ce": loss_font_ce,
            "loss_char_ce": loss_char_ce,
            "font_acc": font_acc,
            "char_acc": char_acc,
            "font_loss_lambda": float(font_loss_lambda),
            "font_label_smoothing": float(self.font_label_smoothing),
            "char_label_smoothing": float(self.char_label_smoothing),
            "qualified_now": float(qualified_now),
        }
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        next_step = self.global_step + 1
        self._set_learning_rate_for_step(next_step)
        self.optimizer.zero_grad(set_to_none=True)
        metrics = self._compute_losses(batch, step=next_step)
        metrics["loss"].backward()
        metrics.update(self._apply_grad_clip(step=next_step))
        self.optimizer.step()
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return _metrics_to_floats(self._compute_losses(batch, step=self.global_step))

    def on_after_train_step(self, batch: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> None:
        if self.checkpoint_dir is None:
            return
        current_train_loss = float(metrics["loss"])
        if self.best_train_loss is not None and current_train_loss >= self.best_train_loss:
            return
        self.best_train_loss = current_train_loss
        best_path = self.checkpoint_dir / "best.pt"
        self.save(best_path)
        (self.checkpoint_dir / "best_train_metrics.json").write_text(
            json.dumps(
                {
                    "step": int(self.global_step),
                    "epoch": int(self.current_epoch),
                    "best_metric_key": "loss",
                    **{key: float(value) for key, value in metrics.items()},
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "font_perceptor",
                "model_state": self.model.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "trainer_config": {
                    "lr_schedule": "cosine",
                    "lr_warmup_steps": int(self.lr_warmup_steps),
                    "lr_min_scale": float(self.lr_min_scale),
                    "weight_decay": float(self.weight_decay),
                    "font_loss_lambda_start": float(self.font_loss_lambda_start),
                    "font_loss_lambda_end": float(self.font_loss_lambda_end),
                    "font_label_smoothing": float(self.font_label_smoothing),
                    "char_label_smoothing": float(self.char_label_smoothing),
                    "qualify_min_char_acc": float(self.qualify_min_char_acc),
                    "best_train_loss": None if self.best_train_loss is None else float(self.best_train_loss),
                    "grad_clip_norm": None if self.grad_clip_norm is None else float(self.grad_clip_norm),
                    "grad_clip_min_norm": None if self.grad_clip_min_norm is None else float(self.grad_clip_min_norm),
                },
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state" not in checkpoint:
            raise RuntimeError("Font perceptor checkpoint is missing 'model_state'.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        trainer_config = checkpoint.get("trainer_config", {})
        if isinstance(trainer_config, dict):
            if "lr_warmup_steps" in trainer_config:
                self.lr_warmup_steps = max(0, int(trainer_config["lr_warmup_steps"]))
            if "lr_min_scale" in trainer_config:
                self.lr_min_scale = float(min(max(0.0, float(trainer_config["lr_min_scale"])), 1.0))
            if "weight_decay" in trainer_config:
                self.weight_decay = max(0.0, float(trainer_config["weight_decay"]))
                _apply_weight_decay_to_optimizer(self.optimizer, self.weight_decay)
            if "font_loss_lambda_start" in trainer_config:
                self.font_loss_lambda_start = max(0.0, float(trainer_config["font_loss_lambda_start"]))
            if "font_loss_lambda_end" in trainer_config:
                self.font_loss_lambda_end = max(0.0, float(trainer_config["font_loss_lambda_end"]))
            if self.font_loss_lambda_end > self.font_loss_lambda_start:
                raise ValueError(
                    "font_loss_lambda_end must be <= font_loss_lambda_start after load, "
                    f"got start={self.font_loss_lambda_start} end={self.font_loss_lambda_end}"
                )
            if "font_label_smoothing" in trainer_config:
                self.font_label_smoothing = min(max(float(trainer_config["font_label_smoothing"]), 0.0), 0.2)
            if "char_label_smoothing" in trainer_config:
                self.char_label_smoothing = min(max(float(trainer_config["char_label_smoothing"]), 0.0), 0.2)
            self.qualify_min_char_acc = float(
                trainer_config.get("qualify_min_char_acc", self.qualify_min_char_acc)
            )
            restored_best_train_loss = trainer_config.get("best_train_loss")
            self.best_train_loss = None if restored_best_train_loss is None else float(restored_best_train_loss)
            if "grad_clip_min_norm" in trainer_config and self.grad_clip_norm is not None:
                restored_clip_min = trainer_config["grad_clip_min_norm"]
                if restored_clip_min is None:
                    self.grad_clip_min_norm = self.grad_clip_norm
                else:
                    self.grad_clip_min_norm = max(0.0, min(float(restored_clip_min), self.grad_clip_norm))
        self.global_step = int(checkpoint.get("step", 0))
        self.current_epoch = int(checkpoint.get("epoch", 0))


class XPredTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        lr_warmup_steps: int = 0,
        lr_decay_start_step: Optional[int] = None,
        lr_schedule: str = "constant",
        lr_min_scale: float = 0.1,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 5e-2,
        noise_scale: float = 1.0,
        sample_steps: int = 20,
        ema_decay: float = 0.9999,
        ema_start_step: Optional[int] = None,
        consistency_lambda: float = 1.0,
        consistency_start_step: Optional[int] = None,
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_min_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
    ) -> None:
        super().__init__(
            model,
            device,
            lr=lr,
            total_steps=total_steps,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_start_step=lr_decay_start_step,
            lr_schedule=lr_schedule,
            lr_min_scale=lr_min_scale,
            log_every_steps=log_every_steps,
            save_every_steps=save_every_steps,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            grad_clip_norm=grad_clip_norm,
            grad_clip_min_norm=grad_clip_min_norm,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            track_best_on_val=True,
        )
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = max(0.0, float(t_eps))
        self.noise_scale = float(noise_scale)
        self.sample_steps = max(1, int(sample_steps))
        self.ema_model: Optional[nn.Module] = None
        self.ema_enabled = False
        self.ema_initialized = False
        self.ema_start_step = self.total_steps + 1
        self._set_ema_decay(float(ema_decay))
        self._set_ema_start_step(ema_start_step)
        self.consistency_lambda = max(0.0, float(consistency_lambda))
        self.consistency_start_step = self.total_steps + 1
        self._set_consistency_start_step(consistency_start_step)

    def _set_ema_decay(self, ema_decay: float) -> None:
        ema_decay = float(ema_decay)
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.ema_decay = ema_decay
        self.ema_enabled = bool(self.ema_decay > 0.0)

    def _set_ema_start_step(self, ema_start_step: Optional[int]) -> None:
        if not self.ema_enabled:
            self.ema_start_step = self.total_steps + 1
            return
        if ema_start_step is None:
            ema_start_step = 40_000
        self.ema_start_step = max(1, min(int(ema_start_step), self.total_steps + 1))

    def _set_consistency_start_step(self, consistency_start_step: Optional[int]) -> None:
        if self.consistency_lambda <= 0.0:
            self.consistency_start_step = self.total_steps + 1
            return
        if consistency_start_step is None:
            self.consistency_start_step = self.total_steps + 1
            return
        self.consistency_start_step = max(1, min(int(consistency_start_step), self.total_steps + 1))

    def _consistency_is_active(self, *, step: int) -> bool:
        return self.consistency_lambda > 0.0 and int(step) >= int(self.consistency_start_step)

    def _sample_t(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.sigmoid(torch.randn(batch_size, device=device) * self.p_std + self.p_mean)

    def _lr_scale_for_step(self, step: int) -> float:
        return super()._lr_scale_for_step(step)

    def _grad_clip_value_for_step(self, step: int) -> Optional[float]:
        if self.grad_clip_norm is None:
            return None
        return _cosine_interpolate(
            step=step,
            total_steps=self.total_steps,
            start_value=self.grad_clip_norm,
            end_value=self.grad_clip_min_norm if self.grad_clip_min_norm is not None else self.grad_clip_norm,
        )

    def _ensure_ema_model(self) -> None:
        if not self.ema_enabled:
            self.ema_model = None
            self.ema_initialized = False
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
        self.ema_initialized = True

    def _ema_is_active(self, *, step: Optional[int] = None) -> bool:
        effective_step = self.global_step if step is None else int(step)
        return (
            self.ema_enabled
            and self.ema_initialized
            and self.ema_model is not None
            and effective_step >= self.ema_start_step
        )

    @torch.no_grad()
    def _update_ema(self, *, step: int) -> None:
        step = int(step)
        if not self.ema_enabled or step < self.ema_start_step:
            return
        self._ensure_ema_model()
        if self.ema_model is None:
            return
        if not self.ema_initialized:
            self._sync_ema_from_model()
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
        if self._ema_is_active():
            return self.ema_model
        return self.model

    def _expand_condition_batch(
        self,
        condition: torch.Tensor,
        condition_index: torch.Tensor,
        *,
        average_grad_by_reuse: bool = False,
    ) -> torch.Tensor:
        expanded = condition.index_select(0, condition_index.to(device=condition.device, dtype=torch.long))
        if average_grad_by_reuse and expanded.requires_grad:
            expanded = _ReuseCountMeanGrad.apply(expanded, condition_index)
        return expanded

    def _encode_conditions(
        self,
        model: nn.Module,
        content: torch.Tensor,
        content_index: torch.Tensor,
        style: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor],
        style_index: torch.Tensor,
    ) -> torch.Tensor:
        unique_content_tokens = model.encode_content_tokens(content)
        content_tokens = self._expand_condition_batch(
            unique_content_tokens,
            content_index,
            average_grad_by_reuse=True,
        )
        unique_content_query = model.precompute_content_query(unique_content_tokens)
        content_query = self._expand_condition_batch(
            unique_content_query,
            content_index,
            average_grad_by_reuse=True,
        )
        style_token_bank, style_token_valid_mask = model.encode_style_token_bank(
            style_img=style,
            style_ref_mask=style_ref_mask,
        )
        style_key, style_value = model.precompute_style_kv(
            style_token_bank,
            token_valid_mask=style_token_valid_mask,
        )
        expanded_style_key = self._expand_condition_batch(
            style_key,
            style_index,
            average_grad_by_reuse=True,
        )
        expanded_style_value = self._expand_condition_batch(
            style_value,
            style_index,
            average_grad_by_reuse=True,
        )
        return model.fuse_content_style_tokens_from_projected(
            content_tokens,
            content_query,
            expanded_style_key,
            expanded_style_value,
        )

    def _prepare_denoising_targets(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = target
        x0 = torch.randn_like(x1) * self.noise_scale
        timesteps = self._sample_t(x1.size(0), device=self.device)
        t_view = timesteps.view(-1, 1, 1, 1).to(dtype=x1.dtype)
        xt = t_view * x1 + (1.0 - t_view) * x0
        denom = (1.0 - t_view).clamp_min(self.t_eps)
        target_v = (x1 - xt) / denom
        return timesteps, xt, denom, target_v, x1

    def _forward_ref_branch(
        self,
        model: nn.Module,
        *,
        content: torch.Tensor,
        content_index: torch.Tensor,
        style: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor],
        style_index: torch.Tensor,
        timesteps: torch.Tensor,
        xt: torch.Tensor,
        denom: torch.Tensor,
        target_v: torch.Tensor,
        x1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        conditioning_tokens = self._encode_conditions(
            model,
            content,
            content_index,
            style,
            style_ref_mask,
            style_index,
        )
        pred_x = model.predict_x(
            xt,
            timesteps,
            conditioning_tokens=conditioning_tokens,
        )
        pred_v = (pred_x - xt) / denom
        loss_v = F.mse_loss(pred_v.float(), target_v.float())
        pred_x_l1 = F.l1_loss(pred_x.float(), x1.float())
        return {
            "pred_x": pred_x,
            "loss_v": loss_v,
            "pred_x_l1": pred_x_l1,
        }

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        model: Optional[nn.Module] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor | float]:
        model = self.model if model is None else model
        if step is None:
            step = int(self.global_step)
        target = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        content_index = batch["content_index"].to(self.device, dtype=torch.long)
        style = batch["style_img"].to(self.device)
        style_index = batch["style_index"].to(self.device, dtype=torch.long)
        style_alt = batch.get("style_img_alt")
        if style_alt is not None:
            style_alt = style_alt.to(self.device)
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)
        style_ref_mask_alt = batch.get("style_ref_mask_alt")
        if style_ref_mask_alt is not None:
            style_ref_mask_alt = style_ref_mask_alt.to(self.device)

        with self._autocast_context():
            timesteps, xt, denom, target_v, x1 = self._prepare_denoising_targets(target)
            dual_ref_active = (
                style_alt is not None
                and style_ref_mask_alt is not None
                and self._consistency_is_active(step=step)
            )
            primary_outputs = self._forward_ref_branch(
                model,
                content=content,
                content_index=content_index,
                style=style,
                style_ref_mask=style_ref_mask,
                style_index=style_index,
                timesteps=timesteps,
                xt=xt,
                denom=denom,
                target_v=target_v,
                x1=x1,
            )
            if dual_ref_active:
                alt_outputs = self._forward_ref_branch(
                    model,
                    content=content,
                    content_index=content_index,
                    style=style_alt,
                    style_ref_mask=style_ref_mask_alt,
                    style_index=style_index,
                    timesteps=timesteps,
                    xt=xt,
                    denom=denom,
                    target_v=target_v,
                    x1=x1,
                )
                alt_weight = float(self.consistency_lambda)
                total_weight = 1.0 + alt_weight
                loss_v = (primary_outputs["loss_v"] + alt_outputs["loss_v"] * alt_weight) / total_weight
                pred_x_l1 = (primary_outputs["pred_x_l1"] + alt_outputs["pred_x_l1"] * alt_weight) / total_weight
                loss_v_ref_alt = alt_outputs["loss_v"]
                pred_x_ref_diff_l1 = F.l1_loss(
                    primary_outputs["pred_x"].float(),
                    alt_outputs["pred_x"].float(),
                )
            else:
                loss_v = primary_outputs["loss_v"]
                pred_x_l1 = primary_outputs["pred_x_l1"]
                loss_v_ref_alt = primary_outputs["loss_v"].new_zeros(())
                pred_x_ref_diff_l1 = pred_x_l1.new_zeros(())
            loss = loss_v
        metrics = {
            "loss": loss,
            "loss_v": loss_v,
            "loss_v_ref_main": primary_outputs["loss_v"],
            "loss_v_ref_alt": loss_v_ref_alt,
            "pred_x_ref_diff_l1": pred_x_ref_diff_l1,
            "pred_x_l1": pred_x_l1,
            "t_mean": timesteps.mean(),
        }
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        next_step = self.global_step + 1
        self._set_learning_rate_for_step(next_step)
        self.optimizer.zero_grad(set_to_none=True)
        dual_ref_active = (
            batch.get("style_img_alt") is not None
            and batch.get("style_ref_mask_alt") is not None
            and self._consistency_is_active(step=next_step)
        )
        if not dual_ref_active:
            metrics = self._compute_losses(
                batch,
                step=next_step,
            )
            metrics["loss"].backward()
        else:
            target = batch["target"].to(self.device)
            content = batch["content"].to(self.device)
            content_index = batch["content_index"].to(self.device, dtype=torch.long)
            style = batch["style_img"].to(self.device)
            style_alt = batch["style_img_alt"].to(self.device)
            style_index = batch["style_index"].to(self.device, dtype=torch.long)
            style_ref_mask = batch.get("style_ref_mask")
            if style_ref_mask is not None:
                style_ref_mask = style_ref_mask.to(self.device)
            style_ref_mask_alt = batch.get("style_ref_mask_alt")
            if style_ref_mask_alt is not None:
                style_ref_mask_alt = style_ref_mask_alt.to(self.device)

            alt_weight = float(self.consistency_lambda)
            total_weight = 1.0 + alt_weight

            with self._autocast_context():
                timesteps, xt, denom, target_v, x1 = self._prepare_denoising_targets(target)
                primary_outputs = self._forward_ref_branch(
                    self.model,
                    content=content,
                    content_index=content_index,
                    style=style,
                    style_ref_mask=style_ref_mask,
                    style_index=style_index,
                    timesteps=timesteps,
                    xt=xt,
                    denom=denom,
                    target_v=target_v,
                    x1=x1,
                )
                primary_weighted_loss = primary_outputs["loss_v"] / total_weight
            primary_weighted_loss.backward()
            pred_x_primary = primary_outputs["pred_x"].detach()
            loss_v_ref_main = primary_outputs["loss_v"].detach()
            pred_x_l1_ref_main = primary_outputs["pred_x_l1"].detach()

            with self._autocast_context():
                alt_outputs = self._forward_ref_branch(
                    self.model,
                    content=content,
                    content_index=content_index,
                    style=style_alt,
                    style_ref_mask=style_ref_mask_alt,
                    style_index=style_index,
                    timesteps=timesteps,
                    xt=xt,
                    denom=denom,
                    target_v=target_v,
                    x1=x1,
                )
                alt_weighted_loss = (alt_outputs["loss_v"] * alt_weight) / total_weight
                pred_x_ref_diff_l1 = F.l1_loss(
                    pred_x_primary.float(),
                    alt_outputs["pred_x"].float(),
                )
            alt_weighted_loss.backward()

            metrics = {
                "loss": (primary_weighted_loss.detach() + alt_weighted_loss.detach()),
                "loss_v": (loss_v_ref_main + alt_outputs["loss_v"].detach() * alt_weight) / total_weight,
                "loss_v_ref_main": loss_v_ref_main,
                "loss_v_ref_alt": alt_outputs["loss_v"].detach(),
                "pred_x_ref_diff_l1": pred_x_ref_diff_l1.detach(),
                "pred_x_l1": (pred_x_l1_ref_main + alt_outputs["pred_x_l1"].detach() * alt_weight) / total_weight,
                "t_mean": timesteps.mean().detach(),
            }
        metrics.update(self._apply_grad_clip(step=next_step))
        self.optimizer.step()
        self._update_ema(step=next_step)
        return _metrics_to_floats(metrics)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        if self.ema_model is not None:
            self.ema_model.eval()
        with torch.inference_mode():
            return _metrics_to_floats(
                self._compute_losses(
                    batch,
                    model=self._inference_model(),
                    step=self.global_step,
                )
            )

    @torch.no_grad()
    def sample(
        self,
        content: torch.Tensor,
        *,
        content_index: torch.Tensor,
        style_img: torch.Tensor,
        style_index: torch.Tensor,
        style_ref_mask: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        sample_model = self._inference_model() if use_ema else self.model
        sample_model.eval()
        content = content.to(self.device)
        content_index = content_index.to(self.device, dtype=torch.long)
        style_img = style_img.to(self.device)
        style_index = style_index.to(self.device, dtype=torch.long)
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)

        batch_size = int(content_index.size(0))
        sample = self.noise_scale * torch.randn(
            batch_size,
            self.model.in_channels,
            self.model.image_size,
            self.model.image_size,
            device=self.device,
        )
        step_count = self.sample_steps if num_inference_steps is None else max(1, int(num_inference_steps))
        with self._autocast_context():
            conditioning_tokens = self._encode_conditions(
                sample_model,
                content,
                content_index,
                style_img,
                style_ref_mask,
                style_index,
            )

            def predict_v(x_t: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
                pred_x = sample_model.predict_x(
                    x_t,
                    t_vec,
                    conditioning_tokens=conditioning_tokens,
                )
                t_view = t_vec.view(-1, 1, 1, 1).to(dtype=x_t.dtype)
                return (pred_x - x_t) / (1.0 - t_view).clamp_min(self.t_eps)

            timesteps = torch.linspace(0.0, 1.0, step_count + 1, device=self.device, dtype=torch.float32)
            for step_idx in range(max(0, step_count - 1)):
                t = torch.full((batch_size,), float(timesteps[step_idx]), device=self.device, dtype=torch.float32)
                t_next = torch.full((batch_size,), float(timesteps[step_idx + 1]), device=self.device, dtype=torch.float32)
                dt = (t_next - t).view(-1, 1, 1, 1).to(dtype=sample.dtype)
                v_t = predict_v(sample, t)
                sample_euler = sample + dt * v_t
                v_t_next = predict_v(sample_euler, t_next)
                sample = sample + dt * (0.5 * (v_t + v_t_next))

            t = torch.full((batch_size,), float(timesteps[-2]), device=self.device, dtype=torch.float32)
            t_next = torch.full((batch_size,), float(timesteps[-1]), device=self.device, dtype=torch.float32)
            sample = sample + (t_next - t).view(-1, 1, 1, 1).to(dtype=sample.dtype) * predict_v(sample, t)

        return sample.clamp(-1.0, 1.0).float()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "stage": "xpred",
                "model_state": self.model.state_dict(),
                "ema_model_state": None if self.ema_model is None else self.ema_model.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state": self.optimizer.state_dict(),
                "trainer_config": {
                    "base_lrs": [float(lr) for lr in self.base_lrs],
                    "lr_schedule": str(self.lr_schedule),
                    "lr_warmup_steps": int(self.lr_warmup_steps),
                    "lr_min_scale": float(self.lr_min_scale),
                    "weight_decay": float(self.weight_decay),
                    "adam_betas": [float(self.adam_beta1), float(self.adam_beta2)],
                    "loss_type": "jit_v_mse+dual_ref_v" if self.consistency_lambda > 0.0 else "jit_v_mse",
                    "p_mean": float(self.p_mean),
                    "p_std": float(self.p_std),
                    "t_eps": float(self.t_eps),
                    "noise_scale": float(self.noise_scale),
                    "sample_steps": int(self.sample_steps),
                    "ode_solver": "heun_last_euler",
                    "ema_decay": float(self.ema_decay),
                    "ema_start_step": int(self.ema_start_step),
                    "consistency_lambda": float(self.consistency_lambda),
                    "consistency_start_step": int(self.consistency_start_step),
                    "grad_clip_norm": None if self.grad_clip_norm is None else float(self.grad_clip_norm),
                    "grad_clip_min_norm": None if self.grad_clip_min_norm is None else float(self.grad_clip_min_norm),
                },
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("stage") != "xpred":
            raise RuntimeError(f"Expected an x-pred checkpoint, got stage={checkpoint.get('stage')!r}.")
        if "model_state" not in checkpoint:
            raise RuntimeError("X-pred checkpoint is missing 'model_state'.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        resume_step = int(checkpoint.get("step", 0))
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        trainer_config = checkpoint.get("trainer_config", {})
        if isinstance(trainer_config, dict):
            if "base_lrs" in trainer_config and isinstance(trainer_config["base_lrs"], list):
                restored_base_lrs = [float(value) for value in trainer_config["base_lrs"]]
                if len(restored_base_lrs) == len(self.optimizer.param_groups):
                    self.base_lrs = restored_base_lrs
            if "lr_warmup_steps" in trainer_config:
                self.lr_warmup_steps = max(0, int(trainer_config["lr_warmup_steps"]))
            if "lr_schedule" in trainer_config:
                self.lr_schedule = str(trainer_config["lr_schedule"])
            if "lr_decay_start_step" in trainer_config:
                lr_decay_start_step = trainer_config["lr_decay_start_step"]
                self.lr_decay_start_step = None if lr_decay_start_step is None else max(0, int(lr_decay_start_step))
            if "lr_min_scale" in trainer_config:
                self.lr_min_scale = float(min(max(0.0, float(trainer_config["lr_min_scale"])), 1.0))
            if "weight_decay" in trainer_config:
                self.weight_decay = max(0.0, float(trainer_config["weight_decay"]))
                _apply_weight_decay_to_optimizer(self.optimizer, self.weight_decay)
            if "adam_betas" in trainer_config:
                adam_betas = trainer_config["adam_betas"]
                if isinstance(adam_betas, (list, tuple)) and len(adam_betas) == 2:
                    self.adam_beta1 = float(adam_betas[0])
                    self.adam_beta2 = float(adam_betas[1])
                    _apply_adam_betas_to_optimizer(
                        self.optimizer,
                        beta1=self.adam_beta1,
                        beta2=self.adam_beta2,
                    )
            if "p_mean" in trainer_config:
                self.p_mean = float(trainer_config["p_mean"])
            if "p_std" in trainer_config:
                self.p_std = float(trainer_config["p_std"])
            if "t_eps" in trainer_config:
                self.t_eps = max(0.0, float(trainer_config["t_eps"]))
            if "noise_scale" in trainer_config:
                self.noise_scale = float(trainer_config["noise_scale"])
            if "sample_steps" in trainer_config:
                self.sample_steps = max(1, int(trainer_config["sample_steps"]))
            if "ema_decay" in trainer_config:
                ema_decay = float(trainer_config["ema_decay"])
                if 0.0 <= ema_decay < 1.0:
                    self._set_ema_decay(ema_decay)
            if "ema_start_step" in trainer_config:
                self._set_ema_start_step(int(trainer_config["ema_start_step"]))
            if "consistency_lambda" in trainer_config:
                self.consistency_lambda = max(0.0, float(trainer_config["consistency_lambda"]))
            if "consistency_start_step" in trainer_config:
                self._set_consistency_start_step(int(trainer_config["consistency_start_step"]))
            if "grad_clip_min_norm" in trainer_config and self.grad_clip_norm is not None:
                restored_clip_min = trainer_config["grad_clip_min_norm"]
                if restored_clip_min is None:
                    self.grad_clip_min_norm = self.grad_clip_norm
                else:
                    self.grad_clip_min_norm = max(0.0, min(float(restored_clip_min), self.grad_clip_norm))
        else:
            self._set_ema_decay(self.ema_decay)
            self._set_ema_start_step(self.ema_start_step)
        ema_state = checkpoint.get("ema_model_state")
        self.global_step = resume_step
        self.current_epoch = int(checkpoint.get("epoch", 0))
        if self.ema_enabled and self.global_step >= self.ema_start_step:
            self._ensure_ema_model()
            if isinstance(ema_state, dict) and self.ema_model is not None:
                self.ema_model.load_state_dict(ema_state, strict=True)
                self.ema_initialized = True
            else:
                self._sync_ema_from_model()
            if self.ema_model is not None:
                self.ema_model.eval()
        else:
            self.ema_model = None
            self.ema_initialized = False
        self._set_learning_rate_for_step(self.global_step)

    @torch.no_grad()
    def sample_and_save(self, batch: Dict[str, torch.Tensor], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = batch["target"][:8].to(self.device)
        content = batch["content"]
        content_index = batch["content_index"][:8]
        style = batch["style_img"]
        style_index = batch["style_index"][:8]
        style_ref_mask = batch.get("style_ref_mask")
        if style_ref_mask is not None:
            style_ref_mask = style_ref_mask.to(self.device)
        sample = self.sample(
            content,
            content_index=content_index,
            style_img=style,
            style_index=style_index,
            style_ref_mask=style_ref_mask,
            num_inference_steps=self.sample_steps,
        )
        content = content.to(self.device)
        content = self._expand_condition_batch(content, content_index.to(self.device, dtype=torch.long))
        vis = torch.cat([(content + 1.0) * 0.5, (target + 1.0) * 0.5, (sample + 1.0) * 0.5], dim=0)
        save_image(vis, out_dir / f"sample_step_{self.global_step}.png", nrow=content.size(0))
