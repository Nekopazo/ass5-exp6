#!/usr/bin/env python3
"""Training utilities for the refactored pixel-space glyph DiT."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

def sample_logit_normal_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    noise = torch.randn(int(batch_size), device=device, dtype=dtype)
    return torch.sigmoid(noise * float(std) + float(mean))


def _metrics_to_floats(metrics: Dict[str, torch.Tensor | float | int]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            output[key] = float(value.detach().item())
        else:
            output[key] = float(value)
    return output

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
        lr_warmup_steps: int = 2000,
        lr_min_ratio: float = 0.1,
        weight_decay: float = 0.0,
        track_best_on_val: bool = False,
        extra_modules: Optional[Iterable[nn.Module]] = None,
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
        max_warmup = max(0, self.total_steps - 1)
        self.lr_warmup_steps = max(0, min(max_warmup, int(lr_warmup_steps)))
        self.lr_min_ratio = min(1.0, max(0.0, float(lr_min_ratio)))
        self.weight_decay = max(0.0, float(weight_decay))
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

        modules = [self.model]
        if extra_modules is not None:
            for module in extra_modules:
                modules.append(module.to(device))
        self.extra_modules = modules[1:]

        params = []
        for module in modules:
            params.extend(param for param in module.parameters() if param.requires_grad)
        if not params:
            raise RuntimeError("No trainable parameters found.")
        self.optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=self.weight_decay)
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

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

    def _lr_scale_for_step(self, step: int) -> float:
        step = max(1, int(step))
        if self.lr_warmup_steps > 0 and step <= self.lr_warmup_steps:
            return float(step) / float(self.lr_warmup_steps)
        if self.total_steps <= self.lr_warmup_steps:
            return 1.0
        decay_start = max(self.lr_warmup_steps, int(math.ceil(self.total_steps * 0.8)))
        if step <= decay_start:
            return 1.0
        decay_steps = max(1, self.total_steps - decay_start)
        progress = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine

    def _set_lr_for_step(self, step: int) -> float:
        lr_scale = self._lr_scale_for_step(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr) * lr_scale
        return lr_scale

    def _backward_and_step(self, loss: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        step = self.global_step + 1
        lr_scale = self._set_lr_for_step(step)
        loss.backward()
        self.optimizer.step()
        return {"lr_scale": float(lr_scale)}

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
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    self._write_step_log(train_row)
                    metric_str = " ".join(
                        f"{key}={value:.4f}" for key, value in train_row.items() if key not in {"step", "epoch"}
                    )
                    print(f"[step] step={self.global_step} epoch={self.current_epoch} {metric_str}", flush=True)

                if (
                    val_dataloader is not None
                    and self.val_every_steps is not None
                    and self.global_step % self.val_every_steps == 0
                ):
                    val_metrics = self.evaluate(val_dataloader)
                    val_row = {
                        "step": int(self.global_step),
                        "epoch": int(self.current_epoch),
                        **{f"val_{key}": float(value) for key, value in val_metrics.items()},
                    }
                    self._write_val_log(val_row)
                    metric_str = " ".join(
                        f"{key}={value:.4f}" for key, value in val_row.items() if key not in {"step", "epoch"}
                    )
                    print(f"[val] step={self.global_step} epoch={self.current_epoch} {metric_str}", flush=True)

                    val_loss = float(val_row.get("val_loss", float("inf")))
                    if self.track_best_on_val and (self.best_val_loss is None or val_loss < self.best_val_loss):
                        self.best_val_loss = val_loss
                        self.save(save_root / "best.pt")
                        print(
                            f"[val] updated best checkpoint step={self.global_step} val_loss={val_loss:.4f}",
                            flush=True,
                        )

                if self.save_every_steps is not None and self.global_step % self.save_every_steps == 0:
                    self.save(save_root / f"ckpt_step_{self.global_step}.pt")

                if (
                    self.sample_every_steps is not None
                    and self.sample_batch is not None
                    and self.sample_dir is not None
                    and self.global_step % self.sample_every_steps == 0
                ):
                    self.sample_and_save(self.sample_batch, self.sample_dir)

            if stop_training:
                break

        self.save(save_root / "last.pt")

    def on_epoch_start(self) -> None:
        return

    def on_after_train_step(self, batch, metrics: Dict[str, float]) -> None:
        return

    def train_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    def eval_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def sample_and_save(self, batch, out_dir: Path) -> None:
        return


class FlowTrainer(_BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-4,
        total_steps: int = 100_000,
        lambda_rf: float = 1.0,
        flow_sample_steps: int = 20,
        flow_sampler: str = "flow_dpm",
        timestep_sampling: str = "logit_normal",
        log_every_steps: int = 100,
        save_every_steps: Optional[int] = None,
        val_every_steps: Optional[int] = None,
        val_max_batches: Optional[int] = 16,
        lr_warmup_steps: int = 2000,
        lr_min_ratio: float = 0.1,
        weight_decay: float = 0.0,
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
            lr_warmup_steps=lr_warmup_steps,
            lr_min_ratio=lr_min_ratio,
            weight_decay=weight_decay,
            track_best_on_val=True,
        )
        self.lambda_rf = float(lambda_rf)
        self.flow_sample_steps = max(1, int(flow_sample_steps))
        self.flow_sampler = str(flow_sampler).strip().lower()
        if self.flow_sampler not in {"flow_dpm", "euler", "heun"}:
            raise ValueError(f"Unsupported flow sampler: {flow_sampler!r}")
        self.timestep_sampling = str(timestep_sampling).strip().lower()
        if self.timestep_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported timestep sampling: {timestep_sampling!r}")

    def _sample_training_timesteps(self, batch_size: int) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            return torch.rand(int(batch_size), device=self.device)
        return sample_logit_normal_timesteps(int(batch_size), device=self.device)

    def _encode_conditions(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.encode_content(content), self.model.encode_style(style)

    def _compute_losses(self, batch) -> Dict[str, torch.Tensor | float]:
        target = batch["target"].to(self.device)
        content = batch["content"].to(self.device)
        style = batch["style_img"].to(self.device)
        with self._autocast_context():
            x1 = target
            x0 = torch.randn_like(x1)
            timesteps = self._sample_training_timesteps(x1.size(0))
            t_view = timesteps.view(-1, 1, 1, 1).to(dtype=x1.dtype)
            x_t = (1.0 - t_view) * x0 + t_view * x1
            v_t = x1 - x0

            content_tokens, style_memory = self._encode_conditions(content, style)
            pred_v = self.model.predict_flow(
                x_t,
                timesteps,
                content_tokens=content_tokens,
                style_memory=style_memory,
            )
            loss_rf = F.mse_loss(pred_v, v_t)
            loss = self.lambda_rf * loss_rf

        return {
            "loss": loss,
            "loss_rf": loss_rf,
            "t_mean": timesteps.mean(),
        }

    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()
        metrics = self._compute_losses(batch)
        metrics.update(self._backward_and_step(metrics["loss"]))
        return _metrics_to_floats(metrics)

    def eval_step(self, batch) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            metrics = self._compute_losses(batch)
        return _metrics_to_floats(metrics)

    def _flow_dpm_step(
        self,
        model_to_use: nn.Module,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        dt: float,
        *,
        content_tokens: torch.Tensor,
        style_memory: torch.Tensor,
    ) -> torch.Tensor:
        dt_tensor = sample.new_tensor(float(dt))
        velocity_0 = model_to_use.predict_flow(
            sample,
            timesteps,
            content_tokens=content_tokens,
            style_memory=style_memory,
        )
        mid_sample = sample + 0.5 * dt_tensor * velocity_0
        mid_t = (timesteps + 0.5 * dt_tensor).clamp_(0.0, 1.0)
        velocity_mid = model_to_use.predict_flow(
            mid_sample,
            mid_t,
            content_tokens=content_tokens,
            style_memory=style_memory,
        )
        return sample + dt_tensor * velocity_mid

    @torch.no_grad()
    def flow_sample(
        self,
        content: torch.Tensor,
        *,
        style_img: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        sampler: Optional[str] = None,
    ) -> torch.Tensor:
        self.model.eval()
        content = content.to(self.device)
        style_img = style_img.to(self.device)

        step_count = self.flow_sample_steps if num_inference_steps is None else max(1, int(num_inference_steps))
        sampler_name = self.flow_sampler if sampler is None else str(sampler).strip().lower()
        if sampler_name not in {"flow_dpm", "euler", "heun"}:
            raise ValueError(f"Unsupported flow sampler: {sampler_name!r}")
        dt = 1.0 / float(step_count)
        batch_size = content.size(0)
        sample = torch.randn(
            batch_size,
            int(self.model.in_channels),
            int(self.model.image_size),
            int(self.model.image_size),
            device=self.device,
        )
        with self._autocast_context():
            content_tokens, style_memory = self._encode_conditions(content, style_img)
            for step_idx in range(step_count):
                t = torch.full(
                    (batch_size,),
                    float(step_idx) / float(step_count),
                    device=self.device,
                    dtype=torch.float32,
                )
                if sampler_name == "euler":
                    pred_v = self.model.predict_flow(
                        sample,
                        t,
                        content_tokens=content_tokens,
                        style_memory=style_memory,
                    )
                    sample = sample + dt * pred_v
                elif sampler_name == "heun":
                    pred_v = self.model.predict_flow(
                        sample,
                        t,
                        content_tokens=content_tokens,
                        style_memory=style_memory,
                    )
                    t_next = torch.full(
                        (batch_size,),
                        float(step_idx + 1) / float(step_count),
                        device=self.device,
                        dtype=torch.float32,
                    ).clamp_(0.0, 1.0)
                    sample_euler = sample + dt * pred_v
                    pred_v_next = self.model.predict_flow(
                        sample_euler,
                        t_next,
                        content_tokens=content_tokens,
                        style_memory=style_memory,
                    )
                    sample = sample + 0.5 * dt * (pred_v + pred_v_next)
                else:
                    sample = self._flow_dpm_step(
                        self.model,
                        sample,
                        t,
                        dt,
                        content_tokens=content_tokens,
                        style_memory=style_memory,
                    )
        return sample.clamp(-1.0, 1.0).float()

    def save(self, path: str | Path) -> None:
        payload = {
            "stage": "flow",
            "model_state": self.model.state_dict(),
            "model_config": self.model.export_config(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "trainer_config": {
                "flow_sample_steps": int(self.flow_sample_steps),
                "flow_sampler": str(self.flow_sampler),
                "timestep_sampling": str(self.timestep_sampling),
                "lr_warmup_steps": int(self.lr_warmup_steps),
                "lr_min_ratio": float(self.lr_min_ratio),
                "weight_decay": float(self.weight_decay),
            },
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state" not in checkpoint:
            raise RuntimeError("Flow checkpoint is missing 'model_state'.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        trainer_config = checkpoint.get("trainer_config")
        if isinstance(trainer_config, dict):
            if "flow_sample_steps" in trainer_config:
                self.flow_sample_steps = max(1, int(trainer_config["flow_sample_steps"]))
            if "flow_sampler" in trainer_config:
                self.flow_sampler = str(trainer_config["flow_sampler"]).strip().lower()
            if "timestep_sampling" in trainer_config:
                self.timestep_sampling = str(trainer_config["timestep_sampling"]).strip().lower()
        if "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[flow] skipped optimizer state load: {exc}", flush=True)
        self.global_step = int(checkpoint.get("step", 0))
        self.current_epoch = int(checkpoint.get("epoch", 0))

    @torch.no_grad()
    def sample_and_save(self, batch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        content = batch["content"][:8].to(self.device)
        target = batch["target"][:8].to(self.device)
        style = batch["style_img"][:8].to(self.device)
        sample = self.flow_sample(
            content,
            style_img=style,
            num_inference_steps=self.flow_sample_steps,
        )
        style_preview = style[:, 0]
        vis = torch.cat([content, style_preview, target, sample], dim=0)
        save_image(vis, out_dir / f"flow_step_{self.global_step}.png", nrow=content.size(0), normalize=True, value_range=(-1, 1))
