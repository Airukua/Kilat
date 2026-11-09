from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque
from typing import Any, Optional

import torch
import torch.nn as nn

from kilat.engine.callbacks import TrainerCallback, TrainerControl, TrainerState
from kilat.engine.checkpoint import capture_rng_state, compute_perplexity
from kilat.models.ffn import DeepSeekMoE


class TrainingDivergedError(RuntimeError):
    """Raised by `FlightRecorderCallback` when a runtime numerical circuit breaker trips."""

    def __init__(self, message: str, forensic_path: Optional[str] = None) -> None:
        self.forensic_path = forensic_path
        super().__init__(message)


def _detach_batch_to_cpu(batch: Any) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu()
    if isinstance(batch, dict):
        return {k: _detach_batch_to_cpu(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_detach_batch_to_cpu(v) for v in batch)
    return batch


class FlightRecorderCallback(TrainerCallback):
    """
    Zero-per-micro-batch-sync runtime telemetry recorder and automatic circuit breaker.

    Responsibilities:
      1. Enriches `logs` at `on_log` intervals (`train/ce_loss`, `train/aux_loss`, `train/ppl`,
         `optim/grad_norm`, `optim/update_to_weight_ratio`, `optim/scaler_scale`,
         `optim/skipped_steps_total`, `moe/router_entropy_norm`, `moe/max_expert_load_ratio`,
         `attn/lambda_min`, `attn/lambda_max`, `sys/tokens_per_sec`, `sys/peak_vram_mb`).
      2. Trips an automatic circuit breaker on:
         - `NaN` or `Inf` loss;
         - `skipped_steps_consecutive >= max_consecutive_skipped_steps` (default 10);
         - `ce_loss > spike_multiplier * rolling_median_loss` for `spike_patience` consecutive steps
           after warmup completes.
      3. Never overwrites `checkpoint-best` or `checkpoint-latest`; dumps an isolated forensic
         snapshot to `<output_dir>/forensics/crash_step_<N>.pt` and raises `TrainingDivergedError`.
    """

    def __init__(
        self,
        max_consecutive_skipped_steps: int = 10,
        spike_multiplier: float = 4.0,
        spike_patience: int = 5,
        rolling_window: int = 25,
        output_dir: Optional[str] = None,
    ) -> None:
        self.max_consecutive_skipped_steps = max_consecutive_skipped_steps
        self.spike_multiplier = spike_multiplier
        self.spike_patience = spike_patience
        self.rolling_losses: deque[float] = deque(maxlen=max(5, rolling_window))
        self.telemetry_history: list[dict[str, float]] = []
        self.output_dir = output_dir

        self._consecutive_spikes: int = 0
        self._last_log_time: Optional[float] = None
        self._tokens_since_log: int = 0
        self._prev_sampled_weight: Optional[torch.Tensor] = None
        self._model_ref: Optional[nn.Module] = None

    def _unwrap_model(self, model: Optional[nn.Module]) -> Optional[nn.Module]:
        if model is None:
            return None
        raw = model.module if hasattr(model, "module") else model
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
        return raw

    def _select_probe_parameter(self, model: Optional[nn.Module]) -> Optional[torch.Tensor]:
        raw = self._unwrap_model(model)
        if raw is None:
            return None
        for p in raw.parameters():
            if p.requires_grad and p.ndim >= 2:
                return p
        return None

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Optional[TrainerControl]:
        _ = (args, state, control)
        self._model_ref = kwargs.get("model")
        self._last_log_time = time.perf_counter()
        self._tokens_since_log = 0
        probe = self._select_probe_parameter(self._model_ref)
        if probe is not None:
            self._prev_sampled_weight = probe.detach().float().clone()
        return None

    def _dump_forensics_and_abort(
        self,
        reason: str,
        remediation: str,
        args: Any,
        state: TrainerState,
        trainer: Any = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        output_dir = getattr(args, "output_dir", "./results")
        forensics_dir = os.path.join(output_dir, "forensics")
        forensic_path: Optional[str] = None

        if state.is_world_process_zero:
            os.makedirs(forensics_dir, exist_ok=True)
            forensic_path = os.path.join(forensics_dir, f"crash_step_{state.global_step}.pt")

            m_in = model if model is not None else self._model_ref
            raw_model = self._unwrap_model(m_in)
            layer_norms: dict[str, dict[str, float]] = {}
            if raw_model is not None:
                for name, p in raw_model.named_parameters():
                    entry: dict[str, float] = {
                        "weight_l2": float(p.detach().float().norm().item()),
                    }
                    if p.grad is not None:
                        entry["grad_l2"] = float(p.grad.detach().float().norm().item())
                    layer_norms[name] = entry

            trigger_batch = (
                _detach_batch_to_cpu(getattr(trainer, "_last_batch", None))
                if trainer is not None
                else None
            )
            model_cfg = (
                raw_model.config.to_dict()
                if raw_model is not None and hasattr(getattr(raw_model, "config", None), "to_dict")
                else {}
            )
            args_dict = args.to_dict() if hasattr(args, "to_dict") else {}

            payload = {
                "step": state.global_step,
                "epoch": state.epoch,
                "reason": reason,
                "remediation": remediation,
                "trigger_batch": trigger_batch,
                "rng_state": capture_rng_state(),
                "layer_norms": layer_norms,
                "model_config": model_cfg,
                "training_args": args_dict,
                "telemetry_tail": list(self.telemetry_history[-10:]),
            }
            torch.save(payload, forensic_path)

        raise TrainingDivergedError(
            f"Circuit breaker tripped at step {state.global_step}: {reason} "
            f"| Forensic bundle: {forensic_path or 'rank!=0'} | Remediation: {remediation}",
            forensic_path=forensic_path,
        )

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Optional[TrainerControl]:
        _ = control
        trainer = kwargs.get("trainer")
        kw_model = kwargs.get("model")
        model = kw_model if kw_model is not None else self._model_ref
        if model is not None:
            self._model_ref = model

        if trainer is not None:
            self._tokens_since_log += int(getattr(trainer, "_last_step_tokens", 0))

        avg_ce_loss = kwargs.get("avg_ce_loss")
        avg_loss = kwargs.get("avg_loss")
        ce_val = avg_ce_loss if avg_ce_loss is not None else avg_loss

        # 1. NaN / Inf loss circuit breaker
        if ce_val is not None and not math.isfinite(float(ce_val)):
            self._dump_forensics_and_abort(
                reason=f"Non-finite loss detected (ce_loss={ce_val}).",
                remediation="Lower learning_rate, inspect trigger_batch for corrupt tokens, or enable z_loss_coef=1e-4.",
                args=args,
                state=state,
                trainer=trainer,
                model=model,
            )

        # 2. Consecutive AMP GradScaler skipped steps circuit breaker
        skipped_consec = int(getattr(trainer, "_skipped_steps_consecutive", 0)) if trainer is not None else 0
        if skipped_consec >= self.max_consecutive_skipped_steps:
            self._dump_forensics_and_abort(
                reason=f"{skipped_consec} consecutive optimizer steps skipped due to Inf/NaN gradients.",
                remediation="Switch precision from fp16 to bf16 or reduce peak learning rate / gradient scale.",
                args=args,
                state=state,
                trainer=trainer,
                model=model,
            )

        # 3. Sustained post-warmup loss spike circuit breaker
        if ce_val is not None:
            ce_f = float(ce_val)
            warmup_steps = (
                args.get_warmup_steps(state.max_steps)
                if hasattr(args, "get_warmup_steps") and state.max_steps > 0
                else int(getattr(args, "warmup_steps", 0))
            )
            if state.global_step > warmup_steps and len(self.rolling_losses) >= 5:
                med = statistics.median(self.rolling_losses)
                if med > 0.0 and ce_f > self.spike_multiplier * med:
                    self._consecutive_spikes += 1
                    if self._consecutive_spikes >= self.spike_patience:
                        self._dump_forensics_and_abort(
                            reason=(
                                f"Sustained loss spike: ce_loss={ce_f:.4f} > "
                                f"{self.spike_multiplier:.1f}x median ({med:.4f}) "
                                f"for {self._consecutive_spikes} consecutive steps."
                            ),
                            remediation="Resume from checkpoint-best with reduced learning_rate and max_grad_norm=1.0.",
                            args=args,
                            state=state,
                            trainer=trainer,
                            model=model,
                        )
                else:
                    self._consecutive_spikes = 0
            self.rolling_losses.append(ce_f)

        return None

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[TrainerControl]:
        _ = (args, control)
        if logs is None:
            return None

        trainer = kwargs.get("trainer")
        kw_model = kwargs.get("model")
        m_in = kw_model if kw_model is not None else self._model_ref
        model = self._unwrap_model(m_in)
        now = time.perf_counter()
        elapsed = max(1e-6, now - (self._last_log_time or now))
        self._last_log_time = now

        ce_loss = float(logs.get("ce_loss", logs.get("loss", 0.0)))
        aux_loss = float(logs.get("aux_loss", 0.0))

        telemetry: dict[str, float] = {
            "step": float(state.global_step),
            "train/ce_loss": ce_loss,
            "train/aux_loss": aux_loss,
            "train/ppl": compute_perplexity(ce_loss),
            "sys/tokens_per_sec": round(float(self._tokens_since_log) / elapsed, 2),
            "sys/peak_vram_mb": (
                round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 2)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        self._tokens_since_log = 0

        if trainer is not None:
            grad_norm_t = getattr(trainer, "_last_grad_norm", None)
            if isinstance(grad_norm_t, torch.Tensor):
                telemetry["optim/grad_norm"] = float(grad_norm_t.item())
            scaler = getattr(trainer, "scaler", None)
            telemetry["optim/scaler_scale"] = float(scaler.get_scale()) if scaler is not None else 1.0
            telemetry["optim/skipped_steps_total"] = float(getattr(trainer, "_skipped_steps_total", 0))

        probe = self._select_probe_parameter(model)
        if probe is not None:
            curr_w = probe.detach().float()
            if self._prev_sampled_weight is not None and self._prev_sampled_weight.shape == curr_w.shape:
                delta_norm = float((curr_w - self._prev_sampled_weight).norm().item())
                w_norm = max(1e-8, float(curr_w.norm().item()))
                telemetry["optim/update_to_weight_ratio"] = delta_norm / w_norm
            self._prev_sampled_weight = curr_w.clone()

        if model is not None:
            entropies: list[float] = []
            max_loads: list[float] = []
            lambdas: list[float] = []
            for mod in model.modules():
                if isinstance(mod, DeepSeekMoE):
                    r_stats = getattr(mod, "_last_router_stats", None)
                    if isinstance(r_stats, dict):
                        if "entropy_norm" in r_stats:
                            entropies.append(float(r_stats["entropy_norm"].item()))
                        if "max_load_ratio" in r_stats:
                            max_loads.append(float(r_stats["max_load_ratio"].item()))
                log_lam = getattr(mod, "log_lambda", None)
                if isinstance(log_lam, nn.Parameter):
                    sig = torch.sigmoid(log_lam.detach().float())
                    lambdas.append(float(sig.min().item()))
                    lambdas.append(float(sig.max().item()))

            if entropies:
                telemetry["moe/router_entropy_norm"] = sum(entropies) / len(entropies)
            if max_loads:
                telemetry["moe/max_expert_load_ratio"] = max(max_loads)
            if lambdas:
                telemetry["attn/lambda_min"] = min(lambdas)
                telemetry["attn/lambda_max"] = max(lambdas)

        logs.update({k: v for k, v in telemetry.items() if k != "step"})
        self.telemetry_history.append(telemetry)
        target_dir = self.output_dir or getattr(args, "output_dir", None)
        if target_dir and getattr(state, "is_world_process_zero", True):
            import json

            os.makedirs(str(target_dir), exist_ok=True)
            with open(os.path.join(str(target_dir), "telemetry.jsonl"), "a", encoding="utf-8") as tf:
                tf.write(json.dumps(telemetry) + "\n")
        return None
