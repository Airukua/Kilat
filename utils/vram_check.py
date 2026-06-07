from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


@dataclass(frozen=True)
class VRAMCheckReport:
    """
    Immutable container for VRAM estimation results.

    The report includes not only a fit/no-fit verdict but also actionable
    recommendations (batch size, sequence length) and a breakdown of where
    memory is consumed. This structure is meant to be consumed both by
    automated scripts and by humans via the `pretty()` method.
    """
    fits: bool
    device: str
    available_bytes: Optional[int]
    estimated_bytes: int
    estimated_breakdown: dict[str, int]
    recommended_train_batch_size: int
    recommended_sequence_length: int
    notes: tuple[str, ...]

    def pretty(self) -> str:
        """Return a human‑readable multi‑line string of the report."""
        lines = [
            f"VRAM check on {self.device}: {'OK' if self.fits else 'OOM risk'}",
            f"Estimated usage: {_format_bytes(self.estimated_bytes)}",
        ]
        if self.available_bytes is not None:
            lines.append(f"Available VRAM:  {_format_bytes(self.available_bytes)}")
        lines.append("Breakdown:")
        for name, value in self.estimated_breakdown.items():
            lines.append(f"  - {name}: {_format_bytes(value)}")
        lines.append(
            f"Recommended batch size: {self.recommended_train_batch_size}"
        )
        lines.append(
            f"Recommended seq length: {self.recommended_sequence_length}"
        )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _format_bytes(num_bytes: int) -> str:
    """Convert bytes to human readable form (B/KB/MB/GB/TB)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"  # fallback (never reached in practice)


def _infer_sequence_length(
    train_dataset: Any | None,
    data_collator: Any | None,
    sequence_length: Optional[int],
) -> int:
    """
    Determine sequence length from user input, dataset, or collator.

    WHY: Users often forget to pass `sequence_length` explicitly. We try to
    infer it from common attributes or by peeking at the first sample.
    This reduces boilerplate and avoids silent misconfiguration.

    Decision order:
    1. Explicit `sequence_length` argument (most reliable).
    2. `train_dataset.sequence_length` attribute.
    3. `data_collator.max_length` attribute.
    4. Sample inspection via `__getitem__` (heuristic, may fail).
    """
    if sequence_length is not None:
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be > 0, got {sequence_length}."
            )
        return sequence_length

    dataset_sequence_length = getattr(train_dataset, "sequence_length", None)
    if isinstance(dataset_sequence_length, int) and dataset_sequence_length > 0:
        return dataset_sequence_length

    collator_max_length = getattr(data_collator, "max_length", None)
    if isinstance(collator_max_length, int) and collator_max_length > 0:
        return collator_max_length

    if train_dataset is not None and hasattr(train_dataset, "__getitem__"):
        # NOTE: Accessing index 0 may fail for iterable-style datasets or
        # sharded data. We catch exceptions and fall back to error.
        try:
            sample = train_dataset[0]
        except Exception:
            sample = None
        inferred = _infer_length_from_sample(sample)
        if inferred is not None:
            return inferred

    raise ValueError(
        "Could not infer sequence length. Pass `sequence_length=` explicitly "
        "or expose `sequence_length` on the dataset/collator."
    )


def _infer_length_from_sample(sample: Any) -> Optional[int]:
    """
    Heuristic to extract sequence length from a single dataset sample.

    WHY: Many HF datasets return dict with "input_ids" or a tuple of tensors.
    We guess the length by looking at the last dimension of the first tensor‑like
    object. This is not 100% reliable, but it's better than failing immediately.
    """
    if sample is None:
        return None

    if isinstance(sample, dict):
        # Common HuggingFace pattern
        input_ids = sample.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
            return int(input_ids.shape[-1])
        if isinstance(input_ids, (list, tuple)) and input_ids:
            first_item = input_ids[0]
            if isinstance(first_item, (list, tuple, torch.Tensor)):
                return len(input_ids)   # list of tokens, each token is maybe scalar
            return len(input_ids)
        return None

    # Fallback: sample is a tuple/list (e.g., (input_ids, attention_mask))
    if isinstance(sample, (tuple, list)) and sample:
        first = sample[0]
        if isinstance(first, torch.Tensor) and first.ndim >= 1:
            return int(first.shape[-1])
        if isinstance(first, (list, tuple)):
            return len(first)

    # Single tensor case (e.g., sample is just the token ids)
    if isinstance(sample, torch.Tensor) and sample.ndim >= 1:
        return int(sample.shape[-1])

    return None


def _get_available_vram_bytes(device: torch.device) -> Optional[int]:
    """Return free VRAM in bytes for a CUDA device, or None if not CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes)


def _extract_probe_input(batch: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        return input_ids, labels
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]
    return batch, None


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {
            key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        return type(batch)(
            value.to(device, non_blocking=True) if hasattr(value, "to") else value
            for value in batch
        )
    return batch.to(device, non_blocking=True) if hasattr(batch, "to") else batch


def _build_probe_batch(
    train_dataset: Any,
    data_collator: Any | None,
    batch_size: int,
) -> Any:
    if train_dataset is None:
        raise ValueError("train_dataset is required for empirical VRAM probing.")

    samples: list[Any] = []

    if getattr(train_dataset, "streaming", False) or not hasattr(train_dataset, "__getitem__"):
        iterator = iter(train_dataset)
        for _ in range(batch_size):
            samples.append(next(iterator))
        if len(samples) == 1:
            return samples[0]
        return data_collator(samples) if data_collator is not None else default_collate(samples)

    if not hasattr(train_dataset, "__len__"):
        raise ValueError("Dataset must be indexable or iterable for VRAM probing.")

    limit = min(batch_size, len(train_dataset))
    for index in range(limit):
        samples.append(train_dataset[index])

    if len(samples) == 1:
        return samples[0]

    return data_collator(samples) if data_collator is not None else default_collate(samples)


def _probe_peak_bytes(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    precision: str,
) -> int:
    was_training = model.training
    model.train(True)
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    batch_on_device = _move_batch_to_device(batch, device)
    autocast_enabled = precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    try:
        with torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            if isinstance(batch_on_device, dict):
                outputs = model(**batch_on_device, return_dict=True)
                loss = outputs.loss
                if loss is None:
                    logits = outputs.logits
                    labels = batch_on_device.get("labels")
                    if labels is None:
                        raise RuntimeError(
                            "Model output has no loss and batch has no labels."
                        )
                    loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100,
                    )
            elif isinstance(batch_on_device, (tuple, list)):
                outputs = model(*batch_on_device, return_dict=True)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model output has no loss for tuple batch.")
            else:
                outputs = model(batch_on_device, return_dict=True)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model output has no loss for tensor batch.")

        loss.backward()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def estimate_training_vram_bytes(
    model: torch.nn.Module,
    args: Any,
    *,
    sequence_length: int,
) -> dict[str, int]:
    """
    Compute a detailed VRAM breakdown for a training configuration.

    Assumptions:
    - We use AdamW optimizer (2 states per parameter → 8 bytes per trainable param).
    - Gradients are stored in fp32.
    - No gradient checkpointing is assumed (hence conservative activation estimate).
    - Mixed precision (AMP) only affects activations and compute, not weights/grads/opt.
    """
    config = getattr(model, "config", None)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    parameter_bytes = int(total_params * 4)
    gradient_bytes = int(trainable_params * 4)
    optimizer_bytes = int(trainable_params * 8)

    return {
        "parameters": parameter_bytes,
        "gradients": gradient_bytes,
        "optimizer": optimizer_bytes,
    }


def check_vram_fit(
    model: torch.nn.Module,
    args: Any,
    *,
    train_dataset: Any | None = None,
    data_collator: Any | None = None,
    sequence_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    safety_margin: float = 0.90,
    raise_on_fail: bool = True,
) -> VRAMCheckReport:
    """
    Public entry point: estimate VRAM and optionally raise if it doesn't fit.

    WHY: Many training scripts OOM half‑way through because they only test with
    small batches. This checker runs before training and gives actionable advice.

    The safety margin leaves headroom for transient allocations (caching, temporary
    tensors) and reduces false positives. 0.90 means "use at most 90% of free VRAM".

    Decision to recommend batch size:
    - Compute per‑sample activation cost by dividing total activation bytes by current batch size.
    - Fixed cost = everything except activations.
    - Then find max batch size such that fixed + batch * per_sample_activation ≤ budget.
    """
    if not 0.0 < safety_margin < 1.0:
        raise ValueError("safety_margin must be between 0 and 1.")

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    inferred_sequence_length = _infer_sequence_length(
        train_dataset,
        data_collator,
        sequence_length,
    )
    available_bytes = _get_available_vram_bytes(device)

    notes: list[str] = []
    train_batch_size = int(getattr(args, "per_device_train_batch_size", 1))
    precision = str(getattr(args, "precision", "fp32"))
    estimated_total = 0
    breakdown: dict[str, int] = {}

    if available_bytes is not None and train_dataset is not None:
        try:
            probe_1 = _build_probe_batch(train_dataset, data_collator, 1)
            probe_2 = _build_probe_batch(train_dataset, data_collator, 2)

            # Temporarily move model to GPU for a real training-memory probe.
            was_on_device = next(model.parameters()).device
            model.to(device)

            peak_1 = _probe_peak_bytes(model, probe_1, device, precision)
            peak_2 = _probe_peak_bytes(model, probe_2, device, precision)

            # Linear fit from real model behavior.
            marginal_per_sample = max(0, peak_2 - peak_1)
            estimated_total = int(peak_1 + max(0, train_batch_size - 1) * marginal_per_sample)
            breakdown = {
                "probe_batch_1_peak": int(peak_1),
                "probe_batch_2_peak": int(peak_2),
                "marginal_per_sample": int(marginal_per_sample),
            }
            notes.append(
                "Empirical estimate from actual forward/backward passes on batches of size 1 and 2."
            )
            notes.append(f"Sequence length used for probing: {inferred_sequence_length}.")
            model.to(was_on_device)
        except Exception as exc:
            notes.append(
                f"Empirical probe failed ({type(exc).__name__}); falling back to analytical estimate."
            )
            breakdown = estimate_training_vram_bytes(
                model,
                args,
                sequence_length=inferred_sequence_length,
            )
            # Conservative fallback: use analytical model if probing fails.
            activation_bytes = int(
                train_batch_size
                * inferred_sequence_length
                * max(
                    1,
                    int(
                        getattr(
                            model.config,
                            "n_embd",
                            getattr(model.config, "hidden_size", 1),
                        )
                    ),
                )
                * max(
                    1,
                    int(
                        getattr(
                            model.config,
                            "n_layer",
                            getattr(model.config, "num_hidden_layers", 1),
                        )
                    ),
                )
                * (2 if precision in {"fp16", "bf16"} else 4)
            )
            estimated_total = sum(breakdown.values()) + activation_bytes
            breakdown["activations_fallback"] = activation_bytes
    else:
        breakdown = estimate_training_vram_bytes(
            model,
            args,
            sequence_length=inferred_sequence_length,
        )
        activation_bytes = int(
            train_batch_size
            * inferred_sequence_length
            * max(
                1,
                int(
                    getattr(
                        model.config,
                        "n_embd",
                        getattr(model.config, "hidden_size", 1),
                    )
                ),
            )
            * max(
                1,
                int(
                    getattr(
                        model.config,
                        "n_layer",
                        getattr(model.config, "num_hidden_layers", 1),
                    )
                ),
            )
            * (2 if precision in {"fp16", "bf16"} else 4)
        )
        estimated_total = sum(breakdown.values()) + activation_bytes
        breakdown["activations_fallback"] = activation_bytes
        notes.append("CUDA unavailable or dataset missing; using analytical fallback only.")

    fits = True
    if available_bytes is not None:
        budget = int(available_bytes * safety_margin)
        fits = estimated_total <= budget
        notes.append(
            f"Safety margin keeps usable memory under {_format_bytes(budget)}."
        )
        if not fits:
            notes.append(
                "Try lowering `per_device_train_batch_size`, `sequence_length`, "
                "or switch to a smaller model / bf16."
            )

    recommended_batch_size = train_batch_size
    if available_bytes is not None:
        if "marginal_per_sample" in breakdown:
            per_batch_bytes = max(1, breakdown["marginal_per_sample"])
            fixed_bytes = int(breakdown["probe_batch_1_peak"] - per_batch_bytes)
        else:
            per_batch_bytes = max(1, breakdown["activations_fallback"] // max(1, train_batch_size))
            fixed_bytes = estimated_total - breakdown["activations_fallback"]
        budget = int(available_bytes * safety_margin)
        if budget > fixed_bytes:
            recommended_batch_size = max(
                1, math.floor((budget - fixed_bytes) / per_batch_bytes)
            )

    report = VRAMCheckReport(
        fits=fits,
        device=str(device),
        available_bytes=available_bytes,
        estimated_bytes=estimated_total,
        estimated_breakdown=breakdown,
        recommended_train_batch_size=recommended_batch_size,
        recommended_sequence_length=inferred_sequence_length,
        notes=tuple(notes),
    )

    if raise_on_fail and not fits:
        # Immediate failure helps integration into training pipelines
        raise RuntimeError(report.pretty())

    return report
