from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Optional
import torch


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


def _estimate_activation_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    precision: str,
    ffn_mode: str,
) -> int:
    """
    Conservative activation memory estimator.

    WHY: Exact activation memory depends on many runtime factors (attention
    implementation, gradient checkpointing, recompute schedule). We use a
    safe multiplier (10x) of the naive per‑token footprint to cover
    attention intermediates, residual connections, MLP activations, and
    autograd metadata.

    Trade‑off: This overestimates for models with heavy recomputation but
    underestimates for very deep or MoE models. We add a MoE bump (1.2×)
    because experts multiply activations.
    """
    # Each element uses 2 bytes for fp16/bf16, 4 bytes for fp32
    activation_dtype_bytes = 2 if precision in {"fp16", "bf16"} else 4
    base = batch_size * sequence_length * hidden_size * num_layers
    multiplier = 10.0
    if ffn_mode == "moe":
        # Mixture of Experts tends to keep multiple expert activations alive
        multiplier *= 1.2
    return int(base * activation_dtype_bytes * multiplier)


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
    # Support both GPT‑style (n_embd, n_layer) and HF transformers (hidden_size, num_hidden_layers)
    hidden_size = int(getattr(config, "n_embd", getattr(config, "hidden_size", 0)))
    num_layers = int(getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0)))
    ffn_mode = str(getattr(config, "ffn_mode", "dense"))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Model weights remain in fp32 in standard training loops; AMP only changes
    # forward/backward compute dtype, not parameter storage.
    parameter_bytes = int(total_params * 4)
    gradient_bytes = int(trainable_params * 4)
    # AdamW stores two moving averages per parameter → 2 * 4 bytes each = 8 bytes/param
    optimizer_bytes = int(trainable_params * 8)
    activation_bytes = _estimate_activation_bytes(
        batch_size=int(args.per_device_train_batch_size),
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision=str(getattr(args, "precision", "fp32")),
        ffn_mode=ffn_mode,
    )

    # Reserve 10% extra for PyTorch internal buffers, kernel launches, and fragmentation
    overhead_bytes = int(
        0.10 * (parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes)
    )

    return {
        "parameters": parameter_bytes,
        "gradients": gradient_bytes,
        "optimizer": optimizer_bytes,
        "activations": activation_bytes,
        "overhead": overhead_bytes,
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
    breakdown = estimate_training_vram_bytes(
        model,
        args,
        sequence_length=inferred_sequence_length,
    )
    estimated_total = sum(breakdown.values())
    available_bytes = _get_available_vram_bytes(device)

    fits = True
    notes: list[str] = [
        "Estimate is conservative and includes model weights, gradients, AdamW state, and activations.",
    ]

    if available_bytes is None:
        notes.append("CUDA not available, so the checker is reporting a theoretical estimate only.")
    else:
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

    train_batch_size = int(getattr(args, "per_device_train_batch_size", 1))
    recommended_batch_size = train_batch_size
    if available_bytes is not None:
        # Avoid division by zero if activation estimate is zero (unlikely)
        per_batch_bytes = max(1, breakdown["activations"] // max(1, train_batch_size))
        fixed_bytes = estimated_total - breakdown["activations"]
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