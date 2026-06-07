from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, default_collate


@dataclass(frozen=True)
class VRAMCheckReport:
    """
    Empirical VRAM probe report based on actual forward/backward runs.

    WHY: Theoretical estimators (like the previous module) are useful but often
    miss framework overheads, kernel launches, and data‑dependent peaks.
    This report runs real batches through the model to measure true peak memory.
    """
    fits: bool
    device: str
    requested_train_batch_size: int
    max_fit_train_batch_size: int
    overflow_train_batch_size: int
    spare_train_batch_size: int
    available_bytes: Optional[int]
    peak_bytes_at_requested: Optional[int]
    peak_bytes_at_max_fit: Optional[int]
    notes: tuple[str, ...]

    def pretty(self) -> str:
        """Human‑readable console output for the health check."""
        status = "OK" if self.fits else "OOM risk"
        lines = [
            f"VRAM check on {self.device}: {status}",
            f"Requested batch size: {self.requested_train_batch_size}",
            f"Max fit batch size:   {self.max_fit_train_batch_size}",
            f"Overflow batch size:   {self.overflow_train_batch_size}",
            f"Spare batch size:      {self.spare_train_batch_size}",
        ]
        if self.available_bytes is not None:
            lines.append(f"Available VRAM:       {_format_bytes(self.available_bytes)}")
        if self.peak_bytes_at_requested is not None:
            lines.append(
                f"Peak at requested:    {_format_bytes(self.peak_bytes_at_requested)}"
            )
        if self.peak_bytes_at_max_fit is not None:
            lines.append(f"Peak at max fit:      {_format_bytes(self.peak_bytes_at_max_fit)}")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _format_bytes(num_bytes: int) -> str:
    """Convert integer bytes to human readable string (B/KB/MB/GB/TB)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"  # fallback (never reached)


def _get_available_vram_bytes(device: torch.device) -> Optional[int]:
    """Return free VRAM in bytes for a CUDA device, else None."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes)


def _take_samples(dataset: Dataset | IterableDataset, count: int) -> list[Any]:
    """
    Extract `count` samples from a dataset, supporting both map‑style and streaming.

    WHY: We need small batches for probing. Streaming datasets (e.g., web streams)
    cannot be indexed, so we iterate. Map‑style datasets are indexed.
    For safety with very small datasets, we wrap around using modulo.

    Edge case: If dataset has length 0 → raise.
    If dataset is iterable but not indexable, we just call `next()` repeatedly.
    """
    if count <= 0:
        raise ValueError("count must be > 0.")

    samples: list[Any] = []
    # Streaming datasets or those without __getitem__ → use iterator
    if getattr(dataset, "streaming", False) or not hasattr(dataset, "__getitem__"):
        iterator = iter(dataset)
        for _ in range(count):
            samples.append(next(iterator))
        return samples

    # Map‑style dataset with __len__
    if not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must be indexable or iterable.")

    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty.")

    # Modulo wrap‑around for tiny datasets (e.g., health check with 1 sample)
    for index in range(count):
        samples.append(dataset[index % total])
    return samples


def _build_batch(
    dataset: Dataset | IterableDataset,
    data_collator: Any | None,
    batch_size: int,
) -> Any:
    """
    Create a single batch of `batch_size` samples, applying the data collator.

    WHY: The trainer uses a data collator to assemble batches (padding, stacking).
    We mimic that exactly to get realistic memory usage.
    For batch_size == 1 we return the raw sample to avoid unnecessary collation overhead.
    """
    samples = _take_samples(dataset, batch_size)
    if batch_size == 1:
        return samples[0]
    if data_collator is not None:
        return data_collator(samples)
    # Fallback to PyTorch default collate (works for tensors, nested structures)
    return default_collate(samples)


def _move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move tensors in a batch to the specified device (non‑blocking).

    WHY: Non‑blocking transfer overlaps CPU→GPU copy with compute,
    but more importantly it matches the behaviour of the real training loop.
    """
    if isinstance(batch, dict):
        return {
            key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        # Preserve the original container type (list or tuple)
        return type(batch)(
            value.to(device, non_blocking=True) if hasattr(value, "to") else value
            for value in batch
        )
    # Single tensor or object with .to() method
    return batch.to(device, non_blocking=True) if hasattr(batch, "to") else batch


def _run_probe(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    precision: str,
) -> int:
    """
    Perform a single forward+backward pass on the batch and return peak allocated memory.

    WHY: This is the core empirical measurement. We reset peak stats, run
    the model with autocast (if fp16/bf16), compute loss, backpropagate,
    and finally read the peak memory. This catches both forward and
    backward activation peaks.

    TRADE‑OFFS:
    - We do NOT update optimizers (just `loss.backward()`). This under‑estimates
      memory by a few dozen MB (optimizer state is already allocated elsewhere),
      but the peak during optimizer step is usually lower than backward peak.
    - We zero gradients before and after to avoid accumulation across probes.
    - The model is put in train mode; we restore original mode afterwards.

    IMPORTANT: We assume the model returns a `loss` field when `return_dict=True`.
    If not, the probe raises a clear error.
    """
    was_training = model.training
    model.train(True)
    model.zero_grad(set_to_none=True)

    # Clear caches and reset stats before measuring
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    batch_on_device = _move_to_device(batch, device)
    enabled = precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    try:
        with torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=enabled,
        ):
            # Support multiple batch formats: dict, (input_ids, labels), or single tensor
            if isinstance(batch_on_device, dict):
                outputs = model(**batch_on_device, return_dict=True)
            elif isinstance(batch_on_device, (tuple, list)):
                if len(batch_on_device) >= 2:
                    outputs = model(
                        input_ids=batch_on_device[0],
                        labels=batch_on_device[1],
                        return_dict=True,
                    )
                else:
                    outputs = model(batch_on_device[0], return_dict=True)
            else:
                outputs = model(batch_on_device, return_dict=True)

            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError(
                    "Model did not return a loss for the probe batch."
                )

        loss.backward()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    finally:
        # Clean up: zero gradients and restore train/eval mode
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _binary_search_max_fit(
    model: torch.nn.Module,
    dataset: Dataset | IterableDataset,
    data_collator: Any | None,
    device: torch.device,
    precision: str,
    requested_batch_size: int,
) -> tuple[int, Optional[int]]:
    """
    Find the largest batch size ≤ `requested_batch_size` that fits in VRAM.

    WHY: When the requested batch OOMs immediately, we need to search downwards
    to find a safe size. Binary search is efficient (log N probes).
    """
    low = 1
    high = requested_batch_size
    best_fit = 0
    best_peak: Optional[int] = None

    while low <= high:
        mid = (low + high) // 2
        batch = _build_batch(dataset, data_collator, mid)
        try:
            peak = _run_probe(model, batch, device, precision)
            best_fit = mid
            best_peak = peak
            low = mid + 1
        except RuntimeError as exc:
            # Only OOM errors are handled; others propagate.
            if "out of memory" not in str(exc).lower():
                raise
            # Clean up after OOM and try smaller batch
            torch.cuda.empty_cache()
            high = mid - 1

    return best_fit, best_peak


def _search_max_fit_above(
    model: torch.nn.Module,
    dataset: Dataset | IterableDataset,
    data_collator: Any | None,
    device: torch.device,
    precision: str,
    start_fit_batch_size: int,
) -> tuple[int, Optional[int]]:
    """
    Find the maximum batch size that fits, starting from a known‑good size.

    WHY: If the requested batch already fits, we want to see how much spare
    capacity exists. We exponentially increase batch size to find an upper
    bound, then binary search inside that interval.

    TRADE‑OFFS:
    - Exponential growth (×2) is fast but may overshoot; binary search then refines.
    - Maximum probe size is capped at 4096 or 64×start_batch to avoid excessive runs.
    """
    best_fit = start_fit_batch_size
    best_peak = _run_probe(
        model,
        _build_batch(dataset, data_collator, start_fit_batch_size),
        device,
        precision,
    )

    probe_size = max(start_fit_batch_size + 1, start_fit_batch_size * 2)
    upper_fail: Optional[int] = None
    # Cap at 4096 or 64× start (whichever is larger but reasonable)
    max_probe_size = max(start_fit_batch_size * 64, start_fit_batch_size + 64, 4096)

    while probe_size <= max_probe_size:
        try:
            peak = _run_probe(
                model,
                _build_batch(dataset, data_collator, probe_size),
                device,
                precision,
            )
            best_fit = probe_size
            best_peak = peak
            probe_size *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            upper_fail = probe_size
            break

    if upper_fail is None:
        # We never hit OOM up to max_probe_size → best_fit is the true max (within cap)
        return best_fit, best_peak

    # Binary search between last known good (best_fit) and first failing (upper_fail)
    low = best_fit + 1
    high = upper_fail - 1
    while low <= high:
        mid = (low + high) // 2
        try:
            peak = _run_probe(
                model,
                _build_batch(dataset, data_collator, mid),
                device,
                precision,
            )
            best_fit = mid
            best_peak = peak
            low = mid + 1
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            high = mid - 1

    return best_fit, best_peak


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
    Empirically check if the model + batch configuration fits in GPU VRAM.

    WHAT IT DOES:
    1. Moves model to GPU (if not already there).
    2. Builds a batch of size `per_device_train_batch_size` from `train_dataset`.
    3. Runs one forward+backward pass, measuring peak allocated memory.
    4. If it fits, searches upward to find the maximum batch size that still fits.
    5. If it OOMs, binary‑searches downward to find the largest fitting batch.
    6. Returns a report with recommended max batch size, overflow, spare.

    WHY EMPIRICAL OVER THEORETICAL:
    - Theoretical estimators (like the previous module) miss framework overhead,
      kernel launches, fragmentation, and data‑dependent peaks (e.g., variable
      sequence lengths).
    - Running an actual batch is the only way to be certain for production.

    ASSUMPTIONS:
    - The model returns a `loss` when `return_dict=True` (standard HF pattern).
    - The dataset is not enormous; we only take a small number of samples per probe.
    - The model's forward/backward behaviour is deterministic enough that one probe
      is representative (true for most decoder‑only models).
    - The `args` object has `per_device_train_batch_size` and `precision` attributes.

    LIMITATIONS (noted in report):
    - Does not account for gradient accumulation (memory scales with batch size,
      accumulation only affects update frequency, not peak).
    - Does not account for activation checkpointing (if enabled, memory would be lower).
    - The probe uses a single batch; if the dataset has highly variable sequence
      lengths, the worst‑case batch may be larger.

    EDGE CASES:
    - CUDA unavailable → returns a dummy report with fits=True (no risk on CPU).
    - Dataset empty → raises ValueError.
    - Batch size 1 → no collation, just a single sample.
    - OOM during probe → cleans up cache and continues search.
    """
    if not 0.0 < safety_margin <= 1.0:
        raise ValueError("safety_margin must be in (0, 1].")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_batch_size = int(getattr(args, "per_device_train_batch_size", 1))
    precision = str(getattr(args, "precision", "fp32"))
    notes: list[str] = []

    # Non‑CUDA case: no empirical probing, just return a dummy report.
    if device.type != "cuda" or not torch.cuda.is_available():
        notes.append("CUDA is unavailable; skipped empirical GPU OOM probing.")
        return VRAMCheckReport(
            fits=True,
            device=str(device),
            requested_train_batch_size=requested_batch_size,
            max_fit_train_batch_size=requested_batch_size,
            overflow_train_batch_size=0,
            spare_train_batch_size=0,
            available_bytes=None,
            peak_bytes_at_requested=None,
            peak_bytes_at_max_fit=None,
            notes=tuple(notes),
        )

    if train_dataset is None:
        raise ValueError("train_dataset is required for empirical VRAM probing.")

    if sequence_length is not None:
        notes.append(f"sequence_length hint: {sequence_length}")

    available_bytes = _get_available_vram_bytes(device)
    peak_at_requested: Optional[int] = None
    peak_at_max_fit: Optional[int] = None

    # Remember original device to restore after probing.
    original_device = next(model.parameters()).device
    model.to(device)

    try:
        requested_batch = _build_batch(
            train_dataset,
            data_collator,
            requested_batch_size,
        )
        try:
            peak_at_requested = _run_probe(model, requested_batch, device, precision)
            notes.append("Requested batch fits empirically on actual data.")
            # It fits → try to see how much spare capacity we have.
            max_fit_batch, peak_at_max_fit = _search_max_fit_above(
                model=model,
                dataset=train_dataset,
                data_collator=data_collator,
                device=device,
                precision=precision,
                start_fit_batch_size=requested_batch_size,
            )
            if max_fit_batch > requested_batch_size:
                notes.append(
                    f"Batch size can likely grow by +{max_fit_batch - requested_batch_size} more."
                )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            notes.append(
                "Requested batch OOMed on actual data; binary search is finding the largest fit."
            )
            if requested_batch_size <= 1:
                max_fit_batch = 0
                peak_at_max_fit = None
            else:
                # OOM → search downward from requested-1
                max_fit_batch, peak_at_max_fit = _binary_search_max_fit(
                    model=model,
                    dataset=train_dataset,
                    data_collator=data_collator,
                    device=device,
                    precision=precision,
                    requested_batch_size=requested_batch_size - 1,
                )
    finally:
        # Restore model to its original device (important if the caller uses CPU offload)
        model.to(original_device)
        torch.cuda.empty_cache()

    overflow = max(0, requested_batch_size - max_fit_batch)
    spare = max(0, max_fit_batch - requested_batch_size)
    fits = overflow == 0

    if available_bytes is not None:
        notes.append(
            f"Free VRAM at check time: {_format_bytes(int(available_bytes * safety_margin))} usable after safety margin."
        )
    if not fits:
        notes.append(
            f"Your requested batch is {overflow} sample(s) above the empirically safe maximum."
        )
        notes.append(
            "Reduce batch size, shorten sequences, or use gradient accumulation."
        )
    elif spare > 0:
        notes.append(
            f"You still have room for +{spare} batch size (up to {max_fit_batch})."
        )

    report = VRAMCheckReport(
        fits=fits,
        device=str(device),
        requested_train_batch_size=requested_batch_size,
        max_fit_train_batch_size=max_fit_batch,
        overflow_train_batch_size=overflow,
        spare_train_batch_size=spare,
        available_bytes=available_bytes,
        peak_bytes_at_requested=peak_at_requested,
        peak_bytes_at_max_fit=peak_at_max_fit,
        notes=tuple(notes),
    )

    if raise_on_fail and not fits:
        raise RuntimeError(report.pretty())

    return report


    