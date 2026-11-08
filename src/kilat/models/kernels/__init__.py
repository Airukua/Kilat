"""Canonical Triton and reference kernels for Kilat models."""

from kilat.models.kernels.flash_decay import (
    HAS_TRITON,
    FlashDecayFunction,
    _cpu_global_decay,
    _reference_global_decay_fp64,
    compute_decay_normalizer_from_logit,
    flash_decay_fwd_kernel,
    triton_global_decay,
)

__all__ = [
    "HAS_TRITON",
    "FlashDecayFunction",
    "flash_decay_fwd_kernel",
    "triton_global_decay",
    "compute_decay_normalizer_from_logit",
    "_cpu_global_decay",
    "_reference_global_decay_fp64",
]
