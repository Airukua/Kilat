from __future__ import annotations
from collections import defaultdict
import torch.nn as nn
from pathlib import Path
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kilat.models.transformer import KilatTransformer

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count total number of parameters in the model.
    
    WHY: Understanding model size is crucial for estimating memory requirements
    and training costs. This function provides a quick way to get parameter counts
    without external profiling tools.
    
    Args:
        model: KilatTransformer instance
        trainable_only: If True, count only parameters with requires_grad=True.
                       Useful for fine-tuning scenarios where some layers are frozen.
    
    Returns:
        Total parameter count as integer
    
    Example:
        >>> model = KilatTransformer(config)
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_breakdown(model: nn.Module):
    """
    Return detailed breakdown of model parameters including shared tensors.
    
    WHY: Many transformer components share weights (e.g., embedding and lm_head
    in weight-tying architectures). This function identifies shared tensors
    and groups them together, preventing double-counting in manual analysis.
    
    WHAT IT DETECTS:
        - Standard parameters: name, shape, size, trainable status
        - Shared tensors: multiple parameter names pointing to same memory
        - Shared groups: which parameters are tied together
    
    Returns:
        Tuple of (rows, shared_groups):
            rows: List of dicts with per-parameter info
            shared_groups: List of lists, each containing names of shared parameters
    
    Example:
        >>> rows, shared = parameter_breakdown(model)
        >>> for row in rows:
        ...     print(f"{row['name']}: {row['numel']:,} params" + 
        ...           (" (shared)" if row['shared'] else ""))
        >>> print(f"Shared groups: {shared}")
    """
    rows = []
    seen_ptrs = set()
    shared = defaultdict(list)

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        is_shared = ptr in seen_ptrs
        if is_shared:
            shared[ptr].append(name)
        else:
            seen_ptrs.add(ptr)

        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "numel": param.numel(),
                "trainable": param.requires_grad,
                "shared": is_shared,
            }
        )

    shared_groups = []
    if shared:
        # Re-scan to include the original owner of each shared tensor.
        # A tensor may appear multiple times; we need all names pointing to it.
        ptr_to_names = defaultdict(list)
        for name, param in model.named_parameters():
            ptr_to_names[param.data_ptr()].append(name)
        shared_groups = [names for names in ptr_to_names.values() if len(names) > 1]

    return rows, shared_groups


def load_model(*args: Any, **kwargs: Any) -> "KilatTransformer":
    """Deprecated shim: use :func:`kilat.models.loading.load_model`.

    Kept for backward compatibility. ``kilat.utils`` is Layer 0 and must not depend on
    ``kilat.models`` at import time, so the canonical implementation is imported lazily.
    """
    warnings.warn(
        "kilat.utils.report.load_model is deprecated; import from kilat.models.loading instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from kilat.models.loading import load_model as _load_model

    return _load_model(*args, **kwargs)
