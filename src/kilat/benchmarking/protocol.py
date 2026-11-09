"""Multi-seed reproducibility and statistical aggregation protocol for Kilat certification.

Adheres to Kilat Invariant 4 (Deterministic Checkpointing & State Recovery) and
experimental standards: never report single-seed performance as an empirical fact.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

DEFAULT_SEEDS: List[int] = [0, 1, 2]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all known RNGs to a deterministic seed state.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables PyTorch and cuDNN deterministic modes.
            Set to False for throughput/efficiency benchmarking where benchmark
            autotuning is needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


@dataclass
class MultiSeedResult:
    """Result container for a single seed execution."""
    seed: int
    value: Dict[str, Any]


def run_with_seeds(
    fn: Callable[[int], Dict[str, Any]],
    seeds: Optional[List[int]] = None,
    deterministic: bool = True,
) -> List[MultiSeedResult]:
    """Execute fn(seed) across multiple seeds with guaranteed RNG reset before each run.

    Args:
        fn: Execution function accepting a seed integer and returning a metrics dict.
        seeds: List of seeds to evaluate (defaults to [0, 1, 2]).
        deterministic: Whether to enforce deterministic cuDNN backends.

    Returns:
        List of MultiSeedResult instances.
    """
    seeds = seeds or list(DEFAULT_SEEDS)
    results = []
    for seed in seeds:
        set_seed(seed, deterministic=deterministic)
        val = fn(seed)
        results.append(MultiSeedResult(seed=seed, value=val))
    return results


def summarize(results: List[MultiSeedResult]) -> Dict[str, Any]:
    """Aggregate metric dicts across seeds into statistical summaries (mean, std, min, max).

    Non-numeric fields (e.g. status strings, sub-dicts) are preserved from the first run.
    """
    if not results:
        return {}

    first_val = results[0].value
    summary: Dict[str, Any] = {}

    for k, v in first_val.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = [float(r.value[k]) for r in results if k in r.value and r.value[k] is not None]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                summary[k] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(arr),
                    "values": vals,
                }
        elif isinstance(v, dict):
            # Recursively summarize nested dictionaries if present
            sub_results = [MultiSeedResult(seed=r.seed, value=r.value.get(k, {})) for r in results]
            summary[k] = summarize(sub_results)
        else:
            summary[k] = v

    return summary
