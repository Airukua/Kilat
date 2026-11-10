"""Backward-compatibility shim.

Canonical architectural location is `kilat.models.base`.
This module re-exports all symbols to prevent breaking existing imports.
"""
from __future__ import annotations

from kilat.models.base import (  # noqa: F401
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    BaseConfig,
    BasePreTrainedModel,
    _shard_state_dict,
    _SAFETENSORS_AVAILABLE,
    _st_load,
    _HF_HUB_AVAILABLE,
    _hf_hub,
)

__all__ = [
    "BaseConfig",
    "BasePreTrainedModel",
    "_shard_state_dict",
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "SAFE_WEIGHTS_NAME",
    "WEIGHTS_INDEX_NAME",
    "SAFE_WEIGHTS_INDEX",
    "_SAFETENSORS_AVAILABLE",
    "_st_load",
]
