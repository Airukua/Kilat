from kilat.utils.banner import get_kilat_banner, print_kilat_banner
from kilat.utils.validators import (
    validate_all_finite_tensors,
    validate_choice,
    validate_divisible,
    validate_finite_tensor,
    validate_less_equal,
    validate_non_negative_float,
    validate_non_negative_int,
    validate_optional_dict,
    validate_positive_int,
    validate_probability,
    validate_sequence_length,
    validate_tensor_last_dim,
    validate_tensor_rank,
    validate_tensor_shape
)


# Backward-compatible re-exports of model-layer symbols. Resolved lazily (PEP 562) so that
# Layer-0 `kilat.utils` never imports Layer-2 `kilat.models` at import time; an eager import
# here creates a circular dependency as soon as `models` uses any `utils` leaf module.
_LAZY_MODEL_BASE_EXPORTS = frozenset({
    "BaseConfig",
    "BasePreTrainedModel",
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "SAFE_WEIGHTS_NAME",
    "SAFE_WEIGHTS_INDEX",
    "WEIGHTS_INDEX_NAME",
    "_SAFETENSORS_AVAILABLE",
    "_st_load",
})


def __getattr__(name: str):
    if name in _LAZY_MODEL_BASE_EXPORTS:
        from kilat.models import base as _model_base

        return getattr(_model_base, name)
    raise AttributeError(f"module 'kilat.utils' has no attribute {name!r}")


__all__ = [
    "get_kilat_banner",
    "print_kilat_banner",
    "BaseConfig",
    "BasePreTrainedModel",
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "SAFE_WEIGHTS_NAME",
    "SAFE_WEIGHTS_INDEX",
    "WEIGHTS_INDEX_NAME",
    "_SAFETENSORS_AVAILABLE",
    "_st_load",
    "validate_all_finite_tensors",
    "validate_choice",
    "validate_divisible",
    "validate_finite_tensor",
    "validate_less_equal",
    "validate_non_negative_float",
    "validate_non_negative_int",
    "validate_optional_dict",
    "validate_positive_int",
    "validate_probability",
    "validate_sequence_length",
    "validate_tensor_last_dim",
    "validate_tensor_rank",
    "validate_tensor_shape",
]
