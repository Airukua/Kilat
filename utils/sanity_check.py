from __future__ import annotations
from typing import Iterable, Sequence
import torch

def validate_positive_int(name: str, value: int) -> int:
    """
    Validate that a value is a positive integer (> 0).

    This is used extensively for hyperparameters and dimensions where
    zero or negative values are nonsensical (e.g., batch_size, num_heads,
    hidden dimensions). The check catches both type errors (floats passed
    where ints expected) and range errors.

    Why not >= 0? Most parameters this validates represent counts or sizes
    that must be at least 1 to be meaningful. For parameters that allow zero,
    use validate_non_negative_float or a custom check.

    Parameters
    ----------
    name : str
        Parameter name for error messages (e.g., "num_heads", "dim").
    value : int
        Value to validate.

    Returns
    -------
    int
        The validated value (pass-through for chaining).

    Raises
    ------
    ValueError
        If value is not an int or is <= 0.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value

def validate_non_negative_float(name: str, value: float) -> float:
    """
    Validate that a value is a non-negative float (>= 0).

    Used for parameters like loss coefficients, dropout rates (before
    probability validation), and other values that can be zero but not negative.
    Does NOT check for upper bounds — combine with validate_probability
    for bounded [0, 1] values.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : float
        Value to validate.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If value < 0.
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value

def validate_probability(name: str, value: float, *, upper_open: bool = True) -> float:
    """
    Validate that a value is a valid probability.

    By default, validates [0, 1) — closed on the left, open on the right.
    This is appropriate for dropout rates and most probability parameters
    where 1.0 would be degenerate (e.g., 100% dropout would zero out
    everything, making training impossible).

    Set upper_open=False for [0, 1] validation when 1.0 is acceptable
    (e.g., keep probability in some contexts, or mixing coefficients
    where 1.0 means "use only one path").

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : float
        Value to validate.
    upper_open : bool
        If True (default), validates [0, 1). If False, validates [0, 1].

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If value is outside the valid range.
    """
    if upper_open:
        valid = 0.0 <= value < 1.0
        bound = "[0, 1)"
    else:
        valid = 0.0 <= value <= 1.0
        bound = "[0, 1]"
    if not valid:
        raise ValueError(f"{name} must be in {bound}, got {value}")
    return value

def validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
    """
    Validate that a string value is one of the allowed choices.

    Used for mode parameters (e.g., mode='dense' vs 'moe', precision='fp16'
    vs 'bf16'). Case-sensitive comparison — "FP16" would fail if choices
    contain "fp16".

    The error message lists all valid choices, making it easy for users
    to see what's available without consulting documentation.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : str
        Value to check.
    choices : Sequence[str]
        Allowed string values.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If value is not in choices.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value

def validate_divisible(name: str, value: int, divisor: int, detail: str | None = None) -> int:
    """
    Validate that an integer is divisible by another integer.

    Critical for architecture validation: n_embd must be divisible by n_head
    so that head_dim = n_embd / n_head is an integer. Also used for tensor
    parallelism checks (world_size must divide num_heads).

    The detail parameter provides context in the error message, useful
    for explaining WHY the divisibility is required (e.g., "n_embd must
    be divisible by n_head to compute head_dim").

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : int
        Value to check.
    divisor : int
        Divisor (must evenly divide value).
    detail : str | None
        Optional explanation appended to error message.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    ValueError
        If value % divisor != 0.
    """
    if value % divisor != 0:
        suffix = f" ({detail})" if detail else ""
        raise ValueError(f"{name} must be divisible by {divisor}, got {value}{suffix}")
    return value

def validate_sequence_length(name: str, values: Sequence[object], expected: int) -> Sequence[object]:
    """
    Validate that a sequence has an exact expected length.

    Used for checking tuple/list shapes or argument groups where the
    number of elements is semantically meaningful and must be exact.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    values : Sequence
        Sequence to check.
    expected : int
        Required length.

    Returns
    -------
    Sequence
        The validated sequence.

    Raises
    ------
    ValueError
        If len(values) != expected.
    """
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(values)}")
    return values


def validate_less_equal(name: str, value: int, upper: int) -> int:
    """
    Validate that an integer is less than or equal to an upper bound.

    Used for constraints like active_experts <= num_experts (can't select
    more experts than exist) or sequence_length <= max_position_embeddings.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : int
        Value to check.
    upper : int
        Maximum allowed value (inclusive).

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    ValueError
        If value > upper.
    """
    if value > upper:
        raise ValueError(f"{name} must be <= {upper}, got {value}")
    return value


def validate_tensor_rank(x: torch.Tensor, expected_rank: int, name: str = "tensor") -> torch.Tensor:
    """
    Validate that a tensor has the expected number of dimensions.

    Most transformer modules expect 3D tensors (batch, seq_len, dim).
    This catches shape errors early, before they propagate through the
    network and produce confusing error messages downstream (e.g., matmul
    dimension mismatch that doesn't clearly indicate the actual problem).

    Parameters
    ----------
    x : torch.Tensor
        Tensor to validate.
    expected_rank : int
        Required number of dimensions.
    name : str
        Tensor name for error messages.

    Returns
    -------
    torch.Tensor
        The validated tensor (pass-through).

    Raises
    ------
    ValueError
        If x.dim() != expected_rank.
    """
    if x.dim() != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {tuple(x.shape)}")
    return x

def validate_tensor_last_dim(x: torch.Tensor, expected_dim: int, name: str = "tensor") -> torch.Tensor:
    """
    Validate that a tensor's last dimension matches the expected size.

    Used before linear layers and other dimension-specific operations to
    catch shape mismatches. Checking the last dim specifically is important
    because batch and sequence dimensions can vary, but the feature dimension
    must match the layer's weight matrix.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to validate.
    expected_dim : int
        Required size of the last dimension.
    name : str
        Tensor name for error messages.

    Returns
    -------
    torch.Tensor
        The validated tensor.

    Raises
    ------
    ValueError
        If x.size(-1) != expected_dim.
    """
    if x.size(-1) != expected_dim:
        raise ValueError(
            f"{name} must have trailing dim {expected_dim}, got shape {tuple(x.shape)}"
        )
    return x

def validate_tensor_shape(x: torch.Tensor, expected: Sequence[int], name: str = "tensor") -> torch.Tensor:
    """
    Validate that a tensor has an exact expected shape.

    Used when both the number of dimensions AND their sizes must match
    exactly (e.g., loss computation, concatenation operations). Unlike
    validate_tensor_rank + validate_tensor_last_dim, this checks all
    dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to validate.
    expected : Sequence[int]
        Required exact shape.
    name : str
        Tensor name for error messages.

    Returns
    -------
    torch.Tensor
        The validated tensor.

    Raises
    ------
    ValueError
        If tuple(x.shape) != tuple(expected).
    """
    if tuple(x.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(x.shape)}")
    return x

def validate_finite_tensor(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Validate that all elements of a tensor are finite (no NaN or Inf).

    This is a critical safety check applied to module outputs. NaN/Inf
    values in tensors can:
    - Corrupt model weights when used in gradient computation
    - Propagate silently through subsequent layers
    - Cause the optimizer to produce NaN parameter updates
    
    By checking at module boundaries (after attention, FFN, normalization),
    we catch numerical issues close to their source rather than discovering
    them tens of layers later when the loss becomes NaN.

    The check uses torch.isfinite() which returns True for normal floats,
    and False for NaN, +Inf, -Inf. Calling .all() requires the ENTIRE
    tensor to be finite — even a single bad value triggers the error.

    Performance note: This validation adds a small overhead (one full-tensor
    check per module call). In production deployment, these checks can be
    disabled via a global flag or stripped entirely, but during development
    and training, the early error detection far outweighs the cost.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to validate.
    name : str
        Tensor name for error messages (e.g., "SwiGLU output", "attention output").

    Returns
    -------
    torch.Tensor
        The validated tensor (pass-through for chaining).

    Raises
    ------
    ValueError
        If any element is NaN or Inf.
    """
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return x

def validate_optional_dict(name: str, value: dict | None) -> dict:
    """
    Validate and normalize an optional dictionary parameter.

    Converts None to an empty dict, enabling callers to write:
        config = validate_optional_dict("override", override)
        final = {**defaults, **config}
    
    without special-casing the None case. Raises TypeError if a non-dict,
    non-None value is passed (e.g., a list or string).

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : dict | None
        Value to validate.

    Returns
    -------
    dict
        The original dict, or {} if value was None.

    Raises
    ------
    TypeError
        If value is neither dict nor None.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict or None, got {type(value).__name__}")
    return value

def validate_all_finite_tensors(pairs: Iterable[tuple[str, torch.Tensor]]) -> None:
    """
    Batch-validate that multiple named tensors are all finite.

    Convenience function for module forward methods that compute multiple
    output tensors. Instead of calling validate_finite_tensor individually
    for each output (which would stop at the first error), this validates
    all tensors and reports which ones failed.

    Note: Currently stops at the first invalid tensor (calls validate_finite_tensor
    sequentially). A future enhancement could collect all failures before raising.

    Parameters
    ----------
    pairs : Iterable[tuple[str, torch.Tensor]]
        (name, tensor) pairs to validate.

    Raises
    ------
    ValueError
        If any tensor contains NaN or Inf.
    """
    for name, tensor in pairs:
        validate_finite_tensor(tensor, name)