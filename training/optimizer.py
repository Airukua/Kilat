from __future__ import annotations
import logging
import math
from typing import Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with decoupled weight decay and parameter groups.

    Why AdamW over Adam:
    Traditional Adam/L2-regularization couples weight decay with the learning rate
    schedule and adaptive gradient scaling, making optimal weight decay dependent on
    the learning rate. AdamW ("Decoupled Weight Decay Regularization", Loshchilov &
    Hutter, 2019) applies weight decay directly to the weights after the gradient
    update, decoupling it from the adaptive mechanism. This makes hyperparameter
    tuning more intuitive and often yields better generalization.

    Parameter Grouping Strategy:
    Weights that undergo multiplicative operations (linear layers, embeddings) should
    receive weight decay, acting as an L2 penalty that prevents weights from growing
    arbitrarily large. Parameters that are additive (biases) or normalization terms
    (LayerNorm, RMSNorm) should NOT be weight-decayed, as penalizing their magnitude
    can hurt the model's ability to shift activations or adjust normalization scales.

    The no_decay_keywords cover:
    - "bias": Additive parameters that shift activations
    - "LayerNorm", "rms": Normalization weight/scale parameters (multiplicative but
      norm-specific, decaying them hurts representational capacity)
    - "rms1", "rms2": Specific RMS normalization layers from Kilat architecture
    - "ln_f": Final layer norm before output projection (GPT-style models)

    Beta values and epsilon:
    Transformer training commonly uses beta1=0.9 and beta2=0.95 to keep
    the moving average responsive to recent gradients. epsilon stabilizes the
    adaptive denominator and is usually left at the PyTorch default 1e-8.

    Parameters
    ----------
    model : nn.Module
        Model with named parameters. Parameters with requires_grad=False are ignored.
    learning_rate : float
        Peak learning rate (used as initial LR for optimizer).
    weight_decay : float
        Weight decay coefficient applied to non-normalization, non-bias parameters.
    beta1, beta2 : float
        Adam moment decay rates.
    epsilon : float
        Adam numerical stability term.

    Returns
    -------
    torch.optim.Optimizer
        Configured AdamW optimizer with parameter-specific weight decay groups.

    Example Usage
    -------------
        >>> optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.01)
    """
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []

    no_decay_keywords = ("bias", "LayerNorm", "rms", "ln_f", "rms1", "rms2")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    scheduler_type: str = "cosine",
    **kwargs: Any,
) -> Any:
    """
    Create a learning rate scheduler by delegating to ``schedulers.get_scheduler``.

    This function is a thin compatibility wrapper. All scheduler logic lives in
    ``schedulers.py``; this entry point exists so that call-sites that previously
    used ``create_scheduler(optimizer, total_steps, warmup_steps)`` continue to
    work without change while now benefiting from the full scheduler registry
    (cosine, linear, wsdlr, rex, inverse_sqrt, etc.).

    Migration path
    --------------
    Old (cosine-only hardcoded):
        scheduler = create_scheduler(optimizer, total_steps=10_000, warmup_steps=500)

    New (any scheduler type):
        scheduler = create_scheduler(
            optimizer,
            total_steps=10_000,
            warmup_steps=500,
            scheduler_type="wsdlr",
            num_decay_steps=1_000,   # forwarded via **kwargs
        )

    Or call get_scheduler() directly for full control:
        from .schedulers import get_scheduler
        scheduler = get_scheduler("wsdlr", optimizer, warmup_steps=500,
                                  num_training_steps=10_000, num_decay_steps=1_000)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to attach the scheduler to.
    total_steps : int
        Total number of optimizer steps (including warmup).
    warmup_steps : int
        Number of linear warmup steps.
    scheduler_type : str
        Scheduler name from the SchedulerType enum (default: "cosine").
        Valid built-ins: "linear", "cosine", "cosine_with_restarts",
        "cosine_with_min_lr", "polynomial", "constant",
        "constant_with_warmup", "inverse_sqrt", "wsdlr", "rex".
    **kwargs
        Extra arguments forwarded to the scheduler constructor:
        - cosine_with_restarts : num_cycles (int)
        - cosine_with_min_lr   : min_lr_ratio (float)
        - polynomial           : power (float), lr_end_ratio (float)
        - wsdlr                : num_decay_steps (int)

    Returns
    -------
    LRScheduler
        Instance of the requested scheduler (subclass of schedulers.LRScheduler,
        which wraps a torch.optim.lr_scheduler.LambdaLR internally).

    Raises
    ------
    ValueError
        If scheduler_type is not a registered SchedulerType value.

    Example Usage
    -------------
        >>> # Default cosine (backward-compatible)
        >>> scheduler = create_scheduler(optimizer, total_steps=10_000, warmup_steps=500)

        >>> # WSD schedule for Kilat pretraining
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     total_steps=100_000,
        ...     warmup_steps=2_000,
        ...     scheduler_type="wsdlr",
        ...     num_decay_steps=5_000,
        ... )
    """
    # Import here to avoid circular imports at module load time.
    # schedulers.py does not import optimizer_utils.py, so no cycle.
    from .schedulers import get_scheduler  # noqa: PLC0415

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        **kwargs,
    )


def resolve_amp_dtype(precision: str) -> Optional[torch.dtype]:
    """
    Map precision string to torch dtype for Automatic Mixed Precision.

    Design rationale:
    - fp16: Uses torch.float16. Requires GradScaler for gradient stability because
      FP16 has limited dynamic range (5-bit exponent, 10-bit mantissa). Gradients
      of small magnitude can underflow to zero, and large gradients can overflow
      to infinity.
    - bf16: Uses torch.bfloat16. Has the same 8-bit exponent as FP32, so it can
      represent the same range of values. No GradScaler needed, which simplifies
      training and improves performance. Only available on Ampere+ GPUs (A100,
      RTX 3090, etc.) and TPUs.
    - fp32: Returns None (no AMP casting). Used for debugging, CPU training, or
      when mixed precision causes instability.

    The None return for fp32 is intentional: it signals to the autocast context
    manager that no dtype conversion should occur, keeping all operations in FP32.

    Parameters
    ----------
    precision : str
        One of "fp16", "bf16", or "fp32" (case-sensitive).

    Returns
    -------
    Optional[torch.dtype]
        torch.float16 for fp16, torch.bfloat16 for bf16, None for fp32.

    Example Usage
    -------------
        >>> dtype = resolve_amp_dtype("bf16")
        >>> with torch.autocast(device_type="cuda", dtype=dtype):
        ...     output = model(input)
    """
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def compute_total_steps(
    training_mode: str,
    max_steps: int,
    num_train_epochs: int,
    dataloader_len: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Compute the total number of optimizer steps for the training run.

    This function handles the fundamental difference between step-based and
    epoch-based training:

    Steps mode:
    Total steps = max_steps (directly specified by user)
    The dataloader length is irrelevant; training stops after exactly max_steps
    optimizer updates regardless of how many epochs that corresponds to.

    Epochs mode:
    Total steps = steps_per_epoch * num_train_epochs
    where steps_per_epoch = dataloader_len // gradient_accumulation_steps
    This converts epoch count to step count for the scheduler, which always
    operates in terms of optimizer steps.

    Why integer division (//):
    If dataloader_len is not perfectly divisible by accumulation steps, the
    remaining batches in an epoch don't trigger an optimizer step. This is correct
    because partial accumulation windows don't produce weight updates.

    max(1, ...) prevents zero steps_per_epoch when dataloader has fewer batches
    than accumulation steps (e.g., tiny test dataset with 3 batches but
    accumulation=4).

    Parameters
    ----------
    training_mode : str
        Either "steps" or "epochs".
    max_steps : int
        Direct step budget (used only in "steps" mode).
    num_train_epochs : int
        Number of epochs (used only in "epochs" mode).
    dataloader_len : int
        Number of batches in the training dataloader.
    gradient_accumulation_steps : int
        Number of forward passes before one optimizer step.

    Returns
    -------
    int
        Total number of optimizer steps the scheduler should plan for.

    Example Usage
    -------------
        >>> total_steps = compute_total_steps(
        ...     training_mode="epochs",
        ...     max_steps=0,
        ...     num_train_epochs=3,
        ...     dataloader_len=1024,
        ...     gradient_accumulation_steps=8,
        ... )
        >>> # total_steps = (1024 // 8) * 3 = 128 * 3 = 384
    """
    if training_mode == "steps":
        return max_steps
    else:
        steps_per_epoch = max(1, dataloader_len // gradient_accumulation_steps)
        return steps_per_epoch * num_train_epochs