from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
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
    - "rms1", "rms2": Likely specific RMS normalization layers from a particular 
      architecture (e.g., some LLM variants)
    - "ln_f": Final layer norm before output projection (present in GPT-style models)

    Beta values (0.9, 0.95):
    Standard transformer training often uses slightly higher beta2 (0.95-0.999) than 
    the PyTorch default (0.999). The lower beta2 means the moving average of squared 
    gradients decays faster, making the optimizer more responsive to recent gradient 
    changes. This is beneficial for large language models where the loss landscape 
    changes rapidly during training. These specific values are common in GPT/LLM 
    training literature.

    Parameters
    ----------
    model : nn.Module
        Model with named parameters. Parameters with requires_grad=False are ignored.
    learning_rate : float
        Peak learning rate for cosine schedule (used as initial LR for optimizer).
    weight_decay : float
        Weight decay coefficient applied to non-normalization, non-bias parameters.

    Returns
    -------
    torch.optim.Optimizer
        Configured AdamW optimizer with parameter-specific weight decay groups.
    """
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []

    # Keywords that identify parameters which should NOT receive weight decay.
    # These are parameters that are additive or normalize activations, where
    # weight decay would be counterproductive.
    no_decay_keywords = ("bias", "LayerNorm", "rms", "ln_f", "rms1", "rms2")

    for name, param in model.named_parameters():
        # Skip frozen parameters entirely - they don't need optimization at all.
        # This supports partial fine-tuning scenarios where some layers are frozen.
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    # Grouped parameters allow different optimization hyperparameters for different
    # parameter types. Here, we use weight_decay=0.0 for normalization/bias parameters
    # to prevent regularizing them, while weight_decay from args is applied to
    # weight matrices and embeddings.
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create cosine learning rate scheduler with linear warmup.

    Schedule Design (standard in transformer training):
    - Warmup phase: LR increases linearly from 0 to peak_lr over warmup_steps.
      This prevents catastrophic early updates when model weights are far from 
      their optimal values, giving the optimizer time to establish reasonable 
      gradient statistics (especially important for Adam's momentum buffers).
    - Cosine decay phase: LR follows a cosine curve from peak_lr to 0 over the 
      remaining steps. This provides smooth, monotonic decay that naturally 
      converges to small learning rates, enabling fine convergence at the end 
      of training. Based on "SGDR: Stochastic Gradient Descent with Warm Restarts" 
      (Loshchilov & Hutter, 2017), adapted for a single cycle.

    The warmup ratio is implicitly defined by warmup_steps / total_steps.
    Common practice: 1-10% of total steps for warmup, depending on model size
    and training stability requirements.

    Edge cases handled:
    - If warmup_steps == total_steps, the schedule never leaves warmup phase
      (reaches LR=1.0 at the end, which means peak_lr from optimizer).
    - max(1, denominator) prevents division by zero if warmup_steps=0 or
      total_steps=warmup_steps, returning a constant LR instead of NaN.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    total_steps : int
        Total number of optimizer steps (including warmup).
    warmup_steps : int
        Number of steps for linear warmup phase.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler that multiplies base LR by lambda values from [0, 1].
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup: Linear interpolation from 0 to 1 over warmup_steps.
        # Using float() ensures Python float division (not integer division).
        # max(1, warmup_steps) prevents division by zero.
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay: Progress from 0 to 1 representing position in decay phase.
        # The formula 0.5 * (1 + cos(pi * progress)) maps [0, 1] -> [1, 0],
        # creating a smooth cosine curve starting at peak LR and ending at 0.
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    because partial accumulation windows don't produce weight updates. For example, 
    if dataloader has 10 batches and accumulation=4, we get 2 full updates per 
    epoch (8 batches used), and the last 2 batches' gradients are carried to the 
    next epoch's first accumulation window.

    max(1, ...) prevents zero steps_per_epoch when dataloader has fewer batches 
    than accumulation steps (e.g., tiny test dataset with 3 batches but 
    accumulation=4). Without this, total_steps=0 and no training occurs.

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
    """
    if training_mode == "steps":
        return max_steps
    else:
        # Convert epochs to optimizer steps accounting for gradient accumulation
        steps_per_epoch = max(
            1,
            dataloader_len // gradient_accumulation_steps,
        )
        return steps_per_epoch * num_train_epochs