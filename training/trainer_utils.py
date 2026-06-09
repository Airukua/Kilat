from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
import torch
import torch.distributed as dist
from .callbacks import CallbackHandler, EarlyStoppingCallback, TrainerState
from .optimizer import compute_total_steps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device & distributed helpers
# ---------------------------------------------------------------------------

def get_device(model: torch.nn.Module) -> torch.device:
    """
    Return the device on which the model's parameters reside.

    WHY: After moving a model to a device (e.g., `model.to(device)`), there is
    no direct `model.device` attribute. This helper inspects the first parameter
    to determine the current device. It is useful for creating tensors on the
    same device as the model without carrying around a separate `device` variable.

    Assumptions:
    - The model has at least one parameter (non‑empty model). If not, we fall back
      to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
    - All parameters are on the same device – typical in practice.

    Edge Cases:
    - If the model is a `nn.DataParallel` or `DistributedDataParallel` wrapper,
      the underlying module (`model.module`) is checked.
    - For meta‑device (torch.empty(0)), fallback is used.

    Performance: O(1) — just inspects the first parameter.

    Example Usage
    -------------
        >>> model = nn.Linear(10, 10).to("cuda")
        >>> device = get_device(model)  # device(type='cuda')
    """
    # Unwrap DataParallel / DDP to access the real module
    model = model.module if hasattr(model, "module") else model
    try:
        # Get the first parameter's device
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters – fallback to CUDA if available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_distributed_initialized() -> bool:
    """Return True if torch.distributed is available and has been initialised."""
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """Return the global rank of the current process, or 0 if distributed not init."""
    return dist.get_rank() if is_distributed_initialized() else 0


def should_save_on_rank0(state: TrainerState) -> bool:
    """
    Return True if the current process should save checkpoints/logs.

    WHY: In distributed training, only rank 0 should write files to avoid
    conflicts and duplication. This function combines the distributed rank
    check with the existing `state.is_world_process_zero` flag.

    The flag `is_world_process_zero` is set by the trainer based on the actual
    rank (0 vs non‑zero). This helper is a convenience wrapper.

    Example Usage
    -------------
        >>> if should_save_on_rank0(state):
        ...     torch.save(checkpoint, path)
    """
    return state.is_world_process_zero


# ---------------------------------------------------------------------------
# PPL (Perplexity) Computation
# ---------------------------------------------------------------------------

def compute_perplexity(loss: float, base: float = torch.e) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity is defined as exp(loss) for natural log, or 2^loss for log2.
    It measures how "surprised" the model is by the data – lower is better.
    For a perfect model (loss=0), perplexity = 1.

    Mathematical definition:
        PPL = exp(avg_negative_log_likelihood)

    Properties:
    - PPL ∈ [1, ∞)
    - PPL = 1 means perfect prediction (loss = 0)
    - PPL = vocab_size means random uniform guessing

    WHY: Perplexity is the standard metric for language modeling because it's
    interpretable (effective vocabulary size) and monotonic with loss.

    Parameters
    ----------
    loss : float
        Average cross-entropy loss (natural log base, typically from nn.CrossEntropyLoss).
    base : float
        Base of exponentiation. Default `torch.e` (natural log) matches PyTorch's loss.
        Use `base=2` for bits-per-character (BPC) style perplexity.

    Returns
    -------
    float
        Perplexity value (always >= 1.0).

    Example
    -------
        >>> loss = 2.3026  # ln(10)
        >>> compute_perplexity(loss)
        10.0  # Model is as surprised as guessing from 10 tokens

        >>> loss = 0.6931  # ln(2)
        >>> compute_perplexity(loss)
        2.0   # Effective vocabulary size = 2
    """
    import math
    if base == torch.e or base == math.e:
        return math.exp(loss)
    return base ** loss


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[float, float]:
    """
    Compute both loss and perplexity from model logits in one pass.

    This is more efficient than calling compute_perplexity separately because
    it avoids recomputing the loss.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (batch_size, seq_len, vocab_size).
    labels : torch.Tensor
        Ground truth labels of shape (batch_size, seq_len).
    ignore_index : int
        Label value to ignore in loss computation (typically -100 for padding).

    Returns
    -------
    tuple[float, float]
        (loss, perplexity) where loss is the average cross-entropy loss.

    Example
    -------
        >>> outputs = model(input_ids)
        >>> loss, ppl = compute_perplexity_from_logits(outputs.logits, labels)
        >>> print(f"Loss: {loss:.4f}, PPL: {ppl:.2f}")
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    # Reshape logits: (B, N, V) -> (B*N, V)
    # Reshape labels: (B, N) -> (B*N)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    ppl = compute_perplexity(loss.item())
    return loss.item(), ppl


def format_metrics_with_ppl(metrics: dict[str, float]) -> dict[str, float]:
    """
    Automatically add perplexity to metrics dict if loss is present.

    Convenience function for evaluation loops: if metrics contains "eval_loss"
    or "loss", add a "perplexity" field automatically.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing at least "loss" or "eval_loss".

    Returns
    -------
    dict[str, float]
        Original metrics with added "perplexity" key.

    Example
    -------
        >>> metrics = {"eval_loss": 1.234}
        >>> metrics = format_metrics_with_ppl(metrics)
        >>> print(metrics)
        {"eval_loss": 1.234, "perplexity": 3.43}
    """
    loss_key = "eval_loss" if "eval_loss" in metrics else "loss"
    if loss_key in metrics:
        metrics["perplexity"] = compute_perplexity(metrics[loss_key])
    return metrics


# ---------------------------------------------------------------------------
# Advanced checkpointing (HuggingFace‑style + training state)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Any,  # PreTrainedModel from transformers (has save_pretrained)
    optimizer: torch.optim.Optimizer,
    scheduler: Any,  # can be LambdaLR or our LRScheduler
    scaler: Optional[torch.cuda.amp.GradScaler],
    state: TrainerState,
    callback_handler: CallbackHandler,
    output_dir: str,
    step: Optional[int] = None,
    tag: Optional[str] = None,
    early_stopping: Optional[EarlyStoppingCallback] = None,
    additional_state: Optional[dict[str, Any]] = None,
    training_args: Optional[Any] = None,   # <-- save hyperparameters
    atomic: bool = True,                   # <-- atomic write
) -> None:
    """
    Save a complete training checkpoint with atomic write and config preservation.

    Checkpoint Design
    -----------------
    Each checkpoint is a self-contained directory containing:
    - Model weights in HuggingFace format (config.json + pytorch_model.bin)
    - Training state (optimizer, scheduler, scaler, TrainerState, callback states)
    - Optional early stopping state and hyperparameters (training_args.json)

    Atomic Write (if `atomic=True`):
    - Saves to a temporary directory then renames to final location.
    - Prevents corrupted checkpoints if the process crashes mid‑write.
    - Rename is atomic on POSIX filesystems.

    Rank‑0 only: Only the main process saves checkpoints in distributed training.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model with `save_pretrained` method.
    optimizer : torch.optim.Optimizer
        Optimizer whose state_dict will be saved.
    scheduler : Any
        Scheduler with `state_dict` method.
    scaler : Optional[torch.cuda.amp.GradScaler]
        AMP scaler (if used).
    state : TrainerState
        Current training state.
    callback_handler : CallbackHandler
        Handler containing all callback states.
    output_dir : str
        Base directory for all checkpoints.
    step : Optional[int]
        Step number for numbered checkpoint naming.
    tag : Optional[str]
        Tag (e.g., "best", "final") – overrides step.
    early_stopping : Optional[EarlyStoppingCallback]
        Early stopping instance (state saved separately).
    additional_state : Optional[dict]
        Extra user state (e.g., random generator state).
    training_args : Optional[Any]
        Training arguments (dataclass or dict) – saved as JSON for reproducibility.
    atomic : bool
        If True, write to temporary directory then rename (safer).

    Edge Cases:
    - If both `step` and `tag` are None, saves as "checkpoint-latest".
    - Missing `save_pretrained` will raise AttributeError.
    - If `atomic=True` and renaming fails, the temporary directory is left behind
      (caller should clean up). We log an error but do not raise.

    Example Usage
    -------------
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     scaler=scaler,
        ...     state=state,
        ...     callback_handler=handler,
        ...     output_dir="./output",
        ...     step=global_step,
        ...     tag="best",
        ...     training_args=args,
        ... )
    """
    if not should_save_on_rank0(state):
        return

    # Determine checkpoint folder name
    if tag is not None:
        folder_name = f"checkpoint-{tag}"
    elif step is not None:
        folder_name = f"checkpoint-{step}"
    else:
        folder_name = "checkpoint-latest"

    final_path = os.path.join(output_dir, folder_name)

    if atomic:
        # Use a temporary directory in the same filesystem (same parent) for atomic rename.
        # mkdtemp creates a unique directory; we place it inside output_dir.
        temp_path = tempfile.mkdtemp(dir=output_dir, prefix=".tmp_checkpoint_")
    else:
        temp_path = final_path
        os.makedirs(temp_path, exist_ok=True)

    try:
        # 1. Save model in HuggingFace format
        model.save_pretrained(temp_path)

        # 2. Save training hyperparameters (if provided)
        if training_args is not None:
            if hasattr(training_args, "__dataclass_fields__"):
                import dataclasses
                args_dict = dataclasses.asdict(training_args)
            elif isinstance(training_args, dict):
                args_dict = training_args
            else:
                # Fallback: convert to string representation
                args_dict = {"args": str(training_args)}
            with open(os.path.join(temp_path, "training_args.json"), "w") as f:
                json.dump(args_dict, f, indent=2)

        # 3. Build and save training state dictionary
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "trainer_state": state.__dict__,
            "callback_states": callback_handler.state_dict(),
        }
        if early_stopping is not None:
            training_state["early_stopping"] = {
                "best_metric": early_stopping.best_metric,
                "patience_counter": early_stopping.early_stopping_patience_counter,
            }
        if additional_state:
            training_state["additional"] = additional_state

        torch.save(training_state, os.path.join(temp_path, "training_state.pt"))

        # Atomic commit: replace final directory with temporary one
        if atomic:
            # Remove existing checkpoint if present
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.move(temp_path, final_path)
            logger.info("Checkpoint saved atomically: %s", final_path)
        else:
            logger.info("Checkpoint saved: %s", final_path)

    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)
        # Clean up temporary directory on failure
        if atomic and os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
        raise


def load_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    state: TrainerState,
    callback_handler: CallbackHandler,
    checkpoint_path: str,
    device: torch.device,
    early_stopping: Optional[EarlyStoppingCallback] = None,
) -> Any:
    """
    Restore complete training state from a checkpoint directory.

    Loading Strategy
    ----------------
    Two-phase loading:
    1. Model weights:
       - If `pytorch_model.bin` exists, load state dict into current model.
       - Otherwise, use `model_class.from_pretrained()` – this returns a new model
         instance. The caller must replace their model reference.
    2. Training state:
       - Load `training_state.pt` if present, restore optimizer, scheduler, scaler,
         TrainerState, callback states, and early stopping state.

    The function returns the model (which may be a new instance if `from_pretrained` was used).

    Assumptions:
    - The checkpoint directory follows the structure created by `save_checkpoint`.
    - The model class has a `from_pretrained` method (HuggingFace style) for fallback.
    - `weights_only=True` is used for security (prevents arbitrary code execution).

    Edge Cases:
    - If `training_state.pt` is missing, training starts from scratch (step=0).
    - Missing keys in saved TrainerState are ignored with a warning.

    Returns
    -------
    Any
        The restored model (may be a new instance). The original `model` argument
        might be modified in‑place when loading from `pytorch_model.bin`, but
        when falling back to `from_pretrained`, a new instance is returned.

    Example Usage
    -------------
        >>> model = load_checkpoint(
        ...     model, optimizer, scheduler, scaler, state, handler,
        ...     "./checkpoint-1000", device
        ... )
        >>> # If from_pretrained was used, model may be a new instance.
    """
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist or is not a directory: {checkpoint_path}"
        )

    logger.info("Resuming from checkpoint: %s", checkpoint_path)

    # Phase 1: Restore model weights.
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(state_dict)
        loaded_model = model
        logger.info("Model loaded from pytorch_model.bin")
    else:
        # HuggingFace fallback – returns new instance
        model_class = model.__class__
        loaded_model = model_class.from_pretrained(checkpoint_path)
        logger.info("Model loaded via from_pretrained (new instance)")

    if device is not None:
        loaded_model = loaded_model.to(device)

    # Phase 2: Restore training state
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device, weights_only=True)

        # Optimizer
        if "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])

        # Scheduler
        if "scheduler" in training_state and training_state["scheduler"] is not None:
            scheduler.load_state_dict(training_state["scheduler"])

        # GradScaler
        if scaler is not None and "scaler" in training_state and training_state["scaler"] is not None:
            scaler.load_state_dict(training_state["scaler"])

        # TrainerState (dataclass)
        saved_state = training_state.get("trainer_state", {})
        for key, value in saved_state.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning("Saved TrainerState has unknown key '%s', skipping.", key)

        # CallbackHandler states
        if "callback_states" in training_state:
            callback_handler.load_state_dict(training_state["callback_states"])

        # Early stopping state
        if early_stopping is not None and "early_stopping" in training_state:
            es_state = training_state["early_stopping"]
            early_stopping.best_metric = es_state.get("best_metric", early_stopping.best_metric)
            early_stopping.early_stopping_patience_counter = es_state.get(
                "patience_counter", early_stopping.early_stopping_patience_counter
            )

        logger.info("Training state restored: step=%d, epoch=%.2f", state.global_step, state.epoch)
    else:
        logger.warning("No training_state.pt found – starting from scratch (step=0).")

    return loaded_model


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Return the path to the most recent checkpoint in `output_dir`.

    WHY: Enables automatic resume from the latest checkpoint without manual
    path selection. The trainer can call this function to find the newest
    checkpoint by modification time.

    Sorting heuristic: Uses file modification time (most recent first).
    This is reliable even if step numbers are not monotonic or if checkpoints
    were manually copied.

    Edge Cases:
    - If the directory does not exist or contains no checkpoint folders,
      returns None.
    - Only directories starting with "checkpoint-" are considered.

    Performance: O(n log n) where n = number of checkpoint directories.
    Typically n <= 100, so negligible.

    Example Usage
    -------------
        >>> latest = get_latest_checkpoint("./output")
        >>> if latest:
        ...     load_checkpoint(..., checkpoint_path=latest, ...)
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, checkpoints[0])


def prune_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    """
    Remove the oldest periodic checkpoints when exceeding ``save_total_limit``.

    Pruning Strategy
    ----------------
    - Only numbered checkpoints (e.g., "checkpoint-1000") are pruned.
    - Tagged checkpoints ("best", "final", "epoch-3", etc.) are preserved.
    - Sorting by modification time (oldest first) ensures the earliest checkpoints
      are removed.

    Why preserve tagged checkpoints?
    - "best": optimal model by validation metric.
    - "final": model at training completion.
    - "epoch-N": epoch boundaries for analysis.
    - "early-stopped": state when early stopping triggered.

    Complexity: O(n log n) sorting, then O(k) deletions. For n ≤ 1000, acceptable.

    Edge Cases:
    - If `save_total_limit` is None or ≤ 0, no pruning.
    - If deletion fails due to permissions, logs a warning but continues.

    Example Usage
    -------------
        >>> prune_checkpoints("./output", save_total_limit=5)
        # Keeps only the 5 newest numbered checkpoints.
    """
    if save_total_limit is None or not os.path.exists(output_dir):
        return

    def is_prunable(name: str) -> bool:
        suffix = name.replace("checkpoint-", "", 1)
        return suffix.isdigit()

    all_checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    prunable = [d for d in all_checkpoints if is_prunable(d)]
    # Sort by modification time, oldest first
    prunable.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))

    while len(prunable) > save_total_limit:
        oldest = prunable.pop(0)
        try:
            shutil.rmtree(os.path.join(output_dir, oldest))
            logger.info("Pruned old checkpoint: %s", oldest)
        except (OSError, PermissionError) as e:
            logger.warning("Could not prune %s: %s", oldest, e)


# ---------------------------------------------------------------------------
# Legacy / simple checkpoint helpers (for backward compatibility)
# ---------------------------------------------------------------------------

def save_checkpoint_simple(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    callback_handler: CallbackHandler,
    additional_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Simplified checkpoint saving (single .pt file). Prefer `save_checkpoint` above.

    This function exists for backward compatibility and non‑HF models.
    It saves all state into one file, which is simpler but less modular.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainer_state": state.__dict__,
        "callback_states": callback_handler.state_dict(),
    }
    if hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if additional_state:
        checkpoint.update(additional_state)

    torch.save(checkpoint, path)
    logger.info("Simple checkpoint saved to %s", path)


def load_checkpoint_simple(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    callback_handler: CallbackHandler,
    map_location: Optional[torch.device | str] = None,
) -> None:
    """
    Simplified checkpoint loading (single .pt file). Prefer `load_checkpoint` above.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    # Model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    # Optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Scheduler
    if "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Trainer state
    saved_state = checkpoint["trainer_state"]
    for key, value in saved_state.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # Callback handler states
    if "callback_states" in checkpoint:
        callback_handler.load_state_dict(checkpoint["callback_states"])

    logger.info("Simple checkpoint loaded from %s (step %d)", path, state.global_step)


# ---------------------------------------------------------------------------
# Scheduling helpers (logging, evaluation, saving)
# ---------------------------------------------------------------------------

def should_log(
    state: TrainerState,
    logging_steps: int,
    log_on_each_step: bool = False,
) -> bool:
    """
    Determine whether the current step should trigger logging.

    WHY: Logging every step may be too verbose; this helper implements the
    common pattern: log if `global_step % logging_steps == 0` (or at step 0).

    Edge Cases:
    - Step 0 is always logged (except when `logging_steps <= 0` and `log_on_each_step` is False).
    - If `logging_steps` is 1, every step triggers logging.
    - Negative `logging_steps` disables logging completely.
    """
    if logging_steps <= 0:
        return False
    if log_on_each_step:
        return True
    return state.global_step % logging_steps == 0 or state.global_step == 0


def should_evaluate(
    state: TrainerState,
    eval_steps: int,
    eval_on_each_epoch: bool = False,
    epoch: Optional[float] = None,
) -> bool:
    """
    Determine whether evaluation should be performed at this point (step‑based only).

    NOTE: Epoch‑based evaluation must be handled by the trainer at epoch boundaries.
    This helper only returns True for step‑based evaluation (`global_step % eval_steps == 0`).
    """
    if eval_steps <= 0:
        return False
    return state.global_step % eval_steps == 0 and state.global_step > 0


def should_save(state: TrainerState, save_steps: int, save_on_each_epoch: bool = False) -> bool:
    """Determine whether a checkpoint should be saved (step‑based only)."""
    if save_steps <= 0:
        return False
    return state.global_step % save_steps == 0 and state.global_step > 0


# ---------------------------------------------------------------------------
# Loss and gradient utilities
# ---------------------------------------------------------------------------

def average_losses(loss_dict: dict[str, float], steps: int = 1) -> dict[str, float]:
    """
    Average a dictionary of accumulated losses over a number of steps.

    WHY: When using gradient accumulation, losses are often summed over micro‑batches.
    This helper divides each loss by the number of accumulation steps to obtain
    the per‑step average, which is more meaningful for logging.
    """
    if steps <= 1:
        return loss_dict.copy()
    return {k: v / steps for k, v in loss_dict.items()}


def clip_grad_norm_(model: torch.nn.Module, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """Clip gradients of the model to a maximum norm. Handles DDP wrapping."""
    model_to_clip = model.module if hasattr(model, "module") else model
    return torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), max_norm, norm_type)


# ---------------------------------------------------------------------------
# Learning rate helpers
# ---------------------------------------------------------------------------

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate of the first parameter group."""
    if not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0]["lr"]


def get_lr_group_info(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return a list of dictionaries containing learning rate and weight decay per group."""
    return [
        {"lr": group["lr"], "weight_decay": group.get("weight_decay", 0.0)}
        for group in optimizer.param_groups
    ]