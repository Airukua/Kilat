from __future__ import annotations
import os
import shutil
from typing import Any, Optional
import torch
from .early_stopping import EarlyStoppingCallback


def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    global_step: int,
    current_epoch: int,
    best_eval_loss: float,
    early_stopping: Optional[EarlyStoppingCallback],
    output_dir: str,
    step: int,
    tag: Optional[str] = None,
) -> None:
    """
    Save a complete training checkpoint for resumption and model export.

    Checkpoint Design
    -----------------
    Each checkpoint is a self-contained directory containing everything needed
    to either resume training exactly where it left off OR load the model for
    inference/evaluation. This dual-purpose design avoids the common pitfall
    of saving inference-only checkpoints that lack training state.

    Two components are saved:

    1. Model weights (via model.save_pretrained):
       - Saves the HuggingFace-standard model files (config.json, model weights)
       - Enables loading with from_pretrained() for inference or fine-tuning
       - Maintains full HF ecosystem compatibility (tokenizer, hub upload, etc.)

    2. Training state (via torch.save):
       - Optimizer state (momentum buffers, parameter-specific values)
       - Scheduler state (current step in LR schedule)
       - GradScaler state (loss scale factor for FP16 training)
       - Counters (global_step, current_epoch, best_eval_loss)
       - Early stopping state (best_loss, fail_counter) if active

    Without the full training state, resuming training would:
    - Reset optimizer momentum buffers → different optimization trajectory
    - Restart LR schedule → potentially skip warmup or be at wrong LR
    - Lose GradScaler's dynamically-adjusted scale → FP16 instability
    - Reset early stopping patience → unintended extended training

    Naming Convention
    -----------------
    Tagged vs numbered checkpoints:
    - Numbered: "checkpoint-1000" (periodic saves at save_steps intervals)
    - Tagged: "checkpoint-best", "checkpoint-final", "checkpoint-epoch-3"
    
    Tagged checkpoints are preserved during pruning; only numbered periodic
    checkpoints are candidates for automatic deletion. This ensures important
    milestones (best model, final model) are never accidentally pruned.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model instance with save_pretrained method.
    optimizer : torch.optim.Optimizer
        Optimizer whose state_dict will be saved.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Scheduler whose state_dict will be saved.
    scaler : torch.amp.GradScaler
        AMP gradient scaler whose state_dict will be saved.
    global_step : int
        Current optimizer step count.
    current_epoch : int
        Current epoch number (for epoch-based training).
    best_eval_loss : float
        Best validation loss achieved so far.
    early_stopping : Optional[EarlyStoppingCallback]
        Early stopping instance (None if eval is disabled).
    output_dir : str
        Base directory for all checkpoints.
    step : int
        Current step number used for numbered checkpoint naming.
    tag : Optional[str]
        Descriptive tag (e.g., "best", "final", "epoch-3") that overrides
        step-based naming when provided.
    """
    # Naming: tagged checkpoints use the tag, periodic checkpoints use step number.
    # This distinction matters for pruning — only numbered checkpoints are
    # candidates for automatic deletion.
    folder_name = f"checkpoint-{step}" if tag is None else f"checkpoint-{tag}"
    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Save model in HuggingFace format for ecosystem compatibility.
    # This writes config.json + model weights, making the checkpoint
    # directly loadable by transformers.AutoModel.from_pretrained().
    model.save_pretrained(save_path)

    # Build training state dict — everything needed to resume training
    # exactly from this point. Using explicit keys rather than a generic
    # dict makes the save format self-documenting.
    training_state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "current_epoch": current_epoch,
        "best_eval_loss": best_eval_loss,
    }
    
    # Early stopping state is conditional because eval may be disabled.
    # Saving it when present is critical: without it, a resumed run would
    # reset the patience counter and potentially train much longer than
    # intended before stopping.
    if early_stopping is not None:
        training_state["early_stopping"] = {
            "best_loss": early_stopping.best_loss,
            "fail_counter": early_stopping.fail_counter,
        }

    torch.save(training_state, os.path.join(save_path, "training_state.pt"))
    print(f"Checkpoint saved: {save_path}")


def resume_from_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    early_stopping: Optional[EarlyStoppingCallback],
    checkpoint_path: str,
    device: torch.device,
) -> tuple[int, int, float]:
    """
    Restore complete training state from a previously saved checkpoint.

    Loading Strategy
    ----------------
    Two-phase loading handles both HuggingFace-native and custom checkpoints:

    Phase 1 - Model weights: Attempts torch.load first (for checkpoints saved
    as raw state dicts), falls back to from_pretrained() (for standard HF
    format). This dual approach ensures compatibility whether the checkpoint
    was saved by this trainer or came from a standard HF source.

    Phase 2 - Training state: Loads optimizer, scheduler, scaler, and counters
    from training_state.pt. If this file is absent (e.g., loading an inference-only
    checkpoint), training starts fresh with step 0 — useful when you want to
    start fine-tuning from a pretrained model rather than resume interrupted
    training.

    Model Loading Fallback
    ----------------------
    Using model.__class__.from_pretrained() instead of the instance method:
    - Instance methods might not handle re-initialization correctly
    - Class method ensures clean model construction
    - Following HuggingFace's own pattern for checkpoint loading

    weights_only=True: Security measure that prevents arbitrary code execution
    during unpickling. Since checkpoint files only contain tensor data, this
    is both safer and slightly faster.

    State Restoration Details
    -------------------------
    - Optimizer state includes per-parameter momentum/variance buffers in Adam.
      Without this, the optimizer "forgets" its adaptive learning rates.
    - Scheduler state includes the current step count. Without this, the LR
      schedule restarts from step 0 (repeating warmup, wrong decay phase).
    - GradScaler state includes the dynamically-adjusted loss scale. Without
      this, FP16 training may experience gradient underflow/overflow as the
      scaler relearns appropriate scales.
    - best_eval_loss uses .get() with default float("inf") for backward
      compatibility with older checkpoints that didn't save this field.
    - Early stopping state is conditionally restored to preserve the exact
      patience counter across preemptions.

    Parameters
    ----------
    model : PreTrainedModel
        Model instance to load weights into (modified in-place).
    optimizer : torch.optim.Optimizer
        Optimizer to restore state into (modified in-place).
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Scheduler to restore state into (modified in-place).
    scaler : torch.amp.GradScaler
        GradScaler to restore state into (modified in-place).
    early_stopping : Optional[EarlyStoppingCallback]
        Early stopping instance to restore (modified in-place if present).
    checkpoint_path : str
        Path to the checkpoint directory (contains model files + training_state.pt).
    device : torch.device
        Device to map tensors to (important for CPU→GPU or GPU→CPU transitions).

    Returns
    -------
    tuple[int, int, float]
        (global_step, current_epoch, best_eval_loss) — the restored training
        position. These values are used to initialize the trainer's counters
        and progress bars.
    """
    print(f"\n{'-'*60}")
    print(f"Resuming from checkpoint: {checkpoint_path}")
    print(f"{'-'*60}")

    # Phase 1: Restore model weights.
    # Prioritize direct state_dict loading (fastest, simplest), but fall back
    # to from_pretrained for HF-native checkpoints that use safetensors or
    # sharded weights. The existence check on pytorch_model.bin determines
    # the loading strategy.
    state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        # weights_only=True for security: prevents arbitrary code execution
        # from potentially malicious checkpoint files.
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Use from_pretrained on the class, not the instance.
        # This is the standard HF pattern for loading models and handles
        # various formats (safetensors, sharded, etc.) automatically.
        model.__class__.from_pretrained(checkpoint_path)

    # Phase 2: Restore training state (optimizer, scheduler, scaler, counters).
    # This file is trainer-specific — absence means we're loading a model
    # for inference or fresh fine-tuning, not resuming interrupted training.
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(
            training_state_path, map_location=device, weights_only=True
        )
        # Restore optimizer and scheduler state in-place.
        # load_state_dict modifies the objects passed to resume_from_checkpoint,
        # so the trainer's references are updated automatically.
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        scaler.load_state_dict(training_state["scaler"])
        global_step = training_state["global_step"]
        current_epoch = training_state["current_epoch"]
        
        # Backward compatibility: older checkpoints may not have best_eval_loss.
        # Default to inf (meaning "no evaluation yet") for smooth migration.
        best_eval_loss = training_state.get("best_eval_loss", float("inf"))

        # Early stopping state restoration: critical for correct patience tracking.
        # Without this, a restarted run gets a fresh patience budget and may train
        # far longer than the original run would have.
        if early_stopping is not None and "early_stopping" in training_state:
            es_state = training_state["early_stopping"]
            early_stopping.best_loss = es_state["best_loss"]
            early_stopping.fail_counter = es_state["fail_counter"]

        print(
            f"Training state restored: step={global_step}, epoch={current_epoch}"
        )
    else:
        # No training state found: start from scratch.
        # This is common when loading a pretrained model for fine-tuning
        # rather than resuming an interrupted run.
        print("No training_state.pt found -- starting optimizer from scratch.")
        global_step = 0
        current_epoch = 0
        best_eval_loss = float("inf")

    return global_step, current_epoch, best_eval_loss


def prune_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    """
    Remove the oldest periodic checkpoints when exceeding ``save_total_limit``.

    Pruning Strategy
    ----------------
    Only numbered (periodic) checkpoints are pruned — tagged checkpoints
    ("best", "final", "epoch-N", "early-stopped") are always preserved.
    This ensures critical milestones survive automatic cleanup.

    Why not prune tagged checkpoints?
    - "best": Represents the optimal model by validation loss — the one you
      likely want to deploy or evaluate.
    - "final": The model state at training completion — useful for comparison
      and as a guaranteed recovery point.
    - "epoch-N": Epoch-boundary saves provide training history for analysis.
    - "early-stopped": The model state when training terminated — important
      for understanding why training stopped.
    - "interrupted": Saves from KeyboardInterrupt — preserves progress from
      cancelled runs.

    Sorting Logic
    -------------
    Checkpoints are sorted by modification time (most recent first) so that
    the oldest checkpoints are removed first. This is preferable to sorting
    by step number because:
    1. Modification time reflects actual save order, which is reliable even
       if step numbers are non-monotonic (shouldn't happen, but defensive).
    2. If checkpoints are manually copied or modified, step-based sorting
       could be misleading.

    The _is_prunable helper identifies numbered checkpoints by checking if
    the suffix after "checkpoint-" consists entirely of digits. This pattern
    matching is intentionally strict to avoid accidentally pruning tagged
    checkpoints that might coincidentally start with digits.

    Parameters
    ----------
    output_dir : str
        Directory containing checkpoint subdirectories.
    save_total_limit : Optional[int]
        Maximum number of periodic (numbered) checkpoints to retain.
        None means unlimited (no pruning). Setting to 0 would prune all
        periodic checkpoints (unusual but technically valid).
    """
    if save_total_limit is None:
        return

    # Collect all checkpoint directories and sort by modification time,
    # most recent first. This preserves the newest checkpoints.
    checkpoint_dirs = sorted(
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ],
        key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
        reverse=True,
    )

    # Only numbered checkpoints (e.g., "checkpoint-1000") are prunable.
    # Tagged checkpoints ("checkpoint-best", "checkpoint-final") are preserved.
    # isdigit() check on the suffix ensures we only match pure numeric names.
    def _is_prunable(name: str) -> bool:
        suffix = name.replace("checkpoint-", "", 1)
        return suffix.isdigit()

    prunable = [d for d in checkpoint_dirs if _is_prunable(d)]

    # Remove oldest prunable checkpoints until we're at or below the limit.
    # pop() removes from the end (oldest, since list is sorted newest-first),
    # so the most recent periodic checkpoints are retained.
    while len(prunable) > save_total_limit:
        oldest = prunable.pop()
        oldest_path = os.path.join(output_dir, oldest)
        print(f"Old checkpoint pruned: {oldest_path}")
        shutil.rmtree(oldest_path)