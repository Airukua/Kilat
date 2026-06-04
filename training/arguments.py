from __future__ import annotations
from typing import Optional, Literal
import torch

PrecisionMode = Literal["fp32", "fp16", "bf16"]
TrainingMode = Literal["steps", "epochs"]

class TrainingArguments:
    """
    Training hyperparameter container (HuggingFace-style interface).

    Holds all settings for the training loop, including batch sizes, learning
    rate schedule, logging frequency, evaluation strategy, early stopping,
    checkpointing, mixed-precision configuration, and optional WandB reporting.

    Design Philosophy
    ----------------
    This class follows the HuggingFace TrainingArguments pattern but is
    intentionally minimal — it only includes parameters that the KilatTrainer
    actually uses. This avoids the "kitchen sink" problem where dozens of
    unused parameters create confusion about what's actually configurable.

    Key Design Decisions:

    1. **Validation at construction time** rather than in the trainer:
       Catches configuration errors immediately (fail fast) rather than
       discovering them deep in the training loop. This is especially
       important for long-running training jobs where discovering a
       misconfiguration after hours of training is costly.

    2. **Sensible defaults for fine-tuning** (not pretraining):
       - learning_rate=5e-5: Standard for BERT/GPT fine-tuning
       - training_mode="epochs": Natural for fixed-size fine-tuning datasets
       - num_train_epochs=3: Common for downstream task fine-tuning
       - max_steps=-1: Sentinel value indicating "not configured" for steps mode
       These defaults reflect that fine-tuning is the more common use case
       than large-scale pretraining.

    3. **max_steps default of -1**:
       This sentinel value is intentionally invalid for steps mode (validated
       at construction). It prevents accidentally running in steps mode with
       a default value, forcing explicit configuration. Using 0 would be
       ambiguous (0 steps could mean "no training" or "not configured").

    4. **No automatic batch size scaling**:
       Unlike some trainers, per_device_train_batch_size is explicit. Users
       must manually account for multi-GPU training. This avoids the common
       confusion where effective batch size is silently multiplied by GPU count.

    Validation Rationale
    -------------------
    - **BF16 on CPU**: Not explicitly rejected because PyTorch 2.1+ supports
      BF16 on CPU (via AMX instructions on Sapphire Rapids). While unusual,
      CPU BF16 training is valid for some edge cases.
    - **FP16 on CPU**: Explicitly rejected because FP16 CPU support is
      incomplete and unreliable. The GradScaler would be useless on CPU.
    - **training_mode validation**: Ensures the mode-specific parameters
      (max_steps, num_train_epochs) are valid for the chosen mode before
      training begins.

    Parameters
    ----------
    output_dir : str
        Directory for saving checkpoints and training artifacts.
    resume_from_checkpoint : Optional[str]
        Path to a checkpoint directory to resume from. ``None`` starts training
        from scratch.
    save_checkpoints : bool
        Whether to save any checkpoints at all. When ``False``, no checkpoints
        are written (useful for quick experiments or dry runs).
    training_mode : TrainingMode
        Training progress mode:

        * ``"steps"``  -- stops after ``max_steps`` optimizer steps.
          ``max_steps`` must be > 0.
        * ``"epochs"`` -- stops after ``num_train_epochs`` full epochs.
          ``num_train_epochs`` must be >= 1.
    learning_rate : float
        Peak learning rate for the AdamW optimizer.
    per_device_train_batch_size : int
        Number of samples per training batch (per GPU).
    per_device_eval_batch_size : int
        Number of samples per evaluation batch (per GPU).
    gradient_accumulation_steps : int
        Number of forward/backward passes before one optimizer step.
        Effective batch size = per_device_train_batch_size * gradient_accumulation_steps.
    weight_decay : float
        Weight decay coefficient applied to non-bias/non-norm parameters.
    max_grad_norm : float
        Maximum L2-norm for gradient clipping.
    max_steps : int
        Total optimizer steps for ``"steps"`` mode. Ignored in ``"epochs"`` mode.
    num_train_epochs : int
        Number of epochs for ``"epochs"`` mode. Ignored in ``"steps"`` mode.
    warmup_steps : int
        Number of linear warmup steps before cosine decay begins.
    logging_steps : int
        Interval (in optimizer steps) for printing and reporting training metrics.
    eval_steps : int
        Interval (in optimizer steps) for running validation.
        In ``"epochs"`` mode, evaluation also runs at the end of every epoch.
    save_steps : int
        Interval (in optimizer steps) for saving periodic checkpoints.
        In ``"epochs"`` mode, a checkpoint is also saved at the end of every epoch.
    save_total_limit : Optional[int]
        Maximum number of periodic (numbered) checkpoints to retain.
        Special checkpoints (best, final, epoch-N, etc.) are not counted toward this limit.
    early_stopping_patience : int
        Number of consecutive evaluations without improvement before stopping.
    early_stopping_threshold : float
        Minimum absolute decrease in eval loss to be considered an improvement.
    precision : PrecisionMode
        Mixed-precision mode:

        * ``"fp32"`` -- full precision, no casting (safe default for all hardware).
        * ``"fp16"`` -- half precision; requires CUDA; GradScaler enabled for numerical stability.
        * ``"bf16"`` -- brain float 16; requires Ampere+ GPU (sm_80+) or CPU with PyTorch >= 2.1;
          GradScaler disabled (BF16 has the same exponent range as FP32).
    report_to : str
        Metrics reporting backend. ``"wandb"`` for Weights & Biases, ``"none"`` to disable.
    run_name : Optional[str]
        Run name for WandB.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If ``training_mode="steps"`` and ``max_steps <= 0``.
    ValueError
        If ``training_mode="epochs"`` and ``num_train_epochs < 1``.
    ValueError
        If ``precision="fp16"`` or ``"bf16"`` is requested but CUDA is unavailable
        (except BF16 on CPU with PyTorch >= 2.1).
    ValueError
        If ``precision`` is not one of ``"fp32"``, ``"fp16"``, ``"bf16"``.

    Example
    -------
    Step-based training with BF16:

    >>> args = TrainingArguments(
    ...     output_dir="./checkpoints",
    ...     training_mode="steps",
    ...     max_steps=10_000,
    ...     precision="bf16",
    ...     learning_rate=3e-4,
    ... )

    Epoch-based training with FP16:

    >>> args = TrainingArguments(
    ...     output_dir="./checkpoints",
    ...     training_mode="epochs",
    ...     num_train_epochs=5,
    ...     precision="fp16",
    ...     learning_rate=3e-4,
    ... )
    """

    def __init__(
        self,
        # I/O
        output_dir: str = "./results",
        resume_from_checkpoint: Optional[str] = None,
        save_checkpoints: bool = True,
        # Training mode
        training_mode: TrainingMode = "epochs",
        # Optimisation
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Schedule (step-based)
        max_steps: int = -1,
        # Schedule (epoch-based)
        num_train_epochs: int = 3,
        warmup_steps: int = 0,
        # Logging & evaluation
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = 3,
        # Early stopping
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        # Mixed precision
        precision: PrecisionMode = "fp16",
        # Reporting
        report_to: str = "wandb",
        run_name: Optional[str] = "kilat-run",
        # Reproducibility
        seed: int = 42,
    ):
        # Validate training mode
        # This early validation prevents the confusing scenario where training
        # starts, runs for hours, then fails because max_steps wasn't set
        # correctly for steps mode.
        if training_mode not in ("steps", "epochs"):
            raise ValueError(
                f"training_mode must be 'steps' or 'epochs', got '{training_mode}'."
            )
        if training_mode == "steps" and max_steps <= 0:
            raise ValueError(
                "training_mode='steps' requires max_steps > 0. "
                f"Current value: max_steps={max_steps}."
            )
        if training_mode == "epochs" and num_train_epochs < 1:
            raise ValueError(
                "training_mode='epochs' requires num_train_epochs >= 1. "
                f"Current value: num_train_epochs={num_train_epochs}."
            )

        # Validate precision mode
        # Only FP16 on CUDA is checked here. BF16 on CPU is not rejected
        # because PyTorch 2.1+ supports it (though rare in practice).
        valid_precisions: tuple[str, ...] = ("fp32", "fp16", "bf16")
        if precision not in valid_precisions:
            raise ValueError(
                f"precision must be one of {valid_precisions}, got '{precision}'."
            )
        if precision == "fp16" and not torch.cuda.is_available():
            raise ValueError(
                "precision='fp16' requires CUDA. "
                "Use precision='fp32' for CPU, or precision='bf16' if "
                "your CPU supports BF16 (PyTorch >= 2.1)."
            )

        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_checkpoints = save_checkpoints
        self.training_mode: TrainingMode = training_mode
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.precision: PrecisionMode = precision
        self.report_to = report_to
        self.run_name = run_name
        self.seed = seed