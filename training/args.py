from __future__ import annotations
from typing import Optional, Literal, Union
import torch
PrecisionMode = Literal["fp32", "fp16", "bf16"]
TrainingMode = Literal["steps", "epochs"]

class TrainingArguments:
    """
    Training hyperparameter container for KilatTrainer.

    Design Philosophy
    -----------------
    Minimal container: only parameters the trainer actually consumes.
    Validated at construction (fail-fast) so errors surface before a long
    training run starts.

    Integration contracts
    ---------------------
    - optimizer_utils.create_optimizer       : learning_rate, weight_decay,
                                               beta1, beta2, epsilon
    - schedulers.get_scheduler               : scheduler_type, warmup_steps
                                               (total_steps computed at runtime)
    - training_utils.compute_total_steps     : training_mode, max_steps,
                                               num_train_epochs
    - training_utils.save_checkpoint         : output_dir, save_total_limit,
                                               atomic_checkpoint, save_checkpoints
    - training_utils.should_log/evaluate/save: logging_steps, eval_steps, save_steps
    - callbacks.EarlyStoppingCallback        : early_stopping_patience,
                                               early_stopping_threshold,
                                               metric_for_best_model,
                                               greater_is_better
    - integrations.get_reporting_integration_callbacks:
                                               report_to (str | list[str] | "all" | "none")
    - TrainingArguments.precision            : resolve_amp_dtype(precision)

    Parameters
    ----------
    output_dir : str
        Directory for checkpoints and artifacts.
    resume_from_checkpoint : Optional[str]
        Path to checkpoint directory to resume from. None = fresh start.
    save_checkpoints : bool
        Master switch for checkpoint writing. False = no files written
        (useful for dry-runs or evaluation-only loops).
    atomic_checkpoint : bool
        If True, write to a temp dir then rename (atomic on POSIX).
        Prevents corrupted checkpoints on crash. Default True.
    training_mode : TrainingMode
        "steps" → stop after max_steps optimizer steps.
        "epochs" → stop after num_train_epochs full passes.
    learning_rate : float
        Peak LR for AdamW.
    beta1, beta2, epsilon : float
        AdamW moment decay rates and numerical stability term.
        Passed directly to create_optimizer().
    per_device_train_batch_size : int
        Samples per training batch (per GPU).
    per_device_eval_batch_size : int
        Samples per evaluation batch (per GPU).
    gradient_accumulation_steps : int
        Forward/backward passes before one optimizer step.
        Effective batch = per_device_train_batch_size × gradient_accumulation_steps.
    weight_decay : float
        AdamW weight decay for non-bias/non-norm parameters.
    max_grad_norm : float
        L2-norm ceiling for gradient clipping.
    max_steps : int
        Optimizer-step budget for "steps" mode. Ignored otherwise.
        Sentinel -1 → not configured (raises if training_mode="steps").
    num_train_epochs : int
        Epoch budget for "epochs" mode. Ignored otherwise.
    warmup_steps : int
        Linear warmup steps passed to get_scheduler().
    scheduler_type : str
        Scheduler name understood by schedulers.get_scheduler() and
        the SchedulerType enum. Default "cosine".
        Valid built-ins: "linear", "cosine", "cosine_with_restarts",
        "cosine_with_min_lr", "polynomial", "constant",
        "constant_with_warmup", "inverse_sqrt", "wsdlr", "rex".
    scheduler_kwargs : dict
        Extra kwargs forwarded to the scheduler constructor
        (e.g. {"num_cycles": 4} for cosine_with_restarts,
              {"min_lr_ratio": 0.05} for cosine_with_min_lr,
              {"num_decay_steps": 1000} for wsdlr).
    logging_steps : int
        Optimizer-step interval for logging. ≤ 0 disables logging.
    eval_steps : int
        Optimizer-step interval for evaluation. ≤ 0 disables step-eval.
        Epoch-end evaluation is always performed in "epochs" mode.
    save_steps : int
        Optimizer-step interval for periodic checkpoints. ≤ 0 disables.
    save_total_limit : Optional[int]
        Max numbered checkpoints to keep. None = keep all.
        Tagged checkpoints ("best", "final", "epoch-N") are always kept.
        Passed to training_utils.prune_checkpoints().
    early_stopping_patience : int
        Consecutive non-improving evaluations before stopping.
        Passed to EarlyStoppingCallback.
    early_stopping_threshold : float
        Minimum improvement to reset patience counter.
        Passed to EarlyStoppingCallback.
    metric_for_best_model : str
        Metric key monitored by EarlyStoppingCallback and used to track
        best_model_checkpoint in TrainerState.
        Default "eval_loss" (lower is better unless overridden by
        greater_is_better).
    greater_is_better : Optional[bool]
        Override improvement direction for early stopping.
        None → auto-detect: "loss" in name → False, else True.
    precision : PrecisionMode
        "fp32" → no AMP; "fp16" → torch.float16 + GradScaler;
        "bf16" → torch.bfloat16, no GradScaler.
        resolve_amp_dtype(precision) maps this to Optional[torch.dtype].
    report_to : str | list[str]
        Logging backend(s) for integrations.get_reporting_integration_callbacks().
        Accepts "none", "all", "wandb", "tensorboard", "mlflow", "comet_ml",
        or a list of those strings.
    run_name : Optional[str]
        Run display name for W&B / MLflow / Comet.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        training_mode="steps" and max_steps <= 0.
        training_mode="epochs" and num_train_epochs < 1.
        precision not in ("fp32", "fp16", "bf16").
        precision="fp16" on a machine without CUDA.
    """

    def __init__(
        self,
        # I/O
        output_dir: str = "./results",
        resume_from_checkpoint: Optional[str] = None,
        save_checkpoints: bool = True,
        atomic_checkpoint: bool = True,
        # Training mode
        training_mode: TrainingMode = "epochs",
        # Optimisation
        learning_rate: float = 5e-5,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Schedule
        max_steps: int = -1,
        num_train_epochs: int = 3,
        warmup_steps: int = 0,
        scheduler_type: str = "cosine",
        scheduler_kwargs: Optional[dict] = None,
        # Logging & evaluation
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = 3,
        # Early stopping
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: Optional[bool] = None,
        # Mixed precision
        precision: PrecisionMode = "fp32",
        # Reporting — accepts str or list[str] for integrations module
        report_to: Union[str, list[str]] = "none",
        run_name: Optional[str] = None,
        # Reproducibility
        seed: int = 42,
    ) -> None:

        # ── training mode ────────────────────────────────────────────────
        if training_mode not in ("steps", "epochs"):
            raise ValueError(
                f"training_mode must be 'steps' or 'epochs', got '{training_mode}'."
            )
        if training_mode == "steps" and max_steps <= 0:
            raise ValueError(
                "training_mode='steps' requires max_steps > 0. "
                f"Got max_steps={max_steps}."
            )
        if training_mode == "epochs" and num_train_epochs < 1:
            raise ValueError(
                "training_mode='epochs' requires num_train_epochs >= 1. "
                f"Got num_train_epochs={num_train_epochs}."
            )

        # ── precision ────────────────────────────────────────────────────
        valid_precisions: tuple[str, ...] = ("fp32", "fp16", "bf16")
        if precision not in valid_precisions:
            raise ValueError(
                f"precision must be one of {valid_precisions}, got '{precision}'."
            )
        if precision == "fp16" and not torch.cuda.is_available():
            raise ValueError(
                "precision='fp16' requires CUDA. "
                "Use 'fp32' for CPU, or 'bf16' if your CPU supports BF16 "
                "(PyTorch >= 2.1 on AMX-capable hardware)."
            )

        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_checkpoints = save_checkpoints
        self.atomic_checkpoint = atomic_checkpoint

        self.training_mode: TrainingMode = training_mode
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs: dict = scheduler_kwargs or {}

        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

        self.precision: PrecisionMode = precision

        # Normalise to list so integrations module always gets the same type
        if isinstance(report_to, str):
            self.report_to: list[str] = [report_to]
        else:
            self.report_to = list(report_to)

        self.run_name = run_name
        self.seed = seed


    @property
    def use_amp(self) -> bool:
        """True if any AMP dtype is active (fp16 or bf16)."""
        return self.precision in ("fp16", "bf16")

    @property
    def use_grad_scaler(self) -> bool:
        """
        True iff GradScaler should be enabled.

        GradScaler is only needed for fp16; bf16 has the same exponent range
        as fp32 so overflow is not a concern.
        """
        return self.precision == "fp16"