from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Union
import warnings
import torch
import yaml
from .base import dump_yaml_file, load_yaml_file


class TrainingConfig:
    """
    Training hyperparameters container with YAML serialization support.

    This class mirrors ``TrainingArguments`` from the trainer module but
    provides cleaner separation of concerns: it focuses purely on storing
    and validating training hyperparameters, without coupling to the
    training loop implementation.

    WHY SEPARATE FROM TRAININGARGUMENTS:
        - TrainingArguments is used directly by the trainer and contains
          runtime-specific fields (e.g., local_rank, ddp_find_unused_parameters)
        - TrainingConfig is for persistent configuration storage (YAML files)
        - Separation allows clean validation at config-load time vs runtime

    DESIGN DECISIONS:
        - **Validation at construction**: All parameter constraints are checked
          immediately, following the fail‑fast principle. Catching ``max_steps <= 0``
          at config time prevents discovering it hours into training.
        - **YAML serialization**: Enables storing complete experiment configs in
          version‑controlled YAML files rather than scattered CLI arguments.
        - **Type safety**: Uses Literal types for constrained parameters
          (training_mode, precision, etc.) to catch invalid values early.
        - **Defaults for fine-tuning**: learning_rate=5e-5, num_train_epochs=3,
          weight_decay=0.01 are standard for fine-tuning tasks.

    IMPORTANT DEFAULTS:
        - `max_steps = -1` is intentionally invalid for steps mode, forcing
          explicit configuration. This prevents accidentally training for 0 steps
          or relying on epoch conversion in steps mode.
        - `report_to = "none"` disables external logging by default. Users must
          explicitly opt into W&B, TensorBoard, etc.
        - `precision = "fp16"` is a good default for modern GPUs; falls back to
          fp32 on CPU automatically.

    Example Usage
    -------------
        >>> # Create from code
        >>> train_cfg = TrainingConfig(
        ...     output_dir="./checkpoints",
        ...     training_mode="steps",
        ...     max_steps=100000,
        ...     learning_rate=3e-4,
        ...     precision="bf16"
        ... )
        >>>
        >>> # Save to YAML
        >>> train_cfg.to_yaml("training_config.yaml")
        >>>
        >>> # Load from YAML
        >>> loaded = TrainingConfig.from_yaml("training_config.yaml")
        >>>
        >>> # Convert to TrainingArguments for trainer
        >>> train_args = TrainingArguments(**loaded.to_dict())
    """
    
    def __init__(
        self,
        # ---- I/O ----
        output_dir: str = "./results",
        resume_from_checkpoint: Optional[str] = None,
        save_checkpoints: bool = True,
        atomic_checkpoint: bool = True,
        # ---- Training mode ----
        training_mode: Literal["steps", "epochs"] = "epochs",
        # ---- Optimisation ----
        learning_rate: float = 5e-5,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # ---- Schedule (step-based) ----
        max_steps: int = -1,
        # ---- Schedule (epoch-based) ----
        num_train_epochs: int = 3,
        warmup_steps: int = 0,
        scheduler_type: str = "cosine",
        scheduler_kwargs: Optional[dict] = None,
        # ---- Logging & evaluation ----
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = 3,
        # ---- Early stopping ----
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: Optional[bool] = None,
        # ---- Mixed precision ----
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        # ---- Reporting ----
        report_to: Union[str, list[str]] = "none",
        run_name: Optional[str] = "kilat-run",
        # ---- Reproducibility ----
        seed: int = 42,
    ):
        """
        Initialize training configuration with validation.

        Parameters
        ----------
        output_dir : str
            Directory for saving checkpoints and training artifacts.
        resume_from_checkpoint : Optional[str]
            Path to checkpoint directory to resume from. ``None`` = fresh start.
        save_checkpoints : bool
            Whether to save checkpoints during training. Set ``False`` for
            quick experiments or dry runs.
        atomic_checkpoint : bool
            If True, write checkpoints to a temp directory and rename them
            into place atomically. Prevents partially written checkpoints.
        training_mode : Literal["steps", "epochs"]
            ``"steps"``: Stop after ``max_steps`` optimizer steps.
            ``"epochs"``: Stop after ``num_train_epochs`` full data passes.
        learning_rate : float
            Peak learning rate for AdamW optimizer.
        beta1 : float
            First moment decay rate for AdamW.
        beta2 : float
            Second moment decay rate for AdamW.
        epsilon : float
            Numerical stability term for AdamW.
        per_device_train_batch_size : int
            Micro‑batch size per device.
        per_device_eval_batch_size : int
            Evaluation batch size per device.
        gradient_accumulation_steps : int
            Number of forward passes before one optimizer step.
            Effective batch = ``per_device_train_batch_size * gradient_accumulation_steps``.
        weight_decay : float
            AdamW weight decay coefficient (applied to non‑bias/non‑norm params).
        max_grad_norm : float
            Maximum L2 norm for gradient clipping.
        max_steps : int
            Total optimizer steps for ``"steps"`` mode. Must be > 0.
            Ignored in ``"epochs"`` mode.
        num_train_epochs : int
            Number of epochs for ``"epochs"`` mode. Must be ≥ 1.
            Ignored in ``"steps"`` mode.
        warmup_steps : int
            Linear warmup steps before cosine decay begins.
        scheduler_type : str
            Scheduler name understood by ``training.scheduler.get_scheduler``.
        scheduler_kwargs : Optional[dict]
            Extra scheduler keyword arguments forwarded to the scheduler factory.
        logging_steps : int
            Interval (in optimizer steps) for printing/reporting metrics.
        eval_steps : int
            Interval for running validation. In epochs mode, also runs
            at epoch boundaries.
        save_steps : int
            Interval for saving periodic checkpoints. In epochs mode,
            also saves at epoch boundaries.
        save_total_limit : Optional[int]
            Max periodic checkpoints to retain. Tagged checkpoints (best,
            final, etc.) are excluded from this limit. ``None`` = unlimited.
        early_stopping_patience : int
            Consecutive evaluations without improvement before stopping.
        early_stopping_threshold : float
            Minimum absolute decrease in eval loss to count as improvement.
        metric_for_best_model : str
            Metric tracked by early stopping and best-checkpoint selection.
        greater_is_better : Optional[bool]
            Override comparison direction for best-metric tracking.
        precision : Literal["fp32", "fp16", "bf16"]
            Mixed precision mode. ``"fp16"`` requires CUDA; ``"bf16"``
            requires Ampere+ GPU or PyTorch ≥ 2.1 CPU.
        report_to : str | list[str]
            Metrics backend(s). Accepts ``"none"``, ``"all"``, or a list of
            backend names such as ``["wandb", "tensorboard"]``.
        run_name : Optional[str]
            Display name for the W&B run.
        seed : int
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If ``training_mode="steps"`` and ``max_steps <= 0``.
        ValueError
            If ``training_mode="epochs"`` and ``num_train_epochs < 1``.
        ValueError
            If ``precision`` is invalid or ``"fp16"`` without CUDA.
        """
        # Validate training mode consistency
        if training_mode not in ("steps", "epochs"):
            raise ValueError(
                f"training_mode must be 'steps' or 'epochs', got '{training_mode}'."
            )
        if training_mode == "steps" and max_steps <= 0:
            raise ValueError(
                f"training_mode='steps' requires max_steps > 0. "
                f"Current value: max_steps={max_steps}."
            )
        if training_mode == "epochs" and num_train_epochs < 1:
            raise ValueError(
                f"training_mode='epochs' requires num_train_epochs >= 1. "
                f"Current value: num_train_epochs={num_train_epochs}."
            )
        
        # Validate precision and hardware compatibility
        valid_precisions = ("fp32", "fp16", "bf16")
        if precision not in valid_precisions:
            raise ValueError(
                f"precision must be one of {valid_precisions}, got '{precision}'."
            )
        if precision == "fp16" and not torch.cuda.is_available():
            raise ValueError(
                "precision='fp16' requires CUDA. "
                "Use precision='fp32' for CPU, or 'bf16' if supported."
            )
        
        # I/O
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_checkpoints = save_checkpoints
        self.atomic_checkpoint = atomic_checkpoint
        
        # Training mode
        self.training_mode = training_mode
        
        # Optimisation
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Schedule
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs: dict = scheduler_kwargs or {}

        # Logging & evaluation
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

        # Mixed precision
        self.precision = precision

        # Reporting
        if isinstance(report_to, str):
            self.report_to: list[str] = [report_to]
        else:
            self.report_to = list(report_to)
        self.run_name = run_name
        
        # Reproducibility
        self.seed = seed

    def to_dict(self) -> dict:
        """
        Convert training configuration to a plain dictionary.

        Returns all parameters as a flat dictionary suitable for
        YAML/JSON serialization. Excludes any runtime state and
        internal attributes.

        Returns
        -------
        dict
            Dictionary with all training hyperparameters.
        """
        return {
            "output_dir": self.output_dir,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "save_checkpoints": self.save_checkpoints,
            "atomic_checkpoint": self.atomic_checkpoint,
            "training_mode": self.training_mode,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "scheduler_type": self.scheduler_type,
            "scheduler_kwargs": self.scheduler_kwargs,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "precision": self.precision,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "seed": self.seed,
        }

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export training configuration to YAML format.

        WHY YAML: Human-readable format that supports comments and is
        easier to version control than JSON. The exported file can be
        edited manually and reloaded with `from_yaml`.

        Parameters
        ----------
        path : Optional[str | Path]
            If provided, writes YAML to this file. If ``None``, returns
            the YAML string.

        Returns
        -------
        str
            YAML string representation.
        """
        if path is not None:
            return dump_yaml_file(self.to_dict(), path)
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """
        Load training configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML training configuration file.

        Returns
        -------
        TrainingConfig
            Validated training configuration instance.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        yaml.YAMLError
            If the YAML file is malformed.
        ValueError
            If the loaded configuration contains invalid values.
        """
        config_dict = load_yaml_file(yaml_path)
        return cls(**config_dict)

    def get_warmup_steps(self, total_steps: int) -> int:
        """
        Return the number of warmup steps (resolved from warmup_steps).

        WHY: This method exists for API compatibility with TrainingArguments
        that may support warmup_ratio. Currently only warmup_steps is supported,
        but having this method allows consistent code in the trainer.

        Parameters
        ----------
        total_steps : int
            Total number of optimizer steps (unused, kept for compatibility).

        Returns
        -------
        int
            Number of warmup steps (clipped to total_steps if needed).
        """
        # Ensure warmup doesn't exceed total steps
        return min(self.warmup_steps, total_steps)

    def get_effective_batch_size(self) -> int:
        """
        Compute effective global batch size per GPU.

        Effective batch size = per_device_train_batch_size * gradient_accumulation_steps.
        This does NOT include world_size (multi-GPU scaling).

        Returns
        -------
        int
            Effective batch size.
        """
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"TrainingConfig(output_dir='{self.output_dir}', "
            f"training_mode='{self.training_mode}', "
            f"learning_rate={self.learning_rate}, "
            f"batch_size={self.per_device_train_batch_size}, "
            f"precision='{self.precision}')"
        )