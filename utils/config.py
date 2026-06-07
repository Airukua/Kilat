from __future__ import annotations
from typing import Optional, Literal
from transformers import PretrainedConfig
import yaml
from pathlib import Path
import torch

class KilatConfig(PretrainedConfig):
    """
    Configuration class for KilatTransformer model architecture.

    Extends HuggingFace's ``PretrainedConfig`` to define all hyperparameters
    needed to construct a complete KilatTransformer model. Supports both
    dense and Mixture‑of‑Experts (MoE) variants, including DeepSeek‑V2
    style architectures with shared experts.

    Design Philosophy
    ----------------
    This config class serves three purposes:

    1. **Model construction**: All parameters needed by ``KilatTransformer.__init__``
       are present, with sensible defaults for each model scale.
    
    2. **Serialization**: Compatible with HF's ``save_pretrained``/``from_pretrained``
       workflow. Additionally supports YAML export for human‑readable configs.
    
    3. **Validation**: Catches architecture inconsistencies (e.g., n_embd not
       divisible by n_head, shared experts without MoE mode) at construction
       time rather than deep in the model forward pass.

    Key Configuration Parameters
    ----------------------------
    - **recall_ratio**: Controls the precision‑efficiency trade‑off. Higher values
      allocate more heads to the precise latent MLA pathway; lower values favor
      the efficient global decay pathway.
    - **latent_dim**: When None, defaults to n_embd // 4 internally, providing
      ~4x KV‑cache compression for the recall pathway.
    - **num_shared_experts**: Distinguishes standard MoE (0 shared experts) from
      DeepSeek‑V2 MoE (>0 shared experts). Shared experts process all tokens
      regardless of routing decisions.
    - **fine_grained_factor**: Implements DeepSeek‑V2's fine‑grained expert
      segmentation. A factor of N splits each "expert" into N smaller sub‑experts,
      enabling finer routing granularity.

    Example (dense model)::
        >>> config = KilatConfig(vocab_size=32000, n_embd=768, n_head=12, n_layer=12)
        >>> config.save_pretrained("./checkpoints/kilat-base")

    Example (DeepSeek‑V2 MoE)::
        >>> config = KilatConfig(
        ...     vocab_size=64000, n_embd=2048, n_head=32, n_layer=32,
        ...     ffn_mode="moe", num_experts=64, active_experts=8,
        ...     num_shared_experts=2
        ... )
    """

    model_type = "kilat_transformer"

    def __init__(
        self,
        # ---- Core architecture ----
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        # ---- Attention ----
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        # ---- Feed‑forward ----
        ffn_mode: Literal["dense", "moe"] = "moe",
        ff_mult: float = 8 / 3,
        ffn_dropout: float = 0.0,
        # ---- MoE ----
        num_experts: int = 8,
        active_experts: int = 2,
        num_shared_experts: int = 0,
        fine_grained_factor: int = 1,
        aux_loss_coef: float = 0.01,
        device_balance_coef: float = 0.001,
        # ---- Regularisation ----
        embd_drop: float = 0.0,
        resid_drop: float = 0.0,
        # ---- HF boilerplate ----
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """
        Initialize KilatTransformer configuration.

        Parameters
        ----------
        vocab_size : int
            Size of the token vocabulary. Must match the tokenizer's vocabulary.
        n_embd : int
            Hidden embedding dimension. Must be divisible by ``n_head``.
        n_layer : int
            Number of transformer blocks (model depth).
        n_head : int
            Number of attention heads per block. Must evenly divide ``n_embd``.
        recall_ratio : float
            Fraction of heads using precise latent MLA attention.
            Remaining heads use efficient global decay. Range: [0, 1].
        latent_dim : Optional[int]
            Bottleneck dimension for low‑rank Q/KV projections in MLA.
            ``None`` defaults to ``n_embd // 4``. Smaller = more compression.
        attn_drop : float
            Dropout probability inside attention softmax. Range: [0, 1).
        ffn_mode : Literal["dense", "moe"]
            ``"dense"`` for single SwiGLU FFN, ``"moe"`` for Mixture‑of‑Experts.
        ff_mult : float
            FFN hidden dimension multiplier. Default 8/3 ≈ 2.67 compensates
            for SwiGLU gating to achieve ~4x effective expansion.
        ffn_dropout : float
            Dropout probability inside FFN layers. Range: [0, 1).
        num_experts : int
            Total number of routed experts in MoE mode.
        active_experts : int
            Number of experts activated per token (Top‑K routing).
            Must be ≤ ``num_experts``.
        num_shared_experts : int
            Number of always‑active shared experts (DeepSeek‑V2 style).
            Set to 0 for standard MoE.
        fine_grained_factor : int
            Expert segmentation factor. 1 = full‑size experts, >1 = smaller
            sub‑experts for finer routing (DeepSeek‑V2 fine‑grained MoE).
        aux_loss_coef : float
            Weight of expert‑level load‑balancing loss. Typical values:
            0.01 for standard MoE, 0.001 for DeepSeek‑V2.
        device_balance_coef : float
            Weight of device‑level balance loss for multi‑GPU training.
            Set to 0 for single‑GPU. Only used when ``num_shared_experts > 0``.
        embd_drop : float
            Dropout probability after token embeddings. Range: [0, 1).
        resid_drop : float
            Dropout probability on FFN residual branch. Range: [0, 1).
        pad_token_id : int
            Token ID for padding shorter sequences.
        bos_token_id : int
            Beginning‑of‑sequence token ID.
        eos_token_id : int
            End‑of‑sequence token ID.
        tie_word_embeddings : bool
            Whether to share weights between input embeddings and output
            projection (saves ``vocab_size * n_embd`` parameters).
        use_cache : bool
            Enable KV‑cache for autoregressive generation. Currently
            experimental pending compressed MLA cache implementation.
        initializer_range : float
            Standard deviation for weight initialization N(0, range).
        **kwargs : dict
            Additional arguments passed to ``PretrainedConfig.__init__``.
        """
        # Hugging Face class validators may run before this initializer fully
        # returns, so seed the internal config flags they expect up front.
        self._output_attentions = False
        self._output_hidden_states = False
        self._use_cache = use_cache
        self._attn_implementation = None
        self._attn_implementation_internal = None
        self._experts_implementation_internal = None

        # Initialize HuggingFace parent class with standard token config
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        # Core architecture
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        # Attention configuration
        self.recall_ratio = recall_ratio
        self.latent_dim = latent_dim
        self.attn_drop = attn_drop

        # Feed‑forward configuration
        self.ffn_mode = ffn_mode
        self.ff_mult = ff_mult
        self.ffn_dropout = ffn_dropout

        # MoE configuration
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.num_shared_experts = num_shared_experts
        self.fine_grained_factor = fine_grained_factor
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef

        # Regularisation
        self.embd_drop = embd_drop
        self.resid_drop = resid_drop

        # Miscellaneous
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        # Run validation after all subclass attributes exist.
        self._validate()

    def __post_init__(self, **kwargs):
        """
        Validate configuration after construction and deserialization.

        HuggingFace calls this automatically after ``__init__`` and after
        loading from checkpoints, ensuring that both freshly created and
        loaded configurations are validated.
        """
        # Newer Hugging Face dataclass wrappers may forward token kwargs here.
        # Validation runs at the end of ``__init__`` instead, because this hook
        # can fire before subclass-specific attributes are assigned.
        return None

    def _validate(self):
        """
        Validate configuration consistency and catch common misconfigurations.

        Checks performed:
        - n_embd must be divisible by n_head (for head_dim calculation)
        - Shared experts require MoE mode
        - Active experts cannot exceed total experts in MoE mode
        - recall_ratio must be in [0, 1]
        - Dropout probabilities must be in [0, 1)

        These checks fail fast with clear error messages rather than letting
        cryptic dimension mismatch errors propagate through the model code.
        """
        # Architecture constraint: head dimension must be an integer
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}) "
                f"to compute integer head_dim = n_embd / n_head."
            )
        
        # Shared experts are only meaningful with MoE routing
        if self.num_shared_experts > 0 and self.ffn_mode != "moe":
            raise ValueError(
                f"num_shared_experts={self.num_shared_experts} requires "
                f"ffn_mode='moe', got ffn_mode='{self.ffn_mode}'."
            )
        
        # MoE routing constraints
        if self.ffn_mode == "moe":
            if self.active_experts > self.num_experts:
                raise ValueError(
                    f"active_experts ({self.active_experts}) must be ≤ "
                    f"num_experts ({self.num_experts})."
                )
            if self.num_experts < 1:
                raise ValueError(
                    f"num_experts must be ≥ 1 in MoE mode, got {self.num_experts}."
                )
        
        # Recall ratio bounds
        if not 0 <= self.recall_ratio <= 1:
            raise ValueError(
                f"recall_ratio must be in [0, 1], got {self.recall_ratio}."
            )
        
        # Dropout probability bounds (all use [0, 1) range)
        for name in ["attn_drop", "ffn_dropout", "embd_drop", "resid_drop"]:
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1), got {value}."
                )

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export model configuration to human‑readable YAML format.

        Unlike JSON (used by ``save_pretrained``), YAML supports comments
        and is more readable for manual editing. The exported YAML excludes
        HuggingFace internal metadata (``transformers_version``, ``model_type``)
        that users don't need to configure.

        Parameters
        ----------
        path : Optional[str | Path]
            If provided, writes YAML to this file. If ``None``, returns
            the YAML string for programmatic use.

        Returns
        -------
        str
            YAML string representation of the configuration.
        """
        config_dict = self.to_dict()
        
        # Strip HF internal keys that are not user‑configurable
        config_dict.pop("transformers_version", None)
        config_dict.pop("model_type", None)
        
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,  # Block style for readability
            sort_keys=False,           # Preserve logical parameter grouping
            allow_unicode=True,
            width=120,                 # Wide lines for parameter + comment
        )
        
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "KilatConfig":
        """
        Load model configuration from a YAML file.

        Enables the workflow:
        1. Edit ``config.yaml`` with desired hyperparameters
        2. Load: ``config = KilatConfig.from_yaml("config.yaml")``
        3. Construct model: ``model = KilatTransformer(config)``

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML configuration file.

        Returns
        -------
        KilatConfig
            Validated configuration instance.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str | Path, **kwargs):
        """
        Save configuration to directory in multiple formats.

        Extends HuggingFace's ``save_pretrained`` to write:
        - ``config.json``: Standard HF format (for ``from_pretrained``)
        - ``config.yaml``: Human‑readable format (for manual editing)

        Parameters
        ----------
        save_directory : str | Path
            Directory to save configuration files.
        **kwargs : dict
            Additional arguments passed to ``PretrainedConfig.save_pretrained``.
        """
        # Save standard HF JSON config
        super().save_pretrained(save_directory, **kwargs)
        
        # Additionally save human‑readable YAML version
        save_dir = Path(save_directory)
        self.to_yaml(save_dir / "config.yaml")


class TrainingConfig:
    """
    Training hyperparameters container with YAML serialization support.

    This class mirrors ``TrainingArguments`` from the trainer module but
    provides cleaner separation of concerns: it focuses purely on storing
    and validating training hyperparameters, without coupling to the
    training loop implementation.

    Design Decisions
    ----------------
    - **Validation at construction**: All parameter constraints are checked
      immediately, following the fail‑fast principle. Catching ``max_steps <= 0``
      at config time prevents discovering it hours into training.
    - **YAML serialization**: Enables storing complete experiment configs in
      version‑controlled YAML files rather than scattered CLI arguments.
    - **Mirrors TrainingArguments**: Parameter names and defaults match exactly,
      making it a drop‑in data source for creating ``TrainingArguments`` instances.

    The ``max_steps = -1`` default is intentionally invalid for steps mode,
    forcing explicit configuration. This prevents accidentally training for
    -1 steps (which would be 0 due to overflow) or relying on epoch conversion
    in steps mode.

    Example::
        >>> train_cfg = TrainingConfig(
        ...     output_dir="./checkpoints",
        ...     training_mode="steps",
        ...     max_steps=100000,
        ...     learning_rate=3e-4,
        ...     precision="bf16"
        ... )
        >>> train_cfg.to_yaml("training_config.yaml")
    """
    
    def __init__(
        self,
        # ---- I/O ----
        output_dir: str = "./results",
        resume_from_checkpoint: Optional[str] = None,
        save_checkpoints: bool = True,
        # ---- Training mode ----
        training_mode: Literal["steps", "epochs"] = "epochs",
        # ---- Optimisation ----
        learning_rate: float = 5e-5,
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
        # ---- Logging & evaluation ----
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = 3,
        # ---- Early stopping ----
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        # ---- Mixed precision ----
        precision: Literal["fp32", "fp16", "bf16"] = "fp16",
        # ---- Reporting ----
        report_to: str = "wandb",
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
        training_mode : Literal["steps", "epochs"]
            ``"steps"``: Stop after ``max_steps`` optimizer steps.
            ``"epochs"``: Stop after ``num_train_epochs`` full data passes.
        learning_rate : float
            Peak learning rate for AdamW optimizer.
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
        precision : Literal["fp32", "fp16", "bf16"]
            Mixed precision mode. ``"fp16"`` requires CUDA; ``"bf16"``
            requires Ampere+ GPU or PyTorch ≥ 2.1 CPU.
        report_to : str
            Metrics backend. ``"wandb"`` for Weights & Biases, ``"none"``
            to disable external logging.
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
        
        # Training mode
        self.training_mode = training_mode
        
        # Optimisation
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Schedule
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        
        # Logging & evaluation
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # Mixed precision
        self.precision = precision
        
        # Reporting
        self.report_to = report_to
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
            "training_mode": self.training_mode,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "precision": self.precision,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "seed": self.seed,
        }

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export training configuration to YAML format.

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
        yaml_str = yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        
        return yaml_str

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
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class MainConfig:
    """
    Complete experiment configuration combining model and training settings.

    Provides a unified configuration interface that bundles model architecture
    hyperparameters with training hyperparameters and experiment metadata.
    This enables storing the complete experiment specification in a single
    version‑controlled YAML file.

    Design Rationale
    ---------------
    Separating model and training configs is standard, but managing two files
    introduces coordination problems:
    - Which model config goes with which training config?
    - Did you remember to update both when changing an experiment?
    - How do you reproduce an experiment from a checkpoint directory?

    ``MainConfig`` solves this by bundling everything together while still
    allowing the individual components to be used independently when needed
    (e.g., loading just the model config for inference).

    YAML Structure
    -------------
    The exported YAML has three top‑level sections::

        experiment:
          name: "my-experiment"
          description: "Experiment description"
          tags: ["tag1", "tag2"]
        model:
          vocab_size: 32000
          n_embd: 768
          ...
        training:
          output_dir: "./checkpoints"
          training_mode: "steps"
          ...

    Example::
        >>> config = MainConfig.from_yaml("experiment_config.yaml")
        >>> model = KilatTransformer(config.model)
        >>> train_args = TrainingArguments(**config.training.to_dict())
        >>> config.save_pretrained("./checkpoints/my-model")
    """
    
    def __init__(
        self,
        model: KilatConfig,
        training: TrainingConfig,
        # ---- Experiment metadata ----
        experiment_name: str = "kilat-experiment",
        description: str = "",
        tags: list[str] = None,
    ):
        """
        Initialize complete experiment configuration.

        Parameters
        ----------
        model : KilatConfig
            Model architecture configuration.
        training : TrainingConfig
            Training hyperparameter configuration.
        experiment_name : str
            Human‑readable experiment name (used for W&B runs, logging).
        description : str
            Optional description of the experiment purpose or changes.
        tags : list[str]
            Tags for experiment categorization and filtering.
        """
        self.model = model
        self.training = training
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []

    def to_yaml(self, path: str | Path) -> str:
        """
        Export complete configuration to a single YAML file.

        The output file includes a header comment with experiment metadata
        for quick identification when viewing the raw file.

        Parameters
        ----------
        path : str | Path
            Path to write the YAML file.

        Returns
        -------
        str
            Complete YAML string.
        """
        config_dict = {
            "experiment": {
                "name": self.experiment_name,
                "description": self.description,
                "tags": self.tags,
            },
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }
        
        # Remove HuggingFace internal metadata from model dict
        # These are runtime artifacts, not user‑configurable parameters
        config_dict["model"].pop("transformers_version", None)
        config_dict["model"].pop("model_type", None)
        
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        
        # Write with header comment for file identification
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# KilatTransformer Full Configuration\n")
            f.write(f"# Experiment: {self.experiment_name}\n")
            if self.description:
                f.write(f"# Description: {self.description}\n")
            if self.tags:
                f.write(f"# Tags: {', '.join(self.tags)}\n")
            f.write(f"\n")
            f.write(yaml_str)
        
        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MainConfig":
        """
        Load complete experiment configuration from YAML.

        The YAML file must contain ``model`` and ``training`` sections.
        The ``experiment`` section is optional and defaults to empty values.

        Parameters
        ----------
        yaml_path : str | Path
            Path to the complete configuration YAML file.

        Returns
        -------
        MainConfig
            Fully validated configuration ready for training.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract experiment metadata with defaults for optional fields
        experiment = config_dict.get("experiment", {})
        
        # Construct model configuration (validates on construction)
        model_config = KilatConfig(**config_dict["model"])
        
        # Construct training configuration (validates on construction)
        training_config = TrainingConfig(**config_dict["training"])
        
        return cls(
            model=model_config,
            training=training_config,
            experiment_name=experiment.get("name", "kilat-experiment"),
            description=experiment.get("description", ""),
            tags=experiment.get("tags", []),
        )

    def save_pretrained(self, save_directory: str | Path):
        """
        Save all configurations to a checkpoint directory.

        Writes three files:
        - ``config.json``: Standard HF model config
        - ``config.yaml``: Human‑readable model config
        - ``training_config.yaml``: Training hyperparameters
        - ``full_config.yaml``: Complete combined configuration

        Parameters
        ----------
        save_directory : str | Path
            Directory to save all configuration files.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model config in both HF JSON and readable YAML formats
        self.model.save_pretrained(save_dir)
        
        # Save training config separately for clarity
        self.training.to_yaml(save_dir / "training_config.yaml")
        
        # Save combined config as the single source of truth
        self.to_yaml(save_dir / "full_config.yaml")
