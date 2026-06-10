from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Any
import yaml
from .base import dump_yaml_file, load_yaml_file
from .dataloader_config import DataLoaderConfig
from .model_config import KilatConfig
from .tokenizer_config import TokenizerConfig
from .training_config import TrainingConfig


class MainConfig:
    """
    Complete experiment configuration aggregating model, tokenizer, dataloader, and training settings.

    WHY THIS EXISTS:
        Maintaining separate configuration files for model architecture, tokenizer,
        data loading, and training hyperparameters leads to coordination problems:
        - Which model config pairs with which training config?
        - How do you exactly reproduce an experiment from a checkpoint directory?
        - Did you update all configs when changing an experiment?

        MainConfig solves this by bundling every experiment parameter into one
        version‑controlled YAML file. It still allows individual components to be
        used independently when needed (e.g., loading just the model config for inference).

    DELEGATION PROPERTIES:
        To reduce boilerplate, this class exposes properties that delegate to the
        underlying sub‑configs (model, dataloader, training). For example:
        - `main_config.pad_token_id` → `main_config.model.pad_token_id`
        - `main_config.train_batch_size` → `main_config.dataloader.train_batch_size`
        - `main_config.learning_rate` → `main_config.training.learning_rate`

        This allows you to pass a single `MainConfig` object to components like
        `KilatDataCollator`, `build_train_dataloader`, and `TrainingArguments`
        without having to write `.model`, `.dataloader`, or `.training` every time.

    YAML STRUCTURE:
        The exported YAML has five top‑level sections:

        experiment:
          name: "my-experiment"
          description: "Experiment description"
          tags: ["tag1", "tag2"]

        model:
          vocab_size: 32000
          n_embd: 768
          n_layer: 12
          n_head: 12
          ...

        tokenizer:
          tokenizer_type: "gpt2"
          tokenizer_name_or_path: "gpt2"
          use_fast: true

        dataloader:
          train_batch_size: 32
          eval_batch_size: 32
          num_workers: 8
          max_seq_length: 1024
          ...

        training:
          output_dir: "./checkpoints"
          training_mode: "steps"
          max_steps: 100000
          learning_rate: 3e-4
          ...

    DESIGN DECISIONS:
        - **Single source of truth** – All experiment parameters live in one YAML file.
        - **Human‑readable** – YAML with block formatting is easy to review and edit.
        - **Validation at construction** – Tokenizer vocabulary size is checked
          against model vocab_size at load time (fail fast, clear error).
        - **Round‑trip safe** – Config saved with `save_pretrained` can be loaded
          back with `from_yaml` without loss.
        - **Delegation properties** – Eliminate repetitive `.model.` prefixes while
          keeping the original sub‑configs accessible for advanced use.

    ASSUMPTIONS:
        - The tokenizer configuration must be consistent with the model's vocab_size.
        - The dataloader configuration must be compatible with the dataset format.
        - All sections (model, tokenizer, dataloader, training) must be present
          in the YAML when using the nested format.

    EDGE CASES:
        - If tokenizer vocab size does not match model vocab_size, a warning is
          emitted but loading continues (graceful degradation).
        - If any required section is missing, a clear `ValueError` is raised with
          actionable message.
        - Empty tags list serialises to `[]` (empty YAML list).

    PERFORMANCE:
        - `from_yaml` reads the file once and constructs sub‑configs; overhead is
          negligible for typical config sizes (few KB). Property access is O(1).

    Example Usage
    -------------
        >>> # Load complete experiment config
        >>> config = MainConfig.from_yaml("experiment_config.yaml")
        >>>
        >>> # Pass directly to components (delegation properties work)
        >>> collator = KilatDataCollator(
        ...     pad_token_id=config.pad_token_id,
        ...     max_length=config.max_seq_length,
        ... )
        >>>
        >>> # Create model (explicit access to model config)
        >>> model = KilatTransformer(config.model)
        >>>
        >>> # Training arguments (explicit or via delegation)
        >>> train_args = TrainingArguments(**config.training.to_dict())
        >>>
        >>> # Save all configs to checkpoint directory
        >>> config.save_pretrained("./checkpoints/my-model")
    """

    def __init__(
        self,
        model: KilatConfig,
        tokenizer: TokenizerConfig,
        training: TrainingConfig,
        dataloader: DataLoaderConfig,
        experiment_name: str = "kilat-experiment",
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize complete experiment configuration.

        Parameters
        ----------
        model : KilatConfig
            Model architecture configuration (vocab_size, n_embd, n_layer, etc.).
        tokenizer : TokenizerConfig
            Tokenizer metadata for reproduction and decode‑time inspection.
        training : TrainingConfig
            Training hyperparameters (learning rate, batch size, scheduler, etc.).
        dataloader : DataLoaderConfig
            Data loading configuration (num_workers, prefetch, packing, etc.).
        experiment_name : str
            Human‑readable experiment name (used for W&B runs, logging).
        description : str
            Optional description of experiment purpose or changes.
        tags : list[str], optional
            Tags for experiment categorization and filtering.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training = training
        self.dataloader = dataloader
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []

    # ========== Delegation properties for model (KilatConfig) ==========
    @property
    def vocab_size(self) -> int:
        """Vocabulary size (delegated from model)."""
        return self.model.vocab_size

    @property
    def n_embd(self) -> int:
        """Hidden embedding dimension (delegated from model)."""
        return self.model.n_embd

    @property
    def n_layer(self) -> int:
        """Number of transformer layers (delegated from model)."""
        return self.model.n_layer

    @property
    def n_head(self) -> int:
        """Number of attention heads (delegated from model)."""
        return self.model.n_head

    @property
    def pad_token_id(self) -> int:
        """Padding token ID (delegated from model)."""
        return self.model.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """End‑of‑sequence token ID (delegated from model)."""
        return self.model.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Beginning‑of‑sequence token ID (delegated from model)."""
        return self.model.bos_token_id

    @property
    def use_cache(self) -> bool:
        """Whether to use KV cache (delegated from model)."""
        return self.model.use_cache

    @property
    def recall_ratio(self) -> float:
        """Recall ratio for hybrid attention (delegated from model)."""
        return self.model.recall_ratio

    @property
    def latent_dim(self) -> Optional[int]:
        """Latent dimension for MLA (delegated from model)."""
        return self.model.latent_dim

    @property
    def attn_drop(self) -> float:
        """Attention dropout (delegated from model)."""
        return self.model.attn_drop

    @property
    def ffn_mode(self) -> str:
        """Feed‑forward mode (dense/moe) (delegated from model)."""
        return self.model.ffn_mode

    @property
    def ff_mult(self) -> float:
        """FFN hidden multiplier (delegated from model)."""
        return self.model.ff_mult

    @property
    def ffn_dropout(self) -> float:
        """FFN dropout (delegated from model)."""
        return self.model.ffn_dropout

    @property
    def num_experts(self) -> int:
        """Number of MoE experts (delegated from model)."""
        return self.model.num_experts

    @property
    def active_experts(self) -> int:
        """Active experts per token (delegated from model)."""
        return self.model.active_experts

    @property
    def num_shared_experts(self) -> int:
        """Shared experts count (delegated from model)."""
        return self.model.num_shared_experts

    @property
    def fine_grained_factor(self) -> int:
        """Expert segmentation factor (delegated from model)."""
        return self.model.fine_grained_factor

    @property
    def aux_loss_coef(self) -> float:
        """Auxiliary loss coefficient (delegated from model)."""
        return self.model.aux_loss_coef

    @property
    def device_balance_coef(self) -> float:
        """Device balance loss coefficient (delegated from model)."""
        return self.model.device_balance_coef

    @property
    def embd_drop(self) -> float:
        """Embedding dropout (delegated from model)."""
        return self.model.embd_drop

    @property
    def resid_drop(self) -> float:
        """Residual dropout (delegated from model)."""
        return self.model.resid_drop

    @property
    def initializer_range(self) -> float:
        """Weight initialisation std (delegated from model)."""
        return self.model.initializer_range

    @property
    def tie_word_embeddings(self) -> bool:
        """Whether to tie embeddings (delegated from model)."""
        return self.model.tie_word_embeddings

    # ========== Delegation properties for dataloader ==========
    @property
    def train_batch_size(self) -> int:
        """Training batch size per device (delegated from dataloader)."""
        return self.dataloader.train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """Evaluation batch size per device (delegated from dataloader)."""
        return self.dataloader.eval_batch_size

    @property
    def num_workers(self) -> int:
        """Number of DataLoader workers (delegated from dataloader)."""
        return self.dataloader.num_workers

    @property
    def pin_memory(self) -> bool:
        """Pin memory for GPU transfer (delegated from dataloader)."""
        return self.dataloader.pin_memory

    @property
    def prefetch_factor(self) -> int:
        """Prefetch factor per worker (delegated from dataloader)."""
        return self.dataloader.prefetch_factor

    @property
    def persistent_workers(self) -> bool:
        """Keep workers alive across epochs (delegated from dataloader)."""
        return self.dataloader.persistent_workers

    @property
    def drop_last(self) -> bool:
        """Drop last incomplete batch (delegated from dataloader)."""
        return self.dataloader.drop_last

    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length (delegated from dataloader)."""
        return self.dataloader.max_seq_length

    @property
    def truncation(self) -> str:
        """Truncation side (left/right) (delegated from dataloader)."""
        return self.dataloader.truncation

    @property
    def use_packing(self) -> bool:
        """Enable sequence packing (delegated from dataloader)."""
        return self.dataloader.use_packing

    @property
    def packed_block_size(self) -> Optional[int]:
        """Block size for packing (delegated from dataloader)."""
        return self.dataloader.packed_block_size

    @property
    def use_distributed_sampler(self) -> bool:
        """Use DistributedSampler in DDP (delegated from dataloader)."""
        return self.dataloader.use_distributed_sampler

    @property
    def distributed_shuffle(self) -> bool:
        """Shuffle in DistributedSampler (delegated from dataloader)."""
        return self.dataloader.distributed_shuffle

    @property
    def train_data_path(self) -> Optional[str]:
        """Training dataset path (delegated from dataloader)."""
        return self.dataloader.train_data_path

    @property
    def eval_data_path(self) -> Optional[str]:
        """Evaluation dataset path (delegated from dataloader)."""
        return self.dataloader.eval_data_path

    @property
    def dataset_format(self) -> str:
        """Dataset format (parquet/memmap/jsonl) (delegated from dataloader)."""
        return self.dataloader.dataset_format

    @property
    def cache_dir(self) -> Optional[str]:
        """Cache directory (delegated from dataloader)."""
        return self.dataloader.cache_dir

    @property
    def prefetch_batches(self) -> int:
        """Prefetch batches in dataset iterator (delegated from dataloader)."""
        return self.dataloader.prefetch_batches

    # ========== Delegation properties for training ==========
    @property
    def output_dir(self) -> str:
        """Output directory for checkpoints (delegated from training)."""
        return self.training.output_dir

    @property
    def resume_from_checkpoint(self) -> Optional[str]:
        """Resume from checkpoint path (delegated from training)."""
        return self.training.resume_from_checkpoint

    @property
    def save_checkpoints(self) -> bool:
        """Whether to save checkpoints (delegated from training)."""
        return self.training.save_checkpoints

    @property
    def atomic_checkpoint(self) -> bool:
        """Atomic checkpoint writes (delegated from training)."""
        return self.training.atomic_checkpoint

    @property
    def training_mode(self) -> str:
        """Training mode (steps/epochs) (delegated from training)."""
        return self.training.training_mode

    @property
    def learning_rate(self) -> float:
        """Peak learning rate (delegated from training)."""
        return self.training.learning_rate

    @property
    def beta1(self) -> float:
        """AdamW beta1 (delegated from training)."""
        return self.training.beta1

    @property
    def beta2(self) -> float:
        """AdamW beta2 (delegated from training)."""
        return self.training.beta2

    @property
    def epsilon(self) -> float:
        """AdamW epsilon (delegated from training)."""
        return self.training.epsilon

    @property
    def per_device_train_batch_size(self) -> int:
        """Per-device training batch size (delegated from training)."""
        return self.training.per_device_train_batch_size

    @property
    def per_device_eval_batch_size(self) -> int:
        """Per-device eval batch size (delegated from training)."""
        return self.training.per_device_eval_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Gradient accumulation steps (delegated from training)."""
        return self.training.gradient_accumulation_steps

    @property
    def weight_decay(self) -> float:
        """Weight decay coefficient (delegated from training)."""
        return self.training.weight_decay

    @property
    def max_grad_norm(self) -> float:
        """Max gradient norm for clipping (delegated from training)."""
        return self.training.max_grad_norm

    @property
    def max_steps(self) -> int:
        """Total steps in steps mode (delegated from training)."""
        return self.training.max_steps

    @property
    def num_train_epochs(self) -> int:
        """Number of epochs in epochs mode (delegated from training)."""
        return self.training.num_train_epochs

    @property
    def warmup_steps(self) -> int:
        """Warmup steps (delegated from training)."""
        return self.training.warmup_steps

    @property
    def scheduler_type(self) -> str:
        """Scheduler type (cosine/linear/etc.) (delegated from training)."""
        return self.training.scheduler_type

    @property
    def scheduler_kwargs(self) -> dict:
        """Extra scheduler kwargs (delegated from training)."""
        return self.training.scheduler_kwargs

    @property
    def logging_steps(self) -> int:
        """Logging interval (steps) (delegated from training)."""
        return self.training.logging_steps

    @property
    def eval_steps(self) -> int:
        """Evaluation interval (steps) (delegated from training)."""
        return self.training.eval_steps

    @property
    def save_steps(self) -> int:
        """Checkpoint interval (steps) (delegated from training)."""
        return self.training.save_steps

    @property
    def save_total_limit(self) -> Optional[int]:
        """Max number of numbered checkpoints (delegated from training)."""
        return self.training.save_total_limit

    @property
    def early_stopping_patience(self) -> int:
        """Early stopping patience (delegated from training)."""
        return self.training.early_stopping_patience

    @property
    def early_stopping_threshold(self) -> float:
        """Early stopping threshold (delegated from training)."""
        return self.training.early_stopping_threshold

    @property
    def metric_for_best_model(self) -> str:
        """Metric for best model selection (delegated from training)."""
        return self.training.metric_for_best_model

    @property
    def greater_is_better(self) -> Optional[bool]:
        """Direction of best metric (delegated from training)."""
        return self.training.greater_is_better

    @property
    def precision(self) -> str:
        """Mixed precision mode (fp32/fp16/bf16) (delegated from training)."""
        return self.training.precision

    @property
    def report_to(self) -> list:
        """Logging backends (delegated from training)."""
        return self.training.report_to

    @property
    def run_name(self) -> Optional[str]:
        """Run display name (delegated from training)."""
        return self.training.run_name

    @property
    def seed(self) -> int:
        """Random seed (delegated from training)."""
        return self.training.seed

    # ========== Serialisation / I/O methods ==========
    def to_yaml(self, path: str | Path) -> str:
        """
        Export complete configuration to a single YAML file.

        Includes a header comment with experiment metadata for quick identification.
        """
        config_dict = {
            "experiment": {
                "name": self.experiment_name,
                "description": self.description,
                "tags": self.tags,
            },
            "model": self.model.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "dataloader": self.dataloader.to_dict(),
            "training": self.training.to_dict(),
        }
        # Remove HuggingFace internal metadata (runtime artifacts)
        config_dict["model"].pop("transformers_version", None)
        config_dict["model"].pop("model_type", None)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# KilatTransformer Full Configuration\n")
            f.write(f"# Experiment: {self.experiment_name}\n")
            if self.description:
                f.write(f"# Description: {self.description}\n")
            if self.tags:
                f.write(f"# Tags: {', '.join(self.tags)}\n")
            f.write("\n")
            f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MainConfig":
        """
        Load complete experiment configuration from YAML.

        Validates that required sections (tokenizer, dataloader, training) exist.
        Warns if tokenizer vocab size does not match model vocab size.
        """
        config_dict = load_yaml_file(yaml_path)

        if "tokenizer" not in config_dict:
            raise ValueError("MainConfig YAML must include a 'tokenizer' section.")
        if "dataloader" not in config_dict:
            raise ValueError("MainConfig YAML must include a 'dataloader' section.")
        if "training" not in config_dict:
            raise ValueError("MainConfig YAML must include a 'training' section.")

        experiment = config_dict.get("experiment", {})

        model_config = KilatConfig(**config_dict["model"])
        tokenizer_config = TokenizerConfig(**config_dict["tokenizer"])
        tokenizer_config.warn_if_vocab_mismatch(model_config.vocab_size)
        dataloader_config = DataLoaderConfig(**config_dict["dataloader"])
        training_config = TrainingConfig(**config_dict["training"])

        return cls(
            model=model_config,
            tokenizer=tokenizer_config,
            training=training_config,
            dataloader=dataloader_config,
            experiment_name=experiment.get("name", "kilat-experiment"),
            description=experiment.get("description", ""),
            tags=experiment.get("tags", []),
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        """
        Save all configurations to a checkpoint directory.

        Writes:
            - config.json / config.yaml (model)
            - tokenizer_config.yaml (tokenizer)
            - dataloader_config.yaml (dataloader)
            - training_config.yaml (training)
            - full_config.yaml (combined)
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.to_yaml(save_dir / "tokenizer_config.yaml")
        self.dataloader.to_yaml(save_dir / "dataloader_config.yaml")
        self.training.to_yaml(save_dir / "training_config.yaml")
        self.to_yaml(save_dir / "full_config.yaml")
    
    # ========== Builder / factory methods ==========
    def build_tokenizer(self) -> Any:
        """Build tokenizer instance from configuration."""
        return self.tokenizer.build()
    
    def build_model(self) -> KilatConfig:
        """Return model configuration."""
        return self.model

    @classmethod
    def from_main_config(cls, main_config: "MainConfig") -> "KilatConfig":
        """
        Extract KilatConfig from a MainConfig object (convenience method).

        This method is kept for backward compatibility. It returns the model
        configuration held inside the MainConfig.

        Raises:
            TypeError: If the input type is not supported.
        """
        if hasattr(main_config, "model"):
            model_dict = main_config.model.__dict__
        elif isinstance(main_config, dict):
            if "model" in main_config:
                model_dict = main_config["model"]
            else:
                model_dict = main_config
        elif hasattr(main_config, "__dict__"):
            model_dict = main_config.__dict__
        else:
            raise TypeError(
                f"Unsupported config type: {type(main_config)}. "
                "Expected MainConfig, dict, or object with model attribute."
            )
        return KilatConfig(**model_dict)

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"MainConfig(experiment='{self.experiment_name}', "
            f"model={self.model.__class__.__name__}, "
            f"tokenizer={self.tokenizer.tokenizer_type}, "
            f"training={self.training.training_mode})"
        )