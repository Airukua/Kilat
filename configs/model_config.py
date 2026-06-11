from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal
import warnings
import yaml
from .base import dump_yaml_file, load_yaml_file
from utils.base_model import BaseConfig

class KilatConfig(BaseConfig):
    """
    Configuration class for KilatTransformer model architecture.

    WHY THIS EXTENDS BaseConfig:
        Inheriting from HuggingFace's BaseConfig provides:
        - Seamless integration with transformers ecosystem (save_pretrained/from_pretrained)
        - Automatic serialisation to JSON (config.json) for model hub compatibility
        - Support for push_to_hub, AutoModel, and other HF tooling
        - Standardised handling of common fields (pad_token_id, tie_word_embeddings, etc.)

    ARCHITECTURE SUPPORT:
        - **Dense transformer**: Standard attention + SwiGLU FFN (ffn_mode="dense")
        - **Mixture of Experts (MoE)**: Sparse routing with configurable experts
        - **DeepSeek-V2 MoE**: Shared experts + fine-grained expert segmentation
        - **Retention/MLA hybrid**: Configurable recall_ratio for efficient long-context

    KEY PARAMETER INTERACTIONS:
        - `latent_dim` defaults to `n_embd // 4` for 4x KV-cache compression in MLA path
        - `recall_ratio` splits heads between precise (latent MLA) and efficient (decay) paths
        - `num_shared_experts` > 0 enables DeepSeek-V2 style (distinct from standard MoE)
        - `fine_grained_factor` splits each expert into smaller sub-experts for finer routing

    VALIDATION:
        All critical constraints are checked at construction time (fail fast):
        - n_embd must be divisible by n_head (integer head_dim)
        - Shared experts only allowed in MoE mode
        - Dropout probabilities in [0, 1)
        - MoE routing constraints (active_experts ≤ num_experts)

    Example Usage
    -------------
        >>> # Dense model
        >>> config = KilatConfig(vocab_size=32000, n_embd=768, n_head=12, n_layer=12)
        >>>
        >>> # DeepSeek-V2 style MoE
        >>> config = KilatConfig(
        ...     vocab_size=64000, n_embd=2048, n_head=32, n_layer=32,
        ...     ffn_mode="moe", num_experts=64, active_experts=8,
        ...     num_shared_experts=2, fine_grained_factor=2
        ... )
        >>>
        >>> # Save for later use
        >>> config.save_pretrained("./models/my-model")
        >>> loaded = KilatConfig.from_pretrained("./models/my-model")
    """

    model_type = "kilat_transformer"

    def __init__(
        self,
        # ---- Core dimensions ----
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        # ---- Attention configuration ----
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        # ---- Feed-forward configuration ----
        ffn_mode: Literal["dense", "moe"] = "moe",
        ff_mult: float = 8 / 3,
        ffn_dropout: float = 0.0,
        # ---- MoE configuration ----
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
        return_dict: bool = True,
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Set internal flags expected by HuggingFace (prevents AttributeError)
        # These are not user-configurable but are required by the base class.
        self._output_attentions = False
        self._output_hidden_states = False
        self._use_cache = use_cache
        self._attn_implementation = None
        self._attn_implementation_internal = None
        self._experts_implementation_internal = None

        # Initialise HF parent with standard token config
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Core dimensions
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        # Attention parameters
        self.recall_ratio = recall_ratio
        self.latent_dim = latent_dim
        self.attn_drop = attn_drop

        # Feed-forward parameters
        self.ffn_mode = ffn_mode
        self.ff_mult = ff_mult
        self.ffn_dropout = ffn_dropout

        # MoE parameters
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.num_shared_experts = num_shared_experts
        self.fine_grained_factor = fine_grained_factor
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef

        # Regularisation
        self.embd_drop = embd_drop
        self.resid_drop = resid_drop

        # Other
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.return_dict = return_dict 

        # Run validation after all attributes are set
        self._validate()

    def __post_init__(self, **kwargs):
        """
        Post-initialisation hook called by HuggingFace after loading from checkpoint.

        WHY: This method is called automatically by HF's `from_pretrained` mechanism.
        However, we already perform validation in `__init__`, so we do nothing here.
        Returning None prevents the base class from running extra validation that
        might fail on our custom fields.
        """
        return None

    def _validate(self):
        """
        Validate configuration consistency and catch common misconfigurations.

        WHY FAIL FAST: A misconfiguration (e.g., n_embd not divisible by n_head)
        will cause cryptic dimension mismatch errors deep in the model code.
        Validating at construction gives clear error messages immediately,
        saving hours of debugging.

        CHECKS PERFORMED:
            1. n_embd must be divisible by n_head (integer head dimension)
            2. Shared experts require MoE mode (not dense)
            3. active_experts ≤ num_experts in MoE mode
            4. num_experts ≥ 1 in MoE mode
            5. recall_ratio in [0, 1]
            6. Dropout probabilities in [0, 1)
        """
        # Architectural constraint: integer head dimension
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}) "
                f"to compute integer head_dim = n_embd / n_head."
            )

        # Shared experts are only meaningful in MoE routing
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

        # Recall ratio bounds (hybrid attention split)
        if not 0 <= self.recall_ratio <= 1:
            raise ValueError(f"recall_ratio must be in [0, 1], got {self.recall_ratio}.")

        # Dropout probability bounds (all share same [0, 1) range)
        for name in ["attn_drop", "ffn_dropout", "embd_drop", "resid_drop"]:
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export configuration to human-readable YAML format.

        WHY: JSON (used by save_pretrained) is machine-friendly but hard to read
        and diff. YAML supports comments and block formatting, making it ideal
        for version-controlled config files that humans review.

        The exported YAML excludes HuggingFace internal fields (transformers_version,
        model_type) that are not user-configurable.

        Parameters
        ----------
        path : Optional[str | Path]
            If provided, writes YAML to this file. If None, returns the YAML string.

        Returns
        -------
        str
            YAML string representation.
        """
        config_dict = self.to_dict()
        # Remove HF internal metadata (runtime artifacts, not user configurable)
        config_dict.pop("transformers_version", None)
        config_dict.pop("model_type", None)

        if path is not None:
            return dump_yaml_file(config_dict, path)
        return yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "KilatConfig":
        """
        Load configuration from a YAML file.

        Enables workflow:
            1. Edit config.yaml with desired hyperparameters
            2. Load: config = KilatConfig.from_yaml("config.yaml")
            3. Create model: model = KilatTransformer(config)

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML configuration file.

        Returns
        -------
        KilatConfig
            Validated configuration instance.
        """
        config_dict = load_yaml_file(yaml_path)
        return cls(**config_dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "KilatConfig":
        """
        Backward-compatible alias for from_yaml.

        WHY: Some existing scripts may call from_file. Keeping this alias
        prevents breaking existing code while encouraging the more explicit
        from_yaml name.
        """
        return cls.from_yaml(path)



    def save_pretrained(self, save_directory: str | Path, **kwargs):
        """
        Save configuration to directory in multiple formats.

        Extends HuggingFace's save_pretrained to write:
        - config.json: Standard HF format (for from_pretrained)
        - config.yaml: Human-readable format (for manual editing)

        WHY: Dual format ensures compatibility with HF ecosystem while providing
        a human-friendly version for configuration management.

        Parameters
        ----------
        save_directory : str | Path
            Directory to save configuration files.
        **kwargs : dict
            Additional arguments passed to BaseConfig.save_pretrained.
        """
        # Save standard HF JSON config (required for from_pretrained)
        super().save_pretrained(save_directory, **kwargs)

        # Additionally save human-readable YAML version
        save_dir = Path(save_directory)
        self.to_yaml(save_dir / "config.yaml")