from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Callable, Any, Union
import warnings
import yaml
import json
from .base import dump_yaml_file, load_yaml_file


class TokenizerConfig:
    """
    Tokenizer configuration for preprocessing and decode-time inspection.

    WHY THIS EXISTS:
        The tokenizer is a critical component of the LLM pipeline that is often
        overlooked in configuration management. This class ensures that:
        - The exact tokenizer used for training is recorded and reproducible
        - Vocabulary size mismatches are caught early (before index errors)
        - Different tokenizer backends (HF, SentencePiece, Custom) have a unified interface

    SUPPORTED BACKENDS:
        - **gpt2**: HuggingFace GPT-2 tokenizer (Byte-level BPE). Default choice
          for most English models. Uses AutoTokenizer.from_pretrained("gpt2").
        - **sentencepiece**: Google's SentencePiece tokenizer. Supports unigram,
          BPE, and char/word models. Requires tokenizer_model_path.
        - **auto**: Auto-detect from tokenizer_name_or_path (HF AutoTokenizer).
          Most flexible but less predictable.
        - **custom**: User-provided custom tokenizer via builder function or class.
          Requires custom_builder or custom_module + custom_class.

    WHY SEPARATE FROM MODEL CONFIG:
        The tokenizer is independent from model architecture. You can:
        - Change tokenizer without changing model architecture
        - Fine-tune a model with a different tokenizer (vocabulary extension)
        - Use the same tokenizer across multiple models
        - Store tokenizer separately for data preprocessing pipelines

    VOCABULARY MISMATCH PROTECTION:
        When loading a complete experiment config, `warn_if_vocab_mismatch()` checks
        that the tokenizer vocabulary size matches the model's `vocab_size`.
        This catches common errors where:
        - The dataset was tokenized with a different tokenizer than the model expects
        - The tokenizer was modified (e.g., added special tokens) after data processing

    PERFORMANCE:
        - `_resolve_vocab_size()` loads the tokenizer only when needed (lazy)
        - The resolved vocab size is NOT cached; called once per config load
        - For sentencepiece, loading the model file adds ~0.1-0.5s overhead

    EDGE CASES:
        - If the tokenizer cannot be resolved (missing file, network issues),
          a warning is emitted but training continues (graceful degradation)
        - `tokenizer_type="auto"` with local_files_only=True will not download
          missing tokenizers; fails gracefully with warning
        - For sentencepiece, tokenizer_model_path overrides tokenizer_name_or_path
        - For custom tokenizers, the builder function must accept a TokenizerConfig
          instance and return a tokenizer object with encode/decode methods

    Example Usage
    -------------
        >>> # HuggingFace tokenizer (most common)
        >>> tokenizer = TokenizerConfig(
        ...     tokenizer_type="gpt2",
        ...     tokenizer_name_or_path="gpt2",
        ...     use_fast=True,
        ... )
        >>>
        >>> # Custom SentencePiece tokenizer
        >>> tokenizer = TokenizerConfig(
        ...     tokenizer_type="sentencepiece",
        ...     tokenizer_model_path="./tokenizers/spm.model",
        ... )
        >>>
        >>> # Custom tokenizer with builder function
        >>> tokenizer = TokenizerConfig(
        ...     tokenizer_type="custom",
        ...     custom_builder="mymodule.tokenizer:build_my_tokenizer",
        ... )
        >>>
        >>> # Check compatibility with model
        >>> tokenizer.warn_if_vocab_mismatch(model_config.vocab_size)
        >>>
        >>> # Build tokenizer instance
        >>> tokenizer_instance = tokenizer.build()
        >>>
        >>> # Save for later use (JSON format for AutoTokenizer compatibility)
        >>> tokenizer.to_json("tokenizer_config.json")
        >>> loaded = TokenizerConfig.from_json("tokenizer_config.json")
    """

    def __init__(
        self,
        tokenizer_type: Literal["gpt2", "sentencepiece", "auto", "custom"] = "gpt2",
        tokenizer_name_or_path: str = "gpt2",
        tokenizer_model_path: Optional[str] = None,
        use_fast: bool = True,
        local_files_only: bool = True,
        # Custom tokenizer parameters
        custom_builder: Optional[str] = None,
        custom_module: Optional[str] = None,
        custom_class: Optional[str] = None,
        custom_kwargs: Optional[dict] = None,
    ):
        """
        Initialise tokenizer configuration with validation.

        Parameters
        ----------
        tokenizer_type : Literal["gpt2", "sentencepiece", "auto", "custom"]
            Type of tokenizer backend:
            - "gpt2": HuggingFace GPT-2 tokenizer (Byte-level BPE)
            - "sentencepiece": Google SentencePiece processor
            - "auto": Auto-detect from tokenizer_name_or_path
            - "custom": User-provided custom tokenizer
        tokenizer_name_or_path : str
            HF model name (e.g., "gpt2", "meta-llama/Llama-2-7b") or local path.
            Required for all types except when using custom_builder directly.
        tokenizer_model_path : Optional[str]
            Path to SentencePiece .model file. Required when tokenizer_type="sentencepiece".
            Ignored for other types.
        use_fast : bool
            Whether to use the fast tokenizer implementation (Rust-based).
            Only applies to HF tokenizers (ignored for sentencepiece).
            Fast tokenizers are 5-10x faster but may have minor behavioural differences.
        local_files_only : bool
            If True, do not download missing tokenizers from HuggingFace Hub.
            Useful for offline environments or to enforce local cache usage.
        custom_builder : Optional[str]
            Import path to a custom tokenizer builder function.
            Format: "module.path:function_name"
            The function must accept a TokenizerConfig instance and return a tokenizer.
            Example: "myapp.tokenizers:build_my_tokenizer"
        custom_module : Optional[str]
            Module path for custom tokenizer class.
            Used together with custom_class.
            Example: "myapp.tokenizers"
        custom_class : Optional[str]
            Class name for custom tokenizer.
            Used together with custom_module.
            The class must have a from_pretrained method or accept tokenizer_name_or_path.
        custom_kwargs : Optional[dict]
            Additional keyword arguments passed to custom tokenizer constructor.

        Raises
        ------
        ValueError
            If tokenizer_type is invalid.
            If tokenizer_name_or_path is empty (when required).
            If tokenizer_type="sentencepiece" but tokenizer_model_path is missing.
            If tokenizer_type="custom" but neither custom_builder nor (custom_module + custom_class) provided.
        """
        # Validate tokenizer_type
        if tokenizer_type not in ("gpt2", "sentencepiece", "auto", "custom"):
            raise ValueError(
                "tokenizer_type must be one of ('gpt2', 'sentencepiece', 'auto', 'custom'), "
                f"got '{tokenizer_type}'."
            )

        # For non-custom types, tokenizer_name_or_path is required
        if tokenizer_type != "custom" and not tokenizer_name_or_path:
            raise ValueError("tokenizer_name_or_path must not be empty for non-custom tokenizers.")

        # SentencePiece requires model path
        if tokenizer_type == "sentencepiece" and not tokenizer_model_path:
            raise ValueError(
                "tokenizer_model_path is required when tokenizer_type='sentencepiece'."
            )

        # Custom tokenizer requires either custom_builder or (custom_module + custom_class)
        if tokenizer_type == "custom":
            if not custom_builder and not (custom_module and custom_class):
                raise ValueError(
                    "tokenizer_type='custom' requires either custom_builder or "
                    "(custom_module + custom_class)."
                )

        self.tokenizer_type = tokenizer_type
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_model_path = tokenizer_model_path
        self.use_fast = use_fast
        self.local_files_only = local_files_only
        self.custom_builder = custom_builder
        self.custom_module = custom_module
        self.custom_class = custom_class
        self.custom_kwargs = custom_kwargs or {}

    def build(self) -> Any:
        """
        Build and return the actual tokenizer instance from configuration.

        WHY: Centralizes tokenizer creation logic so users don't need to write
        conditional code (if sentencepiece else AutoTokenizer). This method handles:
        - HuggingFace tokenizers (gpt2, auto, or custom path)
        - SentencePiece tokenizers
        - Custom tokenizers via builder function or class

        Returns
        -------
        Any
            Tokenizer instance with methods: encode, decode, __call__, etc.
            For HuggingFace: returns AutoTokenizer with save_pretrained method.
            For SentencePiece: returns SentencePieceProcessor instance.
            For custom: returns whatever the builder/class returns.

        Raises
        ------
        ImportError
            If required library (transformers or sentencepiece) is not installed.
        FileNotFoundError
            If tokenizer model file doesn't exist for sentencepiece.
        ValueError
            If tokenizer_type is unknown or custom builder fails.
        """
        if self.tokenizer_type == "sentencepiece":
            return self._build_sentencepiece()
        elif self.tokenizer_type in ("gpt2", "auto"):
            return self._build_huggingface()
        elif self.tokenizer_type == "custom":
            return self._build_custom()
        else:
            raise ValueError(
                f"Unknown tokenizer_type: {self.tokenizer_type}. "
                "Supported: 'gpt2', 'auto', 'sentencepiece', 'custom'"
            )

    def _build_sentencepiece(self) -> Any:
        """Build SentencePiece tokenizer."""
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError:
            raise ImportError(
                "sentencepiece is required for SentencePiece tokenizer. "
                "Install with: pip install sentencepiece"
            )

        model_path = self.tokenizer_model_path or self.tokenizer_name_or_path
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"SentencePiece model file not found: {model_path}"
            )

        processor = SentencePieceProcessor()
        processor.Load(model_path)
        return processor

    def _build_huggingface(self) -> Any:
        """Build HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace tokenizer. "
                "Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            use_fast=self.use_fast,
            local_files_only=self.local_files_only,
        )

        # Ensure pad_token is set (GPT-2 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _build_custom(self) -> Any:
        """
        Build custom tokenizer from user-provided builder or class.

        Supports two modes:
        1. Builder function: custom_builder = "module.path:function_name"
           Function signature: def builder(config: TokenizerConfig) -> tokenizer
        2. Class: custom_module + custom_class
           Class must have from_pretrained method or accept tokenizer_name_or_path
        """
        # Mode 1: Builder function
        if self.custom_builder:
            return self._build_from_builder()

        # Mode 2: Class with module + class name
        if self.custom_module and self.custom_class:
            return self._build_from_class()

        raise ValueError(
            "Cannot build custom tokenizer: neither custom_builder nor "
            "(custom_module + custom_class) is configured."
        )

    def _build_from_builder(self) -> Any:
        """Import and call custom builder function."""
        if not self.custom_builder:
            raise ValueError("custom_builder is not configured.")

        # Parse "module.path:function_name"
        parts = self.custom_builder.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid custom_builder format: '{self.custom_builder}'. "
                "Expected 'module.path:function_name'"
            )

        module_path, func_name = parts
        import importlib
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{module_path}' for custom tokenizer: {e}"
            )

        if not hasattr(module, func_name):
            raise AttributeError(
                f"Module '{module_path}' has no function '{func_name}'"
            )

        builder_func = getattr(module, func_name)
        try:
            return builder_func(self)
        except Exception as e:
            raise ValueError(
                f"Custom tokenizer builder '{self.custom_builder}' failed: {e}"
            ) from e

    def _build_from_class(self) -> Any:
        """Import and instantiate custom tokenizer class."""
        if not self.custom_module or not self.custom_class:
            raise ValueError("custom_module and custom_class are required.")

        import importlib
        try:
            module = importlib.import_module(self.custom_module)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{self.custom_module}' for custom tokenizer: {e}"
            )

        if not hasattr(module, self.custom_class):
            raise AttributeError(
                f"Module '{self.custom_module}' has no class '{self.custom_class}'"
            )

        tokenizer_class = getattr(module, self.custom_class)

        # Try from_pretrained first, then direct constructor
        try:
            if hasattr(tokenizer_class, "from_pretrained"):
                return tokenizer_class.from_pretrained(
                    self.tokenizer_name_or_path,
                    **self.custom_kwargs
                )
            else:
                return tokenizer_class(
                    self.tokenizer_name_or_path,
                    **self.custom_kwargs
                )
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate custom tokenizer '{self.custom_class}': {e}"
            ) from e

    def save_tokenizer(self, tokenizer: Any, save_directory: str | Path) -> None:
        """
        Save a tokenizer instance to a directory.

        This method handles saving both HuggingFace tokenizers, SentencePiece
        tokenizers, and custom tokenizers (if they implement save_pretrained).

        IMPORTANT: Also saves tokenizer_config.json for AutoTokenizer compatibility.

        Parameters
        ----------
        tokenizer : Any
            Tokenizer instance (from build() or external).
        save_directory : str | Path
            Directory to save the tokenizer.

        Example
        -------
            >>> tokenizer_config.save_tokenizer(tokenizer, "./checkpoint/tokenizer")
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the tokenizer config itself for reproducibility (JSON format for AutoTokenizer)
        self.to_json(save_dir / "tokenizer_config.json")
        
        # Also save as YAML for human readability
        self.to_yaml(save_dir / "tokenizer_config.yaml")

        if self.tokenizer_type == "sentencepiece":
            # Save the .model file
            model_path = self.tokenizer_model_path or self.tokenizer_name_or_path
            if model_path and Path(model_path).exists():
                import shutil
                shutil.copy2(model_path, save_dir / "spm.model")
        elif self.tokenizer_type in ("gpt2", "auto"):
            # HuggingFace tokenizer has save_pretrained
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(save_dir)
        elif self.tokenizer_type == "custom":
            # For custom tokenizers, try save_pretrained if available
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(save_dir)
            else:
                # At least save the config for reproducibility
                warnings.warn(
                    f"Custom tokenizer {type(tokenizer).__name__} does not have "
                    "save_pretrained method. Only config was saved.",
                    UserWarning,
                    stacklevel=2,
                )

    def to_dict(self) -> dict:
        """
        Convert tokenizer configuration to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with all tokenizer hyperparameters, suitable for
            JSON/YAML serialisation.
        """
        result = {
            "tokenizer_type": self.tokenizer_type,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_model_path": self.tokenizer_model_path,
            "use_fast": self.use_fast,
            "local_files_only": self.local_files_only,
        }
        # Add custom tokenizer fields only if they exist
        if self.custom_builder:
            result["custom_builder"] = self.custom_builder
        if self.custom_module:
            result["custom_module"] = self.custom_module
        if self.custom_class:
            result["custom_class"] = self.custom_class
        if self.custom_kwargs:
            result["custom_kwargs"] = self.custom_kwargs
        return result

    def to_json(self, path: str | Path) -> None:
        """
        Export tokenizer configuration to JSON format.

        WHY JSON: This is the standard format expected by HuggingFace's
        AutoTokenizer and our custom AutoTokenizer. JSON is machine-friendly
        and widely supported.

        Parameters
        ----------
        path : str | Path
            Path to write the JSON file.

        Example
        -------
            >>> tokenizer_config.to_json("tokenizer_config.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "TokenizerConfig":
        """
        Load tokenizer configuration from a JSON file.

        This is the primary method for loading tokenizer configs saved by
        the converter or trainer. Used by AutoTokenizer.from_pretrained().

        Parameters
        ----------
        json_path : str | Path
            Path to JSON tokenizer configuration file.

        Returns
        -------
        TokenizerConfig
            Validated tokenizer configuration instance.

        Example
        -------
            >>> tokenizer_config = TokenizerConfig.from_json("tokenizer_config.json")
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export tokenizer configuration to YAML format.

        WHY YAML: Human-readable format that preserves comments and supports
        easy manual editing. The exported file can be version-controlled
        alongside experiment configurations.

        Parameters
        ----------
        path : Optional[str | Path]
            If provided, writes YAML to this file. If None, returns the YAML string.

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
    def from_yaml(cls, yaml_path: str | Path) -> "TokenizerConfig":
        """
        Load tokenizer configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML tokenizer configuration file.

        Returns
        -------
        TokenizerConfig
            Validated tokenizer configuration instance.

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

    def warn_if_vocab_mismatch(self, expected_vocab_size: int) -> None:
        """
        Warn when the configured tokenizer vocabulary does not match the model.

        WHY: A mismatch between tokenizer vocab size and model vocab_size is a
        common source of silent errors:
        - IndexError when model expects a token ID beyond its vocabulary
        - Incorrect decoding (wrong special tokens, missing tokens)
        - Dataset tokenized with different tokenizer yields garbage tokens

        The check is best-effort: if the tokenizer cannot be loaded locally,
        a warning is emitted and validation continues without failing training.

        WHEN THIS IS CALLED:
            Typically during MainConfig loading, after both model and tokenizer
            configs are parsed, to catch mismatches before training starts.

        Parameters
        ----------
        expected_vocab_size : int
            The model's vocab_size (expected to match tokenizer vocabulary size).

        Edge Cases
        ----------
        - If expected_vocab_size <= 0, the check is skipped (invalid model config)
        - If tokenizer resolution fails (missing file, network error), warning
          is emitted and function returns (graceful degradation)
        - Warning is emitted with stacklevel=3 so it appears as coming from
          the caller, not from inside this method
        """
        if expected_vocab_size <= 0:
            return

        resolved_vocab_size = self._resolve_vocab_size()
        if resolved_vocab_size is None:
            warnings.warn(
                "Tokenizer vocabulary could not be resolved locally, so vocab size mismatch could not be checked.",
                UserWarning,
                stacklevel=3,
            )
            return

        if resolved_vocab_size != expected_vocab_size:
            warnings.warn(
                f"Tokenizer vocab size ({resolved_vocab_size}) does not match model vocab_size ({expected_vocab_size}). "
                "This can cause index errors or incorrect decoding if the dataset was tokenized with a different tokenizer.",
                UserWarning,
                stacklevel=3,
            )

    def _resolve_vocab_size(self) -> Optional[int]:
        """
        Resolve the actual vocabulary size from the configured tokenizer.

        WHY: Different tokenizer backends provide vocab size via different APIs.
        This method abstracts those differences and provides a uniform interface.

        Returns
        -------
        Optional[int]
            The vocabulary size if resolution succeeds, None otherwise.

        Note
        ----
        This method may:
        - Import libraries (transformers, sentencepiece) lazily
        - Load model files from disk
        - Access the HuggingFace Hub (if local_files_only=False)
        - Call custom tokenizer builder if needed

        Exceptions are caught and logged as None (graceful failure).
        """
        try:
            if self.tokenizer_type == "sentencepiece":
                from sentencepiece import SentencePieceProcessor

                model_path = self.tokenizer_model_path or self.tokenizer_name_or_path
                if not model_path:
                    return None
                processor = SentencePieceProcessor()
                if not processor.load(model_path):
                    return None
                return processor.get_piece_size()

            elif self.tokenizer_type in ("gpt2", "auto"):
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    use_fast=self.use_fast,
                    local_files_only=self.local_files_only,
                )
                return len(tokenizer)

            elif self.tokenizer_type == "custom":
                # For custom tokenizers, try to build and get vocab size
                tokenizer = self.build()
                if hasattr(tokenizer, "vocab_size"):
                    return tokenizer.vocab_size
                elif hasattr(tokenizer, "get_vocab_size"):
                    return tokenizer.get_vocab_size()
                elif hasattr(tokenizer, "vocab"):
                    return len(tokenizer.vocab)
                else:
                    # Cannot determine vocab size, return None
                    return None

            else:
                return None
        except Exception:
            # Catch-all: any failure (import, file not found, network) returns None
            return None

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        if self.tokenizer_type == "custom":
            if self.custom_builder:
                builder_info = f", builder={self.custom_builder}"
            else:
                builder_info = f", class={self.custom_class}"
            return (
                f"TokenizerConfig(type=custom{builder_info})"
            )
        return (
            f"TokenizerConfig(type={self.tokenizer_type}, "
            f"name_or_path={self.tokenizer_name_or_path})"
        )