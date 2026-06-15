"""
Auto tokenizer for KilatTransformer checkpoints.

WHY THIS EXISTS:
    - AutoTokenizer.from_pretrained() fails because KilatTransformer is not
      a registered model type in Hugging Face's registry.
    - This auto tokenizer reads tokenizer_config.json from checkpoint and builds
      the tokenizer using TokenizerConfig.build() method.
    - Supports ALL tokenizer backends: gpt2, auto, sentencepiece, CUSTOM

HOW IT WORKS:
    1. Look for tokenizer_config.json in checkpoint directory
    2. Load TokenizerConfig from that file (includes custom_builder, custom_module, etc.)
    3. Call TokenizerConfig.build() which handles:
       - HuggingFace tokenizers (gpt2, auto)
       - SentencePiece tokenizers
       - CUSTOM tokenizers with builder functions or classes
    4. If tokenizer_config.json not found, fallback to config.yaml
    5. If still not found, fallback to default GPT-2 tokenizer

USAGE:
    from generation.auto_tokenizer import AutoTokenizer
    
    # Load tokenizer from checkpoint (works for ALL types)
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/checkpoint-best")
    
    # Now use it normally
    tokens = tokenizer.encode("Hello world")
    text = tokenizer.decode(tokens)
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Optional, Union
from kilat.configs.tokenizer_config import TokenizerConfig


class AutoTokenizer:
    """
    Auto tokenizer for KilatTransformer checkpoints.
    
    This class provides a unified interface to load tokenizers from Kilat
    checkpoints, similar to Hugging Face's AutoTokenizer but designed to work
    with Kilat's custom tokenizer configuration system.
    
    It supports:
        - HuggingFace tokenizers (gpt2, auto)
        - SentencePiece tokenizers
        - Custom tokenizers with builder functions or classes
    
    Example
    -------
        >>> from generation.auto_tokenizer import AutoTokenizer
        >>>
        >>> # Load from checkpoint
        >>> tokenizer = AutoTokenizer.from_pretrained("./checkpoints/checkpoint-best")
        >>>
        >>> # Tokenize and decode
        >>> tokens = tokenizer.encode("Hello world")
        >>> text = tokenizer.decode(tokens)
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        local_files_only: Optional[bool] = None,  # <-- CHANGE: None = auto-detect
        **kwargs,
    ) -> Any:
        """
        Load a tokenizer from a KilatTransformer checkpoint directory.
        
        This method reads tokenizer configuration from the checkpoint and
        builds the tokenizer using TokenizerConfig.build().
        
        Supports ALL tokenizer types defined in TokenizerConfig:
            - gpt2: HuggingFace GPT-2 tokenizer
            - auto: AutoTokenizer from HuggingFace
            - sentencepiece: Google SentencePiece processor
            - custom: User-provided custom tokenizer with builder function/class
        
        Priority order:
            1. tokenizer_config.json in checkpoint directory (FULL SUPPORT)
            2. config.yaml (MainConfig) in checkpoint directory
            3. config.json (HF standard format) with tokenizer info
            4. Auto-detect from tokenizer files in directory
            5. Custom builder from environment variable
            6. Default GPT-2 tokenizer (fallback)
        
        DYNAMIC MODE:
            - If local_files_only = None (default): Auto-detect
              * If checkpoint has tokenizer files → local_files_only=True
              * If tokenizer needs download → local_files_only=False
            - If local_files_only = True: Force use local files only
            - If local_files_only = False: Force download from Hub
        
        Parameters
        ----------
        pretrained_model_name_or_path : str or Path
            Path to the checkpoint directory containing tokenizer_config.json
            or config.yaml.
        local_files_only : bool or None
            If None, auto-detect. If True, only use local files. If False, download.
        **kwargs
            Additional arguments passed to tokenizer builder.
        
        Returns
        -------
        Any
            Tokenizer instance. For custom tokenizers, returns whatever the
            builder function returns (must have encode/decode methods).
        
        Raises
        ------
        FileNotFoundError
            If no tokenizer configuration found and fallback fails.
        ImportError
            If custom builder module cannot be imported.
        AttributeError
            If custom builder function or class not found in module.
        """
        checkpoint_path = Path(pretrained_model_name_or_path)
        
        # ================================================================
        # PRIORITY 1: tokenizer_config.json (saved by Trainer or Converter)
        # ================================================================
        tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            print(f"Loading tokenizer config from: {tokenizer_config_path}")
            try:
                # Load the JSON config
                with open(tokenizer_config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Create TokenizerConfig from dict
                tokenizer_config = TokenizerConfig(**config_dict)
                
                # Debug: show what type of tokenizer we're loading
                if tokenizer_config.tokenizer_type == "custom":
                    if tokenizer_config.custom_builder:
                        print(f"  Custom tokenizer with builder: {tokenizer_config.custom_builder}")
                    elif tokenizer_config.custom_module and tokenizer_config.custom_class:
                        print(f"  Custom tokenizer with class: {tokenizer_config.custom_module}.{tokenizer_config.custom_class}")
                else:
                    print(f"  Tokenizer type: {tokenizer_config.tokenizer_type}")
                    print(f"  Name/path: {tokenizer_config.tokenizer_name_or_path}")
                
                # Auto-detect local_files_only for this tokenizer
                auto_local_only = cls._should_use_local_only(tokenizer_config, checkpoint_path)
                
                # Override if user specified
                if local_files_only is not None:
                    auto_local_only = local_files_only
                
                # Set local_files_only in config
                tokenizer_config.local_files_only = auto_local_only
                
                # Merge kwargs with config
                if kwargs:
                    for key, value in kwargs.items():
                        if hasattr(tokenizer_config, key):
                            setattr(tokenizer_config, key, value)
                
                # Build tokenizer using TokenizerConfig.build()
                return tokenizer_config.build()
            except Exception as e:
                print(f"  Warning: Failed to load tokenizer_config.json: {e}")
        
        # ================================================================
        # PRIORITY 2: config.yaml (MainConfig style)
        # ================================================================
        config_path = checkpoint_path / "config.yaml"
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # Check if config has tokenizer section
                if "tokenizer" in config_dict:
                    tokenizer_dict = config_dict["tokenizer"]
                    print(f"  Found tokenizer section in config.yaml")
                    print(f"  Tokenizer type: {tokenizer_dict.get('tokenizer_type', 'unknown')}")
                    print(f"  Name/path: {tokenizer_dict.get('tokenizer_name_or_path', 'unknown')}")
                    
                    tokenizer_config = TokenizerConfig(**tokenizer_dict)
                    
                    # Auto-detect local_files_only
                    auto_local_only = cls._should_use_local_only(tokenizer_config, checkpoint_path)
                    if local_files_only is not None:
                        auto_local_only = local_files_only
                    tokenizer_config.local_files_only = auto_local_only
                    
                    return tokenizer_config.build()
                elif "model" in config_dict and "tokenizer_name_or_path" in config_dict["model"]:
                    # Some configs store tokenizer info in model section
                    tokenizer_name = config_dict["model"].get("tokenizer_name_or_path")
                    if tokenizer_name:
                        print(f"  Found tokenizer_name_or_path in model section: {tokenizer_name}")
                        # Auto-detect: check if this is a local path
                        tokenizer_path = Path(tokenizer_name)
                        auto_local_only = tokenizer_path.exists()
                        if local_files_only is not None:
                            auto_local_only = local_files_only
                        
                        tokenizer_config = TokenizerConfig(
                            tokenizer_type="auto",
                            tokenizer_name_or_path=tokenizer_name,
                            local_files_only=auto_local_only,
                        )
                        return tokenizer_config.build()
            except Exception as e:
                print(f"  Warning: Failed to load config.yaml: {e}")
        
        # ================================================================
        # PRIORITY 3: config.json (HF standard format)
        # ================================================================
        config_json_path = checkpoint_path / "config.json"
        if config_json_path.exists():
            print(f"Checking config.json for tokenizer info...")
            try:
                with open(config_json_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Look for tokenizer info in config.json
                tokenizer_name = None
                if "tokenizer_name_or_path" in config_dict:
                    tokenizer_name = config_dict["tokenizer_name_or_path"]
                elif "_name_or_path" in config_dict:
                    # Sometimes tokenizer info is in _name_or_path
                    tokenizer_name = config_dict["_name_or_path"]
                
                if tokenizer_name and tokenizer_name != "gpt2":
                    print(f"  Found tokenizer_name_or_path in config.json: {tokenizer_name}")
                    
                    # Auto-detect: check if this is a local path
                    tokenizer_path = Path(tokenizer_name)
                    auto_local_only = tokenizer_path.exists() or (checkpoint_path / tokenizer_name).exists()
                    if local_files_only is not None:
                        auto_local_only = local_files_only
                    
                    tokenizer_config = TokenizerConfig(
                        tokenizer_type="auto",
                        tokenizer_name_or_path=tokenizer_name,
                        local_files_only=auto_local_only,
                        **kwargs,
                    )
                    return tokenizer_config.build()
            except Exception as e:
                print(f"  Warning: Failed to load config.json: {e}")
        
        # ================================================================
        # PRIORITY 4: Auto-detect from tokenizer files
        # ================================================================
        if cls._has_tokenizer_files(checkpoint_path):
            print(f"Found tokenizer files in {checkpoint_path}, auto-detecting...")
            
            # Check for SentencePiece
            if (checkpoint_path / "spm.model").exists():
                print(f"  Detected SentencePiece tokenizer (spm.model)")
                # SentencePiece is always local
                auto_local_only = True
                if local_files_only is not None:
                    auto_local_only = local_files_only
                    
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="sentencepiece",
                    tokenizer_model_path=str(checkpoint_path / "spm.model"),
                    local_files_only=auto_local_only,
                    **kwargs,
                )
                return tokenizer_config.build()
            
            # Check for HuggingFace tokenizer files
            hf_files = ["tokenizer.json", "vocab.json", "merges.txt"]
            if any((checkpoint_path / f).exists() for f in hf_files):
                print(f"  Detected HuggingFace tokenizer files")
                # Local HF tokenizer files exist
                auto_local_only = True
                if local_files_only is not None:
                    auto_local_only = local_files_only
                    
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="auto",
                    tokenizer_name_or_path=str(checkpoint_path),
                    local_files_only=auto_local_only,
                    **kwargs,
                )
                return tokenizer_config.build()
        
        # ================================================================
        # PRIORITY 5: Custom builder from environment variable
        # ================================================================
        custom_builder_env = os.environ.get("KILAT_TOKENIZER_BUILDER")
        if custom_builder_env:
            print(f"Using custom tokenizer builder from env: {custom_builder_env}")
            tokenizer_config = TokenizerConfig(
                tokenizer_type="custom",
                custom_builder=custom_builder_env,
                local_files_only=local_files_only if local_files_only is not None else False,
                **kwargs,
            )
            return tokenizer_config.build()
        
        # ================================================================
        # FALLBACK: GPT-2 tokenizer (last resort)
        # ================================================================
        print("No tokenizer config found, using default GPT-2 tokenizer")
        # For fallback, auto-detect if GPT-2 is cached
        auto_local_only = cls._is_gpt2_cached()
        if local_files_only is not None:
            auto_local_only = local_files_only
            
        return cls._fallback_tokenizer(auto_local_only, **kwargs)
    
    @classmethod
    def _should_use_local_only(cls, tokenizer_config: TokenizerConfig, checkpoint_path: Path) -> bool:
        """
        Auto-detect whether to use local_files_only based on tokenizer configuration.
        
        Returns:
            True if tokenizer files exist locally, False if needs download.
        """
        # Custom tokenizers are always "local" (defined by code)
        if tokenizer_config.tokenizer_type == "custom":
            return True
        
        # SentencePiece always uses local model file
        if tokenizer_config.tokenizer_type == "sentencepiece":
            if tokenizer_config.tokenizer_model_path:
                return Path(tokenizer_config.tokenizer_model_path).exists()
            return False
        
        # For HuggingFace tokenizers (gpt2, auto)
        tokenizer_path = Path(tokenizer_config.tokenizer_name_or_path)
        
        # Check if it's a local directory
        if tokenizer_path.exists():
            return True
        
        # Check if it's a file in checkpoint directory
        if (checkpoint_path / tokenizer_config.tokenizer_name_or_path).exists():
            return True
        
        # Check if it's in HuggingFace cache
        if cls._is_model_cached(tokenizer_config.tokenizer_name_or_path):
            return True
        
        # Otherwise, needs download
        return False
    
    @classmethod
    def _is_model_cached(cls, model_name: str) -> bool:
        """Check if a HuggingFace model is already cached locally."""
        try:
            from transformers import AutoTokenizer
            # Try to load with local_files_only=True
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
            return True
        except Exception:
            return False
    
    @classmethod
    def _is_gpt2_cached(cls) -> bool:
        """Check if GPT-2 tokenizer is cached locally."""
        return cls._is_model_cached("gpt2")
    
    @classmethod
    def _has_tokenizer_files(cls, path: Path) -> bool:
        """Check if directory contains tokenizer files."""
        tokenizer_files = [
            "tokenizer.json", "vocab.json", "merges.txt",
            "spm.model", "tokenizer_config.json", "tokenizer.model"
        ]
        return any((path / f).exists() for f in tokenizer_files)
    
    @classmethod
    def _fallback_tokenizer(cls, local_files_only: bool = False, **kwargs) -> Any:
        """Fallback to default GPT-2 tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            # Try with local_files_only first, fallback to download if needed
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "gpt2",
                    local_files_only=local_files_only,
                    **kwargs,
                )
            except Exception as e:
                if local_files_only:
                    # Local only failed, try downloading
                    print(f"  GPT-2 not found locally, downloading from HuggingFace...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        "gpt2",
                        local_files_only=False,
                        **kwargs,
                    )
                else:
                    raise e
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Fallback tokenizer: GPT-2, vocab size: {len(tokenizer)}")
            return tokenizer
        except Exception as e:
            raise FileNotFoundError(
                f"No tokenizer found. Fallback to GPT-2 also failed: {e}"
            ) from e


def save_tokenizer(
    tokenizer: Any,
    save_directory: Union[str, Path],
    tokenizer_config: Optional[TokenizerConfig] = None,
) -> None:
    """
    Save tokenizer to a directory for later use with AutoTokenizer.from_pretrained().
    
    This function saves both the tokenizer instance AND the configuration
    so that the tokenizer can be faithfully reconstructed later.
    
    For HuggingFace tokenizers: saves via save_pretrained()
    For SentencePiece tokenizers: saves the .model file
    For custom tokenizers: saves config only (user must provide builder)
    
    Parameters
    ----------
    tokenizer : Any
        Tokenizer instance to save (HuggingFace, SentencePiece, or custom).
    save_directory : str or Path
        Path to the directory where tokenizer will be saved.
    tokenizer_config : Optional[TokenizerConfig]
        Tokenizer configuration. If None, attempts to create from tokenizer.
    
    Example
    -------
        >>> from generation.auto_tokenizer import save_tokenizer
        >>>
        >>> # Save tokenizer to checkpoint
        >>> save_tokenizer(tokenizer, "./checkpoints/checkpoint-best")
        >>>
        >>> # Later, load with:
        >>> tokenizer = AutoTokenizer.from_pretrained("./checkpoints/checkpoint-best")
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create tokenizer config if not provided
    if tokenizer_config is None:
        tokenizer_config = _create_tokenizer_config_from_tokenizer(tokenizer)
    
    # Save tokenizer config as JSON (for easy loading)
    tokenizer_config.to_yaml(save_path / "tokenizer_config.json")
    print(f"Saved tokenizer config to {save_path / 'tokenizer_config.json'}")
    
    # For custom tokenizers, we only save the config
    if tokenizer_config.tokenizer_type == "custom":
        print(f"Custom tokenizer config saved. Tokenizer will be rebuilt via builder function.")
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_path)
            print(f"  Also saved tokenizer files via save_pretrained")
        return
    
    # Save tokenizer files for non-custom types
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_path)
        print(f"Saved HuggingFace tokenizer to {save_path}")
    elif hasattr(tokenizer, "Save"):
        tokenizer.Save(str(save_path / "spm.model"))
        print(f"Saved SentencePiece tokenizer to {save_path}")
    else:
        print(f"Warning: Tokenizer {type(tokenizer).__name__} "
              "does not have save_pretrained or Save method. "
              "Only config was saved.")


def _create_tokenizer_config_from_tokenizer(tokenizer: Any) -> TokenizerConfig:
    """
    Infer TokenizerConfig from tokenizer instance.
    
    This attempts to extract tokenizer configuration from an existing
    tokenizer object, which is useful when saving a tokenizer that was
    created manually rather than from a config.
    """
    tokenizer_type = "auto"
    tokenizer_name_or_path = "gpt2"
    tokenizer_model_path = None
    
    # Detect SentencePiece tokenizer
    if hasattr(tokenizer, "GetPieceSize") and hasattr(tokenizer, "Load"):
        tokenizer_type = "sentencepiece"
        tokenizer_model_path = getattr(tokenizer, "model_file", None)
    
    # Detect HuggingFace tokenizer
    elif hasattr(tokenizer, "name_or_path"):
        tokenizer_name_or_path = tokenizer.name_or_path
        # Check if it's a local path or a Hub model
        if Path(tokenizer_name_or_path).exists():
            tokenizer_type = "auto"
    
    # Detect if it's a custom tokenizer with vocab_size attribute
    elif hasattr(tokenizer, "vocab_size") and not hasattr(tokenizer, "name_or_path"):
        tokenizer_type = "custom"
    
    return TokenizerConfig(
        tokenizer_type=tokenizer_type,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_model_path=tokenizer_model_path,
        use_fast=getattr(tokenizer, "is_fast", True),
        local_files_only=True,
    )


def get_vocab_size(tokenizer: Any) -> int:
    """
    Get vocabulary size from tokenizer instance.
    
    Handles different tokenizer types uniformly:
    - HuggingFace tokenizers: tokenizer.vocab_size or len(tokenizer)
    - SentencePiece: tokenizer.GetPieceSize()
    - Custom: tokenizer.vocab_size or len(tokenizer.vocab)
    
    Example
    -------
        >>> from generation.auto_tokenizer import get_vocab_size
        >>> vocab_size = get_vocab_size(tokenizer)
    """
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    elif hasattr(tokenizer, "get_vocab_size"):
        return tokenizer.get_vocab_size()
    elif hasattr(tokenizer, "vocab"):
        return len(tokenizer.vocab)
    elif hasattr(tokenizer, "GetPieceSize"):
        return tokenizer.GetPieceSize()
    else:
        # Last resort: try len()
        try:
            return len(tokenizer)
        except TypeError:
            raise ValueError(f"Cannot determine vocabulary size for {type(tokenizer).__name__}")