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
from configs.tokenizer_config import TokenizerConfig


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
        local_files_only: bool = True,
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
        
        Parameters
        ----------
        pretrained_model_name_or_path : str or Path
            Path to the checkpoint directory containing tokenizer_config.json
            or config.yaml.
        local_files_only : bool
            If True, only use local files (no download from Hub).
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
                    return tokenizer_config.build()
                elif "model" in config_dict and "tokenizer_name_or_path" in config_dict["model"]:
                    # Some configs store tokenizer info in model section
                    tokenizer_name = config_dict["model"].get("tokenizer_name_or_path")
                    if tokenizer_name:
                        print(f"  Found tokenizer_name_or_path in model section: {tokenizer_name}")
                        tokenizer_config = TokenizerConfig(
                            tokenizer_type="auto",
                            tokenizer_name_or_path=tokenizer_name,
                            local_files_only=local_files_only,
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
                    tokenizer_config = TokenizerConfig(
                        tokenizer_type="auto",
                        tokenizer_name_or_path=tokenizer_name,
                        local_files_only=local_files_only,
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
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="sentencepiece",
                    tokenizer_model_path=str(checkpoint_path / "spm.model"),
                    local_files_only=local_files_only,
                    **kwargs,
                )
                return tokenizer_config.build()
            
            # Check for HuggingFace tokenizer files
            hf_files = ["tokenizer.json", "vocab.json", "merges.txt"]
            if any((checkpoint_path / f).exists() for f in hf_files):
                print(f"  Detected HuggingFace tokenizer files")
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="auto",
                    tokenizer_name_or_path=str(checkpoint_path),
                    local_files_only=local_files_only,
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
                local_files_only=local_files_only,
                **kwargs,
            )
            return tokenizer_config.build()
        
        # ================================================================
        # FALLBACK: GPT-2 tokenizer (last resort)
        # ================================================================
        print("No tokenizer config found, using default GPT-2 tokenizer")
        return cls._fallback_tokenizer(local_files_only, **kwargs)
    
    @classmethod
    def _has_tokenizer_files(cls, path: Path) -> bool:
        """Check if directory contains tokenizer files."""
        tokenizer_files = [
            "tokenizer.json", "vocab.json", "merges.txt",
            "spm.model", "tokenizer_config.json", "tokenizer.model"
        ]
        return any((path / f).exists() for f in tokenizer_files)
    
    @classmethod
    def _fallback_tokenizer(cls, local_files_only: bool = True, **kwargs) -> Any:
        """Fallback to default GPT-2 tokenizer."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                local_files_only=local_files_only,
                **kwargs,
            )
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