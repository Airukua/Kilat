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
            3. Default GPT-2 tokenizer (fallback)
        
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
        
        Example
        -------
            >>> # Load tokenizer from checkpoint
            >>> tokenizer = AutoTokenizer.from_pretrained("./checkpoints/checkpoint-best")
            >>>
            >>> # For custom tokenizer, the builder function is called automatically
            >>> tokens = tokenizer.encode("Hello world")
            >>> text = tokenizer.decode(tokens)
        """
        checkpoint_path = Path(pretrained_model_name_or_path)
        
        # Priority 1: tokenizer_config.json (saved by Trainer)
        tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            print(f"Loading tokenizer config from: {tokenizer_config_path}")
            tokenizer_config = TokenizerConfig.from_yaml(tokenizer_config_path)
            
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
        
        # Priority 2: config.yaml (MainConfig style)
        config_path = checkpoint_path / "config.yaml"
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            from configs.main_config import MainConfig
            main_config = MainConfig.from_yaml(config_path)
            
            if hasattr(main_config, 'build_tokenizer'):
                print(f"  Using MainConfig.build_tokenizer()")
                return main_config.build_tokenizer()
            
            print(f"  Using MainConfig.tokenizer.build()")
            return main_config.tokenizer.build()
        
        # Priority 3: Look for tokenizer files directly
        if cls._has_tokenizer_files(checkpoint_path):
            print(f"Found tokenizer files in {checkpoint_path}, attempting to load...")
            
            if (checkpoint_path / "spm.model").exists():
                print(f"  Detected SentencePiece tokenizer")
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="sentencepiece",
                    tokenizer_model_path=str(checkpoint_path / "spm.model"),
                    local_files_only=local_files_only,
                    **kwargs,
                )
            else:
                print(f"  Detected HuggingFace tokenizer files")
                tokenizer_config = TokenizerConfig(
                    tokenizer_type="auto",
                    tokenizer_name_or_path=str(checkpoint_path),
                    local_files_only=local_files_only,
                    **kwargs,
                )
            return tokenizer_config.build()
        
        # Priority 4: Check for custom builder in environment variable
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
        
        # Fallback: try default GPT-2 tokenizer
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
    
    # Save tokenizer config as YAML
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
    """Infer TokenizerConfig from tokenizer instance."""
    tokenizer_type = "auto"
    tokenizer_name_or_path = "gpt2"
    tokenizer_model_path = None
    
    if hasattr(tokenizer, "GetPieceSize") and hasattr(tokenizer, "Load"):
        tokenizer_type = "sentencepiece"
    elif hasattr(tokenizer, "name_or_path"):
        tokenizer_name_or_path = tokenizer.name_or_path
        if Path(tokenizer_name_or_path).exists():
            tokenizer_type = "auto"
    
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
    
    Handles different tokenizer types uniformly.
    
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
        return len(tokenizer)