"""
Model and tokenizer loading utilities.

Supports multiple checkpoint formats:
- Standard HuggingFace directory with config.json and model weights (safetensors or bin)
- YAML configuration (config.yaml) + weights in same directory
- MainConfig YAML (full_config.yaml) which contains both model and training settings

Why support YAML? During development, it's convenient to keep model configs
in a human‑editable YAML file.  The `MainConfig` also allows storing training
hyperparameters alongside the model for reproducibility.
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings
import torch

from data.tokenizer import KilatTokenizer
from arc.model import KilatTransformer
from utils.config import KilatConfig, MainConfig, TokenizerConfig


def _checkpoint_has_tokenizer_files(path: Path) -> bool:
    """Return True if a checkpoint directory already contains tokenizer artifacts."""
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
    )
    return any((path / name).exists() for name in tokenizer_files)


def _load_tokenizer_config(checkpoint_path: Path) -> Optional[TokenizerConfig]:
    """Load tokenizer metadata if the checkpoint saved it alongside the model."""
    full_config_path = checkpoint_path / "full_config.yaml"
    if full_config_path.exists():
        return MainConfig.from_yaml(full_config_path).tokenizer

    tokenizer_config_path = checkpoint_path / "tokenizer_config.yaml"
    if tokenizer_config_path.exists():
        return TokenizerConfig.from_yaml(tokenizer_config_path)

    return None


def _resolve_tokenizer_source(
    checkpoint_path: Path,
    config: KilatConfig,
    explicit_tokenizer_path: Optional[str],
) -> tuple[str, dict]:
    """
    Resolve the tokenizer source and init kwargs.

    Preference order:
    1. Explicit --tokenizer_path
    2. tokenizer_config.yaml / full_config.yaml saved with the checkpoint
    3. Tokenizer files already present in the checkpoint directory
    4. Bundled GPT-2 tokenizer when the checkpoint matches the standard 50,257 vocab
    """
    if explicit_tokenizer_path:
        return explicit_tokenizer_path, {}

    tokenizer_config = _load_tokenizer_config(checkpoint_path)
    if tokenizer_config is not None:
        if tokenizer_config.tokenizer_type == "sentencepiece":
            model_path = tokenizer_config.tokenizer_model_path or tokenizer_config.tokenizer_name_or_path
            return model_path, {"tokenizer_type": "sentencepiece"}

        return tokenizer_config.tokenizer_name_or_path, {
            "use_fast": tokenizer_config.use_fast,
            "local_files_only": tokenizer_config.local_files_only,
        }

    if _checkpoint_has_tokenizer_files(checkpoint_path):
        return str(checkpoint_path), {}

    bundled_gpt2 = Path(__file__).resolve().parents[2] / "tokenizers" / "gpt2"
    if bundled_gpt2.exists() and config.vocab_size == 50257:
        warnings.warn(
            "No tokenizer files were found in the checkpoint directory; "
            f"falling back to bundled GPT-2 tokenizer at {bundled_gpt2}. "
            "Use --tokenizer_path to override this if the checkpoint was trained "
            "with a different tokenizer.",
            UserWarning,
            stacklevel=2,
        )
        return str(bundled_gpt2), {"local_files_only": True}

    raise FileNotFoundError(
        "Could not find tokenizer files in the checkpoint directory. "
        "Pass --tokenizer_path to point at a tokenizer directory, or save "
        "tokenizer artifacts alongside the model checkpoint."
    )


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    use_yaml_config: bool = False,
    tokenizer_path: Optional[str] = None,
) -> Tuple[KilatTransformer, KilatTokenizer]:
    """
    Load a KilatTransformer model and its tokenizer from a checkpoint.

    Args:
        checkpoint_path: Directory containing model files
        device: Target device (auto-detects CUDA if None)
        use_yaml_config: Force YAML loading (bypasses auto-detection)

    Returns:
        Loaded model in eval mode with tokenizer

    Loading strategy priority (highest to lowest):
        1. If use_yaml_config=True: Load config from config.yaml
        2. Else if full_config.yaml exists: Load MainConfig (model + training params)
        3. Else: Standard HF config.json
        
    Why this order? full_config.yaml is the most complete (preserves training
    hyperparameters for reproducibility). YAML takes precedence over JSON
    because it's human-editable during development. JSON is the fallback for
    compatibility with standard HF checkpoints.
    """
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Determine model configuration ---
    # Priority ordering supports development workflow: YAML for active tweaking,
    # MainConfig for checkpointing experiments, JSON for HF ecosystem compat.
    if use_yaml_config:
        yaml_path = checkpoint_path / "config.yaml" if checkpoint_path.is_dir() else checkpoint_path
        config = KilatConfig.from_yaml(yaml_path)
    elif (checkpoint_path / "full_config.yaml").exists():
        full_config = MainConfig.from_yaml(checkpoint_path / "full_config.yaml")
        config = full_config.model
    else:
        # Standard HF format - used when sharing models with non-Kilat code
        config = KilatConfig.from_pretrained(str(checkpoint_path))

    # --- Load tokenizer ---
    # The checkpoint directory may only contain model weights/config, so we
    # resolve the tokenizer source separately instead of assuming it lives there.
    resolved_tokenizer_path, tokenizer_kwargs = _resolve_tokenizer_source(
        checkpoint_path,
        config,
        tokenizer_path,
    )
    tokenizer = KilatTokenizer.from_pretrained(
        str(resolved_tokenizer_path),
        **tokenizer_kwargs,
    )

    # --- Instantiate model ---
    model = KilatTransformer(config)

    # --- Load weights with fallback strategies ---
    # Three loading methods in order of preference:
    # 1. HF from_pretrained (handles sharded weights, safetensors, index.json)
    # 2. Legacy pytorch_model.bin (single file, pre-safetensors)
    # 3. Error if neither exists
    if checkpoint_path.is_dir():
        try:
            # Preferred: Handles modern checkpoint formats transparently
            model = KilatTransformer.from_pretrained(str(checkpoint_path))
        except Exception:
            # Fallback: Older checkpoints from early development
            state_dict_path = checkpoint_path / "pytorch_model.bin"
            if state_dict_path.exists():
                # weights_only=True prevents arbitrary code execution from pickle
                # Required for security when loading untrusted checkpoints
                state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint path must be a directory, got {checkpoint_path}")

    # Best-effort warning if tokenizer and model disagree on vocabulary size.
    # This is helpful when a checkpoint is paired with a different tokenizer
    # than the one used during training.
    tokenizer_vocab_size = tokenizer.vocab_size
    model_vocab_size = getattr(config, "vocab_size", None)
    if model_vocab_size is not None and tokenizer_vocab_size != model_vocab_size:
        warnings.warn(
            f"Loaded tokenizer vocab size ({tokenizer_vocab_size}) does not match "
            f"model vocab_size ({model_vocab_size}). Decoding and generation may "
            "behave incorrectly if this checkpoint was trained with a different tokenizer.",
            UserWarning,
            stacklevel=2,
        )

    model.to(device)
    model.eval()  # Disables dropout - critical for deterministic inference

    # User feedback for debugging and performance expectations
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Device: {device}")

    return model, tokenizer
