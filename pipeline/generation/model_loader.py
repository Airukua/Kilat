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
from utils.config import KilatConfig, MainConfig


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    use_yaml_config: bool = False,
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

    # --- Load tokenizer ---
    # Generation now uses the shared tokenizer wrapper from data/tokenizer.py so
    # inference and preprocessing stay in sync.
    tokenizer_path = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    tokenizer = KilatTokenizer.from_pretrained(str(tokenizer_path))

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
