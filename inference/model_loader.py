"""
Model and tokenizer loading utilities.

Supports multiple checkpoint formats:
- Standard HuggingFace directory with config.json and model weights (safetensors or bin)
- YAML configuration (config.yaml) + weights in same directory
- FullConfig YAML (full_config.yaml) which contains both model and training settings

Why support YAML? During development, it's convenient to keep model configs
in a human‑editable YAML file.  The `FullConfig` also allows storing training
hyperparameters alongside the model for reproducibility.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer

from model import KilatTransformerHF
from utils.config import KilatConfig, FullConfig


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    use_yaml_config: bool = False,
) -> Tuple[KilatTransformerHF, AutoTokenizer]:
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
        2. Else if full_config.yaml exists: Load FullConfig (model + training params)
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
    # Tokenizer is always loaded from HF format (tokenizer.json, vocab.json, etc.)
    # Design assumption: Tokenizer files co-locate with model weights or in parent
    # directory. This handles both direct directory paths and paths pointing to
    # config files (e.g., /path/to/config.yaml - use parent for tokenizer).
    if checkpoint_path.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path.parent))

    # Critical: Many pretrained tokenizers (GPT-2, LLaMA) lack pad_token
    # Without this, padding batches would use token 0 (often <unk>), corrupting
    # attention. Using eos_token is safe because it's never used as padding
    # for real tokens (EOS appears only at sequence ends).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Determine model configuration ---
    # Priority ordering supports development workflow: YAML for active tweaking,
    # FullConfig for checkpointing experiments, JSON for HF ecosystem compat.
    if use_yaml_config:
        yaml_path = checkpoint_path / "config.yaml" if checkpoint_path.is_dir() else checkpoint_path
        config = KilatConfig.from_yaml(yaml_path)
    elif (checkpoint_path / "full_config.yaml").exists():
        full_config = FullConfig.from_yaml(checkpoint_path / "full_config.yaml")
        config = full_config.model
    else:
        # Standard HF format - used when sharing models with non-Kilat code
        config = KilatConfig.from_pretrained(str(checkpoint_path))

    # --- Instantiate model ---
    model = KilatTransformerHF(config)

    # --- Load weights with fallback strategies ---
    # Three loading methods in order of preference:
    # 1. HF from_pretrained (handles sharded weights, safetensors, index.json)
    # 2. Legacy pytorch_model.bin (single file, pre-safetensors)
    # 3. Error if neither exists
    if checkpoint_path.is_dir():
        try:
            # Preferred: Handles modern checkpoint formats transparently
            model = KilatTransformerHF.from_pretrained(str(checkpoint_path))
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

    model.to(device)
    model.eval()  # Disables dropout - critical for deterministic inference

    # User feedback for debugging and performance expectations
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Device: {device}")

    return model, tokenizer