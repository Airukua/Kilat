"""Model loading entrypoint (canonical location; formerly ``kilat.utils.report.load_model``)."""
from __future__ import annotations

import sys
from pathlib import Path

from kilat.configs.model_config import KilatConfig
from kilat.models.transformer import KilatTransformer


def load_model(
    checkpoint: str | None = None,
    config_path: str | None = None,
    device: str = "cpu"
) -> KilatTransformer:
    """
    Load KilatTransformer model from checkpoint or config.
    
    WHY: Provides flexible model loading for different use cases:
        - From trained checkpoint (with weights) → inference or continued training
        - From config only (random init) → training from scratch
        - With neither → default config (quick testing)
    
    Loading logic:
        1. If checkpoint provided → load pretrained model with weights
        2. Else if config_path provided → create model from config (random weights)
        3. Else → create model with default config (minimal for testing)
    
    Args:
        checkpoint: Path to directory containing model.safetensors/pytorch_model.bin
        config_path: Path to config.yaml or config.json
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        Loaded KilatTransformer model (already moved to device)
    
    Raises:
        ValueError: If both checkpoint and config_path are provided (ambiguous)
        FileNotFoundError: If checkpoint or config file doesn't exist
    
    Example:
        >>> # Load trained model
        >>> model = load_model(checkpoint="./checkpoints/best")
        >>> 
        >>> # Create new model from config
        >>> model = load_model(config_path="./configs/small.yaml")
        >>> 
        >>> # Default model for testing
        >>> model = load_model()
    """
    if checkpoint and config_path:
        raise ValueError(
            "Cannot specify both --checkpoint and --config. "
            "Use --checkpoint to load a pretrained model with weights, "
            "or --config to create a new model from configuration."
        )

    if checkpoint:
        # Load pretrained model (weights + config from checkpoint directory)
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        model = KilatTransformer.from_pretrained(str(checkpoint_path))

    elif config_path:
        # Create model from config only (random weights, for training from scratch)
        config_file = Path(config_path).resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Load config in the appropriate format
        if config_file.suffix in {".yaml", ".yml"}:
            cfg = KilatConfig.from_yaml(str(config_file))
        else:
            # Assume JSON format (from save_pretrained)
            cfg = KilatConfig.from_pretrained(str(config_file))
        model = KilatTransformer(cfg)

    else:
        # Default model with standard config (useful for quick testing)
        print("Warning: No checkpoint or config specified. Using default KilatConfig.", file=sys.stderr)
        cfg = KilatConfig()
        model = KilatTransformer(cfg)

    # Move model to target device
    model = model.to(device)
    return model
