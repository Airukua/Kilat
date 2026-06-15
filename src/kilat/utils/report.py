from __future__ import annotations
from collections import defaultdict
import torch.nn as nn
from pathlib import Path
from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
import sys

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count total number of parameters in the model.
    
    WHY: Understanding model size is crucial for estimating memory requirements
    and training costs. This function provides a quick way to get parameter counts
    without external profiling tools.
    
    Args:
        model: KilatTransformer instance
        trainable_only: If True, count only parameters with requires_grad=True.
                       Useful for fine-tuning scenarios where some layers are frozen.
    
    Returns:
        Total parameter count as integer
    
    Example:
        >>> model = KilatTransformer(config)
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_breakdown(model: nn.Module):
    """
    Return detailed breakdown of model parameters including shared tensors.
    
    WHY: Many transformer components share weights (e.g., embedding and lm_head
    in weight-tying architectures). This function identifies shared tensors
    and groups them together, preventing double-counting in manual analysis.
    
    WHAT IT DETECTS:
        - Standard parameters: name, shape, size, trainable status
        - Shared tensors: multiple parameter names pointing to same memory
        - Shared groups: which parameters are tied together
    
    Returns:
        Tuple of (rows, shared_groups):
            rows: List of dicts with per-parameter info
            shared_groups: List of lists, each containing names of shared parameters
    
    Example:
        >>> rows, shared = parameter_breakdown(model)
        >>> for row in rows:
        ...     print(f"{row['name']}: {row['numel']:,} params" + 
        ...           (" (shared)" if row['shared'] else ""))
        >>> print(f"Shared groups: {shared}")
    """
    rows = []
    seen_ptrs = set()
    shared = defaultdict(list)

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        is_shared = ptr in seen_ptrs
        if is_shared:
            shared[ptr].append(name)
        else:
            seen_ptrs.add(ptr)

        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "numel": param.numel(),
                "trainable": param.requires_grad,
                "shared": is_shared,
            }
        )

    shared_groups = []
    if shared:
        # Re-scan to include the original owner of each shared tensor.
        # A tensor may appear multiple times; we need all names pointing to it.
        ptr_to_names = defaultdict(list)
        for name, param in model.named_parameters():
            ptr_to_names[param.data_ptr()].append(name)
        shared_groups = [names for names in ptr_to_names.values() if len(names) > 1]

    return rows, shared_groups


def load_model(
    checkpoint: str | None = None,
    config_path: str | None = None,
    device: str = "cpu"
) -> 'KilatTransformer':
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
    from kilat.arc.model import KilatTransformer
    from kilat.configs.model_config import KilatConfig
    
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
