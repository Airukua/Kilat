"""
Convert KilatTransformer checkpoint to Hugging Face compatible format.

WHY THIS CONVERTER EXISTS:
    KilatTrainer saves checkpoints in a custom format that includes:
    - Model weights (via save_pretrained for HF compatibility)
    - Training state (optimizer, scheduler, scaler, callback states)
    - Configuration (YAML or JSON)
    
    However, when users want to deploy the model for inference only, they don't need
    the training state. This converter extracts the pure model weights and saves them
    in a format that Hugging Face's transformers library can load directly via
    `AutoModel.from_pretrained()`. This eliminates the dependency on Kilat's training
    code for inference.

CONVERSION STRATEGY:
    The converter follows a priority-based approach to handle diverse checkpoint formats:
    
    Priority 1: Use user-provided config file (most reliable, user knows best)
    Priority 2: Infer config from checkpoint/config.yaml (human-readable)
    Priority 3: Infer config from checkpoint/config.json (HF standard)
    Priority 4: Infer config from training_args.json (may contain model hyperparams)
    Priority 5: Fall back to default KilatConfig (last resort, may not work)

    For weights:
    Priority 1: model.safetensors (recommended, secure, faster)
    Priority 2: pytorch_model.bin (standard HF format)
    Priority 3: training_state.pt (Kilat trainer format with model_state_dict)

ERROR HANDLING PHILOSOPHY:
    - Never crash silently: always provide actionable error messages
    - Fail gracefully: try all possible fallbacks before giving up
    - Provide troubleshooting guidance: tell users what to do when something fails
    - Use strict=False by default: allow conversion even with minor mismatches
    - Log warnings for suspicious but non-fatal issues

Assumptions:
    - The checkpoint directory contains at least one valid weight file
    - The model architecture matches the configuration (same vocab_size, n_embd, etc.)
    - For weight tying, lm_head.weight may be missing and should be copied from wte.weight

Edge Cases Handled:
    - DDP training prefixes ('module.') are automatically stripped
    - Tied weights (lm_head = wte) are resolved automatically
    - Missing config files fall back to defaults with warning
    - Corrupted weight files raise specific errors with recovery suggestions
    - CUDA memory errors suggest using CPU instead
    - Device-side assert errors point to vocabulary size mismatch

Performance:
    - Model is loaded on CPU by default to avoid GPU memory issues
    - Validation forward pass uses small dummy input (10 tokens) for quick check
    - Safe tensors format is used when available for faster loading

Example Usage:
    # Basic conversion (infer config from checkpoint)
    python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model
    
    # With explicit config file (recommended for reproducibility)
    python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model --config ./configs/model.yaml
    
    # Debug mode with verbose logging
    python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model -v
    
    # Skip validation for faster conversion (trust that model works)
    python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model --skip-validation
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import torch
import yaml

# Add project root to path for local imports.
# This allows the script to run from any directory while still finding Kilat modules.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import KilatConfig
from arc.model import KilatTransformer

# Configure logging for both console and file.
# The format includes timestamp, module name, severity, and message for easy debugging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions (following HF pattern)
# ============================================================================

class ConversionError(Exception):
    """Base exception for conversion errors."""
    pass


class ConfigNotFoundError(ConversionError):
    """Raised when configuration file is not found."""
    pass


class WeightFileNotFoundError(ConversionError):
    """Raised when weight file is not found in checkpoint directory."""
    pass


class WeightLoadingError(ConversionError):
    """Raised when weight loading fails with specific details."""
    pass


class ValidationError(ConversionError):
    """Raised when model validation fails after loading."""
    pass


# ============================================================================
# Configuration Loading (with graceful fallbacks)
# ============================================================================

def load_config_from_checkpoint(checkpoint_dir: Path) -> KilatConfig:
    """
    Load KilatConfig from checkpoint directory with priority ordering.
    
    WHY THIS EXISTS: Checkpoint directories may contain config in various formats
    depending on when the checkpoint was saved. This function tries all possible
    locations to find valid configuration, making the converter tolerant of
    different save formats.
    
    Priority (highest to lowest):
        1. config.yaml - Human-readable YAML, preferred for manual inspection
        2. config.json - HF standard JSON format, used by save_pretrained
        3. training_args.json - May embed model config inside training args
        4. Default config - Last resort, may produce mismatched model
    
    Edge Cases:
        - If config.yaml exists but is malformed, we log warning and continue
        - If training_args.json contains model config under 'model_config' key
        - If none found, return default config with warning (better than crashing)
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        KilatConfig instance
        
    Raises:
        ConfigNotFoundError: If no valid config found and no defaults usable
    """
    # Priority 1: config.yaml (human-readable, preferred)
    yaml_path = checkpoint_dir / "config.yaml"
    if yaml_path.exists():
        logger.info(f"Loading config from {yaml_path}")
        try:
            return KilatConfig.from_yaml(str(yaml_path))
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
            # Continue to next option instead of failing immediately

    # Priority 2: config.json (HF standard format)
    json_path = checkpoint_dir / "config.json"
    if json_path.exists():
        logger.info(f"Loading config from {json_path}")
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            # Clean HF internal fields that may cause issues
            config_dict.pop("_name_or_path", None)
            config_dict.pop("transformers_version", None)
            config_dict.pop("model_type", None)
            return KilatConfig(**config_dict)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load config.json: {e}")

    # Priority 3: training_args.json (may contain model config)
    training_args_path = checkpoint_dir / "training_args.json"
    if training_args_path.exists():
        logger.info(f"Attempting to load config from {training_args_path}")
        try:
            with open(training_args_path, 'r') as f:
                args_dict = json.load(f)
            if "model_config" in args_dict:
                return KilatConfig(**args_dict["model_config"])
            # Try to extract model parameters by matching known keys
            model_keys = ["vocab_size", "n_embd", "n_head", "n_layer", "n_embd"]
            if any(k in args_dict for k in model_keys):
                logger.info("Found model parameters in training_args.json")
                config_dict = {k: v for k, v in args_dict.items() if k in model_keys}
                if config_dict:
                    return KilatConfig(**config_dict)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load training_args.json: {e}")

    # Fallback to defaults with warning
    logger.warning(f"No valid config found in {checkpoint_dir}, using default KilatConfig")
    logger.warning("Model may not work correctly if default config doesn't match training")
    return KilatConfig()


# ============================================================================
# Weight Loading (handling tied weights and DDP prefixes)
# ============================================================================

def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean state dict by handling DDP prefixes and tied weights.
    
    WHY THIS EXISTS: 
        - Distributed Data Parallel (DDP) saves weights with 'module.' prefix
        - Weight tying saves lm_head and wte separately but they should share weights
        - The converter must normalise these to match the model's expected keys
    
    What this function does:
        1. Remove 'module.' prefix from all keys (from DDP training)
        2. Add lm_head.weight from wte.weight if missing (weight tying)
        3. Log warnings for missing critical keys
    
    Edge Cases:
        - If 'module.' appears in middle of key (not just prefix) – not handled, rare
        - If both wte.weight and lm_head.weight are missing – warning but continue
        - If only wte.weight exists and lm_head.weight expected – we copy it
    
    Returns:
        Cleaned state dict with proper key names
    """
    cleaned = {}
    
    # Handle DDP 'module.' prefix
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.'
        else:
            new_key = key
        cleaned[new_key] = value
    
    # Handle weight tying: if lm_head.weight missing but wte.weight exists
    if "lm_head.weight" not in cleaned and "wte.weight" in cleaned:
        logger.info("  lm_head.weight not found in checkpoint, using wte.weight (weight tying)")
        cleaned["lm_head.weight"] = cleaned["wte.weight"]
    
    # Handle potential tied weights keys mapping
    if "lm_head.weight" not in cleaned and "wte.weight" not in cleaned:
        logger.warning("  Neither lm_head.weight nor wte.weight found in checkpoint")
    
    return cleaned


def load_model_weights(
    model: KilatTransformer, 
    checkpoint_dir: Path, 
    device: str = "cpu",
    strict: bool = False
) -> Tuple[KilatTransformer, Dict[str, Union[list, str]]]:
    """
    Load model weights from checkpoint directory with robust error handling.
    
    WHY THREE FORMATS: Different training scripts save weights differently:
        - KilatTrainer saves training_state.pt with full training state
        - Hugging Face save_pretrained saves pytorch_model.bin or model.safetensors
        - This converter supports all to be compatible with various sources
    
    Priority order (tries each format until one succeeds):
        1. model.safetensors – fastest, secure, recommended
        2. pytorch_model.bin – standard HF format
        3. training_state.pt – Kilat trainer format
    
    Error Handling Strategy:
        - If safetensors requested but package not installed, explicit error with install command
        - If file exists but corrupt, specific error with recovery steps
        - If no weight file found, raise actionable error listing expected files
        - Use strict=False by default to allow missing/unexpected keys
    
    Args:
        model: Initialized KilatTransformer model
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load weights to (use 'cpu' for conversion)
        strict: If True, fail on missing/unexpected keys
        
    Returns:
        Tuple of (model, metadata dict with loading info)
        
    Raises:
        WeightFileNotFoundError: If no weight file found
        WeightLoadingError: If weights cannot be loaded with specific details
    """
    checkpoint_path = Path(checkpoint_dir)
    metadata = {"source": None, "missing_keys": [], "unexpected_keys": []}
    
    # Priority 1: safetensors (recommended, safer, faster)
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.exists():
        logger.info(f"Loading weights from {safetensors_path}")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            metadata["source"] = "safetensors"
        except ImportError:
            raise WeightLoadingError(
                "safetensors package not installed. Please install with: pip install safetensors"
            )
        except Exception as e:
            raise WeightLoadingError(f"Failed to load safetensors file: {e}")

    # Priority 2: pytorch_model.bin (HF standard)
    elif (checkpoint_path / "pytorch_model.bin").exists():
        bin_path = checkpoint_path / "pytorch_model.bin"
        logger.info(f"Loading weights from {bin_path}")
        try:
            state_dict = torch.load(str(bin_path), map_location=device)
            metadata["source"] = "pytorch_model.bin"
        except Exception as e:
            raise WeightLoadingError(f"Failed to load pytorch_model.bin: {e}")

    # Priority 3: training_state.pt (Kilat trainer format)
    elif (checkpoint_path / "training_state.pt").exists():
        training_state_path = checkpoint_path / "training_state.pt"
        logger.info(f"Loading weights from {training_state_path}")
        try:
            training_state = torch.load(str(training_state_path), map_location=device)
            # Extract state dict from training state (different possible structures)
            if "model_state_dict" in training_state:
                state_dict = training_state["model_state_dict"]
            elif "model" in training_state:
                state_dict = training_state["model"]
            else:
                state_dict = training_state
            metadata["source"] = "training_state.pt"
        except Exception as e:
            raise WeightLoadingError(f"Failed to load training_state.pt: {e}")

    else:
        raise WeightFileNotFoundError(
            f"No weight file found in {checkpoint_dir}. "
            f"Expected one of: model.safetensors, pytorch_model.bin, or training_state.pt"
        )
    
    # Clean and prepare state dict
    state_dict = _clean_state_dict(state_dict)
    logger.info(f"Loaded state dict with {len(state_dict)} keys")
    
    # Load with strict=False to capture missing/unexpected keys without crashing
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    metadata["missing_keys"] = missing_keys
    metadata["unexpected_keys"] = unexpected_keys
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    if strict and (missing_keys or unexpected_keys):
        raise WeightLoadingError(
            f"Strict loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}"
        )
    
    return model, metadata


# ============================================================================
# Model Validation
# ============================================================================

def validate_model(
    model: KilatTransformer,
    config: KilatConfig,
    device: str = "cpu",
    num_tokens: int = 10
) -> Dict[str, Any]:
    """
    Validate model with dummy input to catch initialization issues.
    
    WHY VALIDATION: After loading weights, we need to ensure the model can
    actually perform a forward pass. This catches:
        - Shape mismatches between config and weights
        - Device compatibility issues (CUDA vs CPU)
        - Vocabulary size problems (input tokens out of range)
        - Missing or corrupted weights
    
    The validation runs a single forward pass with:
        - Batch size = 1 (minimal memory)
        - Sequence length = min(num_tokens, vocab_size - 1) (prevents out-of-range)
        - No gradient computation (torch.no_grad() for speed)
    
    Edge Cases Handled:
        - CUDA out of memory -> suggests using CPU or smaller input
        - Device-side assert -> indicates vocab size mismatch, provides helpful hint
        - Any other RuntimeError -> wrapped with context for debugging
    
    Args:
        model: Loaded model
        config: Model configuration
        device: Device to run validation on
        num_tokens: Number of tokens in dummy input
        
    Returns:
        Dictionary with validation results including success status and logits shape
        
    Raises:
        ValidationError: If validation fails with actionable error message
    """
    logger.info("Validating model with dummy input...")
    
    try:
        model.eval()
        with torch.no_grad():
            # Create dummy input with safe values (within vocab range)
            # Using min() ensures we never request a token ID beyond vocab_size
            dummy_input = torch.randint(
                0, config.vocab_size, 
                (1, min(num_tokens, config.vocab_size - 1)), 
                device=device
            )
            output = model(dummy_input)
            
            # Check output shape matches expectations
            expected_shape = (1, dummy_input.shape[1], config.vocab_size)
            if output.logits.shape != expected_shape:
                raise ValidationError(
                    f"Output logits shape mismatch. Expected {expected_shape}, "
                    f"got {output.logits.shape}"
                )
            
            logger.info(f"Validation successful - logits shape: {output.logits.shape}")
            
            return {
                "success": True,
                "logits_shape": output.logits.shape,
                "has_loss": output.loss is not None
            }
            
    except RuntimeError as e:
        # Check for CUDA-specific errors with user-friendly messages
        if "CUDA out of memory" in str(e):
            raise ValidationError(
                f"CUDA out of memory during validation. Try running on CPU by setting "
                f"device='cpu' in the script or reducing num_tokens."
            ) from e
        elif "device-side assert" in str(e):
            raise ValidationError(
                f"CUDA device-side assert triggered. This often indicates a mismatch between "
                f"model vocabulary size ({config.vocab_size}) and the tokenizer or input values. "
                f"Try running on CPU first for more detailed error message."
            ) from e
        else:
            raise ValidationError(f"Validation failed: {e}") from e
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}") from e


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_checkpoint_to_hf(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    use_safetensors: bool = True,
    device: str = "cpu",
    strict: bool = False,
    skip_validation: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convert KilatTransformer checkpoint to Hugging Face format.
    
    This is the main entry point for programmatic conversion. It performs
    these steps in order:
        1. Load or infer configuration
        2. Create model with that configuration
        3. Load weights from checkpoint (trying multiple formats)
        4. Optionally validate with dummy forward pass
        5. Save model in HF format using save_pretrained
        6. Verify saved model can be loaded by AutoModel
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Output directory for converted model
        config_path: Optional path to config file (overrides checkpoint config)
        use_safetensors: Whether to save in safetensors format (recommended)
        device: Device to load model on ('cpu' recommended for conversion)
        strict: If True, fail on missing/unexpected keys
        skip_validation: If True, skip dummy forward pass validation
        verbose: If True, enable debug logging
        
    Returns:
        Dictionary with conversion metadata including success status,
        config source, weight source, missing/unexpected keys, and validation results
        
    Raises:
        ConversionError: If conversion fails at any step
    """
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    logger.info(f"Converting checkpoint from: {checkpoint_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    results = {
        "success": False,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "config_source": None,
        "weight_source": None,
        "missing_keys": [],
        "unexpected_keys": [],
        "validation": None
    }
    
    try:
        # Step 1: Load configuration
        if config_path:
            config_path = Path(config_path)
            logger.info(f"Using provided config from: {config_path}")
            if config_path.suffix in ['.yaml', '.yml']:
                config = KilatConfig.from_yaml(str(config_path))
            else:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = KilatConfig(**config_dict)
            results["config_source"] = str(config_path)
        else:
            config = load_config_from_checkpoint(checkpoint_dir)
            results["config_source"] = "inferred"
        
        logger.info(f"Model config: vocab_size={config.vocab_size}, "
                   f"n_embd={config.n_embd}, n_layer={config.n_layer}")
        
        # Step 2: Create model
        logger.info("Initializing model...")
        try:
            model = KilatTransformer(config)
            model = model.to(device)
            model.eval()
        except Exception as e:
            raise ConversionError(f"Failed to initialize model: {e}") from e
        
        # Step 3: Load weights
        logger.info("Loading weights...")
        try:
            model, weight_metadata = load_model_weights(model, checkpoint_dir, device, strict)
            results["weight_source"] = weight_metadata["source"]
            results["missing_keys"] = weight_metadata["missing_keys"]
            results["unexpected_keys"] = weight_metadata["unexpected_keys"]
        except (WeightFileNotFoundError, WeightLoadingError) as e:
            raise ConversionError(f"Weight loading failed: {e}") from e
        
        # Step 4: Validate model (optional)
        if not skip_validation:
            try:
                results["validation"] = validate_model(model, config, device)
            except ValidationError as e:
                logger.warning(f"Validation failed but continuing: {e}")
                results["validation"] = {"success": False, "error": str(e)}
        
        # Step 5: Save model
        logger.info(f"Saving model to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model.save_pretrained(
                str(output_dir),
                safe_serialization=use_safetensors,
            )
            # Also save YAML config for human readability
            config.to_yaml(output_dir / "config.yaml")
        except Exception as e:
            raise ConversionError(f"Failed to save model: {e}") from e
        
        # Step 6: Verify saved model (attempt to load back with HF AutoModel)
        logger.info("Verifying saved model...")
        try:
            from transformers import AutoModel
            loaded = AutoModel.from_pretrained(str(output_dir), trust_remote_code=True)
            logger.info("Model can be loaded with AutoModel!")
            results["auto_model_verification"] = True
        except Exception as e:
            logger.warning(f"AutoModel verification failed: {e}")
            logger.warning("Model saved but may require custom loading code")
            results["auto_model_verification"] = False
        
        # Log output structure with file sizes
        logger.info("\n" + "="*60)
        logger.info("CONVERSION COMPLETED SUCCESSFULLY")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("\nOutput structure:")
        total_size = 0
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                total_size += size
                size_mb = size / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.2f} MB)")
        logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        logger.info("="*60)
        
        results["success"] = True
        return results
        
    except ConversionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        logger.debug(traceback.format_exc())
        raise ConversionError(f"Conversion failed: {e}") from e


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """
    Command-line interface for checkpoint conversion.
    
    Parses command-line arguments and calls convert_checkpoint_to_hf().
    Provides helpful error messages and troubleshooting guidance on failure.
    """
    parser = argparse.ArgumentParser(
        description="Convert KilatTransformer checkpoint to Hugging Face format",
        epilog="""
Examples:
  %(prog)s -c ./checkpoints/checkpoint-best -o ./converted_model
  %(prog)s -c ./checkpoints/checkpoint-best -o ./converted_model --config ./configs/model.yaml
  %(prog)s -c ./checkpoints/checkpoint-best -o ./converted_model -v --skip-validation
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., ./checkpoints/checkpoint-best)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config_path",
        help="Optional path to config file (overrides checkpoint config)"
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Use pytorch_model.bin instead of safetensors (not recommended)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on (default: cpu)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing or unexpected keys in checkpoint"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dummy forward pass validation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    args = parser.parse_args()
    
    try:
        results = convert_checkpoint_to_hf(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            config_path=args.config_path,
            use_safetensors=not args.no_safetensors,
            device=args.device,
            strict=args.strict,
            skip_validation=args.skip_validation,
            verbose=args.verbose,
        )
        
        # Print summary on success
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Checkpoint: {results['checkpoint_dir']}")
        print(f"Output: {results['output_dir']}")
        print(f"Config source: {results['config_source']}")
        print(f"Weight source: {results['weight_source']}")
        if results.get("missing_keys"):
            print(f"Missing keys: {len(results['missing_keys'])}")
        if results.get("unexpected_keys"):
            print(f"Unexpected keys: {len(results['unexpected_keys'])}")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            traceback.print_exc()
        
        # Provide troubleshooting guidance
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("If conversion failed, try the following:")
        print("  1. Check that all required files exist in checkpoint directory")
        print("  2. Verify config.yaml or config.json matches model architecture")
        print("  3. Try running with --device cpu if using GPU")
        print("  4. Use --skip-validation to bypass forward pass check")
        print("  5. Enable verbose mode with -v for more details")
        print("="*60)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
