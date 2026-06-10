#!/usr/bin/env python3
"""
Convert KilatTransformer checkpoint to Hugging Face compatible format.

WHY THIS CONVERTER EXISTS:
    KilatTrainer saves checkpoints in a custom format that includes:
    - Model weights (via save_pretrained for HF compatibility)
    - Training state (optimizer, scheduler, scaler, callback states)
    - Configuration (YAML or JSON)
    - Tokenizer files (for local/custom tokenizers)
    
    This converter extracts the pure model weights AND tokenizer, saving them
    in a format that Hugging Face's transformers library can load directly via
    `AutoModel.from_pretrained()` and `AutoTokenizer.from_pretrained()`.
    This eliminates the dependency on Kilat's training code for inference.

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

    For tokenizer:
    Priority 1: tokenizer files in checkpoint directory (saved by KilatTrainer)
    Priority 2: tokenizer_config.json in checkpoint (reference to Hub tokenizer)
    Priority 3: Create from config.tokenizer_name_or_path (fallback)

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
    - Tokenizer may be saved in checkpoint (spm.model, tokenizer_config.json, or HF files)

Edge Cases Handled:
    - DDP training prefixes ('module.') are automatically stripped
    - Tied weights (lm_head = wte) are resolved automatically
    - Missing config files fall back to defaults with warning
    - Corrupted weight files raise specific errors with recovery suggestions
    - CUDA memory errors suggest using CPU instead
    - Device-side assert errors point to vocabulary size mismatch
    - Tokenizer from Hub vs local tokenizer

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
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import torch
import yaml

# Add project root to path for local imports.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import KilatConfig
from arc.model import KilatTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
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
# Tokenizer Handling (NEW)
# ============================================================================

def copy_tokenizer_from_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    config: KilatConfig,
) -> bool:
    """
    Copy tokenizer files from checkpoint to output directory.
    
    WHY: When users save checkpoints with KilatTrainer, tokenizer files may be
    saved alongside the model (for local/custom tokenizers) or referenced via
    config (for Hub tokenizers). This function copies the tokenizer files to
    the output directory so the converted model is self-contained.
    
    Priority:
        1. If tokenizer files exist in checkpoint (spm.model, tokenizer.json, etc.)
        2. If tokenizer_config.json exists with reference to Hub tokenizer
        3. Skip (tokenizer will be loaded from Hub during inference)
    
    Returns:
        True if tokenizer was copied, False otherwise.
    """
    tokenizer_copied = False
    
    # Check for SentencePiece model file
    spm_path = checkpoint_dir / "spm.model"
    if spm_path.exists():
        shutil.copy2(spm_path, output_dir / "spm.model")
        logger.info(f"Copied SentencePiece tokenizer: {spm_path}")
        tokenizer_copied = True
    
    # Check for HuggingFace tokenizer files
    hf_tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
    for filename in hf_tokenizer_files:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            logger.info(f"Copied tokenizer file: {filename}")
            tokenizer_copied = True
    
    # Check for tokenizer_config.json (reference to Hub tokenizer)
    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists() and not tokenizer_copied:
        # This may contain reference to Hub tokenizer (e.g., "gpt2")
        # Copy it so inference can use the same config
        shutil.copy2(tokenizer_config_path, output_dir / "tokenizer_config.json")
        logger.info("Copied tokenizer_config.json (Hub tokenizer reference)")
        tokenizer_copied = True
    
    return tokenizer_copied


def save_tokenizer_config(
    output_dir: Path,
    config: KilatConfig,
    checkpoint_dir: Path,
) -> None:
    """
    Save tokenizer configuration for inference.
    
    Creates a tokenizer_config.json file that can be used to load the tokenizer
    during inference with `AutoTokenizer.from_pretrained(output_dir)`.
    """
    # First try to copy existing tokenizer config from checkpoint
    existing_config = checkpoint_dir / "tokenizer_config.json"
    if existing_config.exists():
        shutil.copy2(existing_config, output_dir / "tokenizer_config.json")
        logger.info(f"Using existing tokenizer config from checkpoint")
        return
    
    # Otherwise create a new one from model config
    # This is a fallback for checkpoints without tokenizer files
    tokenizer_config = {
        "tokenizer_type": "auto",  # default to auto
        "tokenizer_name_or_path": getattr(config, "tokenizer_name_or_path", "gpt2"),
        "use_fast": True,
        "local_files_only": False,
        "pad_token_id": getattr(config, "pad_token_id", 0),
        "eos_token_id": getattr(config, "eos_token_id", 2),
        "bos_token_id": getattr(config, "bos_token_id", 1),
    }
    
    # Also save as YAML for human readability
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    logger.info(f"Created tokenizer_config.json (from model config)")


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config_from_checkpoint(checkpoint_dir: Path) -> KilatConfig:
    """
    Load KilatConfig from checkpoint directory with priority ordering.
    
    Priority (highest to lowest):
        1. config.yaml - Human-readable YAML, preferred for manual inspection
        2. config.json - HF standard JSON format, used by save_pretrained
        3. training_args.json - May embed model config inside training args
        4. Default config - Last resort, may produce mismatched model
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        KilatConfig instance
    """
    # Priority 1: config.yaml
    yaml_path = checkpoint_dir / "config.yaml"
    if yaml_path.exists():
        logger.info(f"Loading config from {yaml_path}")
        try:
            return KilatConfig.from_yaml(str(yaml_path))
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")

    # Priority 2: config.json
    json_path = checkpoint_dir / "config.json"
    if json_path.exists():
        logger.info(f"Loading config from {json_path}")
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("transformers_version", None)
            config_dict.pop("model_type", None)
            return KilatConfig(**config_dict)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load config.json: {e}")

    # Priority 3: training_args.json
    training_args_path = checkpoint_dir / "training_args.json"
    if training_args_path.exists():
        logger.info(f"Attempting to load config from {training_args_path}")
        try:
            with open(training_args_path, 'r') as f:
                args_dict = json.load(f)
            if "model_config" in args_dict:
                return KilatConfig(**args_dict["model_config"])
            model_keys = ["vocab_size", "n_embd", "n_head", "n_layer"]
            if any(k in args_dict for k in model_keys):
                logger.info("Found model parameters in training_args.json")
                config_dict = {k: v for k, v in args_dict.items() if k in model_keys}
                if config_dict:
                    return KilatConfig(**config_dict)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load training_args.json: {e}")

    # Fallback to defaults
    logger.warning(f"No valid config found in {checkpoint_dir}, using default KilatConfig")
    logger.warning("Model may not work correctly if default config doesn't match training")
    return KilatConfig()


# ============================================================================
# Weight Loading
# ============================================================================

def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clean state dict by handling DDP prefixes and tied weights."""
    cleaned = {}
    
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        cleaned[new_key] = value
    
    if "lm_head.weight" not in cleaned and "wte.weight" in cleaned:
        logger.info("  lm_head.weight not found, using wte.weight (weight tying)")
        cleaned["lm_head.weight"] = cleaned["wte.weight"]
    
    if "lm_head.weight" not in cleaned and "wte.weight" not in cleaned:
        logger.warning("  Neither lm_head.weight nor wte.weight found in checkpoint")
    
    return cleaned


def load_model_weights(
    model: KilatTransformer, 
    checkpoint_dir: Path, 
    device: str = "cpu",
    strict: bool = False
) -> Tuple[KilatTransformer, Dict[str, Union[list, str]]]:
    """Load model weights from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    metadata = {"source": None, "missing_keys": [], "unexpected_keys": []}
    
    # Priority 1: safetensors
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.exists():
        logger.info(f"Loading weights from {safetensors_path}")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            metadata["source"] = "safetensors"
        except ImportError:
            raise WeightLoadingError("safetensors not installed. Run: pip install safetensors")
        except Exception as e:
            raise WeightLoadingError(f"Failed to load safetensors: {e}")

    # Priority 2: pytorch_model.bin
    elif (checkpoint_path / "pytorch_model.bin").exists():
        bin_path = checkpoint_path / "pytorch_model.bin"
        logger.info(f"Loading weights from {bin_path}")
        try:
            state_dict = torch.load(str(bin_path), map_location=device)
            metadata["source"] = "pytorch_model.bin"
        except Exception as e:
            raise WeightLoadingError(f"Failed to load pytorch_model.bin: {e}")

    # Priority 3: training_state.pt
    elif (checkpoint_path / "training_state.pt").exists():
        training_state_path = checkpoint_path / "training_state.pt"
        logger.info(f"Loading weights from {training_state_path}")
        try:
            training_state = torch.load(str(training_state_path), map_location=device)
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
            f"Expected: model.safetensors, pytorch_model.bin, or training_state.pt"
        )
    
    state_dict = _clean_state_dict(state_dict)
    logger.info(f"Loaded state dict with {len(state_dict)} keys")
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    metadata["missing_keys"] = missing_keys
    metadata["unexpected_keys"] = unexpected_keys
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    if strict and (missing_keys or unexpected_keys):
        raise WeightLoadingError(f"Strict loading failed. Missing: {missing_keys}")
    
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
    """Validate model with dummy input."""
    logger.info("Validating model with dummy input...")
    
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randint(
                0, config.vocab_size, 
                (1, min(num_tokens, config.vocab_size - 1)), 
                device=device
            )
            output = model(dummy_input)
            
            expected_shape = (1, dummy_input.shape[1], config.vocab_size)
            if output.logits.shape != expected_shape:
                raise ValidationError(f"Shape mismatch. Expected {expected_shape}, got {output.logits.shape}")
            
            logger.info(f"Validation successful - logits shape: {output.logits.shape}")
            
            return {
                "success": True,
                "logits_shape": output.logits.shape,
                "has_loss": output.loss is not None
            }
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise ValidationError("CUDA OOM. Try running on CPU with --device cpu")
        elif "device-side assert" in str(e):
            raise ValidationError(f"Vocab size mismatch. Model vocab: {config.vocab_size}")
        else:
            raise ValidationError(f"Validation failed: {e}")
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}")


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
    skip_tokenizer: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convert KilatTransformer checkpoint to Hugging Face format.
    
    Now includes tokenizer copying alongside model conversion.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Output directory for converted model
        config_path: Optional path to config file (overrides checkpoint config)
        use_safetensors: Whether to save in safetensors format
        device: Device to load model on ('cpu' recommended)
        strict: If True, fail on missing/unexpected keys
        skip_validation: If True, skip dummy forward pass validation
        skip_tokenizer: If True, skip copying tokenizer files
        verbose: If True, enable debug logging
        
    Returns:
        Dictionary with conversion metadata
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
        "tokenizer_copied": False,
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
            raise ConversionError(f"Failed to initialize model: {e}")
        
        # Step 3: Load weights
        logger.info("Loading weights...")
        try:
            model, weight_metadata = load_model_weights(model, checkpoint_dir, device, strict)
            results["weight_source"] = weight_metadata["source"]
            results["missing_keys"] = weight_metadata["missing_keys"]
            results["unexpected_keys"] = weight_metadata["unexpected_keys"]
        except (WeightFileNotFoundError, WeightLoadingError) as e:
            raise ConversionError(f"Weight loading failed: {e}")
        
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
            config.to_yaml(output_dir / "config.yaml")
        except Exception as e:
            raise ConversionError(f"Failed to save model: {e}")
        
        # Step 6: Copy tokenizer (NEW)
        if not skip_tokenizer:
            results["tokenizer_copied"] = copy_tokenizer_from_checkpoint(
                checkpoint_dir, output_dir, config
            )
            if not results["tokenizer_copied"]:
                # Create tokenizer config from model config
                save_tokenizer_config(output_dir, config, checkpoint_dir)
                logger.info("Created tokenizer config (fallback)")
        
        # Step 7: Verify saved model
        logger.info("Verifying saved model...")
        try:
            from transformers import AutoModel
            loaded = AutoModel.from_pretrained(str(output_dir), trust_remote_code=True)
            logger.info("Model can be loaded with AutoModel!")
            results["auto_model_verification"] = True
        except Exception as e:
            logger.warning(f"AutoModel verification failed: {e}")
            results["auto_model_verification"] = False
        
        # Log output structure
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
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        raise ConversionError(f"Conversion failed: {e}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for checkpoint conversion."""
    parser = argparse.ArgumentParser(
        description="Convert KilatTransformer checkpoint to Hugging Face format",
        epilog="""
Examples:
  python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model
  python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model --config ./configs/model.yaml
  python convert_to_hf.py -c ./checkpoints/checkpoint-best -o ./converted_model --skip-tokenizer
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-c", "--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                       help="Output directory for converted model")
    parser.add_argument("--config", type=str, default=None, dest="config_path",
                       help="Optional path to config file")
    parser.add_argument("--no_safetensors", action="store_true",
                       help="Use pytorch_model.bin instead of safetensors")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to load model on")
    parser.add_argument("--strict", action="store_true",
                       help="Fail on missing/unexpected keys")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip dummy forward pass validation")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip copying tokenizer files")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
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
            skip_tokenizer=args.skip_tokenizer,
            verbose=args.verbose,
        )
        
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print(f"Checkpoint: {results['checkpoint_dir']}")
        print(f"Output: {results['output_dir']}")
        print(f"Config source: {results['config_source']}")
        print(f"Weight source: {results['weight_source']}")
        print(f"Tokenizer copied: {results.get('tokenizer_copied', False)}")
        if results.get("missing_keys"):
            print(f"Missing keys: {len(results['missing_keys'])}")
        print("="*60)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("If conversion failed, try the following:")
        print("  1. Check that all required files exist in checkpoint directory")
        print("  2. Verify config.yaml or config.json matches model architecture")
        print("  3. Try running with --device cpu if using GPU")
        print("  4. Use --skip-validation to bypass forward pass check")
        print("  5. Use --skip-tokenizer if tokenizer files are missing")
        print("="*60)
        
        sys.exit(1)


if __name__ == "__main__":
    main()