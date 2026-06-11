from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Third-party dependencies with graceful degradation pattern
# Following Hugging Face's approach: check availability at import time,
# warn users, and provide clear installation instructions [citation:7]
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False
    # Warning level is appropriate here - safetensors is recommended for production
    # due to security benefits (no arbitrary code execution during load)
    logger.warning("safetensors not installed. Install with: pip install safetensors")

try:
    import huggingface_hub as _hf_hub
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


# Standard file names as defined by Hugging Face Hub specification
# These constants are used across the ecosystem for compatibility
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX = "model.safetensors.index.json"


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    max_shard_bytes: int = 10 * 1024**3,
) -> List[Dict[str, torch.Tensor]]:
    """
    Split a state dict into smaller shards for manageable file sizes.
    
    Why sharding is necessary:
    - GitHub has file size limits (typically 100MB)
    - Hugging Face Hub recommends shards ≤5GB for optimal handling
    - Some filesystems struggle with very large single files [citation:4]
    
    The algorithm uses a greedy approach with size-based packing.
    This is optimal for our use case because:
    1. Tensor order doesn't matter for loading (shards can be loaded independently)
    2. We want to minimize the number of shards while respecting size limits
    
    Args:
        state_dict: Model parameters to shard
        max_shard_bytes: Maximum size per shard (default 10GB).
            This matches Hugging Face's default shard size for transformers.
    
    Returns:
        List of state dict shards, each under the size limit.
    
    Note: A shard may still exceed max_shard_bytes if a single tensor
    is larger than the limit. This is rare for typical model parameters
    but could occur with embedding tables. Callers should be aware of this edge case.
    """
    shards: List[Dict[str, torch.Tensor]] = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0

    for key, tensor in state_dict.items():
        # Calculate actual memory footprint including tensor overhead
        # Using numel() * element_size() gives the raw tensor data size
        # This doesn't include Python object overhead, which is acceptable
        # because tensor data dominates memory usage
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        # Start new shard if current would exceed limit AND we have existing tensors
        # The second condition prevents infinite loops with giant single tensors
        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
            
        current_shard[key] = tensor
        current_size += tensor_bytes

    if current_shard:
        shards.append(current_shard)

    return shards


class BaseConfig:
    """
    Base configuration class for all models.
    
    Provides serialization/deserialization to/from JSON format,
    following the design pattern established by Hugging Face's PretrainedConfig.
    
    This class intentionally uses dynamic attribute assignment to remain flexible
    for different model architectures while maintaining a consistent interface.
    
    This version is enhanced to be fully compatible with KilatConfig,
    supporting all HF-specific attributes like _output_attentions, _use_cache,
    and token-related IDs.
    
    Attributes:
        model_type: Class-level identifier for model architecture.
            Subclasses MUST override this.
    """
    model_type: str = ""

    def __init__(self, **kwargs):
        """
        Initialize configuration with dynamic keyword arguments.
        
        Why dynamic attributes? Different model architectures have
        different configuration needs (e.g., num_layers, num_heads for transformers,
        kernel_size for CNNs). This approach allows maximum flexibility without
        forcing a rigid schema.
        
        Args:
            **kwargs: Arbitrary configuration parameters.
                Common parameters across models include:
                - hidden_size: Dimension of hidden layers
                - num_attention_heads: For transformer models
                - vocab_size: For language models
                - dtype: Default tensor dtype for the model
                Also supports HF-compatible attributes:
                - pad_token_id, bos_token_id, eos_token_id
                - tie_word_embeddings, use_cache
                - _output_attentions, _output_hidden_states
        """
        # Set all provided kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize HF-compatible internal flags with defaults if not set
        if not hasattr(self, '_output_attentions'):
            self._output_attentions = False
        if not hasattr(self, '_output_hidden_states'):
            self._output_hidden_states = False
        if not hasattr(self, '_use_cache'):
            self._use_cache = getattr(self, 'use_cache', False)
        if not hasattr(self, '_attn_implementation'):
            self._attn_implementation = None
        if not hasattr(self, '_attn_implementation_internal'):
            self._attn_implementation_internal = None
        if not hasattr(self, '_experts_implementation_internal'):
            self._experts_implementation_internal = None
        
        # Initialize token-related attributes with defaults if not set
        if not hasattr(self, 'pad_token_id'):
            self.pad_token_id = 0
        if not hasattr(self, 'bos_token_id'):
            self.bos_token_id = 1
        if not hasattr(self, 'eos_token_id'):
            self.eos_token_id = 2
        if not hasattr(self, 'tie_word_embeddings'):
            self.tie_word_embeddings = True
        if not hasattr(self, 'use_cache'):
            self.use_cache = False
        
        # Handle backward compatibility for torch_dtype -> dtype
        # This fixes the "torch_dtype is deprecated" warning
        if hasattr(self, 'torch_dtype') and not hasattr(self, 'dtype'):
            self.dtype = self.torch_dtype
        elif hasattr(self, 'dtype') and not hasattr(self, 'torch_dtype'):
            self.torch_dtype = self.dtype

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a JSON-serializable dictionary.
        
        Explicitly excludes private attributes (starting with '_') to avoid
        serializing internal state that shouldn't be persisted.
        
        Returns:
            Dictionary containing all public configuration attributes,
            including the model_type for proper deserialization.
        """
        output = {}
        for key, value in self.__dict__.items():
            # Skip private attributes - they're implementation details
            if not key.startswith("_"):
                output[key] = value
        output["model_type"] = self.__class__.model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> BaseConfig:
        """
        Create configuration instance from dictionary.
        
        The model_type key is removed because it's a class-level attribute,
        not an instance attribute. This maintains backward compatibility
        with serialized configs that include the type.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            
        Returns:
            Configuration instance with attributes set from the dictionary.
        """
        config_dict = dict(config_dict)
        # model_type is metadata for the class, not an instance attribute
        config_dict.pop("model_type", None)
        return cls(**config_dict)

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save configuration to a directory.
        
        Args:
            save_directory: Directory where config.json will be created.
                Will be created if it doesn't exist.
                
        Note: This method follows the Hugging Face convention for
        save_pretrained, allowing BaseConfig to be used interchangeably
        with PretrainedConfig in existing pipelines.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config_file = save_dir / CONFIG_NAME
        with open(config_file, "w", encoding="utf-8") as f:
            # indent=2 for human readability when debugging
            # ensure_ascii=False preserves non-ASCII characters in comments/descriptions
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {config_file}")

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> BaseConfig:
        """
        Load configuration from a directory.
        
        Args:
            pretrained_path: Directory containing config.json
            
        Returns:
            Configuration instance populated from the saved config.
            
        Raises:
            FileNotFoundError: If config.json doesn't exist at the specified path.
        """
        config_file = Path(pretrained_path) / CONFIG_NAME
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found at {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """
        Human-readable representation for debugging.
        
        Excludes model_type because it's a class-level attribute that would
        appear in every instance's representation, creating clutter.
        """
        attrs = ", ".join(
            f"{k}={v!r}"
            for k, v in self.to_dict().items()
            if k != "model_type"
        )
        return f"{self.__class__.__name__}({attrs})"


class BasePreTrainedModel(nn.Module):
    """
    Abstract base class for all pretrained models.
    
    Provides core functionality for:
    - Model loading/saving (with sharding support)
    - Weight tying management
    - Gradient checkpointing
    - Hugging Face Hub integration
    
    This class intentionally mirrors Hugging Face's PreTrainedModel API
    to ensure compatibility with existing tools and workflows.
    
    Design decisions:
    - Inherits from nn.Module for PyTorch ecosystem compatibility
    - Uses composition over inheritance for config (config is an attribute)
    - Supports both safetensors and pickle-based serialization
    
    Attributes:
        config_class: The configuration class used by this model.
            Must be set by subclasses.
        base_model_prefix: Prefix for the base model attribute
            (e.g., "transformer" or "encoder").
        supports_gradient_checkpointing: Whether gradient checkpointing
            is implemented. Subclasses should set to True if they override
            _set_gradient_checkpointing.
        _tied_weights_keys: Maps redundant parameter names to canonical names.
            Used to avoid saving duplicate weights. Format: {redundant: canonical}
    """
    config_class = BaseConfig
    base_model_prefix: str = ""
    supports_gradient_checkpointing: bool = False
    _tied_weights_keys: Optional[Dict[str, str]] = None

    def __init__(self, config: Any):
        """
        Initialize the model with configuration.
        
        Args:
            config: Configuration object. Can be any type, but typically
                an instance of config_class or a compatible dict-like object.
        """
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights.
        
        Why override? Different architectures have different optimal
        initialization strategies (e.g., Xavier for tanh, Kaiming for ReLU,
        special schemes for transformers).
        
        Subclasses should override this method to implement their specific
        initialization logic. The default implementation does nothing,
        relying on PyTorch's default initialization.
        
        Args:
            module: PyTorch module to initialize.
        """
        pass

    def _tie_weights(self):
        """
        Tie (share) weights between different parameters.
        
        Weight tying is common in language models where input and output
        embeddings share the same weight matrix. This reduces memory usage
        and can improve convergence.
        
        Implementation pattern:
        - Check if the model has both parameters (e.g., embed_tokens and lm_head)
        - If they should be tied, assign: module2.weight = module1.weight
        
        Note: This method is called after weight initialization but before
        validation. Subclasses must ensure the tied weights are actually
        the same tensor object, not just equal values.
        """
        pass

    @property
    def _no_split_modules(self) -> List[str]:
        """
        Modules that should not be split during model parallelism.
        
        Returns a list of module class names that should be kept intact
        when the model is split across devices. Typically this includes
        transformer blocks, which contain internal residual connections
        that cross device boundaries would break.
        
        Returns:
            List of module names that should not be split.
            Default is empty list (no restrictions on splitting).
        """
        return []

    def _validate_tied_weights(self):
        """
        Validate that tied weight shapes are compatible.
        
        This prevents silent failures where tied weights have mismatched shapes
        (e.g., if vocab_size differs between input and output embeddings).
        
        The validation happens post-initialization to catch configuration errors
        early rather than during training or inference.
        
        Raises:
            ValueError: If any tied weight pair has incompatible shapes.
        """
        if not self._tied_weights_keys:
            return
        for redundant_key, canonical_key in self._tied_weights_keys.items():
            if hasattr(self, redundant_key) and hasattr(self, canonical_key):
                w_redundant = getattr(self, redundant_key)
                w_canonical = getattr(self, canonical_key)
                # Handle modules (like nn.Linear) that have .weight attributes
                if hasattr(w_redundant, 'weight') and hasattr(w_canonical, 'weight'):
                    if w_redundant.weight.shape != w_canonical.weight.shape:
                        raise ValueError(
                            f"Tied weight shape mismatch: {redundant_key}.weight "
                            f"{w_redundant.weight.shape} vs {canonical_key}.weight "
                            f"{w_canonical.weight.shape}. Check vocab_size consistency."
                        )

    def post_init(self):
        """
        Initialize model after all modules are created.
        
        This method follows the template method pattern:
        1. Initialize all weights
        2. Tie weights between modules
        3. Validate tied weights
        
        Subclasses should NOT override this method unless they need to
        modify the initialization flow. Override _init_weights, _tie_weights,
        or _validate_tied_weights instead.
        """
        self.apply(self._init_weights)
        self._tie_weights()
        self._validate_tied_weights()

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        max_shard_bytes: int = 10 * 1024**3,
        safe_serialization: bool = True,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Save model weights and configuration to a directory.
        
        This method handles:
        - Automatic sharding for large models (>max_shard_bytes)
        - Safetensors format for secure loading (preferred for production)
        - Fallback to pickle-based format when safetensors is unavailable
        
        Why default to safe_serialization=True?
        - Pickle files can execute arbitrary code during loading (security risk)
        - Safetensors only loads tensor data, no code execution
        - Production deployments should always use safetensors when possible
        
        Args:
            save_directory: Directory to save the model.
            max_shard_bytes: Maximum size per shard file (default 10GB).
            safe_serialization: Use safetensors format if available.
            state_dict: Optional state dict to save. If None, uses self.state_dict().
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        # Two possible interfaces for maximum compatibility
        if hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(save_dir)
        elif hasattr(self.config, "to_dict"):
            with open(save_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            # Don't fail silently - missing config breaks model loading
            logger.warning("Config has no save_pretrained or to_dict method. Config will not be saved.")

        if state_dict is None:
            state_dict = self.state_dict()

        # Remove redundant tied weights before saving to save space
        state_dict = self._remove_tied_weights(state_dict)

        shards = _shard_state_dict(state_dict, max_shard_bytes)
        use_safetensors = safe_serialization and _SAFETENSORS_AVAILABLE

        if len(shards) == 1:
            # Single file - no sharding needed
            if use_safetensors:
                weights_path = save_dir / SAFE_WEIGHTS_NAME
                _st_save(shards[0], str(weights_path))
            else:
                weights_path = save_dir / WEIGHTS_NAME
                torch.save(shards[0], weights_path)
            logger.info(f"Model saved to {weights_path}")
        else:
            # Multiple shards - create index file following HF convention
            ext = ".safetensors" if use_safetensors else ".bin"
            prefix = "model" if use_safetensors else "pytorch_model"
            index: Dict[str, Any] = {"metadata": {}, "weight_map": {}}
            total_params = sum(t.numel() for t in state_dict.values())
            index["metadata"]["total_size"] = total_params

            for shard_idx, shard in enumerate(shards, start=1):
                shard_name = f"{prefix}-{shard_idx:05d}-of-{len(shards):05d}{ext}"
                shard_path = save_dir / shard_name
                if use_safetensors:
                    _st_save(shard, str(shard_path))
                else:
                    torch.save(shard, shard_path)
                for key in shard:
                    index["weight_map"][key] = shard_name

            index_name = SAFE_WEIGHTS_INDEX if use_safetensors else WEIGHTS_INDEX_NAME
            with open(save_dir / index_name, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            logger.info(f"Model sharded into {len(shards)} files at {save_dir}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        config: Optional[Any] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        ignore_keys: Optional[List[str]] = None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        force_download: bool = False,  # <-- ADDED: Control cache behavior
        local_files_only: bool = False,  # <-- ADDED: Skip download if True
        resume_download: bool = True,  # <-- ADDED: Resume interrupted downloads
        **kwargs,
    ) -> BasePreTrainedModel:
        """
        Load a pretrained model from a local directory or Hugging Face Hub.
        
        This is the primary method for loading models and handles:
        - Local vs Hub resolution (auto-downloads if not found locally)
        - Config loading with proper class instantiation
        - State dict loading from single file or sharded format
        - Device placement and dtype conversion
        
        The method uses a "fail-forward" strategy: if the path doesn't exist locally,
        it automatically attempts to download from Hugging Face Hub (if available).
        This provides a seamless user experience.
        
        Args:
            pretrained_model_name_or_path: Local path or Hub model identifier
                (e.g., "bert-base-uncased" or "./my_model/").
            config: Optional config override. If None, loads from the model directory.
            map_location: Device to load tensors onto initially.
            strict: Whether to strictly enforce that state_dict matches the model.
            ignore_keys: List of parameter keys to ignore during loading.
            dtype: Target dtype for model parameters.
                - "auto": Use config.dtype if available, otherwise detect from weights
                - torch.float32, etc.: Explicit dtype
            device_map: Device placement strategy.
                - "auto": Use accelerate for automatic device mapping
                - Dict: Manual device mapping
                - None: Keep on map_location device
            force_download: If True, download even if already cached.
            local_files_only: If True, only use local files, don't download.
            resume_download: If True, resume interrupted downloads.
            **kwargs: Additional arguments passed to the model constructor.
            
        Returns:
            Loaded model instance.
            
        Raises:
            FileNotFoundError: If config is missing and can't be found locally or on Hub.
            ImportError: If Hub download is needed but huggingface_hub is not installed.
            RuntimeError: If loading fails due to missing keys (when strict=True).
        """
        pretrained_path = Path(pretrained_model_name_or_path)

        # Check local existence first, fall back to Hub download
        if not pretrained_path.exists():
            if not _HF_HUB_AVAILABLE:
                raise ImportError(
                    f"Path '{pretrained_path}' not found locally, "
                    "and huggingface_hub not installed. "
                    "Install with: pip install huggingface_hub"
                )
            logger.info(f"Downloading '{pretrained_model_name_or_path}' from HuggingFace Hub...")
            # snapshot_download handles caching and incremental downloads
            # Added cache control parameters for better performance
            pretrained_path = Path(
                _hf_hub.snapshot_download(
                    str(pretrained_model_name_or_path),
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            )

        # Load configuration
        if config is None:
            config_file = pretrained_path / CONFIG_NAME
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Config not found at '{pretrained_path}'. "
                    "Directory must contain config.json"
                )
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # Support two common config class interfaces
            if hasattr(cls.config_class, "from_dict"):
                config = cls.config_class.from_dict(config_dict)
            elif hasattr(cls.config_class, "from_pretrained"):
                config = cls.config_class.from_pretrained(pretrained_path)
            else:
                raise RuntimeError(
                    f"{cls.config_class} must have from_dict() or from_pretrained() method."
                )

        # Create model instance (weights not yet loaded)
        model = cls(config, **kwargs)

        # Load state dict from disk (always load to CPU first for safety)
        # Why CPU first? Prevents OOM errors when multiple models are loaded
        # and allows explicit device placement after loading
        state_dict = cls._load_state_dict(pretrained_path, map_location="cpu")

        # Remove ignored keys before loading
        if ignore_keys:
            for key in list(state_dict.keys()):
                if any(key.startswith(ik) for ik in ignore_keys):
                    del state_dict[key]

        # Load weights with strict=False to check missing/unexpected separately
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Filter out tied weights from missing keys (they don't need to be loaded)
        tied_keys = set((model._tied_weights_keys or {}).keys())
        missing_real = [k for k in missing if k not in tied_keys]

        # Strict validation if requested
        if strict and missing_real:
            raise RuntimeError(
                f"Missing keys when loading state_dict:\n{missing_real}\n"
                "Use strict=False to ignore missing keys."
            )
        if strict and unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading state_dict:\n{unexpected}\n"
                "Use strict=False to ignore unexpected keys."
            )

        if missing_real:
            logger.warning(f"Missing keys (ignored): {missing_real}")
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {unexpected}")

        # Tie weights after loading (for any that weren't saved)
        model._tie_weights()

        # Handle dtype conversion
                # Handle dtype conversion
        # Map string dtype names to torch.dtype objects
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "float": torch.float32,
            "half": torch.float16,
        }
        
        # Convert dtype to torch.dtype if it's a string
        if isinstance(dtype, str):
            dtype_lower = dtype.lower()
            if dtype_lower in dtype_map:
                dtype = dtype_map[dtype_lower]
            elif dtype_lower == "auto":
                dtype = "auto"  # Keep as auto for now
            else:
                try:
                    dtype = getattr(torch, dtype_lower)
                except AttributeError:
                    logger.warning(f"Unknown dtype string: '{dtype}', defaulting to float32")
                    dtype = torch.float32
        
        # Handle "auto" case
        if dtype == "auto":
            dtype = getattr(config, "dtype", None)
            if dtype is None and state_dict:
                first_param = next(iter(state_dict.values()))
                dtype = first_param.dtype
            # If still None, default to float32
            if dtype is None:
                dtype = torch.float32
        
        # Ensure dtype is a torch.dtype
        if dtype is not None:
            if not isinstance(dtype, torch.dtype):
                # Try one more time to convert
                if isinstance(dtype, str):
                    dtype = dtype_map.get(dtype.lower(), torch.float32)
                else:
                    logger.warning(f"dtype is not a torch.dtype: {dtype}, using float32")
                    dtype = torch.float32
        
        # Apply dtype conversion
        if dtype is not None and isinstance(dtype, torch.dtype):
            model = model.to(dtype=dtype)
            logger.info(f"Model cast to {dtype}")


        # Handle device placement
        # device_map="auto" requires accelerate for intelligent device mapping
        if device_map == "auto":
            try:
                from accelerate import dispatch_model
                model = dispatch_model(model, device_map=device_map)
                logger.info(f"Auto device map applied")
            except ImportError:
                logger.warning("accelerate not installed, device_map='auto' ignored")
        elif isinstance(device_map, dict):
            from accelerate import dispatch_model
            model = dispatch_model(model, device_map=device_map)
        elif device_map is not None:
            model = model.to(device_map)

        logger.info(f"Model loaded from '{pretrained_path}'")
        return model

    @classmethod
    def _load_state_dict(
        cls,
        pretrained_path: Path,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Load state dict from disk, handling both single-file and sharded formats.
        
        Search order (prioritizing safetensors for security):
        1. Single safetensors file
        2. Single pickle file  
        3. Sharded safetensors (index file)
        4. Sharded pickle (index file)
        
        This ordering ensures that if multiple formats exist, the safest format
        (safetensors single file) is preferred.
        
        Args:
            pretrained_path: Directory containing weights files.
            map_location: Device to load tensors onto.
            
        Returns:
            Combined state dict from all shards.
            
        Raises:
            FileNotFoundError: If no weights file is found in any supported format.
            ImportError: If safetensors is required but not installed.
        """
        candidates = [
            (pretrained_path / SAFE_WEIGHTS_NAME,   "safetensors", False),
            (pretrained_path / WEIGHTS_NAME,         "bin",         False),
            (pretrained_path / SAFE_WEIGHTS_INDEX,   "safetensors", True),
            (pretrained_path / WEIGHTS_INDEX_NAME,   "bin",         True),
        ]

        for weights_file, fmt, is_index in candidates:
            if not weights_file.exists():
                continue

            if not is_index:
                # Single file - load directly
                if fmt == "safetensors":
                    if not _SAFETENSORS_AVAILABLE:
                        continue
                    return _st_load(str(weights_file), device=str(map_location))
                else:
                    # weights_only=True prevents arbitrary code execution during load
                    # Added fallback for older models that may not support weights_only
                    try:
                        return torch.load(weights_file, map_location=map_location, weights_only=True)
                    except Exception as e:
                        logger.warning(f"Failed to load with weights_only=True: {e}. Trying without...")
                        return torch.load(weights_file, map_location=map_location, weights_only=False)
            else:
                # Index file - load all shards and merge
                with open(weights_file, "r", encoding="utf-8") as f:
                    index = json.load(f)

                # Determine unique shard files (multiple keys may point to same shard)
                shard_files = sorted(set(index["weight_map"].values()))
                merged: Dict[str, torch.Tensor] = {}
                for shard_name in shard_files:
                    shard_path = pretrained_path / shard_name
                    if not shard_path.exists():
                        raise FileNotFoundError(f"Shard file not found: {shard_path}")
                    if fmt == "safetensors":
                        if not _SAFETENSORS_AVAILABLE:
                            raise ImportError("safetensors required to load this file.")
                        shard_dict = _st_load(str(shard_path), device=str(map_location))
                    else:
                        # Added fallback for sharded pickle files too
                        try:
                            shard_dict = torch.load(
                                shard_path, map_location=map_location, weights_only=True
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load shard with weights_only=True: {e}. Trying without...")
                            shard_dict = torch.load(
                                shard_path, map_location=map_location, weights_only=False
                            )
                    merged.update(shard_dict)
                return merged

        raise FileNotFoundError(
            f"No weights file found at '{pretrained_path}'.\n"
            f"Searched for: {SAFE_WEIGHTS_NAME}, {WEIGHTS_NAME}, "
            f"{SAFE_WEIGHTS_INDEX}, {WEIGHTS_INDEX_NAME}"
        )

    def _remove_tied_weights(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Remove redundant tied weights from state dict before saving.
        
        Why? If weights are tied (shared), they're the same tensor object.
        Saving both would waste disk space and cause confusion during loading.
        
        This method removes the redundant keys, keeping only the canonical version.
        During load, _tie_weights() will re-establish the sharing relationship.
        
        Args:
            state_dict: Original state dict that may contain redundant entries.
            
        Returns:
            State dict with redundant tied weight entries removed.
        """
        if not self._tied_weights_keys:
            return state_dict
        result = dict(state_dict)
        for redundant_key in self._tied_weights_keys:
            result.pop(redundant_key, None)
        return result

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict] = None,
    ):
        """
        Enable gradient checkpointing to reduce memory usage at the cost of compute.
        
        Gradient checkpointing recomputes activations during the backward pass
        instead of storing them. This reduces peak memory by O(sqrt(n)) for
        transformers, enabling larger batch sizes or models.
        
        The trade-off: about 20-30% slower training due to recomputation.
        
        Args:
            gradient_checkpointing_kwargs: Additional kwargs passed to the
                checkpointing function (e.g., use_reentrant for torch.utils.checkpoint).
        
        Raises:
            ValueError: If the model doesn't support gradient checkpointing.
                Subclasses must set supports_gradient_checkpointing=True and
                implement _set_gradient_checkpointing.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing. "
                "Set supports_gradient_checkpointing = True and implement checkpointing in block layers."
            )
        self._set_gradient_checkpointing(enable=True)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (restore full memory usage)."""
        if self.supports_gradient_checkpointing:
            self._set_gradient_checkpointing(enable=False)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Optional[Callable] = None
    ):
        """
        Recursively set gradient checkpointing on all modules that support it.
        
        This method traverses the module hierarchy and sets the gradient_checkpointing
        attribute on any module that has it. Subclasses should override this if
        they need custom checkpointing behavior beyond a simple boolean flag.
        
        Args:
            enable: Enable or disable checkpointing.
            gradient_checkpointing_func: Custom checkpoint function to use.
                If provided, overrides the default torch.utils.checkpoint.
        
        Raises:
            ValueError: If no module with gradient_checkpointing attribute is found.
        """
        is_set = False
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                if enable and gradient_checkpointing_func:
                    module.gradient_checkpointing = gradient_checkpointing_func
                else:
                    module.gradient_checkpointing = enable
                is_set = True
        if not is_set:
            raise ValueError(
                f"No module in {self.__class__.__name__} has 'gradient_checkpointing' attribute. "
                "Add this attribute to your Block layer."
            )

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is currently enabled."""
        return any(
            getattr(m, "gradient_checkpointing", False) for m in self.modules()
        )

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            only_trainable: If True, count only parameters with requires_grad=True.
            
        Returns:
            Total number of parameters (or trainable parameters).
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    def get_memory_footprint(self) -> str:
        """
        Calculate the approximate memory usage of the model in human-readable format.
        
        Includes both parameters and buffers. Returns a string with appropriate
        units (bytes, KB, MB, GB).
        
        Note: This is an estimate. Actual memory usage may be higher due to:
        - Python object overhead
        - CUDA-specific memory alignment
        - Intermediate activations (not included)
        
        Returns:
            Human-readable string with memory footprint (e.g., "1.23 GB").
        """
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        total_bytes += sum(
            b.numel() * b.element_size() for b in self.buffers()
        )
        for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
            if total_bytes >= threshold:
                return f"{total_bytes / threshold:.2f} {unit}"
        return f"{total_bytes} bytes"

    @property
    def device(self) -> torch.device:
        """
        Get the device of the first parameter.
        
        This is a convenience property that returns the device where the
        model's parameters are stored. Useful for moving tensors to the
        same device as the model without tracking device separately.
        
        Returns:
            torch.device: Device of the first parameter, or CPU if no parameters.
        """
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        """
        Override to ensure config stays attached when moving to device.
        
        When moving a model to a different device, we also need to move any
        tensors stored in the config (e.g., some configs store tensor attributes).
        This override ensures everything moves together.
        
        Args:
            *args, **kwargs: Arguments passed to nn.Module.to()
            
        Returns:
            The model instance (self) after moving.
        """
        device = super().to(*args, **kwargs)
        # Move config tensors if config has a to() method
        if hasattr(self.config, 'to'):
            self.config.to(*args, **kwargs)
        return device

    def __repr__(self) -> str:
        """
        Extended representation with parameter count and memory footprint.
        
        This provides useful debugging information at a glance, especially
        for large models where parameter count affects performance decisions.
        """
        n_params = self.num_parameters()
        n_trainable = self.num_parameters(only_trainable=True)
        base_repr = super().__repr__()
        return (
            f"{base_repr}\n"
            f"Total parameters: {n_params:,} "
            f"({n_trainable:,} trainable, {n_params - n_trainable:,} frozen)\n"
            f"Memory footprint: {self.get_memory_footprint()}"
        )

    def add_model_tags(self, tags: Union[str, List[str]]):
        """
        Add tags to the model for discovery on Hugging Face Hub.
        
        Tags are stored in _hub_tags and can be pushed to the Hub using push_to_hub.
        Common tags include task names ("text-classification", "image-generation"),
        frameworks ("pytorch", "jax"), or custom categories.
        
        Args:
            tags: Tag string or list of tags to add.
            
        Raises:
            ImportError: If huggingface_hub is not installed.
        """
        if not _HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub required for tagging")
        existing = getattr(self, "_hub_tags", [])
        new_tags = [tags] if isinstance(tags, str) else tags
        self._hub_tags = list(set(existing + new_tags))

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str = "Upload model",
        private: bool = False,
        token: Optional[str] = None,
        safe_serialization: bool = True,
    ):
        """
        Upload the model to Hugging Face Hub.
        
        This method creates a repository (if needed), saves the model to a
        temporary directory, and uploads all files.
        
        Args:
            repo_id: Hub repository ID (e.g., "username/model-name").
            commit_message: Git commit message for the upload.
            private: Whether the repository should be private.
            token: Hugging Face API token. If None, uses cached token.
            safe_serialization: Use safetensors format for weights.
            
        Raises:
            ImportError: If huggingface_hub is not installed.
            
        Note: This method uses a temporary directory to avoid polluting
        the local filesystem with intermediate files.
        """
        if not _HF_HUB_AVAILABLE:
            raise ImportError(
                "push_to_hub requires huggingface_hub. "
                "Install with: pip install huggingface_hub"
            )

        api = _hf_hub.HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        # Use temporary directory to avoid leaving artifacts on failure
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, safe_serialization=safe_serialization)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )

        logger.info(f"Model uploaded to https://huggingface.co/{repo_id}")