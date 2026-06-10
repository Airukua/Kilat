"""
Generation configuration for text generation with KilatTransformer.

WHY THIS EXISTS:
    - Separates generation parameters from model configuration
    - Follows Hugging Face's GenerationConfig pattern for compatibility
    - Enables serialization/deserialization of generation settings
    - Allows different generation strategies (greedy, sampling, beam search)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import json
from pathlib import Path


@dataclass
class GenerationConfig:
    """
    Configuration for text generation with KilatTransformer.

    WHY DATACLASS:
        - Immutable by default (though fields can be modified)
        - Automatic __init__, __repr__, __eq__ generation
        - Easy to convert to dict for JSON serialization
        - Type hints enforced at construction

    DESIGN PHILOSOPHY:
        This config follows Hugging Face's GenerationConfig design patterns
        to ensure compatibility with existing HF tooling and mental models.
        Users familiar with transformers library will find these parameters
        immediately recognizable.

    PARAMETER INTERACTIONS:
        - `do_sample=False, num_beams=1` → Greedy decoding (fastest)
        - `do_sample=True, num_beams=1` → Multinomial sampling
        - `do_sample=False, num_beams>1` → Beam search
        - `do_sample=True, num_beams>1` → Beam-search sampling

    IMPORTANT DEFAULTS:
        - `do_sample=False`: Greedy by default (deterministic, reproducible)
        - `top_k=50`: Common default from GPT-2/LLaMA
        - `temperature=1.0`: No scaling (uniform sampling when do_sample=True)
        - `repetition_penalty=1.0`: No penalty (can be increased to 1.1-1.2)

    Example Usage
    -------------
        >>> # Greedy decoding (fast, deterministic)
        >>> config = GenerationConfig(do_sample=False, max_new_tokens=100)
        >>>
        >>> # Sampling with temperature (creative, stochastic)
        >>> config = GenerationConfig(
        ...     do_sample=True,
        ...     temperature=0.8,
        ...     top_k=50,
        ...     max_new_tokens=100
        ... )
        >>>
        >>> # Save for later use
        >>> config.save_pretrained("./checkpoints/my-model")
        >>> loaded = GenerationConfig.from_pretrained("./checkpoints/my-model")
    """

    # ========== Length Control ==========
    max_length: int = 100
    """
    Maximum total length of generated sequence (including input prompt).
    Final sequence length = min(max_length, input_length + max_new_tokens).
    Default: 100 tokens total.
    """
    
    max_new_tokens: Optional[int] = None
    """
    Maximum number of new tokens to generate (excluding input prompt).
    If None, uses max_length - input_length. Takes precedence over max_length
    when both are specified. Default: None.
    """
    
    min_length: int = 0
    """
    Minimum total length of generated sequence (excluding EOS token).
    Generation continues until at least min_length tokens are produced,
    even if EOS token is generated earlier. Default: 0 (disabled).
    """

    # ========== Sampling Strategies ==========
    do_sample: bool = False
    """
    Whether to use sampling (stochastic) vs greedy (deterministic).
    - True: Sample next token from probability distribution
    - False: Always pick token with highest probability
    Default: False (greedy decoding).
    """
    
    temperature: float = 1.0
    """
    Temperature for sampling (only used when do_sample=True).
    Higher values = more random (flatter distribution)
    Lower values = more deterministic (sharper peaks)
    Range: (0, ∞). Typical values: 0.7-1.2.
    Default: 1.0 (no scaling).
    """
    
    top_k: int = 50
    """
    Top-K filtering: keep only the K most likely tokens.
    - 0 = disabled (keep all tokens)
    - >0 = only sample from top K tokens
    Typical values: 40-60. Default: 50.
    """
    
    top_p: float = 1.0
    """
    Top-P (nucleus) filtering: keep tokens with cumulative probability >= top_p.
    - 1.0 = disabled (keep all tokens)
    - 0.9 = keep tokens comprising 90% of probability mass
    Typical values: 0.85-0.95. Default: 1.0.
    """

    # ========== Beam Search ==========
    num_beams: int = 1
    """
    Number of beams for beam search.
    - 1 = disabled (greedy or sampling)
    - >1 = beam search with K beams
    Trade-off: higher beams = better quality but slower.
    Default: 1.
    """
    
    num_beam_groups: int = 1
    """
    Number of groups for diverse beam search.
    Groups generate diverse outputs by dividing beams into groups.
    Only used when num_beams > 1 and diversity_penalty > 0.
    Default: 1.
    """
    
    diversity_penalty: float = 0.0
    """
    Diversity penalty for diverse beam search.
    Higher values encourage more diverse outputs across groups.
    Range: [0, ∞). Typical: 0.5-1.0.
    Default: 0.0 (disabled).
    """
    
    length_penalty: float = 1.0
    """
    Length penalty for beam search.
    - <1.0: encourage longer sequences
    - =1.0: neutral
    - >1.0: encourage shorter sequences
    Typical values: 0.8-1.2. Default: 1.0.
    """

    early_stopping: bool = False 
    """
    Whether to stop beam search when at least `num_beams` beams are finished.
    - True: Stop early when enough beams are done
    - False: Continue until max length is reached
    Default: False.
    """

    # ========== Repetition Control ==========
    repetition_penalty: float = 1.0
    """
    Penalty for repeating tokens.
    - 1.0 = disabled
    - >1.0 = penalize repetition (typical: 1.1-1.2)
    - <1.0 = encourage repetition (rarely used)
    Applied by dividing logits of already-seen tokens.
    Default: 1.0.
    """
    
    no_repeat_ngram_size: int = 0
    """
    Prevent repetition of n-grams of this size.
    - 0 = disabled
    - >0 = block any n-gram that has already appeared
    Typical values: 2-4. Default: 0.
    """

    # ========== Token IDs ==========
    eos_token_id: Optional[int] = None
    """
    End-of-sequence token ID. Generation stops when this token is produced.
    If None, uses model's default eos_token_id.
    Default: None.
    """
    
    pad_token_id: Optional[int] = None
    """
    Padding token ID. Used to pad sequences to same length.
    If None, uses model's default pad_token_id.
    Default: None.
    """
    
    bos_token_id: Optional[int] = None
    """
    Beginning-of-sequence token ID. Prefixed to input if add_special_tokens=True.
    If None, uses model's default bos_token_id.
    Default: None.
    """

    # ========== Output Configuration ==========
    num_return_sequences: int = 1
    """
    Number of independently generated sequences to return.
    For beam search: must be <= num_beams.
    For sampling: can be any positive integer.
    Default: 1.
    """
    
    output_scores: bool = False
    """
    Whether to return token scores (log probabilities) with generated sequences.
    Useful for debugging or computing confidence scores.
    Default: False.
    """
    
    return_dict_in_generate: bool = False
    """
    Whether to return a GenerateOutput dataclass instead of just tensor.
    When True, returns object with .sequences, .scores, .past_key_values.
    Default: False.
    """

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        WHY: Enables JSON/YAML serialization while filtering out private
        attributes (starting with underscore) that shouldn't be saved.

        Returns
        -------
        dict
            Dictionary containing all public configuration fields.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save generation configuration to a directory.

        WHY: Allows generation config to be stored alongside model checkpoint,
        ensuring consistent generation settings when loading the model later.

        The config is saved as `generation_config.json` in the specified
        directory. This follows Hugging Face's convention, making the
        checkpoint compatible with transformers' `from_pretrained`.

        Parameters
        ----------
        save_directory : str | Path
            Directory to save the configuration file. Created if doesn't exist.

        Example
        -------
            >>> config = GenerationConfig(do_sample=True, temperature=0.8)
            >>> config.save_pretrained("./checkpoints/my-model")
            >>> # Later: loaded = GenerationConfig.from_pretrained("./checkpoints/my-model")
        """
        path = Path(save_directory) / "generation_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "GenerationConfig":
        """
        Load generation configuration from a directory.

        WHY: Enables loading generation settings that were saved alongside
        a model checkpoint. If no config exists, returns default config
        (graceful fallback).

        Parameters
        ----------
        model_path : str | Path
            Path to model directory containing generation_config.json.

        Returns
        -------
        GenerationConfig
            Loaded configuration, or default if file not found.

        Example
        -------
            >>> config = GenerationConfig.from_pretrained("./checkpoints/my-model")
            >>> output = model.generate(input_ids, generation_config=config)
        """
        path = Path(model_path) / "generation_config.json"
        if path.exists():
            with open(path) as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        return cls()

    def __post_init__(self):
        """
        Post-initialization validation and default resolution.

        WHY: Validates that parameters are within reasonable ranges and
        resolves conflicts (e.g., max_new_tokens vs max_length).

        Edge Cases Handled:
            - If both max_new_tokens and max_length are None, set default
            - Ensure top_k is non-negative
            - Clamp temperature to positive values
        """
        # All fields already have default values, no None handling needed
        # but we still validate ranges
        
        # Validate ranges
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")

    def get_max_new_tokens(self, input_length: int) -> int:
        """
        Calculate the actual number of new tokens to generate.

        WHY: Resolves the priority between max_new_tokens and max_length.
        max_new_tokens takes precedence if specified; otherwise uses
        max_length - input_length (bounded by positive values).

        Parameters
        ----------
        input_length : int
            Length of input prompt (number of tokens).

        Returns
        -------
        int
            Number of new tokens to generate (always >= 1).
        """
        if self.max_new_tokens is not None:
            return max(1, self.max_new_tokens)
        
        if self.max_length is not None:
            return max(1, self.max_length - input_length)
        
        # Default fallback
        return 20

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        sampling_info = f"do_sample={self.do_sample}"
        if self.do_sample:
            sampling_info += f", temp={self.temperature}, top_k={self.top_k}, top_p={self.top_p}"
        
        beam_info = f", beams={self.num_beams}" if self.num_beams > 1 else ""
        penalty_info = f", rep_penalty={self.repetition_penalty}" if self.repetition_penalty != 1.0 else ""
        
        return f"GenerationConfig({sampling_info}{beam_info}{penalty_info})"