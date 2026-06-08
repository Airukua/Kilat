"""
Generation hyperparameters for autoregressive text generation.

This module defines the sampling and stopping parameters used by the generator.
The dataclass encapsulates all knobs that control the trade‑off between
diversity, coherence, and repetition.

Design references:
- Temperature: (Ackley et al., 1985; used in GPT-2/3)
- Top‑k sampling: (Fan et al., 2018) – truncates to k most likely tokens
- Top‑p (nucleus) sampling: (Holtzman et al., 2020) – dynamically selects smallest set
- Repetition penalty: (Keskar et al., 2019) – divides logits of already generated tokens
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GenerationConfig:
    """
    Controls the generation process with battle-tested sampling parameters.
    
    These parameters work in concert to shape output quality:
    - Temperature: Controls randomness concentration. Lower = more deterministic.
    - Top‑k + top‑p: Dual filters that discard improbable tokens BEFORE temperature
      is applied, preventing the model from considering nonsensical completions.
    - Repetition penalty: Addresses the common failure mode where models get stuck
      in loops (e.g., "I love I love I love...").
    
    Why both top-k and top-p? They serve different purposes:
    - top-k prevents the model from considering extremely unlikely tokens
    - top-p dynamically adjusts the candidate set based on distribution peakedness
    - Using both (apply top-k first, then top-p) is standard practice
    
    Implementation notes:
    - The actual sampling logic lives in Generator._sample_next_token()
    - This config is passed through unchanged to preserve reproducibility
    """
    
    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled (use full vocabulary)
    top_p: float = 1.0      # 1.0 = disabled (use all tokens after top-k)
    repetition_penalty: float = 1.0   # 1.0 = disabled
    
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    # Reserved for future: stop_tokens allows early termination when specific
    # token IDs appear (e.g., EOS, custom separators). Currently unused because
    # the generator hardcodes EOS detection, but kept for API consistency.
    stop_tokens: Optional[List[int]] = None
    
    do_sample: bool = True
    # Not implemented in v1: Would require running generation multiple times
    # with different random seeds. Left here as a future extension point.
    num_return_sequences: int = 1

    def __post_init__(self):
        """
        Validate parameter ranges to catch silent failures early.
        
        Why these specific bounds?
        - temperature: Negative values invert probability rankings, producing
          garbage. 0 is valid (greedy/argmax), but should not sample.
        - top_k: 0 conveniently means "no limit" without needing Optional[int].
        - top_p: Must be positive to form a probability mass; 1.0 = no filtering.
        - repetition_penalty: < 1.0 would boost repeated tokens (opposite intent).
        - max_new_tokens: 0 would generate nothing; 1 is minimum meaningful output.
        
        Note: Does NOT validate do_sample + temperature=0 edge case. That's
        handled in the generator's sampling logic (argmax vs. multinomial).
        """
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")