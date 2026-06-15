"""
Stopping criteria for text generation with KilatTransformer.

WHY THIS EXISTS:
    Text generation needs to know when to stop. Different criteria determine
    when generation should end based on various conditions: maximum length,
    EOS token, time limit, or custom conditions. This module provides a
    composable set of stopping criteria that can be combined and used
    together with the generation loop.

DESIGN PHILOSOPHY:
    - Each criterion is a callable class implementing a simple interface
    - Criteria can be composed (AND/OR logic) via StoppingCriteriaList
    - Lazy evaluation: only compute what's needed when checking
    - Batched operations for efficiency with multiple sequences
    - Compatible with Hugging Face's stopping criteria API

STOPPING CRITERIA TYPES:
    1. MaxLengthCriteria - Stop when sequence reaches maximum length
    2. MaxNewTokensCriteria - Stop after generating N new tokens
    3. EosTokenCriteria - Stop when EOS token is generated
    4. MinLengthCriteria - Don't stop before minimum length (for proper generation)
    5. TimeoutCriteria - Stop after a specified time limit
    6. StoppingCriteriaList - Combine multiple criteria (all must be satisfied OR any)
    7. CustomCriteria - User-defined stopping condition

PERFORMANCE:
    - All criteria are designed to be fast (O(1) or O(batch_size) per check)
    - Criteria that require tensor operations are batched
    - No redundant computations across criteria
"""

from __future__ import annotations
import math
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional,Any, Union
import torch


# ============================================================================
# Base Classes
# ============================================================================

class StoppingCriteria(ABC):
    """
    Abstract base class for all stopping criteria.

    WHY: Provides a consistent interface for all criteria. Each criterion
    implements a `__call__` method that takes input_ids and scores and returns
    a boolean tensor indicating whether generation should stop.

    Design:
        - Subclasses must implement `__call__` method
        - Criteria can maintain internal state (e.g., for timeout)
        - Return shape: (batch_size,) boolean tensor
        - True = stop generation for that batch, False = continue

    Example:
        >>> class MyCriteria(StoppingCriteria):
        ...     def __call__(self, input_ids, scores):
        ...         return input_ids.shape[-1] >= 100  # stop at length 100
    """

    @abstractmethod
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[bool, torch.Tensor]:
        """
        Check if generation should stop.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Current logits/scores of shape (batch_size, vocab_size) or None.
        **kwargs
            Additional arguments (e.g., generated_tokens, start_time).

        Returns
        -------
        Union[bool, torch.Tensor]
            If scalar bool: stop generation for all batches.
            If tensor of shape (batch_size,): stop per batch.
        """
        raise NotImplementedError


class StoppingCriteriaList(List[StoppingCriteria]):
    """
    List of stopping criteria that can be evaluated together.

    WHY: Allows combining multiple stopping criteria. You can use AND logic
    (all criteria must be satisfied) or OR logic (any criterion satisfied).

    Features:
        - Inherits from list for familiarity
        - Supports both AND and OR evaluation modes
        - __call__ applies all criteria and combines results
        - Efficient: breaks early for OR mode when any criterion is True

    Example:
        >>> criteria = StoppingCriteriaList([MaxLengthCriteria(100), EosTokenCriteria(2)])
        >>> should_stop = criteria(input_ids, scores)  # OR by default
        >>> should_stop_and = criteria(input_ids, scores, mode="and")
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        mode: str = "or",
        **kwargs,
    ) -> torch.Tensor:
        """
        Evaluate all stopping criteria and combine results.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Current logits/scores of shape (batch_size, vocab_size).
        mode : str
            Combination mode: "or" (any criteria True) or "and" (all criteria True).
        **kwargs
            Additional arguments passed to each criterion.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) indicating stop status.
        """
        if not self:
            # Empty list: never stop
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        # Evaluate all criteria
        results = []
        for criterion in self:
            result = criterion(input_ids, scores, **kwargs)

            # Convert scalar to tensor if needed
            if isinstance(result, bool):
                result = torch.full(
                    (input_ids.shape[0],), result, dtype=torch.bool, device=input_ids.device
                )

            results.append(result)

        # Combine results based on mode
        if mode == "or":
            # OR: stop if any criterion is True
            combined = torch.stack(results).any(dim=0)
        elif mode == "and":
            # AND: stop only if all criteria are True
            combined = torch.stack(results).all(dim=0)
        else:
            raise ValueError(f"mode must be 'or' or 'and', got {mode}")

        return combined

    def add_criterion(self, criterion: StoppingCriteria) -> None:
        """Add a stopping criterion to the list."""
        self.append(criterion)


# ============================================================================
# Length-Based Criteria
# ============================================================================

class MaxLengthCriteria(StoppingCriteria):
    """
    Stop when the sequence reaches maximum length.

    WHY: The most basic stopping criterion. Ensures generation doesn't exceed
    a maximum length, preventing infinite loops and controlling output size.

    Example:
        >>> criteria = MaxLengthCriteria(max_length=100)
        >>> # Stops when input_ids.shape[-1] >= 100
    """

    def __init__(self, max_length: int):
        """
        Initialize MaxLengthCriteria.

        Parameters
        ----------
        max_length : int
            Maximum total length (including input prompt).
            Generation stops when sequence length >= max_length.
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")

        self.max_length = max_length

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if sequence length has reached maximum.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) where True = should stop.
        """
        current_length = input_ids.shape[1]
        should_stop = current_length >= self.max_length

        return torch.full(
            (input_ids.shape[0],), should_stop, dtype=torch.bool, device=input_ids.device
        )


class MaxNewTokensCriteria(StoppingCriteria):
    """
    Stop after generating a specified number of new tokens.

    WHY: More flexible than max_length when you care about generated output
    length rather than total sequence length. Useful for tasks where input
    length varies significantly.

    Example:
        >>> criteria = MaxNewTokensCriteria(max_new_tokens=50, input_length=10)
        >>> # Stops after generating 50 tokens (total length 60)
    """

    def __init__(self, max_new_tokens: int, input_length: Optional[int] = None):
        """
        Initialize MaxNewTokensCriteria.

        Parameters
        ----------
        max_new_tokens : int
            Maximum number of new tokens to generate.
        input_length : Optional[int]
            Length of input prompt. If None, will be determined on first call.
        """
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

        self.max_new_tokens = max_new_tokens
        self.input_length = input_length
        self._initialized = False

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if max new tokens have been generated.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) where True = should stop.
        """
        # Initialize input length on first call if not provided
        if not self._initialized:
            if self.input_length is None:
                self.input_length = input_ids.shape[1]
            self._initialized = True

        generated_tokens = input_ids.shape[1] - self.input_length
        should_stop = generated_tokens >= self.max_new_tokens

        return torch.full(
            (input_ids.shape[0],), should_stop, dtype=torch.bool, device=input_ids.device
        )


class MinLengthCriteria(StoppingCriteria):
    """
    Ensure minimum length before allowing EOS to stop.

    WHY: Prevents generation from stopping too early. EOS token is suppressed
    until minimum length is reached. This is typically used as a processor,
    but can also be used as a stopping criterion for completeness.

    Example:
        >>> criteria = MinLengthCriteria(min_length=20)
        >>> # Returns False for length < 20, True otherwise
    """

    def __init__(self, min_length: int):
        """
        Initialize MinLengthCriteria.

        Parameters
        ----------
        min_length : int
            Minimum total length (including input).
        """
        if min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {min_length}")

        self.min_length = min_length

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if minimum length has been reached.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor where True = length >= min_length (can stop).
        """
        current_length = input_ids.shape[1]
        can_stop = current_length >= self.min_length

        return torch.full(
            (input_ids.shape[0],), can_stop, dtype=torch.bool, device=input_ids.device
        )


# ============================================================================
# Token-Based Criteria
# ============================================================================

class EosTokenCriteria(StoppingCriteria):
    """
    Stop when EOS token is generated.

    WHY: Standard stopping criterion for language models. Stops generation
    when the model outputs the end-of-sequence token, indicating natural
    completion.

    Design:
        - Supports multiple EOS tokens (e.g., for different languages)
        - Works per-batch: different sequences can stop at different times
        - Efficient: only checks the last token per sequence

    Example:
        >>> criteria = EosTokenCriteria(eos_token_id=2)
        >>> # Stops when any token equals 2
        >>> criteria = EosTokenCriteria(eos_token_id=[2, 50256, 50257])
        >>> # Stops when token is any of these IDs
    """

    def __init__(
        self,
        eos_token_id: Union[int, List[int], torch.Tensor],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EosTokenCriteria.

        Parameters
        ----------
        eos_token_id : int or list[int] or torch.Tensor
            Token ID(s) that indicate end of sequence.
        device : Optional[torch.device]
            Device for storing EOS tensor. If None, will use input device.
        """
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        if isinstance(eos_token_id, list):
            eos_token_id = torch.tensor(eos_token_id, device=device)

        if not isinstance(eos_token_id, torch.Tensor):
            raise TypeError(f"eos_token_id must be int, list, or tensor, got {type(eos_token_id)}")

        self.eos_token_id = eos_token_id

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if EOS token has been generated.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) where True = EOS generated.
        """
        # Get the last token of each sequence
        last_tokens = input_ids[:, -1]

        # Check if any last token matches EOS IDs
        # Use isin for efficient multi-token matching
        eos_on_device = self.eos_token_id.to(last_tokens.device)
        should_stop = torch.isin(last_tokens, eos_on_device)

        return should_stop


class EndStringCriteria(StoppingCriteria):
    """
    Stop when a specific string pattern appears in the generated text.

    WHY: Useful for task-specific stopping conditions (e.g., stop at "</s>",
    "END", "---", or custom delimiters). Works across batch elements.

    Note: This criterion requires decoding token IDs to strings, which can
    be computationally expensive. Use with caution for long sequences.

    Example:
        >>> criteria = EndStringCriteria(tokenizer, end_strings=["</s>", "\n\n"])
        >>> # Stops when generated text contains "</s>" or "\n\n"
    """

    def __init__(
        self,
        tokenizer,
        end_strings: Union[str, List[str]],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EndStringCriteria.

        Parameters
        ----------
        tokenizer : Any
            Tokenizer with decode method (must convert token IDs to strings).
        end_strings : str or list[str]
            String(s) that indicate end of generation.
        device : Optional[torch.device]
            Device for storing stop status (unused, for API consistency).
        """
        if isinstance(end_strings, str):
            end_strings = [end_strings]

        self.tokenizer = tokenizer
        self.end_strings = end_strings
        self._decoded_cache = {}  # Cache decoded strings for efficiency

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if any end string appears in generated text.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size, sequence_length).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) where True = end string found.
        """
        should_stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for i in range(input_ids.shape[0]):
            # Get sequence for this batch item
            seq = input_ids[i].tolist()

            # Use cache key to avoid repeated decoding
            cache_key = tuple(seq)
            if cache_key in self._decoded_cache:
                decoded = self._decoded_cache[cache_key]
            else:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=False)
                self._decoded_cache[cache_key] = decoded

            # Check for any end string
            for end_string in self.end_strings:
                if end_string in decoded:
                    should_stop[i] = True
                    break

        # Limit cache size to prevent memory blowup
        if len(self._decoded_cache) > 1000:
            # Keep only half of the cache
            keys = list(self._decoded_cache.keys())
            for key in keys[:500]:
                del self._decoded_cache[key]

        return should_stop


# ============================================================================
# Time-Based Criteria
# ============================================================================

class TimeoutCriteria(StoppingCriteria):
    """
    Stop after a specified time limit.

    WHY: Useful for interactive applications or when generation must complete
    within a fixed time budget. Ensures the model doesn't take too long.

    Example:
        >>> criteria = TimeoutCriteria(timeout_seconds=30.0)
        >>> # Stops after 30 seconds regardless of other criteria
    """

    def __init__(self, timeout_seconds: float):
        """
        Initialize TimeoutCriteria.

        Parameters
        ----------
        timeout_seconds : float
            Maximum time (in seconds) for generation.
            Stops when elapsed time >= timeout_seconds.
        """
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")

        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if timeout has been exceeded.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs (unused, for API compatibility).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (may contain 'start_time').

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size,) where True = timeout exceeded.
        """
        # Get start time from kwargs or use internal
        if self.start_time is None:
            self.start_time = kwargs.get("start_time", time.time())

        elapsed = time.time() - self.start_time
        should_stop = elapsed >= self.timeout_seconds

        return torch.full(
            (input_ids.shape[0],), should_stop, dtype=torch.bool, device=input_ids.device
        )

    def reset(self):
        """Reset the start time for a new generation."""
        self.start_time = None


class IterationLimitCriteria(StoppingCriteria):
    """
    Stop after a maximum number of generation steps.

    WHY: Alternative to time-based stopping for deterministic behavior.
    Useful for debugging or when you want to limit iterations regardless
    of how many tokens are actually generated.

    Example:
        >>> criteria = IterationLimitCriteria(max_iterations=100)
        >>> # Stops after 100 steps even if EOS not reached
    """

    def __init__(self, max_iterations: int):
        """
        Initialize IterationLimitCriteria.

        Parameters
        ----------
        max_iterations : int
            Maximum number of generation iterations.
        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

        self.max_iterations = max_iterations
        self._iteration = 0

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Check if iteration limit has been reached.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs (unused, for API compatibility).
        scores : Optional[torch.Tensor]
            Unused for this criterion.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Boolean tensor where True = iteration limit reached.
        """
        self._iteration += 1
        should_stop = self._iteration >= self.max_iterations

        return torch.full(
            (input_ids.shape[0],), should_stop, dtype=torch.bool, device=input_ids.device
        )

    def reset(self):
        """Reset iteration counter for new generation."""
        self._iteration = 0


# ============================================================================
# Custom and Composite Criteria
# ============================================================================

class CustomCriteria(StoppingCriteria):
    """
    Wrapper for user-defined stopping function.

    WHY: Allows users to provide custom stopping logic without subclassing.
    Great for quick experiments or task-specific stopping conditions.

    Example:
        >>> def stop_when_question_mark(input_ids, scores, **kwargs):
        ...     # Stop when last token is '?'
        ...     return input_ids[:, -1] == tokenizer.encode("?")[0]
        >>>
        >>> criteria = CustomCriteria(stop_when_question_mark)
    """

    def __init__(self, custom_fn: Callable):
        """
        Initialize CustomCriteria.

        Parameters
        ----------
        custom_fn : Callable
            Function that takes (input_ids, scores, **kwargs) and returns
            a boolean or boolean tensor indicating whether to stop.
        """
        self.custom_fn = custom_fn

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[bool, torch.Tensor]:
        """
        Call the custom stopping function.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs.
        scores : Optional[torch.Tensor]
            Current logits/scores.
        **kwargs
            Additional arguments passed to the custom function.

        Returns
        -------
        Union[bool, torch.Tensor]
            Stop indicator from custom function.
        """
        return self.custom_fn(input_ids, scores, **kwargs)


class CompositeCriteria(StoppingCriteria):
    """
    Combine multiple stopping criteria with custom logic.

    WHY: For complex stopping conditions that can't be expressed as simple
    AND/OR (e.g., "stop when EOS is generated AND length > 20").

    Example:
        >>> criteria = CompositeCriteria(
        ...     [MaxLengthCriteria(100), EosTokenCriteria(2)],
        ...     combine_fn=lambda results: results[0] & results[1]
        ... )
    """

    def __init__(
        self,
        criteria: List[StoppingCriteria],
        combine_fn: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
    ):
        """
        Initialize CompositeCriteria.

        Parameters
        ----------
        criteria : List[StoppingCriteria]
            List of criteria to evaluate.
        combine_fn : Optional[Callable]
            Function that takes list of boolean tensors and returns combined
            boolean tensor. If None, uses OR logic.
        """
        self.criteria = criteria
        self.combine_fn = combine_fn

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Evaluate all criteria and combine results.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs.
        scores : Optional[torch.Tensor]
            Current logits/scores.
        **kwargs
            Additional arguments passed to each criterion.

        Returns
        -------
        torch.Tensor
            Combined stop indicator.
        """
        results = []
        for criterion in self.criteria:
            result = criterion(input_ids, scores, **kwargs)
            if isinstance(result, bool):
                result = torch.full(
                    (input_ids.shape[0],), result, dtype=torch.bool, device=input_ids.device
                )
            results.append(result)

        if self.combine_fn is None:
            # Default: OR logic
            combined = torch.stack(results).any(dim=0)
        else:
            combined = self.combine_fn(results)

        return combined


# ============================================================================
# Utility Functions
# ============================================================================

def get_stopping_criteria(
    generation_config,
    input_ids: torch.Tensor,
    tokenizer: Optional[Any] = None,
    **kwargs,
) -> StoppingCriteriaList:
    """
    Factory function to build a StoppingCriteriaList from generation config.

    WHY: Convenience function that mirrors Hugging Face's _get_stopping_criteria.
    Creates standard criteria: max length, max new tokens, EOS token, etc.

    Parameters
    ----------
    generation_config : GenerationConfig
        Generation configuration with stopping parameters.
    input_ids : torch.Tensor
        Initial input tokens (for determining input length).
    tokenizer : Optional[Any]
        Tokenizer for string-based criteria (e.g., EndStringCriteria).
    **kwargs
        Additional criteria (e.g., custom criteria functions).

    Returns
    -------
    StoppingCriteriaList
        List of stopping criteria to evaluate.

    Example
    -------
        >>> criteria = get_stopping_criteria(config, input_ids)
        >>> should_stop = criteria(input_ids, scores)
    """
    criteria = StoppingCriteriaList()

    # Max length criteria
    if generation_config.max_length is not None:
        criteria.add_criterion(MaxLengthCriteria(generation_config.max_length))

    # Max new tokens criteria (if specified and not covered by max_length)
    if generation_config.max_new_tokens is not None:
        # Only add if max_length wouldn't already cover it
        if generation_config.max_length is None:
            criteria.add_criterion(
                MaxNewTokensCriteria(generation_config.max_new_tokens, input_ids.shape[1])
            )
        else:
            # Add both; the stricter one will trigger first
            criteria.add_criterion(
                MaxNewTokensCriteria(generation_config.max_new_tokens, input_ids.shape[1])
            )

    # EOS token criteria
    if generation_config.eos_token_id is not None:
        criteria.add_criterion(EosTokenCriteria(generation_config.eos_token_id))

    # End string criteria (if tokenizer provided)
    if tokenizer is not None and hasattr(generation_config, "end_strings"):
        if generation_config.end_strings:
            criteria.add_criterion(
                EndStringCriteria(tokenizer, generation_config.end_strings)
            )

    # Custom criteria from kwargs
    if "custom_stopping_criteria" in kwargs:
        for custom_criterion in kwargs["custom_stopping_criteria"]:
            if isinstance(custom_criterion, StoppingCriteria):
                criteria.add_criterion(custom_criterion)
            elif callable(custom_criterion):
                criteria.add_criterion(CustomCriteria(custom_criterion))

    return criteria


def should_stop(
    input_ids: torch.Tensor,
    generation_config,
    scores: Optional[torch.Tensor] = None,
    tokenizer: Optional[Any] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to check if generation should stop.

    Parameters
    ----------
    input_ids : torch.Tensor
        Current token IDs.
    generation_config : GenerationConfig
        Generation configuration.
    scores : Optional[torch.Tensor]
        Current logits/scores.
    tokenizer : Optional[Any]
        Tokenizer for string-based criteria.
    **kwargs
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (batch_size,) indicating stop status.
    """
    criteria = get_stopping_criteria(generation_config, input_ids, tokenizer, **kwargs)
    return criteria(input_ids, scores, **kwargs)