"""
Logits processors for controlling text generation.

WHY THIS EXISTS:
    Logits processors modify the model's output logits before sampling/decoding,
    enabling controlled generation without modifying the model itself.
    This follows Hugging Face's design patterns for maximum compatibility [citation:3][citation:9].

DESIGN PHILOSOPHY:
    - Processor: modifies logits unconditionally (e.g., repetition penalty)
    - Warper: modifies logits during sampling (e.g., temperature, top-k, top-p)
    - Both inherit from the same base class for API consistency 
    - Processors are applied sequentially via LogitsProcessorList 
"""

from __future__ import annotations
import inspect
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
import torch.nn.functional as F



class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during generation.

    WHY: Provides a consistent interface for modifying logits. Processors can be
    chained together via LogitsProcessorList to apply multiple transformations
    sequentially [citation:9].

    Design:
        - Processors are called with (input_ids, scores) and return modified scores
        - Processors modify logits BEFORE warpers (temperature, top-k, top-p)
        - Processors are applied regardless of sampling strategy

    Example Usage:
        >>> class MyProcessor(LogitsProcessor):
        ...     def __call__(self, input_ids, scores):
        ...         scores[:, 123] = -float("inf")  # ban token 123
        ...         return scores
    """

    @abstractmethod
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply processor to logits.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, sequence_length).
        scores : torch.Tensor
            Logits of shape (batch_size, vocab_size).

        Returns
        -------
        torch.Tensor
            Modified logits of same shape as input scores.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper(LogitsProcessor):
    """
    Abstract base class for logit warpers used during multinomial sampling.

    WHY: Warpers are conceptually similar to processors but are specifically
    applied during sampling (when do_sample=True). The distinction is largely
    semantic but helps organize code [citation:2].

    Examples of warpers:
        - TemperatureLogitsWarper: scales logits by temperature
        - TopKLogitsWarper: keeps only top-k tokens
        - TopPLogitsWarper: keeps tokens above cumulative probability threshold
    """

    pass  # Same interface as LogitsProcessor, semantic distinction only


class LogitsProcessorList(list):
    """
    A list of logits processors that can be called sequentially.

    WHY: Chains multiple processors together, applying them in order [citation:9].
    This allows composing simple processors into complex pipelines.

    Features:
        - Inherits from list for familiarity
        - __call__ applies all processors in sequence [citation:3]
        - Supports additional kwargs for processors that need them

    Example Usage:
        >>> processors = LogitsProcessorList()
        >>> processors.append(MinLengthLogitsProcessor(10, eos_id))
        >>> processors.append(RepetitionPenaltyLogitsProcessor(1.2))
        >>> scores = processors(input_ids, scores)  # applies both
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply all processors in the list to the scores.

        Each processor is called sequentially, with the output of one becoming
        the input to the next. Processors are applied in the order they were
        added to the list.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, sequence_length).
        scores : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        **kwargs
            Additional kwargs passed to processors that support them.

        Returns
        -------
        torch.Tensor
            Processed logits.
        """
        for processor in self:
            # Check if processor expects additional kwargs [citation:9]
            sig = inspect.signature(processor.__call__)
            if len(sig.parameters) > 2:
                # Processor has kwargs parameter or additional args
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    """
    Enforces minimum generation length by suppressing EOS token.

    WHY: Prevents the model from ending generation too early. Useful for
    tasks requiring outputs of at least a certain length.

    How it works:
        - Sets the score of EOS token to -inf if current length < min_length
        - Makes it impossible for the model to choose EOS until min_length reached
        - Length includes the prompt (for decoder-only models) [citation:9]

    Example:
        >>> processor = MinLengthLogitsProcessor(min_length=50, eos_token_id=2)
        >>> # Model cannot output EOS until at least 50 tokens are generated
    """

    def __init__(
        self,
        min_length: int,
        eos_token_id: Union[int, List[int], torch.Tensor],
        device: str = "cpu",
    ):
        """
        Initialize MinLengthLogitsProcessor.

        Parameters
        ----------
        min_length : int
            Minimum total length below which EOS is forbidden.
        eos_token_id : int or list[int] or torch.Tensor
            Token ID(s) for end-of-sequence token.
        device : str
            Device to allocate tensors on (default: "cpu").
        """
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"min_length must be a non-negative integer, got {min_length}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply min-length constraint by suppressing EOS if below threshold."""
        # Create mask for EOS token positions
        vocab_range = torch.arange(scores.shape[-1], device=scores.device)
        eos_mask = torch.isin(vocab_range, self.eos_token_id)

        # Clone scores to avoid modifying original
        processed_scores = scores.clone()

        # Suppress EOS if below minimum length
        if input_ids.shape[-1] < self.min_length:
            processed_scores = torch.where(eos_mask, -float("inf"), scores)

        return processed_scores


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    """
    Enforces minimum NEW tokens length, ignoring the prompt.

    WHY: Unlike MinLengthLogitsProcessor which includes the prompt, this only
    counts newly generated tokens. Useful when you care about the length of
    generated output, not including the input.

    How it works:
        - Tracks prompt length separately from generated tokens
        - Only suppresses EOS when generated tokens < min_new_tokens
        - Requires prompt_length_to_skip (usually input_length) [citation:9]

    Example:
        >>> # Ensure at least 100 new tokens are generated
        >>> processor = MinNewTokensLengthLogitsProcessor(
        ...     prompt_length_to_skip=10, min_new_tokens=100, eos_token_id=2
        ... )
    """

    def __init__(
        self,
        prompt_length_to_skip: int,
        min_new_tokens: int,
        eos_token_id: Union[int, List[int], torch.Tensor],
        device: str = "cpu",
    ):
        """
        Initialize MinNewTokensLengthLogitsProcessor.

        Parameters
        ----------
        prompt_length_to_skip : int
            Length of the input prompt (tokens to exclude from counting).
        min_new_tokens : int
            Minimum number of NEW tokens below which EOS is forbidden.
        eos_token_id : int or list[int] or torch.Tensor
            Token ID(s) for end-of-sequence token.
        device : str
            Device to allocate tensors on (default: "cpu").
        """
        if not isinstance(prompt_length_to_skip, int) or prompt_length_to_skip < 0:
            raise ValueError(f"prompt_length_to_skip must be >= 0, got {prompt_length_to_skip}")

        if not isinstance(min_new_tokens, int) or min_new_tokens < 0:
            raise ValueError(f"min_new_tokens must be >= 0, got {min_new_tokens}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply min-new-tokens constraint by suppressing EOS if below threshold."""
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip

        # Create mask for EOS token positions
        vocab_range = torch.arange(scores.shape[-1], device=scores.device)
        eos_mask = torch.isin(vocab_range, self.eos_token_id)

        processed_scores = scores.clone()

        if new_tokens_length < self.min_new_tokens:
            processed_scores = torch.where(eos_mask, -float("inf"), scores)

        return processed_scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Penalizes tokens that have already appeared in the generated sequence.

    WHY: Reduces repetitive output by lowering the probability of tokens that
    have been used before. The penalty is applied by dividing logits of seen
    tokens [citation:3].

    How it works:
        - For tokens in the current sequence, apply penalty factor
        - For negative logits: multiply by penalty (makes them more negative)
        - For positive logits: divide by penalty (reduces probability)
        - Penalty > 1.0 discourages repetition [citation:3]
        - Penalty < 1.0 encourages repetition (rarely used)

    Typical values: 1.1 to 1.2 (slight discouragement) [citation:3]

    Example:
        >>> processor = RepetitionPenaltyLogitsProcessor(penalty=1.15)
        >>> # Tokens already seen are 15% less likely to be repeated
    """

    def __init__(self, penalty: float):
        """
        Initialize RepetitionPenaltyLogitsProcessor.

        Parameters
        ----------
        penalty : float
            Penalty factor (>1 discourages repetition, <1 encourages).
            Must be > 0. Default: 1.0 (no effect).
        """
        if penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply repetition penalty to tokens already seen."""
        # Find unique tokens in the current sequence (excluding the last token)
        # because we haven't generated it yet, but penalizing it would be self-censoring
        # Actually, HF implementation penalizes all seen tokens including the last one?
        # We'll follow the standard approach: penalize all tokens in the sequence
        processed_scores = scores.clone()

        for batch_idx in range(input_ids.shape[0]):
            # Get unique tokens from this batch's sequence
            unique_tokens = torch.unique(input_ids[batch_idx])

            for token_id in unique_tokens:
                if processed_scores[batch_idx, token_id] < 0:
                    # Negative logits: make them more negative
                    processed_scores[batch_idx, token_id] *= self.penalty
                else:
                    # Positive logits: make them smaller
                    processed_scores[batch_idx, token_id] /= self.penalty

        return processed_scores


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Prevents repetition of n-grams.

    WHY: Blocks the model from generating the same n-gram twice, which is
    effective at stopping repetitive loops (e.g., "I love you I love you").

    How it works:
        - Tracks all n-grams in the generated sequence
        - For the next token, forbids any token that would create a duplicate n-gram
        - size=2 blocks bigrams, size=3 blocks trigrams, etc.

    Typical value: 3 or 4 (blocks trigrams/quadgrams) [citation:4]

    Example:
        >>> processor = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=3)
        >>> # Prevents any trigram from appearing twice
    """

    def __init__(self, no_repeat_ngram_size: int):
        """
        Initialize NoRepeatNGramLogitsProcessor.

        Parameters
        ----------
        no_repeat_ngram_size : int
            Size of n-grams to block repetition. Must be > 0.
        """
        if no_repeat_ngram_size <= 0:
            raise ValueError(f"no_repeat_ngram_size must be > 0, got {no_repeat_ngram_size}")

        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Suppress tokens that would create duplicate n-grams."""
        # This is a simplified implementation. Full implementation requires
        # tracking n-grams across the batch and beam dimensions.
        # For now, we'll implement a basic version that works for greedy.

        processed_scores = scores.clone()
        batch_size, vocab_size = scores.shape

        for batch_idx in range(batch_size):
            # Extract the current sequence for this batch
            seq = input_ids[batch_idx].tolist()
            seq_len = len(seq)

            if seq_len < self.no_repeat_ngram_size:
                continue

            # Get the n-gram that would be completed by the next token
            # The last (n-1) tokens form the prefix
            prefix = seq[-(self.no_repeat_ngram_size - 1):]

            # Find tokens that would create a duplicate n-gram
            # Check all positions where this prefix appears
            for pos in range(seq_len - self.no_repeat_ngram_size + 1):
                current_ngram = seq[pos:pos + self.no_repeat_ngram_size]

                # If the prefix matches and the n-gram is at a different position
                if current_ngram[:-1] == prefix and pos + self.no_repeat_ngram_size != seq_len:
                    # The token that would duplicate the n-gram
                    forbidden_token = current_ngram[-1]
                    processed_scores[batch_idx, forbidden_token] = -float("inf")

        return processed_scores

class TemperatureLogitsWarper(LogitsProcessor):
    """
    Applies temperature scaling to logits.

    WHY: Controls the randomness of the output distribution [citation:3].
    Higher temperature = more random (flatter distribution).
    Lower temperature = more deterministic (sharper peaks).

    Mathematical formula:
        logits = logits / temperature

    Effect:
        - temperature → 0: collapses to argmax (greedy)
        - temperature = 1: unchanged (standard sampling)
        - temperature → ∞: uniform distribution

    Typical values: 0.7 (creative but coherent), 0.9 (slightly creative),
                    1.2 (more random) [citation:9]

    Example:
        >>> warper = TemperatureLogitsWarper(temperature=0.8)
        >>> # Reduces randomness slightly, making output more focused
    """

    def __init__(self, temperature: float):
        """
        Initialize TemperatureLogitsWarper.

        Parameters
        ----------
        temperature : float
            Temperature value (>0). Lower = less random, higher = more random.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return scores / self.temperature


class TopKLogitsWarper(LogitsProcessor):
    """
    Keeps only the top K tokens by probability.

    WHY: Restricts sampling to only the K most likely tokens, ignoring
    very low-probability tokens that could lead to incoherent output [citation:3].

    How it works:
        - Sorts logits in descending order
        - Keeps only the first K tokens
        - Sets all other tokens to -inf

    Common values:
        - 0: disabled (keep all tokens)
        - 40-60: standard for GPT-2 style models
        - 10-20: more restrictive (less creative)

    Example:
        >>> warper = TopKLogitsWarper(top_k=50)
        >>> # Only the 50 most likely tokens can be sampled
    """

    def __init__(self, top_k: int):
        """
        Initialize TopKLogitsWarper.

        Parameters
        ----------
        top_k : int
            Number of top tokens to keep. Must be >= 0.
            If 0, disabled (keeps all tokens).
        """
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")

        self.top_k = top_k

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Keep only top-k tokens, set others to -inf."""
        if self.top_k <= 0:
            return scores

        # Get threshold value for top-k
        # topk returns values and indices, we only need the values
        top_k_values = torch.topk(scores, self.top_k, dim=-1)[0]

        # Get the minimum value among top-k (the threshold)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)

        # Set all values below threshold to -inf
        scores_processed = torch.where(scores < min_top_k, -float("inf"), scores)

        return scores_processed


class TopPLogitsWarper(LogitsProcessor):
    """
    Keeps tokens with cumulative probability >= p (nucleus sampling).

    WHY: Dynamic alternative to Top-K that adapts to the distribution shape [citation:3].
    If the distribution is sharp (few high-probability tokens), only a few tokens kept.
    If distribution is flat (many similar probability tokens), more tokens kept.

    How it works:
        - Sort tokens by probability (descending)
        - Keep tokens until cumulative probability exceeds threshold p
        - Set all other tokens to -inf

    Typical values:
        - 1.0: disabled (keep all tokens)
        - 0.9: nucleus sampling with 90% probability mass
        - 0.95: less restrictive

    This is the standard sampling method used in GPT-4, LLaMA, etc.

    Example:
        >>> warper = TopPLogitsWarper(top_p=0.92)
        >>> # Keeps the smallest set of tokens whose cumulative probability >= 92%
    """

    def __init__(self, top_p: float):
        """
        Initialize TopPLogitsWarper.

        Parameters
        ----------
        top_p : float
            Cumulative probability threshold. Must be in (0, 1].
            If 1.0, disabled (keeps all tokens).
        """
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        self.top_p = top_p

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Keep tokens with cumulative probability above threshold."""
        if self.top_p >= 1.0:
            return scores

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find indices to remove (cumulative prob > top_p)
        # Shift so that the first token above threshold is also removed
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask and apply to original logits
        indices_to_remove = torch.zeros_like(scores, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        scores_processed = scores.clone()
        scores_processed = scores_processed.masked_fill(indices_to_remove, -float("inf"))

        return scores_processed


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    Forces the first generated token to be a specific token.

    WHY: Useful for prompt-based generation where you want to control
    the start of the output (e.g., forced JSON start, XML tag).

    How it works:
        - At the first generation step, sets all logits to -inf except the forced token
        - After the first step, does nothing

    Example:
        >>> processor = ForcedBOSTokenLogitsProcessor(bos_token_id=1)
        >>> # Forces the first generated token to always be token 1
    """

    def __init__(self, bos_token_id: int):
        """
        Initialize ForcedBOSTokenLogitsProcessor.

        Parameters
        ----------
        bos_token_id : int
            Token ID to force as the first generated token.
        """
        self.bos_token_id = bos_token_id
        self.first_step = True

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Force the first token to be the specified BOS token."""
        if self.first_step:
            self.first_step = False
            processed_scores = torch.full_like(scores, -float("inf"))
            processed_scores[:, self.bos_token_id] = scores[:, self.bos_token_id]
            return processed_scores
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    Forces the last token before EOS to be a specific token.

    WHY: Ensures that a sequence ends with a particular token structure,
    useful for JSON, XML, or other structured output.

    How it works:
        - Calculates how many tokens are left until max length
        - If only one token remains, forces it to be EOS
    """

    def __init__(self, max_length: int, eos_token_id: int):
        """
        Initialize ForcedEOSTokenLogitsProcessor.

        Parameters
        ----------
        max_length : int
            Maximum sequence length.
        eos_token_id : int
            Token ID to force at the end.
        """
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Force EOS token when reaching max length."""
        if input_ids.shape[-1] == self.max_length - 1:
            processed_scores = torch.full_like(scores, -float("inf"))
            processed_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]
            return processed_scores
        return scores


class SuppressTokensLogitsProcessor(LogitsProcessor):
    """
    Suppresses specific tokens by setting their logits to -inf.

    WHY: Allows blacklisting specific tokens that should never be generated
    (e.g., profanity, control characters, unwanted special tokens).

    Example:
        >>> processor = SuppressTokensLogitsProcessor([50256, 50257, 50258])
        >>> # These 3 tokens will never be generated
    """

    def __init__(self, suppress_tokens: List[int]):
        """
        Initialize SuppressTokensLogitsProcessor.

        Parameters
        ----------
        suppress_tokens : List[int]
            List of token IDs to suppress (set to -inf).
        """
        self.suppress_tokens = suppress_tokens

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Suppress specific tokens by setting them to -inf."""
        processed_scores = scores.clone()
        for token_id in self.suppress_tokens:
            processed_scores[:, token_id] = -float("inf")
        return processed_scores


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    """
    Constraints generation to tokens allowed by a prefix function.

    WHY: Enforces grammar or structure constraints (e.g., JSON schema,
    arithmetic expressions) by dynamically determining allowed next tokens.

    How it works:
        - Call prefix_allowed_tokens_fn(batch_id, input_ids) to get allowed tokens
        - Suppresses all other tokens

    Example:
        >>> def json_allowed_tokens_fn(batch_id, input_ids):
        ...     # Return tokens that would produce valid JSON
        ...     return allowed_token_ids

        >>> processor = PrefixConstrainedLogitsProcessor(json_allowed_tokens_fn)
        >>> # Only tokens that maintain valid JSON structure are allowed
    """

    def __init__(self, prefix_allowed_tokens_fn, num_beams: int = 1):
        """
        Initialize PrefixConstrainedLogitsProcessor.

        Parameters
        ----------
        prefix_allowed_tokens_fn : callable
            Function that takes (batch_id, input_ids) and returns list of allowed token IDs.
        num_beams : int
            Number of beams for beam search.
        """
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = num_beams

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Filter logits to only allowed tokens from prefix function."""
        processed_scores = scores.clone()
        mask = torch.full_like(scores, -float("inf"))

        for batch_idx in range(scores.shape[0]):
            # Determine which beam this batch corresponds to
            beam_idx = batch_idx // self.num_beams

            # Get allowed tokens for this prefix
            allowed_tokens = self.prefix_allowed_tokens_fn(beam_idx, input_ids[batch_idx])

            if allowed_tokens:
                mask[batch_idx, allowed_tokens] = scores[batch_idx, allowed_tokens]

        return mask


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    """
    Constrains generation using a grammar (e.g., for structured output).

    WHY: Enforces formal grammar constraints (e.g., JSON, arithmetic, programming
    languages) by tracking the parsing state and only allowing valid next tokens.
    """

    def __init__(self, grammar_constraint):
        """
        Initialize GrammarConstrainedLogitsProcessor.

        Parameters
        ----------
        grammar_constraint : object
            Grammar constraint object with:
            - init_stacks(): initialize parser state
            - accept_token_ids(prefix, stacks): advance parser
            - batch_filter_vocab(stacks, device): return mask of allowed tokens
        """
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None
        self.last_length = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Filter logits to tokens allowed by grammar."""
        # Initialize stacks on first call
        if self.batch_stacks is None:
            self.batch_stacks = [
                self.grammar_constraint.init_stacks() for _ in range(len(input_ids))
            ]

        # Process incremental tokens
        if self.last_length is None:
            # First call: parse entire prefix
            for i, single_input_ids in enumerate(input_ids):
                self.batch_stacks[i] = self.grammar_constraint.accept_token_ids(
                    single_input_ids.tolist(), self.batch_stacks[i]
                )
        elif len(input_ids[0]) == self.last_length + 1:
            # Incremental update: one new token
            for i, single_input_ids in enumerate(input_ids):
                self.batch_stacks[i] = self.grammar_constraint.accept_token_id(
                    single_input_ids[-1].item(), self.batch_stacks[i]
                )
        else:
            # Unexpected length change
            raise RuntimeError(
                "Input IDs length inconsistent with processor state. "
                "If processing a new sequence, create a new processor instance."
            )

        # Get allowed token mask from grammar
        acceptance = self.grammar_constraint.batch_filter_vocab(self.batch_stacks, scores.device)

        # Filter logits
        processed_scores = scores.clone()
        processed_scores[~acceptance] = -float("inf")

        self.last_length = len(input_ids[0])
        return processed_scores

def get_logits_warper(
    generation_config,  # GenerationConfig instance
    **kwargs,
) -> LogitsProcessorList:
    """
    Create a list of logits warpers from generation configuration.

    WHY: Convenience function that mirrors Hugging Face's _get_logits_warper [citation:4].
    Creates the standard warper pipeline: temperature, top-k, top-p.

    Example:
        >>> config = GenerationConfig(do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
        >>> warpers = get_logits_warper(config)
        >>> scores = warpers(input_ids, scores)  # apply all warpers
    """

    warpers = LogitsProcessorList()

    # Warpers only apply when sampling
    if not generation_config.do_sample:
        return warpers

    # Temperature warper
    if generation_config.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(generation_config.temperature))

    # Top-K warper
    if generation_config.top_k > 0:
        warpers.append(TopKLogitsWarper(generation_config.top_k))

    # Top-P warper (nucleus sampling)
    if generation_config.top_p < 1.0:
        warpers.append(TopPLogitsWarper(generation_config.top_p))

    return warpers


def get_logits_processor(
    generation_config,  # GenerationConfig instance
    input_ids_length: int,
    **kwargs,
) -> LogitsProcessorList:
    """
    Create a list of logits processors from generation configuration.

    WHY: Convenience function that mirrors Hugging Face's _get_logits_processor [citation:4].
    Creates standard processor pipeline: repetition penalty, min length, etc.

    Example:
        >>> config = GenerationConfig(repetition_penalty=1.15, min_length=20)
        >>> processors = get_logits_processor(config, input_ids_length=10)
        >>> scores = processors(input_ids, scores)  # apply all processors
    """

    processors = LogitsProcessorList()

    # Repetition penalty
    if generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(generation_config.repetition_penalty))

    # N-gram repetition blocking
    if generation_config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))

    # Min length constraint (total length including prompt)
    if generation_config.min_length > 0 and generation_config.eos_token_id is not None:
        processors.append(
            MinLengthLogitsProcessor(
                generation_config.min_length,
                generation_config.eos_token_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    # Prefix-constrained generation
    if kwargs.get("prefix_allowed_tokens_fn") is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                kwargs["prefix_allowed_tokens_fn"],
                num_beams=generation_config.num_beams,
            )
        )

    return processors