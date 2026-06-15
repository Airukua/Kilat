# generation/sampler.py
"""
Sampling strategies for text generation with KilatTransformer.

WHY THIS EXISTS:
    Sampling strategies determine how to select the next token from a probability
    distribution. Different strategies offer different trade-offs between quality,
    diversity, and computational cost. This module provides a comprehensive set
    of sampling methods compatible with Hugging Face's generation API.

SAMPLING STRATEGIES COVERED:
    1. Multinomial sampling - Standard random sampling from distribution
    2. Greedy sampling - Always pick the most likely token (deterministic)
    3. Top-K sampling - Sample only from K most likely tokens
    4. Top-P (nucleus) sampling - Sample from smallest set with cumulative prob ≥ p
    5. Typical sampling - Sample from tokens with entropy near typical value
    6. Contrastive search - Compare current and previous token probabilities
    7. Mirostat sampling - Dynamically adjust temperature based on surprise

DESIGN PHILOSOPHY:
    - Each sampler is a pure function or callable class
    - No side effects: input logits → output token
    - Easy to compose with logits processors and warpers
    - Batched operations for efficiency
    - Compatible with Hugging Face generation patterns

REFERENCE:
    - Holtzman et al. (2019): "The Curious Case of Neural Text Degeneration"
    - Meister et al. (2022): "Typical Decoding for Natural Language Generation"
    - Su et al. (2022): "A Contrastive Framework for Neural Text Generation"
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ============================================================================
# Base Sampling Classes
# ============================================================================

class Sampler:
    """
    Base class for all samplers.

    WHY: Provides a consistent interface for all sampling strategies.
    Each sampler implements a `__call__` method that takes logits and returns
    the next token ID(s) and optionally the probabilities.

    Usage:
        >>> sampler = MultinomialSampler(temperature=0.8)
        >>> next_token = sampler(logits)  # (batch_size, 1)
    """

    def __call__(
        self,
        logits: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample next token(s) from logits.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        **kwargs
            Additional sampler-specific parameters.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If return_probs=False: token IDs of shape (batch_size, 1)
            If return_probs=True: (token_ids, probabilities) tuple
        """
        raise NotImplementedError


class ReturnValue:
    """Helper class for sampler return values."""

    def __init__(self, tokens: torch.Tensor, probs: Optional[torch.Tensor] = None):
        self.tokens = tokens
        self.probs = probs


# ============================================================================
# Basic Sampling Methods
# ============================================================================

class MultinomialSampler(Sampler):
    """
    Standard multinomial sampling from the probability distribution.

    WHY: The most basic sampling method. After applying temperature scaling,
    softmax converts logits to probabilities, then samples according to these
    probabilities. This provides stochastic diversity in generation.

    Mathematical formula:
        p_i = softmax(logits / temperature)_i
        next_token ~ Multinomial(p)

    Temperature effect:
        - T < 1: Sharper distribution (less random, more focused)
        - T = 1: Original distribution
        - T > 1: Flatter distribution (more random, more diverse)

    Example:
        >>> sampler = MultinomialSampler(temperature=0.8)
        >>> next_token = sampler(logits)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize MultinomialSampler.

        Parameters
        ----------
        temperature : float
            Temperature for scaling logits. Must be > 0.
            Lower = less random, higher = more random.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from multinomial distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample
        next_tokens = torch.multinomial(probs, num_samples=1)

        if return_probs:
            # Get probability of sampled tokens
            sampled_probs = probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


class GreedySampler(Sampler):
    """
    Greedy sampling: always pick the most likely token.

    WHY: Fastest and most deterministic sampling method. Equivalent to
    multinomial sampling with temperature → 0. Useful for tasks where
    reproducibility is more important than diversity (e.g., code generation,
    translation, QA).

    Mathematical formula:
        next_token = argmax(softmax(logits))

    Example:
        >>> sampler = GreedySampler()
        >>> next_token = sampler(logits)  # always picks highest probability
    """

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Greedy selection: pick token with highest probability.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probability of selected token.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1, keepdim=True)

        if return_probs:
            sampled_probs = probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


# ============================================================================
# Filtering-Based Sampling Methods
# ============================================================================

class TopKSampler(Sampler):
    """
    Top-K sampling: restrict to K most likely tokens.

    WHY: Prevents sampling from very low-probability tokens that could lead to
    incoherent output. By filtering to only the top K tokens, we maintain
    diversity while ensuring quality.

    How it works:
        1. Keep only the K tokens with highest logits
        2. Set all other tokens to -inf
        3. Sample from the filtered distribution

    Common values:
        - K = 0: disabled (no filtering)
        - K = 40-60: standard for GPT-2 style models
        - K = 10-20: more restrictive (less creative)

    Reference: Fan et al. (2018) "Hierarchical Neural Story Generation"

    Example:
        >>> sampler = TopKSampler(top_k=50, temperature=0.8)
        >>> next_token = sampler(logits)  # samples from top 50 tokens
    """

    def __init__(self, top_k: int = 50, temperature: float = 1.0):
        """
        Initialize TopKSampler.

        Parameters
        ----------
        top_k : int
            Number of top tokens to keep. Must be >= 0.
            0 = disabled (no filtering).
        temperature : float
            Temperature for scaling logits before filtering.
        """
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.top_k = top_k
        self.temperature = temperature

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from top-k filtered distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        if self.top_k <= 0:
            # Fall back to multinomial sampling
            sampler = MultinomialSampler(temperature=self.temperature)
            return sampler(logits, return_probs=return_probs)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Get top-k values
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Create filtered logits (set all non-top-k to -inf)
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits.scatter_(-1, top_k_indices, top_k_values)

        # Convert to probabilities and sample
        probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        if return_probs:
            sampled_probs = probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


class TopPSampler(Sampler):
    """
    Top-P (nucleus) sampling: sample from smallest set with cumulative prob ≥ p.

    WHY: Dynamically adapts the number of tokens considered based on the
    shape of the probability distribution. If the distribution is sharp
    (few high-probability tokens), fewer tokens are considered. If it's flat
    (many similar-probability tokens), more tokens are considered.

    How it works:
        1. Sort tokens by probability (descending)
        2. Keep tokens until cumulative probability exceeds threshold p
        3. Normalize and sample from this dynamic set

    This is the standard sampling method used in GPT-4, LLaMA, and most
    modern LLMs because it adapts to the model's confidence.

    Reference: Holtzman et al. (2019) "The Curious Case of Neural Text Degeneration"

    Example:
        >>> sampler = TopPSampler(top_p=0.92, temperature=0.8)
        >>> next_token = sampler(logits)  # nucleus sampling
    """

    def __init__(self, top_p: float = 0.95, temperature: float = 1.0):
        """
        Initialize TopPSampler.

        Parameters
        ----------
        top_p : float
            Cumulative probability threshold. Must be in (0, 1].
            1.0 = disabled (keep all tokens).
        temperature : float
            Temperature for scaling logits before filtering.
        """
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.top_p = top_p
        self.temperature = temperature

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from nucleus filtered distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        if self.top_p >= 1.0:
            # Fall back to multinomial sampling
            sampler = MultinomialSampler(temperature=self.temperature)
            return sampler(logits, return_probs=return_probs)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability > top_p
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # Shift so that the first token above threshold is also removed
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

        # Apply mask
        filtered_logits = logits.masked_fill(indices_to_remove, -float("inf"))

        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        if return_probs:
            sampled_probs = probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


class TopKTopPSampler(Sampler):
    """
    Combined Top-K and Top-P sampling.

    WHY: Applies both filtering strategies sequentially for even more control.
    First filters to top K tokens, then applies nucleus sampling on the result.
    This is commonly used in practice (e.g., LLaMA uses both).

    Example:
        >>> sampler = TopKTopPSampler(top_k=50, top_p=0.95, temperature=0.8)
        >>> next_token = sampler(logits)  # top-k then top-p
    """

    def __init__(self, top_k: int = 50, top_p: float = 0.95, temperature: float = 1.0):
        """
        Initialize TopKTopPSampler.

        Parameters
        ----------
        top_k : int
            Number of top tokens to keep (first filter).
        top_p : float
            Cumulative probability threshold (second filter).
        temperature : float
            Temperature for scaling logits.
        """
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from top-k then top-p filtered distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Step 1: Top-K filtering
        if self.top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            filtered_logits = torch.full_like(logits, -float("inf"))
            filtered_logits.scatter_(-1, top_k_indices, top_k_values)
        else:
            filtered_logits = logits

        # Step 2: Top-P filtering (if enabled)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)

            # Compute cumulative probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

            filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float("inf"))

        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        if return_probs:
            sampled_probs = probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


# ============================================================================
# Advanced Sampling Methods
# ============================================================================

class TypicalSampler(Sampler):
    """
    Typical sampling: sample from tokens with entropy near the typical value.

    WHY: Avoids both "too random" (high entropy) and "too predictable"
    (low entropy) tokens. Selects tokens whose log-probability is within
    a threshold of the expected log-probability.

    Mathematical formula:
        - Compute entropy H = -Σ p_i log(p_i)
        - Keep tokens where | -log(p_i) - H | < threshold
        - Sample from remaining tokens

    Reference: Meister et al. (2022) "Typical Decoding for Natural Language Generation"

    Example:
        >>> sampler = TypicalSampler(temperature=0.8, typical_p=0.95)
        >>> next_token = sampler(logits)  # typical sampling
    """

    def __init__(self, temperature: float = 1.0, typical_p: float = 0.95):
        """
        Initialize TypicalSampler.

        Parameters
        ----------
        temperature : float
            Temperature for scaling logits.
        typical_p : float
            Proportion of probability mass to keep for typical sampling.
            In (0, 1]. 1.0 = disabled.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0 < typical_p <= 1:
            raise ValueError(f"typical_p must be in (0, 1], got {typical_p}")

        self.temperature = temperature
        self.typical_p = typical_p

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from typical distribution.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        if self.typical_p >= 1.0:
            sampler = MultinomialSampler(temperature=self.temperature)
            return sampler(logits, return_probs=return_probs)

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Get probabilities and log probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute entropy: H = -Σ p_i * log(p_i)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        # Compute shifted log-probabilities
        shifted_log_probs = log_probs + entropy

        # Keep tokens where |shifted_log_probs| < threshold
        # The threshold is determined by typical_p
        sorted_shifted, sorted_indices = torch.sort(shifted_log_probs, descending=True)
        sorted_probs = probs.gather(-1, sorted_indices)

        # Compute cumulative sum and determine cutoff
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumsum < self.typical_p
        cutoff_mask[..., -1] = True  # Ensure at least one token

        # Create mask for kept tokens
        kept_indices = torch.zeros_like(probs, dtype=torch.bool)
        kept_indices.scatter_(-1, sorted_indices, cutoff_mask)

        # Filter logits
        filtered_logits = logits.masked_fill(~kept_indices, -float("inf"))

        # Sample
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_probs, num_samples=1)

        if return_probs:
            sampled_probs = filtered_probs.gather(1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


class ContrastiveSampler(Sampler):
    """
    Contrastive sampling: compare current and previous token probabilities.

    WHY: Encourages diversity by penalizing tokens that are too predictable
    given the previous token. Uses an adaptive penalty based on the model's
    confidence.

    Mathematical formula:
        score = (1 - α) * p(x_t | x_<t) - α * max(0, p(x_t | x_<t) - p(x_t | x_<t-1))

    Where α controls the strength of the contrastive penalty.

    Reference: Su et al. (2022) "A Contrastive Framework for Neural Text Generation"

    Note: Requires access to both current and previous step logits. This
    sampler maintains internal state and is best used in the generation loop
    where previous logits are available.

    Example:
        >>> sampler = ContrastiveSampler(contrastive_penalty=0.5, top_k=4)
        >>> next_token = sampler(logits, prev_logits=prev_logits)
    """

    def __init__(self, contrastive_penalty: float = 0.5, top_k: int = 4):
        """
        Initialize ContrastiveSampler.

        Parameters
        ----------
        contrastive_penalty : float
            Strength of contrastive penalty. In [0, 1].
            Higher = more penalty for degenerate tokens.
        top_k : int
            Number of top tokens to consider for contrastive selection.
        """
        if not 0 <= contrastive_penalty <= 1:
            raise ValueError(f"contrastive_penalty must be in [0, 1], got {contrastive_penalty}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.contrastive_penalty = contrastive_penalty
        self.top_k = top_k

    def __call__(
        self,
        logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample using contrastive search.

        Parameters
        ----------
        logits : torch.Tensor
            Current logits of shape (batch_size, vocab_size).
        prev_logits : Optional[torch.Tensor]
            Previous step logits (for computing degeneration penalty).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        if prev_logits is None:
            warnings.warn(
                "ContrastiveSampler requires prev_logits for best results. "
                "Falling back to TopKSampler for first step.",
                UserWarning,
            )
            sampler = TopKSampler(top_k=self.top_k)
            return sampler(logits, return_probs=return_probs)

        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        prev_probs = F.softmax(prev_logits, dim=-1)

        # Get top-k candidate tokens and their probabilities
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Compute contrastive scores
        # For each candidate token, get its probability from previous step
        prev_probs_for_candidates = prev_probs.gather(-1, top_k_indices)

        # Calculate contrastive score: (1 - α) * current - α * max(0, current - previous)
        degeneration = torch.max(
            torch.zeros_like(top_k_probs),
            top_k_probs - prev_probs_for_candidates,
        )
        scores = (1 - self.contrastive_penalty) * top_k_probs - self.contrastive_penalty * degeneration

        # Select token with highest score
        best_idx = torch.argmax(scores, dim=-1, keepdim=True)
        next_tokens = top_k_indices.gather(-1, best_idx)

        if return_probs:
            sampled_probs = probs.gather(-1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


class MirostatSampler(Sampler):
    """
    Mirostat sampling: dynamically adjust temperature based on surprise.

    WHY: Automatically adapts the generation to achieve a target perplexity
    (surprise level). This balances randomness and coherence without manual
    tuning of temperature.

    How it works:
        1. Compute the entropy of the current distribution
        2. Calculate the target surprise based on desired perplexity
        3. Adjust temperature to move towards target
        4. Sample from the adjusted distribution

    Reference: Basu et al. (2021) "Mirostat: A Neural Text decoding algorithm that
               directly controls perplexity"

    Example:
        >>> sampler = MirostatSampler(target_perplexity=10, tau=2.0, top_k=40)
        >>> next_token = sampler(logits)  # dynamically adjusted temperature
    """

    def __init__(
        self,
        target_perplexity: float = 10.0,
        tau: float = 2.0,
        top_k: int = 40,
        max_temperature: float = 10.0,
        min_temperature: float = 0.1,
    ):
        """
        Initialize MirostatSampler.

        Parameters
        ----------
        target_perplexity : float
            Desired perplexity of generated text. Lower = more coherent.
            Typical range: 5-20.
        tau : float
            Adaptation rate. Higher = faster adaptation.
        top_k : int
            Number of top tokens to consider for sampling.
        max_temperature : float
            Maximum allowed temperature.
        min_temperature : float
            Minimum allowed temperature.
        """
        if target_perplexity <= 0:
            raise ValueError(f"target_perplexity must be > 0, got {target_perplexity}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        self.target_perplexity = target_perplexity
        self.target_surprise = math.log(target_perplexity)
        self.tau = tau
        self.top_k = top_k
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature = 1.0  # Start with neutral temperature
        self._step = 0

    def reset(self):
        """Reset internal state for new generation."""
        self.temperature = 1.0
        self._step = 0

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample using Mirostat algorithm.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        # Apply current temperature
        scaled_logits = logits / self.temperature

        # Get top-k tokens
        top_k_values, top_k_indices = torch.topk(scaled_logits, self.top_k, dim=-1)

        # Compute probabilities for top-k
        top_k_probs = F.softmax(top_k_values, dim=-1)

        # Sample from top-k
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_tokens = top_k_indices.gather(-1, sampled_idx)

        # Compute observed surprise for sampled token
        sampled_prob = top_k_probs.gather(-1, sampled_idx)
        observed_surprise = -torch.log(sampled_prob).mean().item()

        # Compute error and adjust temperature
        error = observed_surprise - self.target_surprise
        self.temperature += self.tau * error
        self.temperature = max(self.min_temperature, min(self.max_temperature, self.temperature))

        self._step += 1

        if return_probs:
            full_probs = F.softmax(scaled_logits, dim=-1)
            sampled_probs = full_probs.gather(-1, next_tokens)
            return next_tokens, sampled_probs

        return next_tokens


# ============================================================================
# Ensemble and Adaptive Methods
# ============================================================================

class AdaptiveSampler(Sampler):
    """
    Adaptive sampling: automatically select best sampling strategy based on entropy.

    WHY: Different sampling strategies work better in different contexts.
    This sampler measures the entropy of the distribution and selects the
    most appropriate strategy dynamically.

    Strategy selection:
        - High entropy (uncertain): use more restrictive sampling (top-p)
        - Medium entropy: use standard sampling (temperature)
        - Low entropy (confident): use greedy or top-k

    Example:
        >>> sampler = AdaptiveSampler(base_temperature=0.8, top_p_threshold=0.95)
        >>> next_token = sampler(logits)  # adapts based on entropy
    """

    def __init__(
        self,
        high_entropy_threshold: float = 2.0,
        low_entropy_threshold: float = 0.5,
        base_temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ):
        """
        Initialize AdaptiveSampler.

        Parameters
        ----------
        high_entropy_threshold : float
            Entropy above this uses top-p sampling.
        low_entropy_threshold : float
            Entropy below this uses greedy/top-k.
        base_temperature : float
            Temperature for medium entropy cases.
        top_p : float
            Top-p value for high entropy cases.
        top_k : int
            Top-k value for low entropy cases.
        """
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold
        self.base_temperature = base_temperature
        self.top_p = top_p
        self.top_k = top_k

        self._top_p_sampler = TopPSampler(top_p=top_p, temperature=base_temperature)
        self._temperature_sampler = MultinomialSampler(temperature=base_temperature)
        self._top_k_sampler = TopKSampler(top_k=top_k, temperature=base_temperature)

    def __call__(
        self,
        logits: torch.Tensor,
        return_probs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample using adaptive strategy based on entropy.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, vocab_size).
        return_probs : bool
            If True, also return probabilities of sampled tokens.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Token IDs and optionally their probabilities.
        """
        # Compute entropy of the distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # Select strategy based on entropy
        if entropy > self.high_entropy_threshold:
            sampler = self._top_p_sampler
        elif entropy < self.low_entropy_threshold:
            sampler = self._top_k_sampler
        else:
            sampler = self._temperature_sampler

        return sampler(logits, return_probs=return_probs)


# ============================================================================
# Factory Function
# ============================================================================

def get_sampler(
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    typical_p: float = 1.0,
    contrastive_penalty: float = 0.0,
    sampling_strategy: str = "multinomial",
    **kwargs,
) -> Sampler:
    """
    Factory function to get the appropriate sampler based on parameters.

    WHY: Provides a unified interface for creating samplers from generation
    configuration parameters. Follows Hugging Face's pattern.

    Parameters
    ----------
    do_sample : bool
        Whether to sample (if False, returns GreedySampler).
    temperature : float
        Temperature for scaling.
    top_k : int
        Top-K value (0 = disabled).
    top_p : float
        Top-P value (1.0 = disabled).
    typical_p : float
        Typical sampling threshold (1.0 = disabled).
    contrastive_penalty : float
        Contrastive search penalty (0.0 = disabled).
    sampling_strategy : str
        Strategy name: "multinomial", "greedy", "top_k", "top_p", "top_k_top_p",
        "typical", "contrastive", "mirostat", "adaptive".
    **kwargs
        Additional sampler-specific parameters.

    Returns
    -------
    Sampler
        Configured sampler instance.

    Example
    -------
        >>> sampler = get_sampler(do_sample=True, temperature=0.8, top_p=0.95)
        >>> next_token = sampler(logits)
    """
    if not do_sample:
        return GreedySampler()

    if sampling_strategy == "greedy":
        return GreedySampler()

    if sampling_strategy == "top_k" or (top_k > 0 and top_p >= 1.0 and typical_p >= 1.0):
        return TopKSampler(top_k=top_k, temperature=temperature)

    if sampling_strategy == "top_p" or (top_p < 1.0 and top_k <= 0 and typical_p >= 1.0):
        return TopPSampler(top_p=top_p, temperature=temperature)

    if sampling_strategy == "top_k_top_p" or (top_k > 0 and top_p < 1.0):
        return TopKTopPSampler(top_k=top_k, top_p=top_p, temperature=temperature)

    if sampling_strategy == "typical" or typical_p < 1.0:
        return TypicalSampler(temperature=temperature, typical_p=typical_p)

    if sampling_strategy == "contrastive" or contrastive_penalty > 0:
        return ContrastiveSampler(contrastive_penalty=contrastive_penalty, **kwargs)

    if sampling_strategy == "mirostat":
        return MirostatSampler(**kwargs)

    if sampling_strategy == "adaptive":
        return AdaptiveSampler(base_temperature=temperature, top_p=top_p, top_k=top_k)

    # Default: standard multinomial sampling
    return MultinomialSampler(temperature=temperature)