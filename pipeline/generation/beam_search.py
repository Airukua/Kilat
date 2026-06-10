"""
Beam search implementation for text generation with Hugging Face compatibility.

WHY BEAM SEARCH EXISTS:
    Greedy decoding picks the single most likely token at each step, which can
    lead to locally optimal but globally suboptimal sequences. Beam search
    maintains multiple candidate sequences (beams) and explores different paths
    simultaneously, selecting the overall best sequence by cumulative score.

WHY SEPARATE MODULE:
    - Beam search logic is complex and deserves its own file
    - Can be used independently of the generation mixin
    - Easier to test and debug in isolation
    - Allows for multiple beam search variants (standard, diverse, group)

ALGORITHM OVERVIEW:
    1. Initialize num_beams copies of the input sequence
    2. For each step:
        a. Forward pass all beams simultaneously (efficient batching)
        b. Expand each beam to all possible next tokens (vocab_size)
        c. Compute log probabilities and cumulative scores
        d. Keep only top_k beams (k = num_beams)
        e. Track finished beams (reached EOS)
    3. When all beams finished or max length reached, return best sequence

COMPLEXITY:
    - Time: O(num_beams * vocab_size * sequence_length)
    - Memory: O(num_beams * sequence_length)
    
    For typical values (num_beams=4, vocab_size=50k, seq_len=100):
    This is ~20M operations per step – acceptable for most use cases.

DESIGN DECISIONS:
    - Uses log probabilities for numerical stability
    - Batch processes all beams together for efficiency
    - Handles variable-length sequences with padding
    - Supports length penalty and early stopping
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class BeamSearchScorer:
    """
    Beam search scorer that tracks scores and finished beams.

    WHY SEPARATE SCORER: Encapsulates the bookkeeping of beam scores,
    finished beams, and candidate selection. This separation allows:
        - Different scoring strategies (length penalty, diversity penalty)
        - Easy testing of scoring logic without model forward passes
        - Reuse across different beam search implementations

    Design:
        - Maintains beam_scores tensor (log probabilities)
        - Tracks which beams are finished (reached EOS)
        - Selects top beams while respecting finished status

    Example:
        >>> scorer = BeamSearchScorer(
        ...     batch_size=2,
        ...     num_beams=4,
        ...     device="cuda",
        ...     length_penalty=1.2,
        ... )
        >>> # After each step, call scorer.process() with new scores
        >>> next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
        ...     input_ids, next_scores, next_tokens, next_indices
        ... )
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False,
        num_beam_hyps_to_keep: int = 1,
        num_beam_groups: int = 1,
    ):
        """
        Initialize BeamSearchScorer.

        Parameters
        ----------
        batch_size : int
            Number of independent sequences in the batch.
        num_beams : int
            Number of beams to maintain per batch.
        device : torch.device
            Device to store tensors on.
        length_penalty : float
            Exponential penalty to sequence length.
            > 1.0 encourages shorter sequences (penalizes length)
            < 1.0 encourages longer sequences
            = 1.0 no penalty (default)
        do_early_stopping : bool
            Whether to stop the beam search when at least `num_beams` beams
            are finished per batch. If False, continues until max length.
        num_beam_hyps_to_keep : int
            Number of best hypotheses to keep after search.
        num_beam_groups : int
            Number of groups for diverse beam search.
        """
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups

        # Initialize beam scores: zeros for first beam, -inf for others
        # This ensures only the first beam is active initially
        self.beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        self.beam_scores[:, 1:] = -1e9

        # Track which beams are finished (reached EOS)
        self.beam_hyps = [
            BeamHypotheses(num_beams, length_penalty, do_early_stopping)
            for _ in range(batch_size)
        ]
        self.done = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

    @property
    def is_done(self) -> bool:
        """Return True if all batches are done."""
        return self.done.all()

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the scores of the next tokens and select top beams.

        This is the core scoring function called after each forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current token IDs of shape (batch_size * num_beams, seq_len).
        next_scores : torch.Tensor
            Scores for next tokens of shape (batch_size, num_beams * vocab_size).
        next_tokens : torch.Tensor
            Token IDs of shape (batch_size, num_beams * vocab_size).
        next_indices : torch.Tensor
            Beam indices of shape (batch_size, num_beams * vocab_size).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - beam_scores: Updated scores for next step
            - beam_tokens: Selected token IDs
            - beam_indices: Selected beam indices
        """
        batch_size = len(self.beam_hyps)

        # Reshape to (batch_size, num_beams, vocab_size)
        beam_scores = next_scores.view(batch_size, self.num_beams, -1)
        beam_tokens = next_tokens.view(batch_size, self.num_beams, -1)
        beam_idx = next_indices.view(batch_size, self.num_beams, -1)

        # Prepare for next step
        beam_scores_out = []
        beam_tokens_out = []
        beam_idx_out = []

        for batch_idx in range(batch_size):
            if self.done[batch_idx]:
                # If this batch is done, pad with zeros
                beam_scores_out.append(
                    torch.zeros(self.num_beams, device=self.device)
                )
                beam_tokens_out.append(
                    torch.zeros(self.num_beams, dtype=torch.long, device=self.device)
                )
                beam_idx_out.append(
                    torch.zeros(self.num_beams, dtype=torch.long, device=self.device)
                )
                continue

            # Get current hypotheses for this batch
            hypotheses = self.beam_hyps[batch_idx]

            # Get scores for this batch
            batch_beam_scores = beam_scores[batch_idx]  # (num_beams, vocab_size)
            batch_beam_tokens = beam_tokens[batch_idx]  # (num_beams, vocab_size)
            batch_beam_idx = beam_idx[batch_idx]        # (num_beams, vocab_size)

            # Flatten to (num_beams * vocab_size)
            flat_scores = batch_beam_scores.view(-1)
            flat_tokens = batch_beam_tokens.view(-1)
            flat_indices = batch_beam_idx.view(-1)

            # Add current scores to the existing beam scores
            current_scores = self.beam_scores[batch_idx]  # (num_beams,)
            current_scores = current_scores.unsqueeze(1).expand_as(batch_beam_scores)
            candidate_scores = current_scores + batch_beam_scores

            # Flatten again after adding
            candidate_scores = candidate_scores.view(-1)

            # Sort by score (descending)
            sorted_indices = torch.argsort(candidate_scores, descending=True)

            # Keep only the top 2*num_beams candidates
            sorted_indices = sorted_indices[: 2 * self.num_beams]

            # Selected candidates
            selected_scores = candidate_scores[sorted_indices]
            selected_tokens = flat_tokens[sorted_indices]
            selected_indices = flat_indices[sorted_indices]

            # Process selected candidates
            next_beam_scores = []
            next_beam_tokens = []
            next_beam_indices = []

            for score, token, beam_idx_val in zip(
                selected_scores, selected_tokens, selected_indices
            ):
                # Check if beam is finished
                if hypotheses.is_done:
                    break

                # Add to hypotheses if token is EOS
                if token == self.eos_token_id:
                    # TODO: Need eos_token_id from generation config
                    hypotheses.add(
                        input_ids[batch_idx * self.num_beams + beam_idx_val].clone(),
                        score,
                    )
                else:
                    # Continue with this beam
                    next_beam_scores.append(score)
                    next_beam_tokens.append(token)
                    next_beam_indices.append(beam_idx_val)

                    if len(next_beam_scores) == self.num_beams:
                        break

            # Pad if we don't have enough beams
            while len(next_beam_scores) < self.num_beams:
                next_beam_scores.append(0.0)
                next_beam_tokens.append(0)
                next_beam_indices.append(0)

            # Convert to tensors
            beam_scores_out.append(torch.tensor(next_beam_scores, device=self.device))
            beam_tokens_out.append(torch.tensor(next_beam_tokens, dtype=torch.long, device=self.device))
            beam_idx_out.append(torch.tensor(next_beam_indices, dtype=torch.long, device=self.device))

            # Update current beam scores
            self.beam_scores[batch_idx] = beam_scores_out[-1]

            # Check if this batch is done
            self.done[batch_idx] = self.done[batch_idx] or hypotheses.is_done

        # Stack results
        beam_scores_out = torch.stack(beam_scores_out)
        beam_tokens_out = torch.stack(beam_tokens_out)
        beam_idx_out = torch.stack(beam_idx_out)

        return beam_scores_out, beam_tokens_out, beam_idx_out

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Finalize the beam search and return the best hypotheses.

        Parameters
        ----------
        input_ids : torch.Tensor
            Final token IDs of shape (batch_size * num_beams, seq_len).
        final_beam_scores : torch.Tensor
            Final scores for all beams.
        final_beam_tokens : torch.Tensor
            Final tokens for all beams.
        final_beam_indices : torch.Tensor
            Final beam indices.
        max_length : int
            Maximum sequence length.

        Returns
        -------
        torch.Tensor
            Best sequences of shape (batch_size, best_seq_len).
        """
        batch_size = len(self.beam_hyps)

        # Add remaining beams to hypotheses
        for batch_idx in range(batch_size):
            hypotheses = self.beam_hyps[batch_idx]
            if not hypotheses.is_done:
                for beam_idx in range(self.num_beams):
                    idx = batch_idx * self.num_beams + beam_idx
                    if final_beam_scores[idx] > -1e9:
                        hypotheses.add(
                            input_ids[idx].clone(),
                            final_beam_scores[idx],
                        )

        # Select the best hypotheses
        best_sequences = []
        for batch_idx in range(batch_size):
            best_hyp = self.beam_hyps[batch_idx].best()
            best_sequences.append(best_hyp)

        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in best_sequences)
        padded_sequences = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(best_sequences):
            padded_sequences[i, : len(seq)] = seq

        return padded_sequences


class BeamHypotheses:
    """
    Container for beam hypotheses (completed sequences).

    WHY: Manages the set of completed beams, keeps only the best ones,
    and provides methods to add new hypotheses and check if done.

    Design:
        - Maintains a min-heap of scores (keeping best ones)
        - Only stores sequences that have reached EOS
        - Length penalty applied at addition time
    """

    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
    ):
        """
        Initialize BeamHypotheses.

        Parameters
        ----------
        num_beams : int
            Number of beams to keep.
        length_penalty : float
            Penalty factor for sequence length.
        early_stopping : bool
            Whether to stop when `num_beams` hypotheses are collected.
        """
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beams = []  # List of (score, sequence)
        self.worst_score = 1e9

    def __len__(self) -> int:
        """Number of hypotheses currently stored."""
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float) -> None:
        """
        Add a new hypothesis.

        Parameters
        ----------
        hyp : torch.Tensor
            Sequence of token IDs.
        sum_logprobs : float
            Cumulative log probability (score) of the sequence.
        """
        # Apply length penalty
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self.beams) < self.num_beams:
            # Store if we have capacity
            self.beams.append((score, hyp))
            if len(self.beams) == self.num_beams:
                # Sort to find worst score
                self.beams.sort(key=lambda x: x[0], reverse=True)
                self.worst_score = self.beams[-1][0]
        else:
            # Replace if better than worst
            if score > self.worst_score:
                self.beams[-1] = (score, hyp)
                self.beams.sort(key=lambda x: x[0], reverse=True)
                self.worst_score = self.beams[-1][0]

    @property
    def is_done(self) -> bool:
        """Return True if we have enough hypotheses and early stopping is enabled."""
        if self.early_stopping and len(self.beams) >= self.num_beams:
            return True
        return False

    def best(self) -> torch.Tensor:
        """Return the best hypothesis (highest score)."""
        if not self.beams:
            return torch.tensor([], dtype=torch.long)
        return self.beams[0][1]


def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_scorer: BeamSearchScorer,
    logits_processor,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    **model_kwargs,
) -> torch.Tensor:
    """
    Perform beam search generation.

    This is the main beam search loop that orchestrates the forward passes
    and scoring. It handles:
        - Batch expansion for multiple beams
        - Forward pass with cache
        - Score processing via beam_scorer
        - Cache reordering for next step

    Parameters
    ----------
    model : nn.Module
        Language model with forward() method supporting past_key_values.
    input_ids : torch.Tensor
        Initial input tokens of shape (batch_size, seq_len).
    beam_scorer : BeamSearchScorer
        Scorer that tracks beam scores and finished beams.
    logits_processor : LogitsProcessorList
        Processors to apply to logits (repetition penalty, etc.).
    max_length : int
        Maximum total sequence length.
    pad_token_id : int
        Padding token ID (used for padding).
    eos_token_id : int
        End-of-sequence token ID.
    **model_kwargs
        Additional keyword arguments for model.forward().

    Returns
    -------
    torch.Tensor
        Generated sequences of shape (batch_size, best_seq_len).
    """
    batch_size = len(beam_scorer.beam_hyps)
    num_beams = beam_scorer.num_beams

    # Expand input_ids for multiple beams: (B, N) -> (B * num_beams, N)
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(-1, input_ids.shape[-1])

    # Expand model_kwargs for multiple beams
    for key, value in model_kwargs.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] == batch_size:
                model_kwargs[key] = value.unsqueeze(1).repeat(1, num_beams, 1).view(-1, *value.shape[1:])

    # Initialize past_key_values (cache)
    past_key_values = None

    # Main generation loop
    while True:
        # Forward pass
        outputs = model(
            input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            **model_kwargs,
        )

        logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)
        past_key_values = outputs.past_key_values

        # Apply logits processor
        logits = logits_processor(input_ids, logits)

        # Get next token scores and log probabilities
        next_scores = F.log_softmax(logits, dim=-1)  # (B*num_beams, vocab_size)

        # Reshape for beam scorer: (batch_size, num_beams * vocab_size)
        next_scores = next_scores.view(batch_size, -1)

        # Get top 2*num_beams tokens and scores
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=-1)

        # Get beam indices for each candidate
        next_indices = torch.div(next_tokens, logits.shape[-1], rounding_mode="floor")
        next_tokens = next_tokens % logits.shape[-1]

        # Process scores with beam scorer
        beam_scores, beam_tokens, beam_indices = beam_scorer.process(
            input_ids, next_scores, next_tokens, next_indices
        )

        # Update input_ids and cache
        input_ids = input_ids.view(batch_size, num_beams, -1)
        input_ids = torch.gather(
            input_ids, 1,
            beam_indices.unsqueeze(-1).expand(-1, -1, input_ids.shape[-1]),
        ).view(-1, input_ids.shape[-1])

        # Append new tokens
        input_ids = torch.cat([input_ids, beam_tokens.view(-1, 1)], dim=-1)

        # Reorder past_key_values for next step
        if past_key_values is not None:
            past_key_values = model._reorder_cache(past_key_values, beam_indices.view(-1))

        # Update model_kwargs with new cache
        model_kwargs["past_key_values"] = past_key_values

        # Check if done
        if beam_scorer.is_done:
            break

        # Check max length
        if input_ids.shape[-1] >= max_length:
            break

    # Finalize and return best sequences
    sequences = beam_scorer.finalize(
        input_ids, beam_scores, beam_tokens, beam_indices, max_length
    )

    return sequences


def beam_search_generate(
    model,
    input_ids: torch.Tensor,
    generation_config,
    logits_processor,
    **model_kwargs,
) -> torch.Tensor:
    """
    Convenience wrapper for beam search generation.

    Creates a BeamSearchScorer with parameters from generation_config and
    calls the main beam_search function.

    Parameters
    ----------
    model : nn.Module
        Language model.
    input_ids : torch.Tensor
        Initial input tokens.
    generation_config : GenerationConfig
        Generation configuration with beam search parameters.
    logits_processor : LogitsProcessorList
        Processors for logits.
    **model_kwargs
        Additional arguments for model.forward().

    Returns
    -------
    torch.Tensor
        Generated sequences.
    """
    # Create beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=input_ids.shape[0],
        num_beams=generation_config.num_beams,
        device=input_ids.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
    )

    # Calculate max length
    max_length = input_ids.shape[1] + generation_config.max_new_tokens

    # Run beam search
    return beam_search(
        model=model,
        input_ids=input_ids,
        beam_scorer=beam_scorer,
        logits_processor=logits_processor,
        max_length=max_length,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        **model_kwargs,
    )