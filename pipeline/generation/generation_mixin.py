# generation/generation_mixin.py
"""
Generation mixin for KilatTransformer with Hugging Face compatible interface.

WHY THIS EXISTS:
    Provides a seamless generate() method that works with the existing KV-cache
    infrastructure. This mixin adds text generation capabilities to any model
    that implements the standard forward() interface with past_key_values support.

DESIGN PHILOSOPHY:
    - Follows Hugging Face's GenerationMixin pattern for familiarity
    - Leverages existing KV-cache for efficient incremental decoding
    - Supports multiple decoding strategies (greedy, sampling, beam search)
    - Composable via logits processors, warpers, samplers, and stopping criteria
    - Zero-copy cache management for optimal performance

ARCHITECTURE:
    GenerationMixin adds a thin layer on top of the model's forward pass:
        1. prepare_inputs_for_generation() - shapes inputs for each step
        2. _update_model_kwargs_for_generation() - updates cache
        3. generate() - orchestrates the decoding loop
        4. Delegates to specialized modules for sampling, beam search, and stopping

INTEGRATION WITH OTHER MODULES:
    - Uses samplers from .sampler for token selection
    - Uses beam_search from .beam_search for beam search strategies
    - Uses stopping_criteria from .stopping_criteria for termination conditions
    - Uses logit_process for logits modification
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .generation_config import GenerationConfig
from .logit_processor import (
    LogitsProcessorList,
    get_logits_processor,
    get_logits_warper,
)
from .sampler import get_sampler
from .beam_search import BeamSearchScorer, beam_search_generate
from .stoping_criteria import (
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    EosTokenCriteria,
    get_stopping_criteria,
)


class GenerationMixin:
    """
    Mixin class adding generate() method to KilatTransformer.

    WHY MIXIN NOT BASE CLASS:
        - Allows existing model class to gain generation capabilities
        - No need to modify the core model inheritance chain
        - Follows HF's pattern of adding generation as a mixin

    IMPLEMENTATION REQUIREMENTS:
        The model must implement:
            - forward(input_ids, past_key_values=None, use_cache=True, ...)
            - prepare_inputs_for_generation() (provided by this mixin)
            - _reorder_cache() (for beam search, provided by this mixin)

    KV-Cache Integration:
        This mixin works seamlessly with the existing KV-cache infrastructure:
            - During generation, use_cache=True is passed to forward()
            - past_key_values are automatically passed between steps
            - The model's KV-cache implementation handles compression/decompression

    Example Usage:
        >>> class KilatTransformer(GenerationMixin, KilatPreTrainedModel):
        ...     # model implementation
        ...     pass
        >>>
        >>> model = KilatTransformer.from_pretrained("./checkpoint")
        >>> output = model.generate(input_ids, max_new_tokens=100)
    """

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the next generation step.

        WHY: When past_key_values exist (cached from previous steps), we only need
        to pass the LAST token of the input sequence, not the entire sequence.
        This is the key optimization that makes generation O(N) instead of O(N²).

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, sequence_length).
        past_key_values : Optional[Tuple]
            Cached key/value states from previous forward passes.
        attention_mask : Optional[torch.Tensor]
            Attention mask for padding (passed through unchanged).
        **kwargs
            Additional keyword arguments (passed through to forward).

        Returns
        -------
        Dict[str, Any]
            Dictionary of arguments to pass to model.forward().
        """
        # If cache exists, we only need the last token for the next prediction
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: Any,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """
        Update keyword arguments for the next generation step.

        WHY: After each forward pass, we need to update the model_kwargs with
        the new past_key_values for the next step.

        Parameters
        ----------
        outputs : Any
            Output from model.forward() (must have .past_key_values attribute).
        model_kwargs : Dict[str, Any]
            Current model keyword arguments (will be updated).
        is_encoder_decoder : bool
            Whether the model is encoder-decoder (unused, for HF compatibility).
        standardize_cache_format : bool
            Whether to standardize cache format (unused).

        Returns
        -------
        Dict[str, Any]
            Updated keyword arguments for the next step.
        """
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values

        return model_kwargs

    def _reorder_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        beam_idx: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Reorder cache for beam search.

        WHY: During beam search, each beam has its own cache state. When we select
        the top beams for the next step, we need to reorder the cache to match
        the new beam ordering.

        For KilatTransformer, each layer's cache is a tuple of (global_state, latent_kv):
            - global_state: (batch_size, n_global_heads, head_dim)
            - latent_kv: (batch_size, sequence_length, latent_dim)

        Parameters
        ----------
        past_key_values : Tuple
            Cached states from all layers.
        beam_idx : torch.Tensor
            Indices of selected beams (shape: batch_size * num_beams).

        Returns
        -------
        Tuple
            Reordered cache with same structure.
        """
        reordered_past = ()

        for layer_past in past_key_values:
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx) for past_state in layer_past
            )
            reordered_past += (reordered_layer,)

        return reordered_past

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        Build logits warper pipeline for sampling.

        WHY: Warpers modify logits during sampling to control randomness.
        The standard pipeline: temperature → top-k → top-p.

        Parameters
        ----------
        generation_config : GenerationConfig
            Generation configuration with sampling parameters.

        Returns
        -------
        LogitsProcessorList
            List of warpers to apply during sampling.
        """
        return get_logits_warper(generation_config)

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_length: int,
        **kwargs,
    ) -> LogitsProcessorList:
        """
        Build logits processor pipeline.

        WHY: Processors modify logits unconditionally (before sampling/warping)
        to enforce constraints like min length, repetition penalty, etc.

        Parameters
        ----------
        generation_config : GenerationConfig
            Generation configuration with processor parameters.
        input_ids_length : int
            Length of input prompt (for length-based processors).

        Returns
        -------
        LogitsProcessorList
            List of processors to apply.
        """
        return get_logits_processor(generation_config, input_ids_length, **kwargs)

    def _get_sampler(
        self,
        generation_config: GenerationConfig,
    ):
        """
        Build sampler for token selection.

        WHY: Sampler determines how to select the next token from logits.
        Different samplers offer different trade-offs between quality and diversity.

        Parameters
        ----------
        generation_config : GenerationConfig
            Generation configuration with sampling parameters.

        Returns
        -------
        Sampler
            Configured sampler instance.
        """
        return get_sampler(
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            typical_p=getattr(generation_config, "typical_p", 1.0),
            contrastive_penalty=getattr(generation_config, "contrastive_penalty", 0.0),
            sampling_strategy=getattr(generation_config, "sampling_strategy", "multinomial"),
        )

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> StoppingCriteriaList:
        """
        Build stopping criteria for generation.

        WHY: Stopping criteria determine when generation should stop (max length,
        EOS token, etc.).

        Parameters
        ----------
        generation_config : GenerationConfig
            Generation configuration with stopping parameters.
        input_ids : torch.Tensor
            Input token IDs (for determining input length).
        **kwargs
            Additional criteria (e.g., custom stopping function).

        Returns
        -------
        StoppingCriteriaList
            List of stopping criteria to evaluate.
        """
        return get_stopping_criteria(generation_config, input_ids, **kwargs)

    def _validate_generation_config(
        self,
        generation_config: GenerationConfig,
        **kwargs,
    ) -> GenerationConfig:
        """
        Validate and merge generation config with kwargs.
        """
        config = copy.deepcopy(generation_config)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set default values if None
        if config.max_new_tokens is not None and config.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {config.max_new_tokens}")
        
        # Handle None values by setting defaults
        if config.temperature is None:
            config.temperature = 1.0
        if config.top_k is None:
            config.top_k = 50
        if config.top_p is None:
            config.top_p = 1.0
        if config.repetition_penalty is None:
            config.repetition_penalty = 1.0
        if config.num_beams is None:
            config.num_beams = 1
        
        # Validate parameters
        if config.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {config.temperature}")
        
        if config.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {config.top_k}")
        
        if not 0 < config.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {config.top_p}")
        
        if config.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {config.repetition_penalty}")
        
        if config.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {config.num_beams}")
        
        # Set default for early_stopping if not present
        if not hasattr(config, 'early_stopping'):
            config.early_stopping = False
        
        return config
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate text autoregressively.

        WHY: Main entry point for text generation. Supports multiple decoding
        strategies:
            - Greedy decoding (do_sample=False, num_beams=1)
            - Multinomial sampling (do_sample=True, num_beams=1)
            - Beam search (num_beams>1, do_sample=False)
            - Beam sampling (num_beams>1, do_sample=True)

        KV-Cache Optimization:
            - Uses the model's built-in KV-cache for O(1) per step
            - For global decay heads: O(1) recurrent update
            - For MLA heads: O(N) attention (optimal for this architecture)

        Parameters
        ----------
        input_ids : Optional[torch.Tensor]
            Input token IDs of shape (batch_size, sequence_length).
        generation_config : Optional[GenerationConfig]
            Generation configuration. If None, uses default config.
        **kwargs
            Override parameters for generation_config.

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            Generated token IDs. If return_dict_in_generate=True, returns a dict
            with 'sequences' and optionally 'scores', 'past_key_values'.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")

        # Setup generation config
        if generation_config is None:
            generation_config = GenerationConfig()

        generation_config = self._validate_generation_config(generation_config, **kwargs)

        # Get model-specific token IDs
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = getattr(self.config, "eos_token_id", None)
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = getattr(self.config, "pad_token_id", 0)
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = getattr(self.config, "bos_token_id", None)

        # Calculate maximum lengths
        input_length = input_ids.shape[1]
        max_new_tokens = generation_config.get_max_new_tokens(input_length)
        max_length = input_length + max_new_tokens

        # Select generation strategy
        do_sample = generation_config.do_sample
        num_beams = generation_config.num_beams

        if num_beams > 1:
            # Beam search or beam sampling
            if do_sample:
                return self._beam_sample(
                    input_ids,
                    generation_config,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )
            else:
                return self._beam_search(
                    input_ids,
                    generation_config,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )
        else:
            # Greedy or sampling (single beam)
            if do_sample:
                return self._sample(
                    input_ids,
                    generation_config,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )
            else:
                return self._greedy_search(
                    input_ids,
                    generation_config,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                )

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        max_length: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Greedy decoding: pick the token with highest probability at each step.

        WHY: Fastest decoding strategy, deterministic. Always picks the argmax
        token. Best for tasks where reproducibility is important.

        Complexity: O(N) per step (dominated by attention)
        """
        generated = input_ids
        past_key_values = None
        eos_token_id = generation_config.eos_token_id

        # Prepare processors
        logits_processor = self._get_logits_processor(
            generation_config, input_ids.shape[1]
        )

        # Prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config, input_ids
        )

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Apply processors
            logits = logits_processor(generated, logits)

            # Greedy: take argmax
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Check stopping criteria
            should_stop = stopping_criteria(generated)
            if should_stop.all():
                break

            # Also check EOS as a fast path
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def _sample(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        max_length: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Multinomial sampling: sample next token from probability distribution.

        WHY: Most flexible decoding strategy, supports temperature, top-k, top-p
        for controlling randomness and diversity. Best for creative tasks.
        """
        generated = input_ids
        past_key_values = None
        eos_token_id = generation_config.eos_token_id

        # Prepare processors, warpers, and sampler
        logits_processor = self._get_logits_processor(
            generation_config, input_ids.shape[1]
        )
        logits_warper = self._get_logits_warper(generation_config)
        sampler = self._get_sampler(generation_config)

        # Prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config, input_ids
        )

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Apply processors and warpers
            logits = logits_processor(generated, logits)
            logits = logits_warper(generated, logits)

            # Sample next token
            next_token = sampler(logits)
            generated = torch.cat([generated, next_token], dim=1)

            # Check stopping criteria
            should_stop = stopping_criteria(generated)
            if should_stop.all():
                break

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        max_length: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Beam search: maintain multiple beams and select the best overall.

        WHY: Higher quality than greedy, but slower. Explores multiple paths
        and selects the one with highest cumulative score. Best for translation,
        summarization, and tasks where quality is critical.
        """
        # Prepare logits processor
        logits_processor = self._get_logits_processor(
            generation_config,
            input_ids.shape[1],
        )

        # Create beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
        )

        # Prepare model kwargs
        model_kwargs = {"use_cache": True}

        # Run beam search
        return beam_search_generate(
            model=self,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            max_length=max_length,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            **model_kwargs,
        )

    def _beam_sample(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        max_length: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Beam sampling: hybrid of beam search and sampling.

        WHY: Combines beam search's exploration with sampling's randomness.
        Useful for diverse generation tasks.
        """
        # Prepare logits processor and warper
        logits_processor = self._get_logits_processor(
            generation_config,
            input_ids.shape[1],
        )
        logits_warper = self._get_logits_warper(generation_config)

        # Create beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
        )

        # Prepare model kwargs
        model_kwargs = {"use_cache": True}

        # Note: Full beam sampling with temperature requires modifications to
        # the beam search algorithm. For now, we fall back to beam search.
        # TODO: Implement true beam sampling with temperature and warpers
        warnings.warn(
            "Beam sampling is not fully implemented. Falling back to beam search.",
            UserWarning,
        )

        return self._beam_search(
            input_ids, generation_config, max_length, max_new_tokens
        )

    def can_generate(self) -> bool:
        """
        Flag for Hugging Face compatibility.

        Returns
        -------
        bool
            True if the model can generate text.
        """
        return True