"""
Autoregressive text generator with full KV‑cache support.

This module implements the core generation loop using incremental decoding.
The generator is responsible for tokenisation, sampling, and managing the
KV‑cache returned by the KilatTransformer model.

Why incremental decoding? Without caching, each new token would require
re‑computing attention for the entire previous sequence – O(N²) per step.
With caching, we only compute attention for the new token, achieving O(N) per step.
This is essential for long‑form generation.

The generator works in two phases:
1. **Prompt processing**: Feed the whole input prompt once to obtain initial logits
   and KV‑caches for all transformer layers.
2. **Token generation loop**: For each new token, feed only the last token together
   with the cached states from previous steps.
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Optional
from transformers import AutoTokenizer

from model import KilatTransformerHF
from .generation_config import GenerationConfig


class KilatGenerator:
    """Efficient generator using the model's incremental KV‑cache."""

    def __init__(
        self,
        model: KilatTransformerHF,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Generation is compute-bound; GPU acceleration provides ~10-100x speedup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        # KV-cache is mandatory for efficient generation. The model's forward()
        # checks this flag to determine whether to return past_key_values.
        self.model.config.use_cache = True

        # Cache frequently used token IDs to avoid repeated attribute lookups
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, List[str]],
        gen_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate completions for one or more prompts.
        
        Args:
            prompt: Single string or list of strings to complete
            gen_config: Sampling parameters (temperature, top-p, etc.)
            **kwargs: Overrides for gen_config fields (e.g., temperature=0.8)
        
        Returns:
            Generated text(s) matching input type (str for str, List[str] for List)
        
        Performance characteristics:
            - First forward pass processes entire prompt at once (parallelized)
            - Each subsequent step processes only 1 token + cached attention states
            - Memory usage scales with batch_size * max_new_tokens for generated tokens
        
        Design decision: Keep finished sequences in the batch rather than dynamically
        removing them. Dynamic removal would require re-packing caches (complex) and
        provides minimal benefit for batch_size < 8. The cost is a few extra forward
        passes on pad_token_id, which is cheap.
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        # Convenience: allow callers to override via kwargs without creating config obj
        for key, value in kwargs.items():
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)

        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else prompt

        # Tokenization with padding to handle variable-length prompts
        # Note: attention_mask is currently ignored by KilatAttention but included
        # for HF model compatibility and future implementations.
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.n_embd,   # Hard upper bound from architecture
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        batch_size, prompt_len = input_ids.shape
        # Tracks which sequences are still generating (haven't hit EOS + satisfied min_new_tokens)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        generated = input_ids.clone()          # Accumulates all tokens (prompt + new)
        past_key_values = None

        # ---------- Phase 1: Process the full prompt ----------
        # Single forward pass computes logits for the last token AND initial KV-caches
        # for all layers. This is O(prompt_len) vs. O(prompt_len²) if done token-by-token.
        outputs = self.model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=None,
        )
        logits = outputs.logits[:, -1, :]            # Only need last position's logits
        past_key_values = outputs.past_key_values    # Tuple of (key, value) per layer

        # ---------- Phase 2: Autoregressive generation ----------
        for step in range(gen_config.max_new_tokens):
            # Early exit: all sequences have reached EOS and satisfied min_new_tokens
            if not unfinished.any():
                break

            # Apply temperature to control randomness concentration
            # Temperature 0 would cause division by zero, but caller guarantees >0
            # when do_sample=True. For greedy case, temperature is ignored anyway.
            if gen_config.temperature > 0:
                logits = logits / gen_config.temperature

            # Repetition penalty: suppresses tokens already seen in the sequence
            # Implemented as division because logits are in log-space; this matches
            # standard practice in Hugging Face and llama.cpp.
            if gen_config.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, gen_config.repetition_penalty
                )

            # Branch: stochastic sampling vs. deterministic greedy
            if gen_config.do_sample and gen_config.temperature > 0:
                next_tokens = self._sample(
                    logits,
                    top_k=gen_config.top_k,
                    top_p=gen_config.top_p,
                )
            else:
                next_tokens = torch.argmax(logits, dim=-1)   # Greedy decoding

            # Finished sequences should output pad_token_id. Their logits are meaningless,
            # but we keep them in the batch for tensor shape consistency.
            next_tokens = torch.where(
                unfinished,
                next_tokens,
                torch.full_like(next_tokens, self.pad_token_id),
            )

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            # Update unfinished mask with EOS detection
            # Key subtlety: min_new_tokens temporarily disables EOS stopping
            if self.eos_token_id is not None:
                eos_reached = (next_tokens == self.eos_token_id)
                if step + 1 >= gen_config.min_new_tokens:
                    unfinished = unfinished & ~eos_reached
                # else: ignore EOS, keep generating even if eos_reached=True

            # Prepare next iteration's input (only for unfinished sequences)
            # Finished sequences still participate to maintain batch structure,
            # but they receive pad_token_id and their logits are discarded.
            if unfinished.any():
                current_input = next_tokens.unsqueeze(-1)        # Shape: (B, 1)
                # Extend attention mask: 1 for real tokens, 0 for padding (unused)
                new_mask = torch.ones_like(next_tokens).unsqueeze(-1)
                attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

                # The key optimization: pass past_key_values to avoid recomputing
                # attention for the entire prefix. Model only processes current_input
                # while using cached keys/values from all previous steps.
                outputs = self.model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                break   # All sequences finished; skip remaining steps

        # Decode final sequences (includes original prompts)
        outputs = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if single_input:
            return outputs[0]
        return outputs

    # --------------------------------------------------------------------
    # Sampling helpers
    # --------------------------------------------------------------------

    def _sample(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample from categorical distribution with top-k and nucleus filtering.
        
        Why both filters? They address different failure modes:
            - Top-k: prevents the model from considering extremely unlikely tokens
            - Top-p (nucleus): adapts to distribution shape; when one token dominates,
              nucleus sampling keeps fewer candidates; when distribution is flat,
              it keeps more.
        
        Implementation order matters: top-k first (reduces vocabulary size), then
        top-p on the remainder. This matches the original nucleus sampling paper
        and modern implementations (GPT-3, LLaMA, etc.).
        
        Edge case: If filtering removes all probability mass (shouldn't happen with
        valid parameters), subsequent renormalization will produce NaNs. The caller
        ensures top-k and top-p values keep at least one token.
        """
        probs = F.softmax(logits, dim=-1)
        batch_size, vocab_size = probs.shape

        # Phase 1: Top-k - keep only k highest-probability tokens
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            topk_values, _ = torch.topk(probs, top_k, dim=-1)
            # threshold = k-th largest probability (scalar per batch item)
            min_topk = topk_values[:, -1].unsqueeze(-1)
            probs = torch.where(probs >= min_topk, probs, torch.zeros_like(probs))

        # Phase 2: Top-p (nucleus) - keep smallest set with cumulative prob > top_p
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens whose cumulative probability exceeds top_p
            # Subtract sorted_probs to exclude the token that crosses the threshold
            nucleus_mask = cumsum - sorted_probs > top_p
            sorted_probs[nucleus_mask] = 0.0
            # Scatter back to original indexing order
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        # Renormalize to account for removed probability mass
        # This maintains a valid distribution for multinomial sampling
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_tokens

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """
        Suppress tokens that have already appeared in the generated sequence.
        
        Why division rather than subtraction? In log-space (where logits live),
        dividing by penalty > 1 is equivalent to subtracting a positive constant.
        Division keeps the penalty scale-invariant: if logits are naturally large
        (e.g., from a model with high-magnitude outputs), subtraction might have
        insufficient effect.
        
        Implementation note: Penalizes all occurrences (not just the first). This
        is simpler and empirically sufficient for preventing repetition loops.
        A frequency-based penalty would require tracking counts per token and
        is left as future optimization if needed.
        
        Assumption: generated contains the full sequence (prompt + generated tokens).
        Prompt tokens are included in penalty because repeating prompt phrases is
        also undesirable.
        """
        batch_size, vocab_size = logits.shape
        penalty_mask = torch.zeros(batch_size, vocab_size, device=logits.device)

        # Build mask: 1 for any token ID present anywhere in the sequence
        for i in range(batch_size):
            unique_tokens = torch.unique(generated[i])   # deduplicate for efficiency
            penalty_mask[i, unique_tokens] = 1.0

        penalized_logits = torch.where(
            penalty_mask.bool(),
            logits / penalty,   # Suppress seen tokens
            logits,              # Leave unseen tokens unchanged
        )
        return penalized_logits