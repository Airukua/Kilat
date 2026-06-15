"""
High-level generation wrapper for KilatTransformer.

This module provides a simple, user-friendly interface for text generation
while still allowing access to low-level generation for advanced users.

WHY THIS WRAPPER:
    - Most users just want to generate text with a few parameters
    - Hides the complexity of KV-cache, beam search, sampling strategies
    - Provides sensible defaults for common use cases
    - Still allows advanced users to access low-level APIs

USAGE:
    # Simple generation
    generator = TextGenerator(model, tokenizer)
    text = generator.generate("Once upon a time", max_new_tokens=100)
    
    # With custom settings
    generator = TextGenerator(model, tokenizer, device="cuda")
    text = generator.generate(
        "The future of AI is",
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
    )
    
    # Batch generation
    texts = generator.batch_generate(
        ["Prompt 1", "Prompt 2", "Prompt 3"],
        max_new_tokens=100,
    )
    
    # Streaming generation (token by token)
    for token in generator.stream("Hello", max_new_tokens=50):
        print(token, end="", flush=True)
"""

from __future__ import annotations
from typing import List, Optional, Union, Iterator, Callable
import torch
from tqdm.auto import tqdm
from .generation_config import GenerationConfig


class TextGenerator:
    """
    High-level wrapper for text generation with KilatTransformer.
    
    This class provides a simple, user-friendly interface for common
    generation tasks. It handles tokenization, device management, and
    decoding automatically.
    
    Attributes
    ----------
    model : KilatTransformer
        The language model to use for generation.
    tokenizer : Any
        Tokenizer for encoding/decoding text.
    device : torch.device
        Device to run generation on.
    default_config : GenerationConfig
        Default generation configuration (can be overridden per call).
    
    Example
    -------
        >>> generator = TextGenerator(model, tokenizer)
        >>> text = generator.generate("Hello", max_new_tokens=50)
        >>> print(text)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[Union[str, torch.device]] = None,
        default_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the text generator.
        
        Parameters
        ----------
        model : KilatTransformer
            The language model (must be in eval mode).
        tokenizer : Any
            Tokenizer with encode/decode methods.
        device : Optional[Union[str, torch.device]]
            Device to run on. If None, auto-detects from model.
        default_config : Optional[GenerationConfig]
            Default generation config. If None, uses sensible defaults.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Detect device from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)
        
        # Set default config with proper values
        if default_config is None:
            self.default_config = GenerationConfig(
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=False,
            )
        else:
            self.default_config = default_config
        
        # Cache for tokenizer special tokens
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    
    def _encode_prompt(
        self,
        prompt: str,
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        """Encode a prompt string to token IDs."""
        tokens = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return tokens.to(self.device)
    
    def _decode_output(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=skip_special_tokens)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        num_beams: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        This is the main method for text generation. All parameters are optional;
        if not provided, they default to the values in default_config.
        
        Parameters
        ----------
        prompt : str
            Input prompt text.
        max_new_tokens : Optional[int]
            Maximum number of new tokens to generate.
        do_sample : Optional[bool]
            Whether to use sampling (True) or greedy (False).
        temperature : Optional[float]
            Sampling temperature (higher = more random).
        top_k : Optional[int]
            Top-k filtering (0 = disabled).
        top_p : Optional[float]
            Top-p nucleus sampling (1.0 = disabled).
        repetition_penalty : Optional[float]
            Penalty for repeated tokens (1.0 = disabled).
        num_beams : Optional[int]
            Number of beams for beam search (1 = disabled).
        **kwargs
            Additional generation parameters.
        
        Returns
        -------
        str
            Generated text (including the original prompt).
        
        Example
        -------
            >>> text = generator.generate("Once upon a time", max_new_tokens=100)
            >>> print(text)
        """
        # Build config for this generation
        config = self._build_config(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            **kwargs,
        )
        
        # Encode prompt
        input_ids = self._encode_prompt(prompt)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                generation_config=config,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
            )
        
        # Decode and return
        return self._decode_output(output_ids)
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Parameters
        ----------
        prompts : List[str]
            List of input prompts.
        max_new_tokens : Optional[int]
            Maximum new tokens per generation.
        show_progress : bool
            Whether to show a progress bar.
        **kwargs
            Additional generation parameters (passed to generate()).
        
        Returns
        -------
        List[str]
            List of generated texts.
        """
        results = []
        iterator = tqdm(prompts, desc="Generating", disable=not show_progress)
        
        for prompt in iterator:
            text = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            results.append(text)
        
        return results
    
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream generated text token by token.
        
        This is useful for real-time applications where you want to show
        output as it's generated.
        
        Parameters
        ----------
        prompt : str
            Input prompt text.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        callback : Optional[Callable[[str], None]]
            Optional callback function called for each token.
        **kwargs
            Additional generation parameters.
        
        Yields
        ------
        str
            Generated text chunks (token by token).
        
        Example
        -------
            >>> for chunk in generator.stream("Hello", max_new_tokens=50):
            ...     print(chunk, end="", flush=True)
        """
        # Build config
        config = self._build_config(max_new_tokens=max_new_tokens, **kwargs)
        
        # Encode prompt
        input_ids = self._encode_prompt(prompt)
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Sample or greedy
                if config.do_sample:
                    probs = torch.softmax(logits / config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode and yield new text
                new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                if callback:
                    callback(new_text)
                
                yield new_text
                
                # Check for EOS
                if self.eos_token_id is not None and next_token.item() == self.eos_token_id:
                    break
    
    def _build_config(self, **kwargs) -> GenerationConfig:
        """Build generation config from defaults and overrides."""
        # Get values with proper defaults
        max_new_tokens = kwargs.get("max_new_tokens", self.default_config.max_new_tokens)
        if max_new_tokens is None:
            max_new_tokens = 100
            
        do_sample = kwargs.get("do_sample", self.default_config.do_sample)
        if do_sample is None:
            do_sample = False
            
        temperature = kwargs.get("temperature", self.default_config.temperature)
        if temperature is None:
            temperature = 1.0
            
        top_k = kwargs.get("top_k", self.default_config.top_k)
        if top_k is None:
            top_k = 50
            
        top_p = kwargs.get("top_p", self.default_config.top_p)
        if top_p is None:
            top_p = 1.0
            
        repetition_penalty = kwargs.get("repetition_penalty", self.default_config.repetition_penalty)
        if repetition_penalty is None:
            repetition_penalty = 1.0
            
        num_beams = kwargs.get("num_beams", self.default_config.num_beams)
        if num_beams is None:
            num_beams = 1
            
        length_penalty = kwargs.get("length_penalty", getattr(self.default_config, 'length_penalty', 1.0))
        if length_penalty is None:
            length_penalty = 1.0
        
        # Build config with all values
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=kwargs.get("early_stopping", getattr(self.default_config, 'early_stopping', False)),
        )
        
        # Override with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_low_level(self):
        """
        Return access to low-level generation APIs for advanced users.
        
        This allows advanced users to bypass the wrapper and use
        model.generate() directly for full control.
        
        Returns
        -------
        dict
            Dictionary with model, tokenizer, and config for low-level access.
        
        Example
        -------
            >>> low = generator.to_low_level()
            >>> output = low["model"].generate(
            ...     low["tokenizer"].encode("Hello", return_tensors="pt"),
            ...     num_beams=5,
            ... )
        """
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "device": self.device,
            "default_config": self.default_config,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    **kwargs,
) -> str:
    """
    One-shot generation for quick experiments.
    
    This is the simplest possible interface for generation.
    
    Example
    -------
        >>> text = quick_generate(model, tokenizer, "Hello", max_new_tokens=50)
        >>> print(text)
    """
    generator = TextGenerator(model, tokenizer)
    return generator.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    **kwargs,
) -> List[str]:
    """
    Batch generation for multiple prompts.
    
    Example
    -------
        >>> texts = batch_generate(model, tokenizer, ["Hello", "Hi"], max_new_tokens=50)
    """
    generator = TextGenerator(model, tokenizer)
    return generator.batch_generate(prompts, max_new_tokens=max_new_tokens, **kwargs)


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    **kwargs,
) -> Iterator[str]:
    """
    Stream generation token by token.
    
    Example
    -------
        >>> for chunk in stream_generate(model, tokenizer, "Hello"):
        ...     print(chunk, end="", flush=True)
    """
    generator = TextGenerator(model, tokenizer)
    return generator.stream(prompt, max_new_tokens=max_new_tokens, **kwargs)