"""
This implementation is inspired by the RoFormer architecture
("RoFormer: Enhanced Transformer with Rotary Position Embedding",
Su et al., 2021), which introduces Rotary Position Embedding (RoPE)
to encode relative positional information via sinusoidal rotation
in feature space.

It follows design patterns commonly used in modern transformer
implementations such as GPT-NeoX and LLaMA-style attention,
where rotary embeddings are applied directly to query and key
representations to improve long-context generalization.

The cosine/sine cache precomputation strategy is inspired by
performance optimizations in production LLM inference systems,
where redundant trigonometric computation is avoided during
autoregressive decoding.

The incremental decoding with sequence offsets is aligned with
standard KV-cache based transformer inference pipelines used
in large-scale language model deployment.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.
    
    Rotary Position Embedding (RoPE) rotates pairs of dimensions in the feature
    space based on absolute positions. The rotation matrix is applied as:
        x_rotated = x * cos + rotate_half(x) * sin
    
    This formulation preserves relative position information while providing
    better length extrapolation compared to absolute position embeddings.
    
    Args:
        x: Input tensor of shape (B, H, N, D) or (B, N, D) or (N, D)
            where B=batch, H=heads, N=sequence length, D=feature dimension (must be even)
        cos: Cosine values of shape broadcastable to x
        sin: Sine values of shape broadcastable to x
    
    Returns:
        Tensor with rotary position embedding applied, same shape as x
    
    Example:
        >>> x = torch.randn(2, 8, 512, 64)  # B=2, H=8, N=512, D=64
        >>> cos, sin = build_rope_cache(512, 64, x.device)
        >>> x_rotated = apply_rotary_pos_emb(x, cos, sin)
    """
    # Split into two halves for rotation
    # For even-dimension tensors, we rotate each pair of dimensions (x_i, x_{i+1})
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    
    # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    # This is equivalent to multiplying by the rotation matrix:
    # [[cos, -sin], [sin, cos]] * [x1, x2]^T
    rotated = torch.cat([-x2, x1], dim=-1)
    
    return x * cos + rotated * sin


def build_rope_cache(
    seq_len: int,
    rope_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine caches for rotary position embedding.
    
    This function implements the standard RoPE frequency formulation from
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021):
        theta_i = base^{-2(i-1)/d} for i = 1, 2, ..., d/2
        pos_emb(p, i) = sin(p * theta_i) or cos(p * theta_i) for position p
    
    The cache is precomputed once per sequence length and reused across all
    attention heads, significantly reducing redundant computation during
    autoregressive generation.
    
    Args:
        seq_len: Length of sequence to precompute positions for
        rope_dim: Dimension of rope features (must be even, typically head_dim)
        device: Device to allocate tensors on
        dtype: Data type for computed tensors (default: float32)
        base: Base frequency for exponential decay (default: 10000.0)
        offset: Starting position index for incremental decoding.
            When decoding sequentially, positions are offset by previously
            generated tokens (e.g., if 5 tokens already exist, offset=5)
    
    Returns:
        Tuple of (cos_cache, sin_cache) each with shape (1, 1, seq_len, rope_dim)
        Shape is chosen for efficient broadcasting: batch and head dimensions are 1
    
    Example:
        >>> cos, sin = build_rope_cache(512, 64, torch.device('cuda'))
        >>> # For incremental decoding starting at position 10
        >>> cos, sin = build_rope_cache(1, 64, torch.device('cuda'), offset=10)
    """
    # Validate rope_dim is even (requirement for proper rotation)
    assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
    
    # Compute frequencies for each dimension pair
    # Standard RoPE formula: theta_i = base^{-2*(i-1)/d} for i=1..d/2
    half_dim = rope_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    
    # Compute positions: [offset, offset+1, ..., offset+seq_len-1]
    positions = torch.arange(offset, offset + seq_len, device=device).float()
    
    # Outer product: positions (N,) * theta (half_dim,) = (N, half_dim)
    # Each element is position * theta_i
    freqs = torch.outer(positions, theta)  # (seq_len, half_dim)
    
    # Duplicate to full rope_dim by concatenating with itself
    # This matches the standard RoPE implementation where each pair (i, i+1)
    # shares the same frequency
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, rope_dim)
    
    # Add broadcast dimensions: (1, 1, seq_len, rope_dim)
    # This allows direct broadcasting with tensors of shape (B, H, N, D)
    cos_cache = emb.cos()[None, None, :, :].to(dtype)
    sin_cache = emb.sin()[None, None, :, :].to(dtype)
    
    return cos_cache, sin_cache


class RotaryPositionalEmbedding(nn.Module):
    """
    Learnable or fixed-cache rotary position embedding module.
    
    This module provides a convenient wrapper around RoPE cache management,
    automatically recomputing caches when sequence length changes and
    handling the complexity of incremental decoding.
    
    The module supports two modes:
    1. Full-sequence mode: Precompute cache for entire sequence length
    2. Incremental mode: Extend cache gradually as new tokens are generated
    
    Memory optimization: Cache is stored as buffers (not parameters),
    so it doesn't participate in gradient computation or optimization.
    
    Args:
        rope_dim: Dimension to apply RoPE to (must be even)
        base: Base frequency (default: 10000.0)
        max_seq_len: Maximum sequence length for cache preallocation.
            If provided, preallocates cache to avoid recomputation.
            If None, caches are built on-demand.
    
    Attributes:
        rope_dim: Dimension for RoPE
        base: Base frequency value
        max_seq_len: Preallocated cache length (or None)
        cos_cache: Cached cosine values (registered as buffer)
        sin_cache: Cached sine values (registered as buffer)
    
    Example:
        >>> rope = RotaryPositionalEmbedding(rope_dim=64, max_seq_len=2048)
        >>> x = torch.randn(2, 8, 512, 64)
        >>> out = rope(x, seq_offset=0)  # Apply RoPE at positions 0-511
        >>> 
        >>> # Incremental decoding
        >>> for pos in range(512, 1024):
        ...     x_new = torch.randn(2, 8, 1, 64)
        ...     out = rope(x_new, seq_offset=pos)  # Position-specific RoPE
    """
    
    def __init__(
        self,
        rope_dim: int,
        base: float = 10000.0,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Preallocate caches if max_seq_len is provided
        if max_seq_len is not None:
            # Register as buffers (non-trainable)
            self.register_buffer(
                "cos_cache",
                torch.zeros(1, 1, max_seq_len, rope_dim)
            )
            self.register_buffer(
                "sin_cache",
                torch.zeros(1, 1, max_seq_len, rope_dim)
            )
            self._cache_built = False
        else:
            # Dynamic caching: caches built on demand
            self.cos_cache = None
            self.sin_cache = None
            self._cache_built = False
    
    def _build_caches(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Build or extend caches for given sequence length.
        
        If max_seq_len was specified and caches are already built,
        this is a no-op. Otherwise, rebuilds caches for the requested length.
        """
        # Check if existing cache is sufficient
        if self.max_seq_len is not None and self._cache_built:
            # Preallocated cache is sufficient for any seq_len <= max_seq_len
            if seq_len <= self.max_seq_len:
                return
            else:
                # Requested length exceeds max_seq_len, need to rebuild
                print(f"Warning: seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}. Rebuilding caches.")
        
        # Build new caches
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            rope_dim=self.rope_dim,
            device=device,
            dtype=dtype,
            base=self.base,
            offset=0,
        )
        
        # Store as buffers
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)
        self._cache_built = True
    
    def forward(
        self,
        x: torch.Tensor,
        seq_offset: int = 0,
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor. Supported shapes:
                - (B, H, N, D): Multi-head attention format
                - (B, N, D): Single-head or aggregated format
                - (N, D): Single batch format
            seq_offset: Starting position index for the sequence.
                Used during incremental decoding to indicate that this
                sequence continues from a previous context.
        
        Returns:
            Tensor with RoPE applied, same shape as input
        
        Example:
            >>> rope = RotaryPositionalEmbedding(64)
            >>> x = torch.randn(2, 8, 10, 64)  # 10 tokens
            >>> out = rope(x, seq_offset=0)    # First 10 positions
            >>> 
            >>> # Next token
            >>> x_next = torch.randn(2, 8, 1, 64)
            >>> out_next = rope(x_next, seq_offset=10)  # Position 10
        """
        # Infer batch, heads, seq_len, dim from input shape
        original_shape = x.shape
        if x.dim() == 4:  # (B, H, N, D)
            B, H, N, D = original_shape
        elif x.dim() == 3:  # (B, N, D)
            # Add head dimension
            x = x.unsqueeze(1)
            B, H, N, D = x.shape
            needs_squeeze = True
        elif x.dim() == 2:  # (N, D)
            x = x.unsqueeze(0).unsqueeze(0)
            B, H, N, D = x.shape
            needs_squeeze = True
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, expected 2, 3, or 4")
        
        # Ensure rope_dim <= D
        if self.rope_dim > D:
            raise ValueError(
                f"rope_dim ({self.rope_dim}) cannot exceed feature dimension ({D})"
            )
        
        # Build or retrieve caches
        total_seq_len = seq_offset + N
        self._build_caches(total_seq_len, x.device, x.dtype)
        
        # Extract slice for current positions
        cos = self.cos_cache[:, :, seq_offset:seq_offset + N, :self.rope_dim]
        sin = self.sin_cache[:, :, seq_offset:seq_offset + N, :self.rope_dim]
        
        # Apply RoPE to the rope_dim portion of the tensor
        x_rope_part = x[..., :self.rope_dim]
        x_rotated = apply_rotary_pos_emb(x_rope_part, cos, sin)
        
        # Combine rotated part with unchanged part
        if self.rope_dim < D:
            result = torch.cat([x_rotated, x[..., self.rope_dim:]], dim=-1)
        else:
            result = x_rotated
        
        # Restore original shape if needed
        if needs_squeeze:
            result = result.squeeze(1) if H == 1 else result
        
        return result
    
    def get_caches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current cosine and sine caches for debugging/inspection. """
        return self.cos_cache, self.sin_cache


# Convenience function for quick RoPE application without full module
def rope(
    x: torch.Tensor,
    rope_dim: int,
    seq_offset: int = 0,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Apply RoPE to tensor using cached precomputation.
    
    This is a functional interface for single-use RoPE applications.
    For repeated use (e.g., across multiple layers), prefer the
    RotaryPositionalEmbedding module for better performance.
    
    Args:
        x: Input tensor of shape (..., N, D) where D >= rope_dim
        rope_dim: Dimension to apply RoPE to (must be even)
        seq_offset: Starting position offset
        base: Base frequency for RoPE
    
    Returns:
        Tensor with RoPE applied
    
    Example:
        >>> q = torch.randn(2, 8, 128, 64)
        >>> q_rope = rope(q, rope_dim=64, seq_offset=512)
    """
    B, H, N, D = x.shape
    cos, sin = build_rope_cache(N, rope_dim, x.device, x.dtype, base, seq_offset)
    
    x_rope = x[..., :rope_dim]
    x_rotated = apply_rotary_pos_emb(x_rope, cos, sin)
    
    if rope_dim < D:
        return torch.cat([x_rotated, x[..., rope_dim:]], dim=-1)
    return x_rotated