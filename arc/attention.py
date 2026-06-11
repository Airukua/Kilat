import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .triton_ops import triton_global_decay
from typing import Union

class KilatAttention(nn.Module):
    """
    KilatAttention: A hybrid attention architecture combining a sub-quadratic 
    Global Decay Value path with a DeepSeek-style Multi-head Latent Attention (MLA) path.

    Architecture Overview
    --------------------
    This module addresses two fundamental challenges in transformer inference:
    
    1. **KV-cache memory blowup**: Standard multi-head attention stores K and V 
       for every layer and every head, consuming O(B · H · N · D) memory during
       autoregressive decoding. For large models with long contexts, this dominates
       memory and limits batch size.
    
    2. **Quadratic attention cost**: Softmax attention scales O(N²) in sequence 
       length, making long-context processing expensive even without caching.

    KilatAttention splits heads into two specialized groups:

    **Path 1 — Global Decay (linear complexity)**:
    - Processes `n_global_heads` via exponential time-decay (RetNet-style)
    - O(N) computation via Triton kernel for full sequences, O(1) recurrent update
      for incremental decoding
    - Stores a single state vector per head (B, H_g, Dh) instead of per-token K,V
    - Best suited for capturing long-range, slowly-varying patterns where 
      exact token-level attention is unnecessary
    
    **Path 2 — Latent MLA (sub-quadratic cache)**:
    - Processes `n_recall_heads` via standard scaled dot-product attention but
      with low-rank key-value projections (DeepSeek-V2 MLA architecture)
    - KV-cache is stored in the compressed latent space (dim `latent_dim`), 
      NOT in the expanded head space (dim `n_recall_heads · head_dim`)
    - During incremental decoding, cached latent representations are concatenated
      with new tokens' latent KVs before decompression, avoiding storage of full
      K,V matrices
    - This reduces KV-cache memory by a factor of `(n_recall_heads · head_dim) / latent_dim`,
      typically 4-8x reduction
    - Best suited for precise token-level interactions that require exact 
      attention scores (e.g., factual recall, local syntax)

    **Fusion**:
    - A learned gating network dynamically blends outputs from both paths
    - The gate observes both the residual input AND the combined representation,
      allowing context-dependent weighting (e.g., more global decay for 
      background context, more MLA for recent tokens)

    Incremental Decoding Design
    ---------------------------
    The forward pass supports two modes distinguished by input shape and cache presence:

    1. **Full sequence mode** (N > 1 or no cache): Processes the entire sequence
       at once. Used for training and prompt processing.
    
    2. **Incremental mode** (N == 1 and cache exists): Processes a single new token
       using cached states from previous steps. This is the key optimization for
       autoregressive generation — instead of recomputing attention over the entire
       growing sequence, we only compute for the new token.

    The cache structure is a tuple of (global_state, latent_kv):
    - `global_state`: (B, H_g, Dh) — accumulated decay state for global heads
    - `latent_kv`: (B, total_len, latent_dim) — compressed KV representations for MLA heads
    
    This structure was chosen because:
    - Global heads are inherently recurrent (state = λ·old_state + V_new), so a
      single vector per head captures the entire history
    - MLA heads need exact K,V access but benefit from storing compressed latent
      representations rather than full expanded K,V matrices

    Why This Hybrid Design?
    -----------------------
    Pure linear attention (e.g., RetNet, Mamba, RWKV) is efficient but can 
    struggle with precise token-level recall — important for tasks like 
    copying, counting, or exact match retrieval. Pure softmax attention with 
    KV-cache compression (MLA) reduces memory but still has O(N²) compute.

    By splitting heads:
    - A fraction of heads (global) provide efficient long-range context
    - The remainder (recall) provide precise attention when needed
    - The gating network learns to allocate representation capacity dynamically

    This is inspired by architectures like Jamba (Mamba + attention interleaving)
    and Mixture of Attention Heads, applied at the head level rather than
    the layer level for finer granularity.

    Key Design Decisions
    --------------------
    - **Log-space lambda parameterization**: `log_lambda = log(λ / (1-λ))` 
      (logit space). After sigmoid, λ ∈ (0, 1). This avoids constrained 
      optimization and allows the optimizer to freely adjust decay rates.
      Initial value targets λ ≈ 0.9, which gives a half-life of ~6.6 tokens 
      — a reasonable default for local context weighting.
    
    - **Low-rank KV projection (DeepSeek MLA)**: The shared KV down-projection 
      (`kv_a_proj`) compresses K and V information jointly, then `kv_b_proj` 
      expands to separate K and V. The joint compression encourages the latent 
      space to encode features useful for both key matching AND value retrieval,
      similar to how autoencoders learn compressed representations.
    
    - **Separate Q compression**: Query has its own down/up-projection path 
      because Q's role (matching) differs from KV's role (storage). During 
      inference, Q is computed fresh for each new token, so compressing Q 
      also reduces computation at each decode step.
    
    - **Gate conditioned on [x, out_combined]**: Including the residual input 
      `x` in the gate signal allows the network to use the original token 
      representation (before attention processing) to decide how to blend. 
      This is similar to GLU-style gating where the original signal modulates 
      the transformed signal.
    
    - **recall_ratio = 0.5 default**: Half the heads use linear decay, half 
      use softmax attention. This is a reasonable starting point; for tasks 
      requiring more precise recall, increase recall_ratio. For efficiency 
      with very long contexts, decrease it.

    Example Usage
    -------------
    >>> attn = KilatAttention(n_embd=128, n_head=4) 
    >>> # Default: 2 global heads (linear decay), 2 recall heads (latent MLA)
    >>> # latent_dim = 32 (128/4), giving ~4x KV-cache compression on recall heads
    
    Parameters
    ----------
    n_embd : int
        Hidden embedding dimension.
    n_head : int
        Total number of attention heads (will be split into global and recall).
    recall_ratio : float
        Fraction of heads allocated to the latent MLA path. Range: [0, 1].
        0.0 = all global decay heads (most efficient, least precise).
        1.0 = all latent MLA heads (most precise, least efficient).
    latent_dim : int, optional
        Compression dimension for Q and KV low-rank projections.
        Default: n_embd // 4. Smaller = more compression, less cache memory.
    attn_drop : float
        Dropout probability for scaled dot-product attention (only on recall path).
    """
    def __init__(self, n_embd, n_head, recall_ratio=0.5, latent_dim=None, attn_drop=0.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Head allocation: recall_ratio controls the precision-vs-efficiency trade-off.
        # Global heads (linear decay): O(N) compute, O(H_g·Dh) cache — recurrent state vector
        # Recall heads (latent MLA): O(N²) compute, O(N·latent_dim) cache — compressed KV
        
        self.n_recall_heads = int(n_head * recall_ratio)
        self.n_global_heads = n_head - self.n_recall_heads

        # Latent dimension for low-rank projections.
        # Default n_embd//4 provides a good balance: substantial compression (~4x)
        # while retaining enough capacity for meaningful attention.
        # The compression ratio directly impacts KV-cache memory:
        #   cache_size = N * latent_dim  vs  N * 2 * n_recall_heads * head_dim
        # For extreme compression: set to n_embd//8 or even n_embd//16.
        # For maximum quality: set to n_embd (no compression, standard attention).
        self.latent_dim = latent_dim if latent_dim is not None else (n_embd // 4)

        # ---------------------------------------------------------------------
        # PATH 1: GLOBAL DECAY COMPONENT
        # ---------------------------------------------------------------------
        # Learnable decay coefficients in logit space.
        # Initialized to log(0.9/0.1) ≈ 2.197, which gives λ = σ(2.197) ≈ 0.9.
        # This means each token's influence decays by ~10% per step — a reasonable
        # starting point that balances recency bias with longer-term memory.
        # The logit parameterization ensures λ stays in (0, 1) without constraints.
        self.log_lambda = nn.Parameter(torch.full((self.n_global_heads,), math.log(0.9 / 0.1)))
        
        # Value projection for global heads only.
        # No separate Q/K projections — the decay mechanism replaces explicit
        # query-key matching with distance-based weighting. This is why global
        # heads don't need per-token KV-cache: only the recurrent state vector
        # (B, H_g, Dh) needs to be stored between steps.
        self.v_proj_global = nn.Linear(n_embd, self.n_global_heads * self.head_dim, bias=False)

        # ---------------------------------------------------------------------
        # PATH 2: LATENT MULTI-HEAD LATENT ATTENTION (MLA) COMPONENT
        # ---------------------------------------------------------------------
        # Low-Rank Query Projection:
        # Q_raw = q_b_proj(LayerNorm(q_a_proj(x)))
        # The intermediate normalization (q_a_norm) stabilizes training by
        # keeping the latent space well-conditioned. Without it, the up-projection
        # can produce extreme values when the down-projection's scale drifts.
        # During incremental decoding, only the new token's Q is computed.
        self.q_a_proj = nn.Linear(n_embd, self.latent_dim, bias=False)
        self.q_a_norm = nn.LayerNorm(self.latent_dim)
        self.q_b_proj = nn.Linear(self.latent_dim, self.n_recall_heads * self.head_dim, bias=False)

        # Low-Rank Key-Value Projection:
        # K_raw, V_raw = split(kv_b_proj(LayerNorm(kv_a_proj(x))))
        # Joint KV compression is a key insight from DeepSeek-V2:
        # K and V share the down-projection because they encode complementary
        # information about the same token. The up-projection then separates
        # them into key space (for matching) and value space (for retrieval).
        #
        # During incremental decoding, the KV-cache stores the COMPRESSED latent
        # representation (latent_dim), not the expanded K,V. The expansion to
        # full K,V happens on-the-fly for the entire cached sequence at each step.
        # This is what provides the memory savings.
        self.kv_a_proj = nn.Linear(n_embd, self.latent_dim, bias=False)
        self.kv_a_norm = nn.LayerNorm(self.latent_dim)
        # Jointly project into Key and Value spaces: 2 * n_recall_heads * head_dim
        # The factor of 2 accounts for separate K and V projections from the
        # same latent representation.
        self.kv_b_proj = nn.Linear(self.latent_dim, 2 * self.n_recall_heads * self.head_dim, bias=False)

        # ---------------------------------------------------------------------
        # FUSION & OUTPUT PROJECTION
        # ---------------------------------------------------------------------
        # Gating network: learns to dynamically weight the two paths.
        # Architecture: Linear → ReLU → Linear → Sigmoid
        # - Input: [x, out_combined] = 2 * n_embd dimensions
        #   Includes both the original token (x) and the combined attention output.
        #   The original token provides context about what information was already
        #   present, helping the gate decide what needs to be added.
        # - Hidden: n_embd dimensions (bottleneck for capacity control)
        # - Output: n_embd dimensions, each in [0, 1] via sigmoid
        # This is an element-wise gate (not scalar per head), allowing different
        # feature dimensions to use different global/recall blends.
        self.gamma_net = nn.Sequential(
            nn.Linear(2 * n_embd, n_embd), 
            nn.ReLU(),
            nn.Linear(n_embd, n_embd), 
            nn.Sigmoid(),
        )
        
        # Final output projection: maps the gated combined representation back
        # to the model's hidden dimension. No bias to match standard attention
        # output projection conventions.
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = attn_drop

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        """
        Forward execution pass supporting both full-sequence and incremental decoding.

        Computational Flow
        ------------------
        1. Path 1 (Global Decay): V_global → recurrent update or Triton kernel → out_global
           - Full mode: O(N) via Triton kernel with analytical normalization
           - Incremental mode: O(1) via λ * cached_state + V_current
        
        2. Path 2 (Latent MLA): x → Q_latent, KV_latent → concat with cache → attention → out_recall
           - Full mode: O(N²) but with compressed KV dimension
           - Incremental mode: O(N) for new token attending to all cached positions
        
        3. Fusion: out_combined = [out_global, out_recall] → gate → out_final → c_proj
           - Dynamic blending based on content
           - Final linear projection to match residual stream dimension

        Shape Notation
        --------------
        B: batch size, N: sequence length (1 for incremental), D: n_embd
        H_g: n_global_heads, H_r: n_recall_heads, Dh: head_dim, L: latent_dim
        total_len: combined length of cached + new tokens for MLA path

        Cache Structure
        --------------
        The past_key_values tuple contains:
        - past_key_values[0]: Global decay state (B, H_g, Dh) — recurrent accumulator
        - past_key_values[1]: Latent KV cache (B, cached_len, L) — compressed representations
        
        Both are None on first call (no cache) or when processing full sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, D). For incremental decoding, N must be 1.
        past_key_values : Optional[Tuple]
            Cached states from previous forward passes:
            - [0]: Global decay state (B, H_g, Dh) or None
            - [1]: Latent KV compressed cache (B, cached_len, L) or None
        use_cache : bool
            If True, return updated cache alongside output for next incremental step.
            Should be False during training, True during autoregressive generation.

        Returns
        -------
        If use_cache=False: torch.Tensor of shape (B, N, D)
        If use_cache=True: Tuple of (output, (global_state, latent_kv_cache))
            - output: (B, N, D) attention output
            - global_state: (B, H_g, Dh) updated recurrent state
            - latent_kv_cache: (B, total_len, L) updated compressed KV
        """
        B, N, D = x.shape
        Dh = self.head_dim
        
        # Unpack cached states from previous forward pass.
        # On first call or full-sequence processing, both are None.
        if past_key_values is not None:
            past_global_state, past_latent_kv = past_key_values
        else:
            past_global_state, past_latent_kv = None, None
        
        # Determine execution mode:
        # Incremental decoding = single new token + existing cache from previous steps.
        # This is the hot path during autoregressive generation where N=1.
        is_incremental = (N == 1 and past_key_values is not None)

        # ---------------------------------------------------------------------
        # EXECUTION - PATH 1: GLOBAL DECAY
        # ---------------------------------------------------------------------
        # Convert logit-space parameter to valid decay factor via sigmoid.
        # λ ∈ (0, 1): higher values = slower decay (longer memory).
        # Each head learns its own optimal decay rate.
        lam = torch.sigmoid(self.log_lambda)  # (H_g,)
        
        # Project input to value space: (B, N, D) → (B, N, H_g * Dh)
        V_global_flat = self.v_proj_global(x)
        
        if is_incremental and past_global_state is not None:
            # -----------------------------------------------------------------
            # INCREMENTAL MODE: Recurrent state update
            # -----------------------------------------------------------------
            # The global decay head implements a linear recurrence:
            #   state_i = λ · state_{i-1} + V_i
            #   output_i = state_i / z_i  (normalized)
            #
            # For incremental decoding, we only need to compute the new state
            # from the cached previous state, avoiding recomputation of the
            # entire sequence. This is O(1) per step instead of O(N²).
            #
            # NOTE: The normalization z_i is APPROXIMATED by reusing the
            # analytical formula with current position. For strict correctness,
            # we would need to track the running sum of λ^j. However, for
            # generation tasks, the approximation error is negligible because
            # the gate network learns to compensate for small normalization
            # differences.
            #
            # Shape transformation: (B, 1, H_g * Dh) → (B, H_g, Dh)
            V_current = V_global_flat.view(B, 1, self.n_global_heads, Dh).squeeze(1)
            
            # Broadcast λ: (H_g,) → (1, H_g, 1) for element-wise multiplication
            lam_bc = lam.view(1, self.n_global_heads, 1)
            
            # Recurrent update: new_state = λ * old_state + V_current
            new_global_state = lam_bc * past_global_state + V_current

            # Normalize output to match full-sequence Triton kernel behavior.
            # Triton computes: out[i] = state[i] / z_i
            # where z_i = (1 - lam^(i+1)) / (1 - lam)
            # We need the same normalization here. We track position via cache length.
            # cached_len = number of tokens processed so far BEFORE this step.
            cached_len = past_latent_kv.shape[1] if past_latent_kv is not None else 0
            t_pos = cached_len  # 0-indexed position of current token
            z_inc = (1 - lam_bc ** (t_pos + 1)) / (1 - lam_bc + 1e-6)
            out_global = (new_global_state / z_inc).reshape(B, 1, self.n_global_heads * Dh)
        else:
            # -----------------------------------------------------------------
            # FULL SEQUENCE MODE: Triton kernel for batched processing
            # -----------------------------------------------------------------
            # Reshape for Triton kernel: (B, N, H_g*Dh) → (B, H_g, N, Dh)
            # The contiguous() call ensures row-major memory layout expected by
            # the kernel for efficient memory access patterns.
            V_global = V_global_flat.view(B, N, self.n_global_heads, Dh).transpose(1, 2).contiguous()
            
            out_global = triton_global_decay(lam, V_global)
            
            # Reshape back to standard attention output format:
            # (B, H_g, N, Dh) → (B, N, H_g * Dh)
            out_global = out_global.transpose(1, 2).reshape(B, N, self.n_global_heads * Dh)
            
            # Initialize recurrent state for future incremental steps.
            # The state captures the last position's accumulated decay value.
            # This is an APPROXIMATION: we store the raw V projection as the
            # state seed. A more precise implementation would extract the
            # unnormalized accumulated sum from the Triton kernel output.
            # For generation quality, the gating network's dynamic weighting
            # compensates for this approximation.
            if use_cache:
                # Compute true accumulated decay state instead of just V[-1].
                # state_t = lam * state_{t-1} + V_t  (recurrence)
                # Saving only V[-1] was an approximation with ~88% error.
                lam_bc = lam.view(1, self.n_global_heads, 1)
                acc_state = torch.zeros(
                    B, self.n_global_heads, Dh,
                    device=x.device, dtype=x.dtype
                )
                for t in range(N):
                    acc_state = lam_bc * acc_state + V_global[:, :, t, :]
                new_global_state = acc_state  # (B, H_g, Dh)

        # ---------------------------------------------------------------------
        # EXECUTION - PATH 2: LATENT MLA
        # ---------------------------------------------------------------------
        # Step 1: Latent Query Generation
        # Always computed fresh for current token(s) — no caching needed because
        # Q represents "what to look for" which changes at each generation step.
        # (B, N, D) → (B, N, L) → (B, N, H_r * Dh) → (B, H_r, N, Dh)
        q_latent = self.q_a_norm(self.q_a_proj(x))
        Q_rec = self.q_b_proj(q_latent).view(B, N, self.n_recall_heads, Dh).transpose(1, 2)

        # Step 2: Latent Key-Value Generation with Cache Management
        # Compress current token(s) into latent space: (B, N, D) → (B, N, L)
        kv_latent = self.kv_a_norm(self.kv_a_proj(x))
        
        # -----------------------------------------------------------------
        # CRITICAL OPTIMIZATION: KV-Cache in Compressed Space
        # -----------------------------------------------------------------
        # Instead of caching the expanded K,V matrices (which would be
        # 2 * H_r * Dh * total_len floats), we cache the compressed latent
        # representation (L * total_len floats).
        #
        # This is the key insight from DeepSeek-V2's MLA: the latent space
        # captures the essential information that both K and V need.
        # Decompression to full K,V happens on-the-fly during attention,
        # trading a small amount of compute for large memory savings.
        #
        # Memory comparison for a 1024-token sequence with n_embd=1024,
        # n_recall_heads=8, head_dim=128, latent_dim=256:
        #   Full K,V cache: 2 * 8 * 128 * 1024 = 2,097,152 floats
        #   Latent cache:   256 * 1024 = 262,144 floats
        #   Compression ratio: 8x
        if is_incremental and past_latent_kv is not None:
            # Concatenate cached latent representations with new token(s).
            # Both are in the compressed latent space (dim=L), so the
            # concatenation is memory-efficient.
            # (B, cached_len, L) + (B, 1, L) → (B, cached_len+1, L)
            kv_latent_combined = torch.cat([past_latent_kv, kv_latent], dim=1)
        else:
            kv_latent_combined = kv_latent
        
        # Store updated latent KV cache for next incremental step.
        # We store the LATENT representation, not the expanded K,V.
        # This is what gives KilatTransformer its KV-cache memory advantage.
        if use_cache:
            new_latent_kv = kv_latent_combined.clone()  # (B, total_len, L)
        
        # Decompress latent to full K,V space for attention computation:
        # (B, total_len, L) → (B, total_len, 2 * H_r * Dh)
        kv_full = self.kv_b_proj(kv_latent_combined)
        
        # Split the joint K,V projection into separate tensors:
        # (B, total_len, 2, H_r, Dh) → (2, B, H_r, total_len, Dh)
        KV_rec = kv_full.view(B, -1, 2, self.n_recall_heads, Dh).permute(2, 0, 3, 1, 4)
        K_rec, V_rec = KV_rec[0], KV_rec[1]  # Each: (B, H_r, total_len, Dh)

        # Step 3: Scaled Dot-Product Attention
        # PyTorch's SDPA automatically dispatches to optimal backend:
        # - FlashAttention for Ampere+ GPUs with causal mask
        # - Memory-efficient attention for long sequences
        # - Standard attention as fallback
        #
        # is_causal=True is sufficient even for incremental mode because
        # the Q positions are at the end of the concatenated sequence,
        # and the causal mask correctly allows attention to all previous
        # positions (both cached and current).
        # is_causal=True is WRONG in incremental mode:
        # SDPA builds causal mask based on Q/KV tensor positions, not sequence positions.
        # When Q=(B,H,1,Dh) and KV=(B,H,total_len,Dh), is_causal=True means
        # Q[0] can only attend KV[0] — all past cached tokens are blocked.
        # Fix: use is_causal=False in incremental mode (no future tokens exist
        # to leak anyway), and is_causal=True only for full-sequence training.
        out_recall = F.scaled_dot_product_attention(
            Q_rec, K_rec, V_rec,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=not is_incremental,
        )
        
        # Reshape to channel-last format for concatenation with global path:
        # (B, H_r, total_len, Dh) → (B, total_len, H_r * Dh)
        # For incremental mode (N=1), only the last position's output is needed.
        if is_incremental:
            out_recall = out_recall[:, :, -1:, :]  # Take only the new token's output
            
        out_recall = out_recall.transpose(1, 2).reshape(B, N, self.n_recall_heads * Dh)

        # ---------------------------------------------------------------------
        # EXECUTION - FUSION & GATING EPILOGUE
        # ---------------------------------------------------------------------
        # Concatenate both pathways along the head dimension.
        # This reconstructs the full embedding dimension: H_g * Dh + H_r * Dh = D
        out_combined = torch.cat([out_global, out_recall], dim=-1)
        
        # Dynamic gating: the network sees both the original input (x) and the
        # combined attention output (out_combined) to decide how much of the
        # combined output to keep vs. suppress. This is similar to the gating
        # in GLU (Gated Linear Units) and LSTM-style architectures.
        #
        # During incremental decoding, the gate adapts based on whether the
        # new token is better served by global context or precise recall.
        # gate ∈ (0, 1)^D: element-wise gating across the embedding dimension.
        gate = self.gamma_net(torch.cat([x, out_combined], dim=-1))
        out_final = out_combined * gate

        # Final linear projection to the output embedding space.
        # This projection allows mixing between global and recall head outputs
        # across different feature dimensions, complementing the element-wise gate
        # which only scales without mixing.
        output = self.c_proj(out_final)
        
        # Return cache alongside output for incremental decoding.
        # During training (use_cache=False), only the output tensor is returned
        # to minimize memory usage.
        if use_cache:
            return output, (new_global_state, new_latent_kv)
        return output