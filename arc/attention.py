import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .triton_ops import triton_global_decay

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
    - O(N) computation via Triton kernel (see `triton_global_decay`)
    - NO KV-cache needed: the recurrent formulation means each position only 
      depends on an accumulated state, not individual past K/V pairs
    - Best suited for capturing long-range, slowly-varying patterns where 
      exact token-level attention is unnecessary
    
    **Path 2 — Latent MLA (sub-quadratic cache)**:
    - Processes `n_recall_heads` via standard scaled dot-product attention but
      with low-rank key-value projections (DeepSeek-V2 MLA architecture)
    - KV-cache is stored in the compressed latent space (dim `latent_dim`), 
      NOT in the expanded head space (dim `n_recall_heads · head_dim`)
    - This reduces KV-cache memory by a factor of `(n_recall_heads · head_dim) / latent_dim`,
      typically 4-8x reduction
    - Best suited for precise token-level interactions that require exact 
      attention scores (e.g., factual recall, local syntax)

    **Fusion**:
    - A learned gating network dynamically blends outputs from both paths
    - The gate observes both the residual input AND the combined representation,
      allowing context-dependent weighting (e.g., more global decay for 
      background context, more MLA for recent tokens)

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
        # Global heads (linear decay): O(N) compute, no KV-cache, good for long-range
        # Recall heads (latent MLA): O(N²) compute, compressed KV-cache, good for precision
        self.n_recall_heads = int(n_head * recall_ratio)
        self.n_global_heads = n_head - self.n_recall_heads

        # Latent dimension for low-rank projections.
        # Default n_embd//4 provides a good balance: substantial compression (~4x)
        # while retaining enough capacity for meaningful attention.
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
        # heads don't need a KV-cache: V is the only per-token state.
        self.v_proj_global = nn.Linear(n_embd, self.n_global_heads * self.head_dim, bias=False)

        # ---------------------------------------------------------------------
        # PATH 2: LATENT MULTI-HEAD LATENT ATTENTION (MLA) COMPONENT
        # ---------------------------------------------------------------------
        # Low-Rank Query Projection:
        # Q_raw = q_b_proj(LayerNorm(q_a_proj(x)))
        # The intermediate normalization (q_a_norm) stabilizes training by
        # keeping the latent space well-conditioned. Without it, the up-projection
        # can produce extreme values when the down-projection's scale drifts.
        self.q_a_proj = nn.Linear(n_embd, self.latent_dim, bias=False)
        self.q_a_norm = nn.LayerNorm(self.latent_dim)
        self.q_b_proj = nn.Linear(self.latent_dim, self.n_recall_heads * self.head_dim, bias=False)

        # Low-Rank Key-Value Projection:
        # K_raw, V_raw = split(kv_b_proj(LayerNorm(kv_a_proj(x))))
        # Joint KV compression is a key insight from DeepSeek-V2:
        # K and V share the down-projection because they encode complementary
        # information about the same token. The up-projection then separates
        # them into key space (for matching) and value space (for retrieval).
        # The output is 2x larger than Q's up-projection because it produces
        # both K and V simultaneously.
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

    def forward(self, x):
        """
        Forward execution pass.

        Computational Flow
        ------------------
        1. Path 1 (Global Decay): V_global → triton_global_decay → out_global
           - O(N) compute, no quadratic attention matrix
           - Handled by custom Triton kernel for hardware efficiency
        
        2. Path 2 (Latent MLA): x → Q_latent, KV_latent → scaled_dot_product_attention → out_recall
           - O(N²) compute but with compressed KV dimension
           - Uses PyTorch's optimized SDPA backend (FlashAttention, etc.)
        
        3. Fusion: out_combined = [out_global, out_recall] → gate → out_final → c_proj
           - Dynamic blending based on content
           - Final linear projection to match residual stream dimension

        Shape Notation
        --------------
        B: batch size, N: sequence length, D: n_embd (hidden dim)
        H_g: n_global_heads, H_r: n_recall_heads, Dh: head_dim

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, D) where B = Batch Size, 
            N = Sequence Length, D = Embedding Dimension (n_embd).

        Returns
        -------
        torch.Tensor
            Output context tensor of shape (B, N, D), suitable for residual
            connection addition in the transformer block.
        """
        B, N, D = x.shape
        Dh = self.head_dim

        # ---------------------------------------------------------------------
        # EXECUTION - PATH 1: GLOBAL DECAY
        # ---------------------------------------------------------------------
        # Convert logit-space parameter to valid decay factor via sigmoid.
        # λ ∈ (0, 1): higher values = slower decay (longer memory).
        # Each head learns its own optimal decay rate.
        lam = torch.sigmoid(self.log_lambda)
        
        # Project input to value space for global heads and reshape for kernel:
        # (B, N, D) → (B, N, H_g * Dh) → (B, H_g, N, Dh)
        # The contiguous() call ensures memory layout expected by the Triton kernel
        # (row-major with head dimension contiguous).
        V_global = self.v_proj_global(x).view(B, N, self.n_global_heads, Dh).transpose(1, 2).contiguous()

        # Invoke custom Triton causal decay kernel.
        # This kernel computes: out[i] = Σ_{j≤i} λ^(i-j) · V[j] / z_i
        # See triton_ops.py for the detailed implementation.
        out_global = triton_global_decay(lam, V_global)
        
        # Reshape back to standard attention output format:
        # (B, H_g, N, Dh) → (B, N, H_g * Dh)
        out_global = out_global.transpose(1, 2).reshape(B, N, self.n_global_heads * Dh)

        # ---------------------------------------------------------------------
        # EXECUTION - PATH 2: LATENT MLA
        # ---------------------------------------------------------------------
        # Step 1: Latent Query Generation
        # x → down-project → LayerNorm → up-project → reshape for multi-head
        # (B, N, D) → (B, N, latent_dim) → (B, N, H_r * Dh) → (B, H_r, N, Dh)
        q_latent = self.q_a_norm(self.q_a_proj(x))
        Q_rec = self.q_b_proj(q_latent).view(B, N, self.n_recall_heads, Dh).transpose(1, 2)

        # Step 2: Latent Key-Value Generation
        # Joint compression of K and V into shared latent space:
        # (B, N, D) → (B, N, latent_dim) → (B, N, 2 * H_r * Dh)
        kv_latent = self.kv_a_norm(self.kv_a_proj(x))
        
        # Split the joint projection into K and V:
        # (B, N, 2 * H_r * Dh) → (B, N, 2, H_r, Dh) → (2, B, H_r, N, Dh)
        # The permute(2, 0, 3, 1, 4) moves the "2" dimension (K vs V) to the front
        # so we can index KV_rec[0] and KV_rec[1] as separate tensors.
        KV_rec = self.kv_b_proj(kv_latent).view(B, N, 2, self.n_recall_heads, Dh).permute(2, 0, 3, 1, 4)
        K_rec, V_rec = KV_rec[0], KV_rec[1]

        # Step 3: Scaled Dot-Product Attention
        # F.scaled_dot_product_attention dispatches to the optimal backend:
        # - FlashAttention for Ampere+ GPUs with causal mask
        # - Memory-efficient attention for long sequences
        # - Standard attention as fallback
        # is_causal=True applies an upper-triangular mask, enforcing that
        # position i can only attend to positions j ≤ i.
        out_recall = F.scaled_dot_product_attention(
            Q_rec, K_rec, V_rec,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape to channel-last format for concatenation with global path:
        # (B, H_r, N, Dh) → (B, N, H_r * Dh)
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
        # gate ∈ (0, 1)^D: element-wise gating across the embedding dimension.
        gate = self.gamma_net(torch.cat([x, out_combined], dim=-1))
        out_final = out_combined * gate

        # Final linear projection to the output embedding space.
        # This projection allows mixing between global and recall head outputs
        # across different feature dimensions, complementing the element-wise gate
        # which only scales without mixing.
        return self.c_proj(out_final)