# 02 — Mathematical Specification & Surgical Bugfix Plan (`M-01` .. `M-07`)

> **Document Purpose**: Provide exact mathematical specifications, IEEE-754 numerical precision bounds, tensor shapes, and code locations for the **seven mathematically verified bugs** in `src/kilat/`.
> **Evidence Standard**: No code fix described in this document is considered complete until it passes `torch.autograd.gradcheck` in `torch.float64` (tolerance $< 10^{-6}$) and reference-equivalence tests.

---

## Executive Summary of the Seven Mathematical Fixes

| ID | Component & Canonical File Location | Mathematical Failure Prevented | Silent Failure if Violated |
| :---: | :--- | :--- | :--- |
| **M-01** | `src/kilat/models/kernels/flash_decay.py` & `src/kilat/models/attention.py` | The CUDA kernel `flash_decay_fwd_kernel` must be wrapped in `FlashDecayFunction` (`autograd.Function`); the naive formula $(1-\lambda^{i+1})/(1-\lambda+10^{-6})$ fails as $\lambda \to 1^-$. | The entire *Global Decay* branch (`log_lambda`, `v_proj_global`) loses gradients on GPU; normalization collapses toward $0$ as $\lambda \to 1$. |
| **M-02** | `src/kilat/models/blocks.py` (`RMSNorm`) | The variance in `RMSNorm.forward` must be upcast to `float32` (while preserving `float64` for `gradcheck`). | Overflow to `Inf` causes `0.0` / `NaN` output when $|x| > 256$ under FP16 AMP. |
| **M-03** | `src/kilat/engine/optimizer.py` (`create_optimizer`) | All parameters with `ndim < 2`, `RMSNorm`, `LayerNorm`, `.bias`, and `log_lambda` must be assigned to the `weight_decay = 0.0` group. | `attn.log_lambda` and `q_a_norm`/`kv_a_norm` are eroded by weight decay, damaging the long-term memory horizon ($\lambda \to 0.5$). |
| **M-04** | `src/kilat/models/ffn.py` (`DeepSeekMoE`) | Normalize `expert_fraction` by $K_{\text{eff}}$; skip Top-1 renormalization to preserve router gradients; support `num_shared_experts >= 0` and `bias_free` mode; compute router logits in FP32. | `aux_loss` is inflated by a factor of $K$; the Top-1 router receives no LM-loss gradient; configurations without shared experts fail. |
| **M-05** | `src/kilat/models/attention.py` (`LayerHybridCache`) | Store `seq_len` explicitly in `LayerHybridCache`; support *chunked prefill* ($N > 1$); store RoPE keys in the cache after rotation. | Generation output can explode by $\sim 10\times$ with `recall_ratio=0.0`; *chunked prefill* loses the preceding context. |
| **M-06** | `src/kilat/models/transformer.py`, `src/kilat/engine/trainer.py`, `src/kilat/engine/checkpoint.py` | Separate `ce_loss` from `aux_loss` so `ppl = exp(ce_loss)` is token-weighted; scale partial accumulation windows by $m_w$; compute `compute_total_steps` with `ceil`; scale residuals by $1/\sqrt{2L}$. | MoE PPL metrics are biased high; gradients in the final epoch step are artificially reduced; the LR scheduler exceeds the step limit; activation variance grows linearly with depth $L$. |
| **M-07** | `src/kilat/models/{blocks,ffn,transformer}.py` | Make zero `.all()` / `.any()` / `.item()` synchronization calls inside `nn.Module.forward()` when `debug_nan_checks=False`. | Stalls the CUDA stream (CPU-GPU synchronization in every sub-layer) and breaks `torch.compile(fullgraph=True)`. |

---

## M-01: Exact Analytical $O(N)$ Autograd & Stability as $\lambda \to 1^-$ in `triton_global_decay`

### 1. Exact Forward Equation (Free of the $1 - \lambda$ Singularity)
For value inputs $V \in \mathbb{R}^{B \times N \times H_g \times D_h}$ and per-head decay parameters $\lambda \in (0, 1)^{H_g}$, where $\lambda_h = \sigma(\alpha_h)$ (`alpha = self.log_lambda`), define the causal recurrence for position indices $i \in \{0, 1, \dots, N-1\}$:

$$s_{b, 0, h} = v_{b, 0, h}, \qquad s_{b, i, h} = \lambda_h s_{b, i-1, h} + v_{b, i, h}$$

$$z_{0, h} = 1, \qquad z_{i, h} = \lambda_h z_{i-1, h} + 1, \qquad z'_{0, h} = 0, \qquad z'_{i, h} = \lambda_h z'_{i-1, h} + z_{i-1, h}$$

$$y_{b, i, h} = \frac{s_{b, i, h}}{z_{i, h}}$$

**Mathematical Advantages of the $(z_i, z'_i)$ Recurrence**:
- Since $\lambda_h \in [0, 1]$, $z_{i,h} \ge 1$ is guaranteed for all $i \ge 0$. **There is no division by $(1 - \lambda_h)$**, so division-by-zero errors and catastrophic cancellation cannot occur, even at the extreme $\lambda_h = 1.0$ ($z_i = i+1$, $z'_i = \frac{i(i+1)}{2}$).
- All recurrence accumulators ($s, z, z'$) in both the Triton kernel and CPU path **must** use `float32` (or `float64` when `V.dtype == torch.float64` for `gradcheck`), then be cast back to `V.dtype` when writing $y$.

### 2. Stable Closed Form for an Index Jump at $t$ (Incremental Decode)
During *incremental decoding* at sequence position $t = \text{cache.seq\_len}$ (without rescanning $0 \dots t$), never compute `(1 - lam**(t+1)) / (1 - lam + 1e-6)`. Use the log-space parameterization $u_h = \ln \sigma(\alpha_h) = -\operatorname{softplus}(-\alpha_h) = \operatorname{logsigmoid}(\alpha_h) \le 0$ in `float32`/`float64` and the smooth analytic function $\varphi_1(w)$:

$$\varphi_1(w) = \begin{cases} \dfrac{\operatorname{expm1}(w)}{w}, & |w| \ge 10^{-4} \\[8pt] 1 + \dfrac{w}{2} + \dfrac{w^2}{6} + \dfrac{w^3}{24}, & |w| < 10^{-4} \end{cases}$$

Then, for any $n = t + 1 \ge 1$ (including $n = 10^6$) and any $\alpha_h \in (-\infty, +\infty)$:

$$z_{t, h} = n \cdot \frac{\varphi_1(n u_h)}{\varphi_1(u_h)}$$

This formula is accurate to machine precision with no upper bound on context length $n$.

### 3. Exact Analytical $O(N)$ Backward Derivation (`FlashDecayFunction.backward`)
Let the upstream gradient be $\frac{\partial \mathcal{L}}{\partial y} = G \in \mathbb{R}^{B \times N \times H_g \times D_h}$. Define the scaled gradient $\tilde{g}_{b, i, h} = \frac{g_{b, i, h}}{z_{i, h}}$.

1. **Gradient with respect to $V$ (reverse scan $i = N-1 \dots 0$)**:
   $$r_{b, N-1, h} = \tilde{g}_{b, N-1, h}, \qquad r_{b, i, h} = \tilde{g}_{b, i, h} + \lambda_h r_{b, i+1, h}$$
   $$\frac{\partial \mathcal{L}}{\partial v_{b, i, h}} = r_{b, i, h}$$

2. **Gradient with respect to $\lambda$ (without an additional $O(BND)$ buffer!)**:
   Using the adjoint identity $\sum_{i=0}^{N-1} \langle \tilde{g}_i, q_i \rangle = \sum_{i=1}^{N-1} \langle r_i, s_{i-1} \rangle$ and $s_{i-1} = z_{i-1} y_{i-1}$, the gradient with respect to $\lambda_h$ can be computed directly during the same *reverse scan*, storing only the forward output $Y$ and the small tables $(z, z') \in \mathbb{R}^{N \times H_g}$:
   $$\frac{\partial \mathcal{L}}{\partial \lambda_h} = \sum_{b=0}^{B-1} \sum_{i=1}^{N-1} \left( z_{i-1, h} \langle r_{b, i, h}, y_{b, i-1, h} \rangle - z'_{i, h} \langle \tilde{g}_{b, i, h}, y_{b, i, h} \rangle \right)$$
   *(Note: for $i=0$, $z'_{0,h} = 0$, so the $i=0$ term is identically zero.)*

### 4. Code Implementation Contract
- Implement `class FlashDecayFunction(torch.autograd.Function)` in `src/kilat/models/kernels/flash_decay.py`:
  - `forward(ctx, V, lam)` supports the `@triton.jit` `flash_decay_fwd_kernel` path when `V.is_cuda and HAS_TRITON and V.dtype != torch.float64`, and the compiled/vectorized PyTorch path on CPU or for `torch.float64`. Save `ctx.save_for_backward(y, lam, z, z_prime)`.
  - `backward(ctx, grad_out)` calls the `@triton.jit` `flash_decay_bwd_kernel` on CUDA (with `tl.float32` accumulation and `tl.atomic_add` or per-program reduction for `grad_lam`) and uses an exact reverse scan on CPU/`float64`. Return `(grad_V.to(V.dtype), grad_lam.to(lam.dtype))`.
- Provide the pure function `_reference_global_decay_fp64(V, lam)` (exact lower-triangular causal matrix $M_{i,j} = \frac{\lambda^{i-j}}{z_i}$ for $j \le i$) specifically for Gate 2 and Gate 3 equivalence tests.

---

## M-02: Upcasting Variance to FP32 in `RMSNorm`

### 1. Location & Problem
A naive implementation of `src/kilat/models/blocks.py` (`RMSNorm.forward`):
```python
norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
x_normed = x * torch.rsqrt(norm_x + self.eps)
return self.weight * x_normed
```
In `torch.float16`, the maximum representable value is $65504$. Once a residual activation element satisfies $|x_k| > 255.9$, then $x_k^2 > 65504 \implies +\infty$, so `torch.rsqrt(inf) == 0.0` (or `inf * 0.0 == NaN`).

### 2. Target Equation & Implementation
For input $x \in \mathbb{R}^{\dots \times D}$ with any dtype $\tau \in \{\text{fp16}, \text{bf16}, \text{fp32}, \text{fp64}\}$:
$$\tilde{x} = \operatorname{cast}_{\text{fp32}}(x) \quad (\text{unless } \tau = \text{fp64}, \text{ in which case preserve fp64})$$
$$\operatorname{RMSNorm}(x) = \operatorname{cast}_{\tau}\!\left(\tilde{x} \cdot \frac{1}{\sqrt{\frac{1}{D}\sum_{k=1}^{D} \tilde{x}_k^2 + \epsilon}}\right) \odot w$$

Exact implementation:
```python
compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
x_up = x.to(compute_dtype)
variance = x_up.pow(2).mean(dim=-1, keepdim=True)
x_normed = (x_up * torch.rsqrt(variance + self.eps)).to(x.dtype)
return x_normed * self.weight
```
*(Note the `torch.float64` guard: it prevents downcasting FP64 $\to$ FP32, which would cause `torch.autograd.gradcheck` to fail.)*

---

## M-03: Weight-Decay Partitioning by Dimension & Module Type (`create_optimizer`)

### 1. Location & Problem
In `src/kilat/engine/optimizer.py`, naive name matching only uses:
`no_decay_keywords = ("bias", "LayerNorm", "rms", "ln_f", "rms1", "rms2")`.
- `attn.log_lambda` has shape `(n_global_heads,)` (`ndim == 1`). With $\lambda = \sigma(\text{log\_lambda})$ initialized to $\ln 9 \approx 2.197$ ($\lambda = 0.9$), *weight decay* pulls `log_lambda` toward $0 \implies \lambda \to 0.5$, even when task gradients favor long memory.
- `attn.q_a_norm.weight` and `attn.kv_a_norm.weight` belong to `nn.LayerNorm`, but their attribute names do not contain the string `"LayerNorm"`, so they are also affected by *weight decay*.

### 2. Exact Partitioning Rules (Karpathy / TorchTitan / OLMo Standard)
A parameter $\theta$ named `param_name` in a `module` belongs to the **`no_decay` group (`weight_decay = 0.0`)** if it satisfies **at least one** of the following conditions:
1. `param.ndim < 2` (covers all `bias` parameters, `RMSNorm`/`LayerNorm` weights, and per-head scalar/vector parameters such as `log_lambda`).
2. `isinstance(module, (nn.LayerNorm, RMSNorm, nn.Embedding))` — note: for `wte` (`nn.Embedding`), provide an option `decay_embeddings: bool = False` (default `False`, consistent with modern LLM practice, or use an explicit flag to avoid breaking older tests that check parameter groups—verify against `tests/unit/features/training/test_training_components.py`).
3. The parameter name ends in `.bias` or contains a normalization keyword (`norm`, `rms`, `ln_`, `log_lambda`, case-insensitive).

> **Anti-Hallucination Note for Existing Tests**: Before enabling the `nn.Embedding` exclusion, read `tests/unit/features/training/test_training_components.py` to ensure parameter-count assertions for the `decay` and `no_decay` groups remain compatible. `param.ndim < 2` plus case-insensitive matching of `("bias", "layernorm", "norm", "rms", "ln_", "log_lambda")` already protects `log_lambda`, `q_a_norm`, `kv_a_norm`, `rms1`, `rms2`, `ln_f`, and all biases without changing how 2D matrices are treated.

---

## M-04: Exact `DeepSeekMoE` Mathematics (Load Balance, Top-1 Gradient, Shared Experts $\ge 0$, & Zero-Sync Dispatch)

### 1. Four Mathematical Invariants in `src/kilat/models/ffn.py`

1. **Router-Logit Precision (`float32`)**:
   For flattened token input $X \in \mathbb{R}^{T \times D}$ ($T = B \cdot S$), compute router logits in FP32 (except in FP64 for `gradcheck`):
   $$Z_{\text{router}} = X_{\text{up}} W_{\text{router}}^\top, \qquad P = \operatorname{softmax}(Z_{\text{router}}, \text{dim}=-1)$$
   where $P \in \mathbb{R}^{T \times N_r}$, $N_r = \text{num\_experts} \times \text{fine\_grained\_factor}$ is the total number of *routed experts*, and $K_{\text{eff}} = \text{active\_experts} \times \text{fine\_grained\_factor}$ is the number of active experts per token.

2. **Router-Gradient Guard when $K_{\text{eff}} == 1$**:
   Select $(W_{\text{top}}, I_{\text{top}}) = \operatorname{TopK}(P, K_{\text{eff}})$.
   $$\tilde{W}_{\text{top}} = \begin{cases} \dfrac{W_{\text{top}}}{\sum_{k=1}^{K_{\text{eff}}} W_{\text{top}, k} + \epsilon}, & \text{if } K_{\text{eff}} > 1 \text{ and } \text{renormalize\_topk}=\text{True} \\[8pt] W_{\text{top}}, & \text{if } K_{\text{eff}} == 1 \end{cases}$$
   Thus, when $K_{\text{eff}} = 1$, $\tilde{W}_{\text{top}} = P_{t, i^*(t)}$ remains dependent on the router parameters, and $\nabla_{W_{\text{router}}} \mathcal{L}_{\text{CE}} \neq \mathbf{0}$.

3. **Exact Load-Balancing Loss Equation (DeepSeek-V2 Eq. 16 / Switch Transformer)**:
   Define expert assignment fraction $f_i$ and mean probability $P_i$ for $i \in \{0, \dots, N_r - 1\}$:
   $$f_i = \frac{N_r}{K_{\text{eff}} \cdot T} \sum_{t=1}^{T} \mathbb{1}\!\left[i \in I_{\text{top}}(t)\right], \qquad \bar{P}_i = \frac{1}{T} \sum_{t=1}^{T} P_{t, i}$$
   $$\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}} \sum_{i=0}^{N_r - 1} f_i \cdot \bar{P}_i$$
   **Required Invariant**: Under perfectly uniform routing ($\bar{P}_i = \frac{1}{N_r}$ and each expert receives exactly $\frac{K_{\text{eff}} T}{N_r}$ assignments), $f_i = \frac{N_r}{K_{\text{eff}} T} \cdot \frac{K_{\text{eff}} T}{N_r} = 1$. Therefore, $\sum_{i=0}^{N_r - 1} f_i \bar{P}_i = \sum_{i=0}^{N_r - 1} 1 \cdot \frac{1}{N_r} = 1.0$, and $\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}}$, **independent of both $K_{\text{eff}}$ and $N_r$**.
   - Count expert assignments with the vectorized `torch.bincount(I_top.reshape(-1), minlength=N_r)` instead of a Python loop.

4. **Support for `num_shared_experts >= 0` & Optional DeepSeek-V3 Bias-Free Balancing**:
   - Replace `validate_positive_int("num_shared_experts", num_shared_experts)` at `ffn.py:198` with `validate_non_negative_int("num_shared_experts", num_shared_experts)`. If `num_shared_experts == 0`, initialize `shared_out = torch.zeros_like(x_flat)`.
   - Remove the synchronizing branch `if expert_mask.any():` at `ffn.py:310`. Sort token indices (`sorted_expert_ids, perm = I_top.reshape(-1).sort()`, `counts = bincount(...)`) or process experts without per-expert host synchronization to avoid GPU stalls (and avoid `.any()` even for small CPU tensors).
   - Add the optional `router_balance_mode: Literal["aux_loss", "bias_free"] = "aux_loss"` to `KilatConfig`: in `"bias_free"` mode (following DeepSeek-V3), select Top-K using $\operatorname{TopK}(P + b_{\text{expert}})$ while computing mixture weights from the original unbiased $P$; set `aux_loss = 0.0`; and update the expert bias $b_{\text{expert}}$ without gradients at the end of the optimization step: $b_i \leftarrow b_i + \gamma \operatorname{sign}(\bar{c} - c_i)$. Keep `"aux_loss"` as the default for full backward compatibility.

---

## M-05: Explicit `HybridCache` State Machine (Fixes for `recall_ratio=0.0`, Chunked Prefill, & RoPE Cache)

### 1. `HybridCache` Data Structure
Instead of passing a raw tuple `(new_global_state, new_latent_kv)` that loses sequence-length information when `n_recall_heads == 0`, define a lightweight `LayerHybridCache` class (while retaining two-element `(global_state, latent_kv)` `__iter__` and `__getitem__` support for full backward compatibility with existing callers):

```python
@dataclass
class LayerHybridCache:
    global_state: torch.Tensor          # (B, H_global, D_head) in float32
    latent_kv: torch.Tensor             # (B, S_past, D_latent)
    seq_len: int                        # Actual number of past tokens S_past >= 0
    rope_k: Optional[torch.Tensor] = None  # (B, H_recall, S_past, D_rope), already RoPE-rotated

    def __iter__(self):
        # Backward compatibility for existing code that does: g_state, l_kv = past_kv
        yield self.global_state
        yield self.latent_kv
```

### 2. Multi-Token (`N >= 1`) Cached Transition Equations
Assume the past cache has length $T_0 = \text{seq\_len} \ge 0$ with unnormalized accumulator $S_{T_0 - 1} = \text{global\_state} \in \mathbb{R}^{B \times H_g \times D_h}$, and the new input has length $N \ge 1$ (either *single-token decode* with $N=1$ or *chunked prefill* with $N > 1$):
1. **Global Decay branch (`n_global_heads > 0`)**:
   For local step $j \in \{0, \dots, N-1\}$ (absolute position $t = T_0 + j$):
   $$S_{T_0 + j} = \lambda S_{T_0 + j - 1} + V_{\text{new}, j} \qquad (S_{-1} = \mathbf{0})$$
   $$y_{T_0 + j} = \frac{S_{T_0 + j}}{z_{T_0 + j}(\lambda)}$$
   where $z_{T_0 + j}(\lambda)$ is computed using the stable $\varphi_1$ formula from **M-01** with $t = T_0 + j$ from `cache.seq_len` (**not** `latent_kv.shape[1]`). Store `new_global_state = S_{T_0 + N - 1}` (FP32) and `new_seq_len = T_0 + N`.
2. **MLA Recall & Decoupled RoPE branch (`n_recall_heads > 0`)**:
   - Concatenate `latent_kv_total = cat([cache.latent_kv, new_latent_kv], dim=1)` with shape `(B, T_0 + N, D_latent)`.
   - In `KilatAttentionRoPE`, apply RoPE **only** to new queries (`pos = T_0 .. T_0 + N - 1`) and new keys (`pos = T_0 .. T_0 + N - 1`), then concatenate `rope_k_total = cat([cache.rope_k, new_rope_k_rotated], dim=2)`. This reduces decode-step complexity from $O(T^2)$ for repeated rotations to $O(1)$ per step and guarantees numerical parity with full *prefill*.
   - For the rectangular causal mask used in *chunked prefill* ($N > 1$ and $T_0 > 0$), a query at local index $j \in [0, N-1]$ may attend to keys at absolute indices $k \in [0, T_0 + j]$.

---

## M-06: Separating $\mathcal{L}_{\text{CE}}$ from $\mathcal{L}_{\text{aux}}$, Token-Weighted PPL, Partial Accumulation, & Residual Scaling by $\frac{1}{\sqrt{2L}}$

### 1. Model Output Contract & Pure Perplexity
In `KilatTransformer.forward` (`src/kilat/models/transformer.py`):
- Compute pure cross-entropy loss in `float32`:
  $$\mathcal{L}_{\text{CE}} = \operatorname{CrossEntropy}\!\left(\operatorname{cast}_{\text{fp32}}(\text{logits}_{:, :-1, :}),\; \text{labels}_{:, 1:},\; \text{ignore\_index}=-100,\; \text{reduction}=\text{"mean"}\right)$$
  and the number of valid tokens $T_{\text{valid}} = \sum_{b, t} \mathbb{1}[\text{labels}_{b, t+1} \neq -100]$.
  - **Empty-Target Guard (`T_valid == 0`)**: If all labels in a *micro-batch* are `-100`, PyTorch's `F.cross_entropy(..., reduction="mean")` returns **`NaN`** ($0/0$). Add a guard: when `T_valid == 0`, return `(logits * 0.0).sum()` (differentiable zero without `NaN`) and log a warning through `FlightRecorder`.
- The total optimization objective is:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}$$
- Store the latest graph-free metrics (`detach()`) or return a `KilatModelOutput` object that **remains fully compatible with two-/three-element tuple unpacking `(logits, loss)` or `(logits, loss, presents)`** (for example, store `_last_loss_breakdown = {"ce_loss": ..., "aux_loss": ..., "valid_tokens": ...}` on the module or use a named `ModelOutput` tuple subclass). This ensures legacy code `logits, loss = model(x, labels=y)` **never breaks**, while allowing `KilatTrainer` to read `ce_loss`, `aux_loss`, and `valid_tokens` exactly.
- **Token-Weighted Evaluation & Pure PPL (`trainer.py`)**:
  In `_default_eval_loop`:
  $$\overline{\mathcal{L}}_{\text{eval, CE}} = \frac{\sum_{m=1}^{M} \mathcal{L}_{\text{CE}, m} \cdot T_{\text{valid}, m}}{\max\!\left(1,\; \sum_{m=1}^{M} T_{\text{valid}, m}\right)}, \qquad \text{eval\_ppl} = \exp\!\left(\min(\overline{\mathcal{L}}_{\text{eval, CE}}, 20.0)\right)$$
  Log these separately: `train_ce_loss`, `train_aux_loss`, `train_loss` (total), `eval_loss` (pure CE), `eval_aux_loss`, and `eval_ppl` (pure $\exp(\overline{\mathcal{L}}_{\text{eval, CE}})$).

### 2. Partial Gradient-Accumulation Alignment & `compute_total_steps`
Let the `train_dataloader` contain $B_{\text{total}}$ *micro-batches* per *epoch*, and let the gradient-accumulation factor be $G = \text{gradient\_accumulation\_steps}$:
1. **Total Optimizer Steps (`optimizer.py` / `trainer_utils.py`)**:
   For `training_mode == "epochs"`:
   $$S_{\text{epoch}} = \left\lceil \frac{B_{\text{total}}}{G} \right\rceil = \frac{B_{\text{total}} + G - 1}{G} \quad (\text{pembagian bulat atas}), \qquad S_{\text{total}} = S_{\text{epoch}} \times E$$
2. **Exact Gradient Scaling for Partial Windows (`trainer.py`)**:
   For accumulation window $w$, calculate its actual size:
   $$m_w = \min\!\left(G,\; B_{\text{total}} - \text{window\_start\_idx}\right)$$
   Divide each *micro-batch* loss in window $w$ by **`m_w`** (not always by `G`).
   - For every full window, $m_w = G$ (unchanged from before).
   - For the remainder window at the end of an *epoch* ($1 \le m_w < G$), averaging gradients over $m_w$ *micro-batches* is **unbiased**: $\frac{1}{m_w}\sum_{j=1}^{m_w} \nabla \ell_j$.

### 3. Residual-Depth Initialization Scaling (GPT-2 / DeepSeek Standard)
In `KilatTransformer._init_weights` (`src/kilat/models/transformer.py`), output projection matrices that add directly to the main residual stream (`attn.c_proj.weight`, `ffn.down.weight` in `SwiGLU`, `shared_experts`, and `routed_experts`) are reinitialized after `self.apply(self._init_weights)` with standard deviation:

$$\sigma_{\text{residual}} = \frac{\text{config.initializer\_range}}{\sqrt{2 \cdot \text{config.n\_layer}}}$$

This keeps residual-stream variance $\operatorname{Var}(h^{(L)}) = \mathcal{O}(1)$ from exploding as `n_layer` increases.

---

## M-07: Removing 450+ CUDA Stream Synchronizations from the `forward()` Hot Path

### 1. Problem
Calls to `validate_finite_tensor(...)` inside `SwiGLU.forward`, `DeepSeekMoE.forward`, `FeedForward.forward`, and `KilatTransformer.forward` execute `if not torch.isfinite(tensor).all():`, implicitly inserting hundreds of `cudaStreamSynchronize` operations in a single *forward* step.

### 2. Fix Specification
- Add a lightweight configuration flag to `KilatConfig`: `debug_nan_checks: bool = False` (default **`False`**).
- Remove unconditional `validate_finite_tensor` calls from `nn.Module.forward()`. Run the validator only when `self.debug_nan_checks` is `True` (for diagnosing divergence).
- Keep `validate_finite_tensor` in `src/kilat/utils/validators.py` so all unit tests in `tests/unit/features/utils/test_validators.py` continue to pass.
- Move production `NaN`/`Inf` detection to the end of the optimization step in `KilatTrainer` and `FlightRecorderCallback` (one check on the loss scalar per optimization step, rather than 450 checks per *micro-batch*).
