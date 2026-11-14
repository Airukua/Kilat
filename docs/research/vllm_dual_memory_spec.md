# Kilat Dual-Memory Serving Specification: PagedAttention + Recurrent State Pool (Phase P3.1)

## 1. Architectural Motivation

Standard Transformer serving engines (such as vLLM and TGI) assume a **homogeneous KV-cache topology**, allocating paged blocks of shape $(B_{\text{blocks}}, N_{\text{KV}}, B_{\text{size}}, D_{\text{head}})$ uniformly across all $L$ layers.

Kilat's **Macro-Synergy** hybrid architecture breaks this assumption by interleaving two mathematically distinct state topologies across layers $l \in \{0, \dots, L-1\}$:

1. **Even Layers ($l \bmod 2 = 0$, `recall_ratio = 0.0`) — Global Exponential Decay**:
   - **Recurrence**:
     $$S_t = \lambda \odot S_{t-1} + V_t, \qquad Z_t = \lambda Z_{t-1} + 1, \qquad O_t = \frac{S_t}{Z_t}$$
   - **State Complexity**: $O(1)$ in sequence length $T$.
   - **Per-Sequence State Footprint**: Fixed tensor $S_t \in \mathbb{R}^{H_{\text{global}} \times D_{\text{head}}}$ plus step counter $t \in \mathbb{Z}_{\ge 0}$ (or normalizer $Z_t \in \mathbb{R}^{H_{\text{global}}}$). Zero token-level KV cache is stored.

2. **Odd Layers ($l \bmod 2 = 1$, `recall_ratio = 1.0`) — Decoupled RoPE Latent MLA**:
   - **Low-Rank Joint KV Compression**:
     $$c_t^{\text{KV}} = W_{\text{DKV}} x_t \in \mathbb{R}^{d_c}, \qquad k_t^{\text{R}} = \text{RoPE}(W_{\text{KR}} x_t) \in \mathbb{R}^{d_r}$$
   - **State Complexity**: $O(T)$ in sequence length $T$, compressed by $d_c \ll H \cdot D_{\text{head}}$.
   - **Per-Token Cache Entry**: Concatenated latent vector $(c_t^{\text{KV}}, k_t^{\text{R}}) \in \mathbb{R}^{d_c + d_r}$, compatible with block-paged memory (`PagedAttention`).

---

## 2. Dual-Memory Pool Topology

A high-throughput Kilat serving engine partitions GPU HBM reservation into two coordinated allocators managed by a unified `HybridScheduler`:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     KILAT DUAL-MEMORY SERVING ENGINE                       │
├─────────────────────────────────────┬──────────────────────────────────────┤
│   POOL A: Recurrent State Pool      │   POOL B: Paged Latent MLA Pool      │
│   (Even Layers: l = 0, 2, ..., L-2) │   (Odd Layers: l = 1, 3, ..., L-1)   │
├─────────────────────────────────────┼──────────────────────────────────────┤
│ • Contiguous Slot Tensor:           │ • Paged Block Table:                 │
│   [M_max_seqs, L_even, H_g, D_h]    │   [N_blocks, L_odd, BlockSize, d_c+r]│
│ • Step Counter Vector:              │ • Logical-to-Physical Block Map:     │
│   seq_lens[M_max_seqs] (int32)      │   block_tables[M_max_seqs, MaxBlocks]│
│ • O(1) allocation per active req    │ • Dynamic block allocation every     │
│   (allocated on prefill, freed EOS) │   `BlockSize` (e.g., 16) tokens      │
└─────────────────────────────────────┴──────────────────────────────────────┘
```

### 2.1 Memory Footprint Reduction vs. Standard MHA
For a sequence of length $T$, model dimension $D = H \cdot D_h$, and $L$ layers ($L/2$ Global Decay, $L/2$ Latent MLA with latent dimension $d_c = D/4$ and RoPE dimension $d_r = D_h/2$):

$$\text{Bytes}_{\text{Kilat}}(T) = \underbrace{\frac{L}{2} \cdot H \cdot D_h \cdot b_{\text{state}}}_{\text{Pool A: } O(1) \text{ Recurrent State}} + \underbrace{\frac{L}{2} \cdot T \cdot (d_c + d_r) \cdot b_{\text{kv}}}_{\text{Pool B: } O(T) \text{ Compressed Latent MLA}}$$

Compared to standard Multi-Head Attention ($\text{Bytes}_{\text{MHA}}(T) = L \cdot T \cdot 2 H D_h \cdot b_{\text{kv}}$), Kilat reduces asymptotic per-token KV memory by:

$$\frac{\text{Asymptotic KV Slope}_{\text{Kilat}}}{\text{Asymptotic KV Slope}_{\text{MHA}}} = \frac{\frac{1}{2}(d_c + d_r)}{2 D} \approx \frac{1}{16} \quad (\sim 93.75\% \text{ KV-cache reduction})$$

---

## 3. Execution Lifecycle & Continuous Batching Invariants

### 3.1 Request Admission & Prefill
1. **Slot Assignment**: When request $r$ enters the running batch, the allocator assigns:
   - One recurrent state slot `slot_id` $\in \{0, \dots, M_{\max}-1\}$ in Pool A, zero-initialized ($S_0 = 0, t = 0$).
   - $\lceil T_{\text{prompt}} / B_{\text{size}} \rceil$ physical pages in Pool B.
2. **Prefill Forward**:
   - **Even Layers**: Execute `triton_global_decay` over the prompt $(1, T_{\text{prompt}}, D)$ and write the terminal recurrent state $S_{T_{\text{prompt}}}$ into `PoolA.states[slot_id, l_even]` and `PoolA.seq_lens[slot_id] = T_prompt`.
   - **Odd Layers**: Project $(c_{1:T}^{\text{KV}}, k_{1:T}^{\text{R}})$, scatter into `PoolB` physical pages via `block_tables[slot_id]`, and compute causal FlashAttention/MLA prefill.

### 3.2 Decode Step (Single-Token Autoregressive Batch)
For a batch of $B$ decoding requests with `slot_ids` $[s_1, \dots, s_B]$:
1. **Even Layers ($O(1)$ Fused Recurrent Step)**:
   - Gather `S_prev = PoolA.states[slot_ids, l_even]` and `t_prev = PoolA.seq_lens[slot_ids]`.
   - Update in-place:
     $$S_{\text{new}} = \lambda \odot S_{\text{prev}} + V_t, \qquad Z_{\text{new}} = \text{compute\_decay\_normalizer\_from\_logit}(\log \lambda, t_{\text{prev}})$$
     $$O_t = \frac{S_{\text{new}}}{Z_{\text{new}}}$$
2. **Odd Layers (Paged Latent MLA Decode)**:
   - Append new $(c_t^{\text{KV}}, k_t^{\text{R}})$ to the tail page in `PoolB`.
   - Run paged latent attention over `[0 .. t_prev]`, up-projecting latent KV via $W_{\text{UK}}, W_{\text{UV}}$ (or absorbed query-key multiplication $q_t (W_{\text{UK}})^\top c_{\le t}^{\text{KV}}$).

### 3.3 Preemption, Swapping, and Prefix Caching Constraints
- **Preemption**: Swapping a request to CPU RAM copies both its fixed-size Pool A slot ($O(1)$ bytes) and its Pool B pages.
- **Prefix Caching**: Unlike pure attention where any token boundary is addressable from paged KV blocks, Global Decay state $S_t$ is a Markovian summary at boundary $t$. Prefix caching on shared system prompts must cache the **pair** $(\text{PoolA\_snapshot}(t_{\text{prefix}}), \text{PoolB\_pages}(1 \dots t_{\text{prefix}}))$ at discrete prefix checkpoints.

---

## 4. Phased Implementation Trajectory

| Tier | Deployment Target | Engine Mechanism | Status |
| :---: | :--- | :--- | :---: |
| **Tier 1** | Single-GPU / Edge SLM & HF Ecosystem | Native `LayerHybridCache` via `KilatForCausalLM.generate()` & `export_to_hf()` standalone bundle (`trust_remote_code=True`). | **Production Ready** |
| **Tier 2** | Multi-Request Async Microservice | FastAPI async batching runner pooling `LayerHybridCache` tuples with dynamic request slot recycling. | **Supported via Native Cache** |
| **Tier 3** | High-Throughput Cluster Serving (vLLM) | Out-of-tree vLLM stateful model plugin (`MambaCacheManager` + MLA `PagedAttention` hybrid backend). | **Specification Complete** |
