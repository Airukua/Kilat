# 04 — Frontier Assurance, Active Chaos Engineering (`Saboteur`), & Runtime `FlightRecorder`

> **Document Purpose**: Evolve Kilat validation beyond syntax-focused *unit testing* into a three-layer **Research-Grade Pre-Training Validation & Chaos Assurance Pipeline** (Pre-Flight Gates, Active Fault Injection, and Runtime Flight Telemetry) aligned with frontier-lab practices (Google DeepMind `Paxml`/`MaxText`, Meta `Lingua`, Microsoft/OpenAI $\mu\text{P}$, and `DeepSeek-V3`).

---

## Kilat's Three-Layer Assurance Architecture

```mermaid
flowchart LR
    subgraph L1["Layer 1: Pre-Flight Validation (kilat verify)"]
        G1["Gate 1: Theory & Init"] --> G2["Gate 2: Numerical & Causal"]
        G2 --> G3["Gate 3: Equivalence"]
        G3 --> G4["Gate 4: Data Forensics"]
        G4 --> G5["Gate 5: Micro-Dynamics & Stability"]
        G5 --> G6["Gate 6: Memory, DDP & Resume Parity"]
        G6 -. "--full (Research)" .-> G7["Stage 7-8: muP & WSD Scaling Readiness"]
    end

    subgraph L2["Layer 2: Active Chaos (kilat chaos)"]
        S["Saboteur Runtime Hooks (C-01 .. C-08)"] -->|Inject Silent Bug| L1
        L1 -->|Must Catch 100%| V["Meta-Verifier: 0% False Negative"]
    end

    subgraph L3["Layer 3: Runtime Protection (FlightRecorder)"]
        T["Training Run"] --> FR["Async Metrics (No GPU Sync)"]
        FR -->|Anomaly Detected| CB["Circuit Breaker + Forensic Dump"]
    end

    L1 -->|Signed Artifact| R["assurance_report.{json,md}"]
```

---

## 1. Layer 1 — Pre-Flight Validation Pipeline (`kilat verify`)

Each gate produces a `PASS` / `FAIL` status, duration in milliseconds, measurable metrics, and a structured failure code (`AssuranceFailureCode`). Gates run sequentially; an early failure prevents wasted computation in later gates (unless `--keep-going` is specified).

### Gate 0: Security & Supply-Chain Posture (`gate0_security.py`) — `G0_SECURITY_SUPPLY_CHAIN`
Runs **first** (low-cost, CPU-only, < 1 second). A mathematically correct model that loads attacker code from a checkpoint is not fit for training.
- **Static AST scan** of the entire `kilat` package: `torch.load` without `weights_only=True`, the `pickle` family, unsafe `yaml.load`, and `numpy.load(allow_pickle=True)` → `ERR_INSECURE_DESERIALIZATION`; `trust_remote_code=True`, `eval`/`exec` → `ERR_REMOTE_CODE_EXECUTION`; `shell=True`/`os.system`/`os.popen` → `ERR_SHELL_INJECTION_RISK`.
- **Config audit**: `tokenizer.custom_builder` / `custom_module` must be on the allowlist (`kilat.*` + `KILAT_TRUSTED_MODULES`) → `ERR_UNTRUSTED_DYNAMIC_IMPORT`.
- **Anti-vacuous**: scan 0 file = FAIL.
- Mapping: OWASP LLM03/LLM04, OWASP ML06/ML10, OWASP A01/A03/A08, NCSC/CISA *Guidelines for Secure AI System Development*. Details: `.agents/skills/core/ml-security-assurance/SKILL.md`.

### Gate 1: Theoretical & Initialization Sanity (`gate1_theory.py`)
Verifies theoretical equations and initialization distributions at step 0, before any optimization step is run:
1. **Unbiased Initial Entropy ($\mathcal{L}_0 \approx \ln V$)**:
   - For uniformly random input $x \sim \mathcal{U}\{0, \dots, V-1\}$, the initial *pure cross-entropy loss* must fall within:
     $$\left|\mathcal{L}_{\text{CE}}(0) - \ln V\right| \le 0.15$$
   - Failure indicates an oversized `lm_head` scale (*overconfident logits*), incorrect scaling by $\sqrt{d_k}$, or a nonzero bias in the final projection.
2. **Residual-Stream Variance Stability Across Depth**:
   - Measure block-output standard deviation $\sigma_\ell = \operatorname{std}(h^{(\ell)})$ for $\ell = 1 \dots L$. The growth ratio $\sigma_L / \sigma_1$ must remain in the healthy range $[0.3, 3.0]$ due to residual scaling by $\frac{\text{initializer\_range}}{\sqrt{2L}}$ (**M-06**).
3. **Audit Grup Weight Decay (`M-03`)**:
   - Ensure **zero parameters** with `ndim < 2`, `log_lambda`, `RMSNorm`, `LayerNorm` (`q_a_norm`, `kv_a_norm`), or `.bias` are in the `weight_decay > 0` group.
4. **Active vs. Total Parameter Accounting & VRAM Estimate**:
   - Compute $\Phi_{\text{total}}$, $\Phi_{\text{active}}$ (accounting for shared and routed MoE proportions), estimated FLOPs/token ($6 \Phi_{\text{active}}$), and estimated static-plus-activation memory footprint relative to device capacity.

### Gate 2: Numerical, Gradient, & Causal Integrity (`gate2_numerical.py`)
1. **Finite-Difference `torch.autograd.gradcheck` (`torch.float64`)**:
   - Test `FlashDecayFunction` (including extreme regimes $\lambda \in \{0.5, 0.9, 0.99999, 1.0 - 10^{-7}\}$), `RMSNorm`, `SwiGLU`, `KilatAttention`, `KilatAttentionRoPE`, and `DeepSeekMoE` (for $K=1$ and $K=2$) at micro dimensions ($B=2, S=6, D=16$) using `torch.autograd.gradcheck(..., eps=1e-6, atol=1e-5, rtol=1e-4)`:
     $$\max_{\theta} \frac{\left\|\nabla^{\text{analytic}}_\theta \mathcal{L} - \nabla^{\text{numeric}}_\theta \mathcal{L}\right\|_\infty}{\max\!\left(1, \left\|\nabla^{\text{numeric}}_\theta \mathcal{L}\right\|_\infty\right)} < 10^{-5}$$
2. **Causal Future-Blindness Perturbation / Jacobian Test**:
   - Run `logits_a = model(x)` in deterministic evaluation mode (`dropout=0`). Replace all future tokens $x_{:, t^*+1 : S}$ with substantially different random tokens $\tilde{x}$, then compute `logits_b = model(x_tilde)`.
   - The hard invariant for all past and current positions $t \le t^*$ (test $t^* \in \{0, S//2, S-2\}$) is:
     $$\max_{t \le t^*} \left\|\text{logits}_a[:, t, :] - \text{logits}_b[:, t, :]\right\|_\infty \le \epsilon_{\text{dtype}} \quad (= 0.0 \text{ for deterministic FP64 / CPU, } \le 10^{-6} \text{ for FP32})$$
   - Also test causal gradients: compute $\mathcal{L}_{\le t^*} = \text{logits}[:, t^*, :].sum()$, call `.backward()` with respect to input embeddings $E$, and ensure $\|(\nabla_E \mathcal{L}_{\le t^*})[:, t^* + 1:, :]\|_\infty == 0.0$.
3. **Batch-Independence Test**:
   - Corrupt samples `1..B-1` in the *batch* with extreme values; `logits[0]` must remain unchanged ($\|\Delta \text{logits}[0]\|_\infty \le 10^{-6}$). This catches `view` vs `transpose` bugs that mix the `B` and `S` dimensions, as well as MoE router-state leakage between samples.
4. **100% Gradient Connectivity for Active Parameters**:
   - Ensure every parameter with `requires_grad=True` has `param.grad is not None`, `torch.isfinite(param.grad).all()`, and $\|\nabla \theta\|_2 > 0$ (especially `attn.log_lambda`, `attn.v_proj_global.weight`, and `ffn.moe.router.weight`), while intentionally frozen parameters (`requires_grad=False`) have `param.grad is None`.

### Gate 3: Implementation Equivalence (`gate3_equivalence.py`)
Proves that optimized paths are mathematically identical to the simplest reference implementations in `torch.float64`:
1. **Triton Kernel / Fast Global Decay vs. Exact FP64 Causal Matrix**:
   - Compare *forward* and *backward* ($\nabla_V, \nabla_\lambda$) using these $L_\infty$ relative-error limits:
     - `FP32`: $E_{\text{fwd}} \le 10^{-5}$, $E_{\text{grad}} \le 5 \times 10^{-5}$
     - `FP16`: $E_{\text{fwd}} \le 5 \times 10^{-3}$, $E_{\text{grad}} \le 10^{-2}$
     - `BF16`: $E_{\text{fwd}} \le 2 \times 10^{-2}$, $E_{\text{grad}} \le 3 \times 10^{-2}$
2. **Full-Prefill vs Step-by-Step `HybridCache` & Chunked-Prefill Equivalence**:
   - Test the full matrix of combinations **`recall_ratio` $\in \{0.0, 0.5, 1.0\}$** $\times$ **`use_rope` $\in \{\text{False}, \text{True}\}$**:
     - Compare `logits_full[:, :S, :]` (one full *prefill* of length $S=12$) with:
       1. Incremental token-by-token generation ($1 + 1 + \dots + 1$).
       2. *Chunked prefill* ($5\text{ initial tokens} + 4\text{ middle tokens} + 3\text{ final tokens}$).
     - The maximum difference across all positions must be $\le 10^{-5}$ in FP32, directly guarding against regression of bug **M-05**.
3. **Vectorized Token-Sorted `DeepSeekMoE` vs. Naive Loop Reference**:
   - Compare layer outputs and `aux_loss` from the synchronization-free vectorized implementation against a simple loop over expert indices: difference $\le 10^{-6}$ in FP32. Also verify $\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}}$ exactly for constant/uniform router logits and $K=1, 2, 4$.

### Gate 4: Data Pipeline & Leakage Forensics (`gate4_data.py`)
1. **Index and Target-Mask Contract**:
   - Scan the first $N_{\text{audit}}$ *batches* from `train_dataloader` and `eval_dataloader`:
     - `input_ids.dtype == torch.long`, `labels.dtype == torch.long`, and identical `(B, S)` shapes.
     - All `input_ids` are in $[0, V-1]$; all `labels` are in $[0, V-1] \cup \{-100\}$.
     - Every position where `attention_mask == 0` has `labels == -100`.
     - Every *batch* contains at least one valid prediction token after causal shifting: `(labels[:, 1:] != -100).sum() > 0`.
2. **Train–Validation Decontamination ($n$-gram Hash Audit)**:
   - Build rolling 13-gram hashes (or full-sequence hashes for short datasets) for validation samples and ensure **zero exact leakage** from validation into the training split (except in `overfit_single_batch=True` debug mode).
3. **Memmap File Integrity**:
   - When using `np.memmap`, verify that the file size is divisible by `dtype.itemsize` and matches `.meta.json`.

### Gate 5: Micro-Training Dynamics & Frontier Stability Proxies (`gate5_dynamics.py`)
1. **Deterministic Single-Batch Memorization**:
   - Train a micro-model for $40$ steps on one fixed *batch*:
     - *Pure CE loss* must fall from $\approx \ln V$ to $< 0.05$ (Dense) or $< 0.10$ (MoE).
     - Normalized entropy of MoE expert assignments:
       $$\hat{H}_{\text{router}} = -\frac{1}{\ln N_r} \sum_{i=0}^{N_r - 1} f_i^{\text{prob}} \ln\!\left(f_i^{\text{prob}} + 10^{-12}\right) > 0.30$$
       (demonstrating that no immediate single-expert collapse occurs).
2. **DeepMind High-LR Instability Proxy (Wortsman et al., ICLR 2024)**:
   - Run a short 15-step sweep at learning-rate multipliers $\eta \in \{1\times, 3\times\}$ of the base LR (and use large FP16 activations $|x| = 300$ to test `RMSNorm` and `GradScaler`):
     - Record $\max |\text{attn\_logits}|$ and the weight-update ratio $\rho_W = \frac{\|\Delta W\|_2}{\|W\|_2}$.
     - Ensure there are no `NaN`/`Inf` values, `GradScaler` recovers cleanly if the first FP16 step overflows, and attention logits remain bounded ($< 80.0$).
     - Support the optional PaLM *z-loss* penalty (`z_loss_coef: float = 0.0`, e.g. $10^{-4} \log^2 Z$ in `KilatConfig`) to stabilize large-scale *pre-training*.

### Gate 6: Memory Leak, Checkpoint Bitwise Parity, & DDP Parity (`gate6_system.py`)
1. **Per-Step Memory-Leak Audit (*Zero Memory Creep*)**:
   - Run 10 consecutive training steps after 2 warmup steps. Ensure active tensor allocations (and `torch.cuda.memory_allocated()` on GPU) do not grow monotonically between steps 3 and 10 (catching bugs such as storing tensors with `grad_fn` in Python history lists).
2. **Bitwise Checkpoint Save & Resume Parity**:
   - **Path A**: Train steps $1 \dots 10$ uninterrupted $\to$ record $\mathcal{L}_{6..10}^{(A)}$ and final weights $\theta_{10}^{(A)}$.
   - **Path B**: Train steps $1 \dots 5$ $\to$ `save_checkpoint` $\to$ re-instantiate the model, optimizer, scheduler, scaler, and RNG from scratch $\to$ `load_checkpoint` $\to$ train steps $6 \dots 10$ $\to$ record $\mathcal{L}_{6..10}^{(B)}$ and $\theta_{10}^{(B)}$.
   - **Strict Pass Criteria**:
     $$\max_{t \in [6, 10]} \left|\mathcal{L}_t^{(A)} - \mathcal{L}_t^{(B)}\right| == 0.0 \quad \text{and} \quad \max_p \left\|\theta_{10, p}^{(A)} - \theta_{10, p}^{(B)}\right\|_\infty == 0.0$$
3. **Single-Process Accumulation vs. Multi-Rank DDP Gradient Parity**:
   - Using the `torch.distributed` `gloo` backend (2 local CPU processes) or `nccl` on multiple GPUs, one step with $B_{\text{global}} = 4$ (1 process, `grad_acc=2`, `micro_batch=2`) must produce parameter gradients and `eval_loss` matching 2-process DDP (`world_size=2`, `grad_acc=1`, `micro_batch=2`) within tolerance $< 10^{-5}$.

### Stages 7 & 8 (`kilat verify --full` / `assurance/scaling.py`): $\mu\text{P}$ Coordinate Check & WSD Scaling Harness
For full-scale *pre-training* readiness before renting a GPU cluster:
- **Stage 7a ($\mu\text{P}$ / Width Coordinate Check)**: Instantiate models at widths $d \in \{64, 128, 256\}$, run 5 steps, and verify that mean activation magnitude $\|h_\ell\|_1 / d$ remains scale-invariant ($\mathcal{O}(1)$) across widths.
- **Stage 7b (Mini Scaling Law with `WSDLR` Scheduler)**: Run micro-training on three model sizes ($N_1 < N_2 < N_3$) using the *Warmup-Stable-Decay* scheduler (`wsdlr`, already available in `scheduler.py`) to verify that validation pure PPL decreases monotonically with the FLOP budget ($\mathcal{L}(N_3) < \mathcal{L}(N_2) < \mathcal{L}(N_1)$).
- **Stage 8 (Final Pre-Training Readiness Sign-Off)**: Validate `MainConfig` completeness, availability of a tokenizer and real dataset, `vocab_size` alignment, integrity of the trial checkpoint's SHA-256 manifest, and sign `assurance_report.json`.

---

## 2. Layer 2 — ML Chaos Engineering / Active Saboteur (`kilat chaos`)

### 2.1 `Saboteur` Design Principle (100% Non-Destructive)
`Saboteur` (`src/kilat/assurance/chaos/saboteur.py`) is **STRICTLY FORBIDDEN** from editing `.py` files on disk. All fault injection must use `contextlib.contextmanager` to install temporary in-memory hooks (`register_forward_hook`, `register_full_backward_hook`, temporary callable wrappers on in-memory object instances, or disruptions in a `tmp_path` directory), then remove them in a `finally:` block.

### 2.2 Required Matrix of Eight Sabotage Scenarios (`C-01` through `C-08`)

| Code | Chaos Scenario | In-Memory Injection Mechanism (`with saboteur.*:`) | Gate / Component That Must Catch It | Expected Failure Code (`AssuranceFailureCode`) |
| :---: | :--- | :--- | :--- | :--- |
| **C-01** | **Silent Autograd Branch Detachment** | Install a *forward hook* on the *Global Decay* branch that returns `output.detach()` (reproducing the old CUDA `triton_ops.py` bug while MLA remains active). | Gate 2 (Gradient Connectivity & `gradcheck`) | `ERR_MISSING_PARAMETER_GRAD` (`log_lambda` & `v_proj_global`) |
| **C-02** | **Subtle Causal Future Leak** | Install a *forward hook* on the attention block that mixes $10^{-3} \times x_{t+1}$ into position $t$. | Gate 2 (Causal Future-Blindness Test) | `ERR_CAUSAL_FUTURE_LEAK` |
| **C-03** | **Cross-Batch Tensor Reshape Mix** | Temporarily replace feature flattening with `.transpose(0, 1).contiguous().view(...)` without restoring the order, mixing samples `b=0` and `b=1`. | Gate 2 (Batch-Isolation Test) | `ERR_BATCH_ISOLATION_BREACH` |
| **C-04** | **FP16 >256 Square Overflow & Inf Grad Storm** | Inject an activation spike $|x| = 512.0$ in `float16` (testing `RMSNorm` robustness) and inject `Inf` into one gradient during the first `GradScaler` step (ensuring `scheduler.step()` does not advance when `scaler.step` is skipped). | Gate 3 / Gate 5 & `FlightRecorder` | If `RMSNorm` is sabotaged to use FP16: `ERR_NON_FINITE_ACTIVATION`; in the `GradScaler` test: verify clean recovery without a premature LR shift. |
| **C-05** | **Decay Horizon Singularity ($\lambda \to 1^-$)** | Force `attn.log_lambda.data.fill_(25.0)` ($\lambda \approx 1 - 1.4 \times 10^{-11}$) to compare naive division against the stable $\varphi_1$ formula. | Gate 2 / Gate 3 | `ERR_DECAY_NORMALIZATION_DIVERGENCE` |
| **C-06** | **MoE Router Collapse & Top-1 Zero-Grad** | Force one router-column weight to $+50.0$ (all tokens route to Expert 0) **and** use a subtest that forcibly normalizes $K=1$ weights to `ones_like`. | Gate 2 & Gate 5 / `FlightRecorder` | `ERR_ROUTER_COLLAPSE` & `ERR_ZERO_PARAMETER_GRAD` |
| **C-07** | **Poisoned Batch (All `-100` & OOV Token)** | Inject one *micro-batch* with all `labels = -100` and one sample with `input_ids = vocab_size + 5`. | Gate 4 & **M-06** loss guard | `ERR_EMPTY_TARGET_BATCH` & `ERR_TOKEN_OUT_OF_BOUNDS` |
| **C-08** | **Mid-Checkpoint Crash & Corrupted State / Split-Brain** | Raise `SimulatedPreemptionError` midway through `save_checkpoint` after half the file has been written to `.tmp_checkpoint_*`, and flip one byte in `pytorch_model.bin` to test SHA-256 verification against `checkpoint_manifest.json`. | Gate 6 & `load_checkpoint` | The old checkpoint remains 100% intact, `.tmp_checkpoint_*` is cleaned up, and the corrupt file is rejected with `ERR_CHECKPOINT_CHECKSUM_MISMATCH`. |

**`kilat chaos` Pass Criteria**:
- `kilat chaos` runs a clean baseline first (which must have **zero errors**), then runs `C-01` through `C-08` one at a time.
- If **any** sabotage scenario is not detected (*false negative*), `kilat chaos` exits with a **failure status (`exit code 1`)**.

---

## 3. Layer 3 — Runtime `FlightRecorderCallback` & Circuit Breaker

Installed as a lightweight `TrainerCallback` (`src/kilat/assurance/flight_recorder.py`) during `kilat train`:

### 3.1 Metrics Recorded Without Per-Micro-Batch Synchronization
All metrics are accumulated as scalar tensors on-device and read to CPU (`.item()`) only when `should_log(step)` is `True`:
1. `train/ce_loss`, `train/aux_loss`, `train/ppl` ($\exp(\text{ce\_loss})$).
2. `optim/grad_norm` (global L2 gradient norm after `unscale_`, before *clipping*).
3. `optim/update_to_weight_ratio` ($\|\eta \cdot \text{update}\|_2 / \|W\|_2$, sampled every $N_{\text{log}}$ steps; healthy range $10^{-5} \dots 10^{-2}$).
4. `optim/scaler_scale` & `optim/skipped_steps_total` (number of steps skipped by `GradScaler`).
5. `moe/router_entropy_norm` ($\hat{H}_{\text{router}} \in [0, 1]$) & `moe/max_expert_load_ratio` ($\max_i f_i$; sehat $\approx 1.0$, bahaya $> 4.0$).
6. `attn/lambda_min`, `attn/lambda_max` (monitor the decay horizon $\sigma(\text{log\_lambda})$ across all layers).
7. `sys/tokens_per_sec`, `sys/peak_vram_mb`.

### 3.2 Automatic Circuit-Breaker Rules
If any of the following emergency conditions occurs:
- `ce_loss` is `NaN` or `Inf`;
- `skipped_steps_consecutive >= 10` (10 consecutive steps with `Inf` gradients under FP16 AMP);
- `ce_loss > 4.0 * rolling_median_loss` for five consecutive steps after *warmup* is complete;

Then `FlightRecorderCallback`:
1. Must **NOT** overwrite a healthy `checkpoint-best` or `checkpoint-latest`.
2. Saves an isolated diagnostic bundle to `<output_dir>/forensics/crash_step_<N>.pt` (containing the triggering `input_ids`/`labels` *batch*, RNG state, per-layer gradient-norm summaries, and configuration).
3. Stops training cleanly with an informative `TrainingDivergedError` and specific remediation recommendations.

---

## 4. Readiness Certificate Schema (`assurance_report.json`)

Each `kilat verify` run writes a structured JSON file and a Markdown summary that can be attached to a PR or experiment repository:

```json
{
  "schema_version": "1.0",
  "kilat_version": "2.0.0",
  "timestamp_utc": "2026-11-09T00:00:00Z",
  "git_commit": "<sha>",
  "config_sha256": "<sha256>",
  "environment": {
    "python": "3.14",
    "torch": "2.x",
    "device": "cpu | cuda",
    "has_triton": true
  },
  "overall_status": "PASS",
  "gates": [
    {
      "gate_id": "G1_THEORY_INIT",
      "status": "PASS",
      "duration_ms": 42.5,
      "metrics": {
        "initial_ce_loss": 4.848,
        "expected_ln_vocab": 4.852,
        "residual_std_ratio_L_over_1": 1.08,
        "leaked_decay_1d_params": 0
      }
    }
  ],
  "chaos_verification": {
    "executed": true,
    "scenarios_total": 8,
    "scenarios_caught": 8,
    "false_negatives": 0
  }
}
```
