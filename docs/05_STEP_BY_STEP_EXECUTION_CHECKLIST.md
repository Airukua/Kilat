# 05 — Verification Protocol & Release Readiness Checklist (`v1.0.0`)

> **Document Purpose**: Atomic engineering checklist and mandatory numerical acceptance gates for every modification, PR, or pre-training launch in Kilat.

---

## 1. Mandatory Pre-Merge & Pre-Training Acceptance Gates

Every code or configuration change in Kilat must pass all 5 gates sequentially before committing or launching a multi-day training run:

| Gate | Command | Mandatory Acceptance Criterion |
| :---: | :--- | :--- |
| **1. Static Lint & Type Safety** | `make check` | `ruff check src` $\to$ `0` errors · `mypy src` $\to$ `0` errors across all canonical modules. |
| **2. Numerical & BDD Test Suite** | `make test` | 100% pass across all 387 unit, seam, pipeline, assurance, and chaos BDD scenarios, **`0` warnings**, total coverage $\ge 80\%$. |
| **3. 8-Stage Pre-Training Assurance** | `kilat verify --config configs/debug_cpu.yaml --full` | Overall status **`PASS`** across `G1_THEORY_INIT`, `G2_NUMERICAL_CAUSAL`, `G3_EQUIVALENCE`, `G4_DATA_FORENSICS`, `G5_MICRO_DYNAMICS`, `G6_SYSTEM_PARITY`, `S7_SCALING_LAWS`, and `S8_READINESS_SIGNOFF`. |
| **4. Active ML Chaos Saboteur** | `kilat chaos --scenarios all` | Baseline clean pass (`True`), **8/8 (100%)** silent ML faults (`C-01`..`C-08`) caught with exact `AssuranceFailureCode`, post-chaos restore pass (`True`). |
| **5. Repository & Hook Hygiene** | `make pre-commit` | 100% pass on `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`, `ruff`, and `mypy`. |

---

## 2. Completed Master Architecture & Assurance Milestones (`v1.0.0`)

- [x] **Phase 0 — Baseline Audit & Anti-Hallucination Lock**: Verified baseline test suite, configuration contracts, and tensor invariants.
- [x] **Phase 1 — Mathematical Surgery (`M-01` .. `M-07`)**:
  - [x] `M-01`: Exact $O(N)$ `FlashDecayFunction` forward/backward autograd + $\varphi_1(w)$ $\lambda \to 1^-$ stabilization (`src/kilat/models/kernels/flash_decay.py`).
  - [x] `M-02` & `M-03`: FP32-upcast `RMSNorm` (`src/kilat/models/blocks.py`) and strict `ndim < 2` / norm `weight_decay=0.0` partitioning (`src/kilat/engine/optimizer.py`).
  - [x] `M-04` & `M-07`: $1/K_{\text{eff}}$ load-balancing normalization, Top-1 router gradient preservation, FP32 router logits, `num_shared_experts >= 0`, `bias_free` DeepSeek-V3 routing, and zero-sync forward passes (`src/kilat/models/ffn.py`).
  - [x] `M-05` & `M-06`: `LayerHybridCache` (`recall_ratio=0.0`, chunked prefill, post-RoPE KV caching), pure token-weighted NLL perplexity (`ce_loss` separated from `aux_loss`), partial accumulation window scaling, `ceil` step accounting, and $1/\sqrt{2L}$ residual initialization (`src/kilat/models/attention.py`, `src/kilat/models/transformer.py`, `src/kilat/engine/trainer.py`).
- [x] **Phase 2 — 8-Stage Pre-Training Assurance (`src/kilat/assurance/`)**: Implemented Gates 1–6, Stages 7–8 (`scaling.py`), `FlightRecorderCallback`, and SHA-256 signed `assurance_report.{json,md}`.
- [x] **Phase 3 — Active ML Chaos Engineering (`src/kilat/assurance/chaos/`)**: Implemented non-destructive `Saboteur` (`C-01`..`C-08`) with 100% meta-detection and clean state restoration.
- [x] **Phase 4 — Unified Config, Zero-Copy Data, DDP, & Atomic Checkpoints**: Connected full `KilatConfig` across blocks, added `MainConfig.apply_overrides`, zero-copy `KilatDataCollator`, `SyntheticTokenDataset`, `DistributedContext`, and atomic Rank-0 checkpoints with full RNG state + `checkpoint_manifest.json` SHA-256 verification (`0.0` bitwise resumption parity).
- [x] **Phase 5 — Clean Modular Codebase Consolidation**: Consolidated all source code into `src/kilat/{models,engine,data,inference,distillation,assurance,configs,utils}` and retired all obsolete legacy directories (`arc/`, `training/`, `pipeline/`, `distiliation/`, `sample_config/`, `experiments/`).
- [x] **Phase 6 — Unified `kilat` CLI & Tiered Presets (`configs/`)**: Shipped `kilat {verify,chaos,prepare-data,train,generate,export-hf}` and documented YAML presets (`debug_cpu.yaml`, `small_dense.yaml`, `small_moe.yaml`, `standard_moe.yaml`).
