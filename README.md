<div align="center">

# ⚡ Kilat

**Kernelized Lightweight Attention Transformer — Pre-Training, Assurance, & Inference Toolkit**

*Built for researchers and ML engineers who care about mathematical rigor under the hood.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Assurance: 8-Stage Verified](https://img.shields.io/badge/assurance-8--stage%20verified-success?style=flat-square)](docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)

<p align="center">
  <img src="images/illustration.png" alt="Kilat illustration" width="720" />
</p>

</div>

---

**Kilat** (*Indonesian: lightning*) is a modular PyTorch toolkit for pre-training, mathematically verifying, distilling, and deploying language models from scratch — spanning dense SwiGLU baselines to fine-grained **DeepSeekMoE** architectures with **Hybrid Sub-Quadratic Global Decay + Multi-Head Latent Attention (MLA)** and **Decoupled RoPE**.

In LLM pre-training, the worst bugs rarely raise an exception: detached autograd branches, FP16 variance overflows, weight decay eroding 1D decay horizons, or MoE router collapse can silently ruin weeks of GPU compute. Kilat pairs a clean, hackable architecture with a built-in **8-Stage Pre-Training Assurance Pipeline (`kilat verify`)**, **Active ML Chaos Saboteur (`kilat chaos`)**, and **Runtime `FlightRecorder` Circuit Breaker**.

```python
from kilat import KilatConfig, KilatTransformer

model = KilatTransformer(
    KilatConfig(vocab_size=50_257, n_embd=768, n_layer=12, n_head=12, ffn_mode="moe", num_experts=8, active_experts=2)
)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M total parameters")
```

---

## Key Capabilities

- **Hybrid Sub-Quadratic + Latent Attention (`kilat.models`)**: Splits heads via `recall_ratio` between $O(N)$ **Global Exponential Decay** (exact analytical forward/reverse-scan autograd + Triton kernel, $O(1)$ decode state) and **Low-Rank Latent MLA** ($4\text{–}8\times$ smaller KV-cache) with **Decoupled RoPE**.
- **Fine-Grained DeepSeekMoE**: Shared + routed SwiGLU experts supporting both differentiable $1/K_{\text{eff}}$ load-balancing loss (`aux_loss`) and **DeepSeek-V3 loss-free dynamic bias balancing (`bias_free`)** in FP32 without host-device stream syncs.
- **Research-Grade Assurance (`kilat.assurance`)**:
  - `kilat verify`: 6 pre-flight gates (Theory/Init, FP64 `gradcheck` & Causal Blindness, Reference Equivalence, Data & Leakage Forensics, Single-Batch Memorization, and `0.0` Bitwise Checkpoint/DDP Parity) + Stages 7–8 ($\mu\text{P}$ coordinate check & WSDLR scaling law harness).
  - `kilat chaos`: Non-destructive `Saboteur` injecting 8 classic silent ML bugs (`C-01`..`C-08`) in memory to prove 100% gate detection with zero false negatives.
  - `FlightRecorderCallback`: Asynchronous training telemetry (`telemetry.jsonl`) and automatic circuit breaker with forensic crash dumps before corrupting healthy checkpoints.
- **Production Pre-Training Engine (`kilat.engine`)**: Mixed precision (`bf16` / `fp16` with `GradScaler`), Warmup-Stable-Decay (`wsdlr`) & cosine schedules, DDP collective evaluation (`all_reduce`), and atomic Rank-0 checkpoints with full RNG state (`python`, `numpy`, `torch`, `cuda`) + SHA-256 manifest verification.
- **Zero-Copy Data, Inference, & Distillation**: Memory-mapped `.bin`/`.npy` ingestion (`kilat prepare-data`), zero-copy `KilatDataCollator`, `LayerHybridCache` incremental/chunked decoding, Hugging Face `safetensors` export (`kilat export-hf`), and Knowledge Distillation (`VanillaKDLoss`, `ReverseKDLoss`, `AdaptiveKDLoss`).

---

## Installation

```bash
# Standard installation
pip install git+https://github.com/Airukua/kilat.git

# Editable developer installation (ruff, mypy, pytest, pre-commit)
git clone https://github.com/Airukua/kilat.git
cd kilat
pip install -e ".[dev]"

# Optional experiment tracking backends (wandb, tensorboard, mlflow, comet_ml)
pip install -e ".[reporting]"
```

---

## 30-Second CLI Quickstart

Kilat installs a unified `kilat` CLI entrypoint. You can verify the entire stack and run a complete CPU training + generation cycle immediately after cloning — **no external dataset or Hub download required**:

```bash
# 1. Run all 8 Pre-Training Assurance stages (emits signed assurance_report.{json,md})
kilat verify --config configs/debug_cpu.yaml --full --output-dir ./verify_report

# 2. Run Active ML Chaos Engineering suite (verifies 8/8 silent ML faults are caught)
kilat chaos --scenarios all

# 3. Train on the offline synthetic config (with optional dot-list overrides)
kilat train --config configs/debug_cpu.yaml \
  --override training.max_steps=20 \
  --override training.output_dir=./checkpoints/quickstart

# 4. Generate text from the saved checkpoint
kilat generate \
  --checkpoint ./checkpoints/quickstart/checkpoint-best \
  --prompt "Kilat" \
  --max-new-tokens 32

# 5. Export checkpoint to Hugging Face format (safetensors / pytorch_model.bin)
kilat export-hf \
  --checkpoint ./checkpoints/quickstart/checkpoint-best \
  --output ./hf_exported_model
```

### Pre-Training on Custom Corpora & Multi-GPU DDP

```bash
# Tokenize Parquet, JSONL, or TXT into a memory-mapped array + SHA-256 .meta.json
kilat prepare-data \
  --input ./raw_corpus.jsonl \
  --output ./data/train.npy \
  --tokenizer gpt2 \
  --dtype int32

# Single-GPU training (Dense SwiGLU or DeepSeekMoE)
kilat train --config configs/small_moe.yaml

# Multi-GPU Distributed Data Parallel (DDP)
torchrun --nproc_per_node=2 -m kilat.cli train --config configs/small_moe.yaml
```

---

## Python API Quickstart

### 1. Config-Driven Pre-Training

```python
from kilat import (
    KilatTrainer,
    KilatTransformer,
    MainConfig,
    build_dataloaders_from_config,
)

config = MainConfig.from_yaml("configs/debug_cpu.yaml")
config.apply_overrides(["training.max_steps=20", "model.router_balance_mode=bias_free"])

model = KilatTransformer(config)
tokenizer = config.build_tokenizer()
train_loader, eval_loader = build_dataloaders_from_config(config)

trainer = KilatTrainer(
    model=model,
    args=config.to_training_arguments(),
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    tokenizer_config=config.tokenizer,
)
trainer.train()
```

### 2. Text Generation with `LayerHybridCache`

```python
from kilat import GenerationConfig, KilatTransformer, TextGenerator
from kilat.data import AutoTokenizer

model = KilatTransformer.from_pretrained("./checkpoints/quickstart/checkpoint-best")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/quickstart/checkpoint-best")

generator = TextGenerator(model, tokenizer)
gen_cfg = GenerationConfig(max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95)
print(generator.generate("Once upon a time", generation_config=gen_cfg))
```

### 3. Knowledge Distillation

```python
from kilat.distillation import AdaptiveKDLoss, BaseStudent, BaseTeacher

kd_criterion = AdaptiveKDLoss(temperature=2.0, alpha=0.5)
# Combine soft teacher-student KL divergence with ground-truth cross-entropy
loss_out = kd_criterion(student_logits, teacher_logits, labels=labels)
```

---

## Configuration & Hardware Presets (`configs/`)

Kilat uses a single validated YAML file (`MainConfig`) as its source of truth across `experiment`, `model`, `tokenizer`, `dataloader`, and `training`. Ready-to-run presets live in [`configs/`](configs/README.md):

| Config File | Target Hardware | Architecture Summary | Data Format |
| :--- | :--- | :--- | :--- |
| [`configs/debug_cpu.yaml`](configs/debug_cpu.yaml) | Any CPU / Laptop (`< 5s`) | 2L Hybrid MLA/Decay + 4-expert MoE (`d=64`) | `synthetic` (offline byte tokenizer) |
| [`configs/small_dense.yaml`](configs/small_dense.yaml) | 1x GPU (`8–16 GB VRAM`) | 12L Dense SwiGLU + Hybrid MLA/Decay + RoPE (`d=768`) | `memmap` (`.npy` / `.bin`) |
| [`configs/small_moe.yaml`](configs/small_moe.yaml) | 1x GPU (`8–16 GB VRAM`) | 10L DeepSeekMoE (8 experts, Top-2, 1 shared, `bias_free`) | `memmap` (`.npy` / `.bin`) |
| [`configs/standard_moe.yaml`](configs/standard_moe.yaml) | Multi-Node FSDP / EP (`BF16`) | 24L DeepSeekMoE (64 experts, Top-8, 2 shared, 54.2B params, FSDP recipe target) | `memmap` + sequence packing |

To resume interrupted training with **exact bitwise state & RNG parity**, set:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/small_moe/checkpoint-latest"
```

---

## Architecture Overview

### Hybrid Global Decay + Latent MLA Attention

Each transformer block splits attention heads into two parallel paths controlled by `recall_ratio` (`0.0` = pure $O(N)$ Global Decay, `1.0` = pure Latent MLA, `0.5` = balanced hybrid) and merges them through a learned context gate:

```text
                     Input x  [B, N, D]
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
  ╔═══════════════════╗           ╔════════════════════╗
  ║   PATH 1          ║           ║   PATH 2           ║
  ║   Global Decay    ║           ║   Latent MLA       ║
  ║   (linear, O(N))  ║           ║   + Decoupled RoPE ║
  ╚═══════════════════╝           ╚════════════════════╝
          │                                │
          ▼                                ▼
   ┌─────────────┐                 ┌──────────────────┐
   │  v_proj     │                 │  q_a → LN → q_b  │
   │  λ = σ(α_h) │                 │  kv_a → LN → kv_b│
   └──────┬──────┘                 └────────┬─────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────────────┐        ┌─────────────────────┐
   │ FlashDecayFunction   │        │ Low-rank KV cache   │
   │ Full seq: O(N) scan  │        │ [B, S, D_latent]    │
   │ Decode  : O(1) state │        │ + Post-RoPE key     │
   │ φ₁(w) λ→1⁻ guard     │        │ Causal SDPA         │
   └──────────┬───────────┘        └──────────┬──────────┘
              │  out_global                   │  out_recall
              └──────────────┬────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  cat + learned gate │
                  │  c_proj (1/√2L init)│
                  └──────────┬──────────┘
                             ▼
                     output [B, N, D]
                  + LayerHybridCache
```

### Feed-Forward & MoE Routing Modes

| `ffn_mode` | `router_balance_mode` | Description |
| :--- | :--- | :--- |
| `dense` | — | Standard two-layer GELU feed-forward network |
| `swiglu` | — | Gated SwiGLU feed-forward network (`ff_mult = 8/3`) |
| `moe` | `aux_loss` | Fine-grained `DeepSeekMoE` with shared + routed experts and $1/K_{\text{eff}}$-normalized auxiliary balance loss |
| `moe` | `bias_free` | **DeepSeek-V3 style** loss-free dynamic expert bias balancing (zero auxiliary gradient interference on LM perplexity) |

### Supported Generation Strategies

| Strategy | Key `GenerationConfig` Parameters |
| :--- | :--- |
| Greedy | `do_sample=False` |
| Multinomial / Temp | `do_sample=True, temperature=0.8` |
| Top-K / Top-P (Nucleus) | `do_sample=True, top_k=50, top_p=0.95` |
| Typical / Mirostat | `sampling_strategy="typical"` or `"mirostat"` |
| Contrastive Search | `sampling_strategy="contrastive", contrastive_penalty=0.5` |
| Beam Search | `num_beams=4, length_penalty=1.0` |

---

## Pre-Training Assurance & Active Chaos Suite

Before spending GPU hours on a large run, `kilat verify` executes an 8-stage research validation pipeline and signs `assurance_report.{json,md}`:

| Stage | Gate ID | What It Proves Numerically |
| :---: | :--- | :--- |
| **1** | `G1_THEORY_INIT` | Unbiased init entropy $|\mathcal{L}_0 - \ln V| \le 0.15$, bounded residual variance across depth, and `0` 1D/norm parameters leaked into `weight_decay`. |
| **2** | `G2_NUMERICAL_CAUSAL` | FP64 finite-difference `gradcheck` across `FlashDecayFunction`, `RMSNorm`, `SwiGLU`, `MLA`, and `DeepSeekMoE`; strict causal future-blindness; batch isolation; 100% non-zero finite gradients. |
| **3** | `G3_EQUIVALENCE` | Triton/fast Global Decay vs exact FP64 causal matrix; full prefill vs 1-by-1 & chunked `LayerHybridCache` across `recall_ratio` $\in \{0.0, 0.5, 1.0\}$; vectorized MoE vs reference loop. |
| **4** | `G4_DATA_FORENSICS` | Token bounds $[0, V-1]$, pad mask alignment (`labels == -100`), non-empty causal targets, memmap `.meta.json` checksum, and 13-gram train/val decontamination. |
| **5** | `G5_MICRO_DYNAMICS` | Deterministic single-batch memorization ($\mathcal{L}_{\text{CE}} \to 0$), MoE router normalized entropy $> 0.30$, and FP16 $|x|=512$ `RMSNorm` + `GradScaler` stability. |
| **6** | `G6_SYSTEM_PARITY` | Zero per-step tensor memory creep, **`0.0` bitwise parameter & loss parity** across mid-run checkpoint save/resume, and gradient accumulation vs sharded parity. |
| **7–8** | `S7_SCALING_LAWS` & `S8` | (`--full`) $\mu\text{P}$ width coordinate invariance, monotonic WSDLR scaling law verification, and SHA-256 readiness certificate sign-off. |

---

## Repository Structure & Documentation

```text
Kilat/
├── configs/                 # Tiered YAML configs (debug_cpu, small_dense, small_moe, standard_moe)
├── docs/                    # Deep-dive architecture, math derivations, & assurance specs
├── src/kilat/
│   ├── cli.py               # Unified `kilat` CLI entrypoint
│   ├── configs/             # MainConfig, KilatConfig, TrainingConfig, DataLoaderConfig, TokenizerConfig
│   ├── models/              # KilatTransformer, Block, RMSNorm, MLA/Decay Attention, DeepSeekMoE, RoPE, Triton
│   ├── engine/              # KilatTrainer, TrainingArguments, DistributedContext, Checkpoint, Optimizer, Scheduler
│   ├── data/                # Datasets, zero-copy KilatDataCollator, DataLoaders, Tokenizers, Memmap Preprocessor
│   ├── inference/           # TextGenerator, GenerationMixin, Samplers, LogitProcessors, StoppingCriteria, HF Export
│   ├── distillation/        # VanillaKDLoss, ReverseKDLoss, AdaptiveKDLoss, BaseStudent, BaseTeacher
│   ├── assurance/           # Gates 1–6, Stages 7–8, Chaos Saboteur (C-01..C-08), FlightRecorderCallback
│   └── utils/               # BaseConfig, BaseModel, validators, parameter breakdown reports
└── tests/                   # 387 BDD unit, seam, pipeline, assurance, and chaos tests
```

Detailed engineering blueprints are available in [`docs/`](docs/README.md):
- [**`docs/README.md`**](docs/README.md) — Architecture & Documentation Index
- [**`docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md`**](docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md) — Module Inventory & API Contracts
- [**`docs/02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md`**](docs/02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md) — Analytical Derivations & Precision Invariants (`M-01`..`M-07`)
- [**`docs/03_CODEBASE_RESTRUCTURE_PLAN.md`**](docs/03_CODEBASE_RESTRUCTURE_PLAN.md) — Package Design, Config Overrides, DDP, & CLI Guide
- [**`docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md`**](docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md) — 8-Stage Verification, `Saboteur`, & `FlightRecorder`
- [**`docs/05_STEP_BY_STEP_EXECUTION_CHECKLIST.md`**](docs/05_STEP_BY_STEP_EXECUTION_CHECKLIST.md) — Release & Pre-Training Acceptance Gates

---

## Development & Verification Commands

```bash
make check        # Ruff lint + Mypy static type check (0 errors)
make test         # Run full 387-test BDD suite (0 warnings, >= 80% coverage)
make verify       # Run Pre-Training Assurance Gates on configs/debug_cpu.yaml
make chaos        # Run Active Chaos Saboteur suite (C-01..C-08)
make pre-commit   # Run all git pre-commit hooks across the repository
```

---

## Roadmap

- [x] Dense, SwiGLU, and Fine-Grained DeepSeekMoE (`aux_loss` & `bias_free`) architectures
- [x] Hybrid attention ($O(N)$ Global Decay with exact autograd + Latent MLA + Decoupled RoPE)
- [x] `LayerHybridCache` incremental & chunked-prefill KV-cache generation
- [x] 8-Stage Pre-Training Assurance (`kilat verify`), Active Chaos `Saboteur` (`kilat chaos`), & `FlightRecorder`
- [x] DDP-aware engine, atomic SHA-256 checkpoints (`0.0` bitwise resume parity), & `WSDLR` scheduler
- [x] Knowledge Distillation (`VanillaKDLoss`, `ReverseKDLoss`, `AdaptiveKDLoss`) & Hugging Face export
- [ ] FSDP2 / Tensor Parallel multi-node sharding
- [ ] Custom CUDA / Triton paged MLA decode kernels
- [ ] ONNX / TorchScript runtime export

---

## Citation

```bibtex
@software{kilat2026,
  author = {Abdul Wahid Rukua},
  title  = {Kilat: Kernelized Lightweight Attention Transformer},
  year   = {2026},
  url    = {https://github.com/Airukua/kilat}
}
```

---

## Author

**Abdul Wahid Rukua** — AI/ML Engineer & Researcher

Focus: efficient sequence modeling, Indonesian NLP, low-resource language technology

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdul--wahid--rukua-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdul-wahid-rukua/)

---

*Licensed under MIT. See [LICENSE](LICENSE) for details.*
