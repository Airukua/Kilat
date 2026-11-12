<div align="center">

# ⚡ Kilat

**Kernelized Lightweight Attention Transformer — Pre-Training, Assurance, & Inference Toolkit**

*Built for researchers and ML engineers who care about mathematical rigor under the hood.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Assurance: 8-Stage Verified](https://img.shields.io/badge/assurance-8--stage%20verified-success?style=flat-square)](docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)

<p align="center">
  <img src="images/illustration.png" alt="Kilat illustration" width="720" />
</p>

</div>

---

**Kilat** (*Indonesian: lightning*) is a modular PyTorch toolkit for pre-training, mathematically certifying, distilling, and deploying sub-quadratic language models from scratch — spanning dense SwiGLU baselines to fine-grained **DeepSeekMoE** architectures with flagship **Macro-Synergy (Interleaved Global Decay + Latent MLA)** and **Decoupled RoPE**.

In LLM pre-training, the worst bugs rarely raise an exception: detached autograd branches, FP16 variance overflows, weight decay eroding 1D decay horizons, or MoE router collapse can silently ruin weeks of GPU compute. Kilat pairs high-throughput sub-quadratic architectures with a built-in **5-Pillar Model Certification Pipeline (`kilat certify`)**, an **8-Stage Pre-Training Assurance Pipeline (`kilat verify`)**, an **Active ML Chaos Saboteur (`kilat chaos`)**, and a **Runtime `FlightRecorder` Circuit Breaker**.

```python
from kilat import KilatConfig, KilatTransformer

model = KilatTransformer(
    KilatConfig(vocab_size=50_257, n_embd=768, n_layer=12, n_head=12, ffn_mode="moe", num_experts=8, active_experts=2)
)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M total parameters")
```

---

## Key Capabilities

- **Flagship Macro-Synergy Architecture (`kilat.models`)**: Interleaved layer stacking alternating between $O(N)$ **Global Exponential Decay Contextualizer** (even layers: zero KV-cache, Triton parallel prefix scan) and **Latent MLA Associative Recall** with Decoupled RoPE (odd layers: $100\%$ capacity dedicated to memory retrieval). Slashes KV-cache by **$6.4\times$** vs standard Transformers (**$2.0\times$** vs pure MLA) and boosts MQAR recall from $15.6\%$ to **$96.1\%$** ($+80.5\%$ absolute gain). Also supports intra-layer split via `recall_ratio`.
- **5-Pillar Model Certification Pipeline (`kilat certify`)**: Unified KMCP suite validating mechanical correctness (causal blindness, FP64 gradcheck), mechanistic capability (Stanford Zoology MQAR, leak-free induction heads, delimited copying), empirical scaling laws, hardware efficiency (prefill tok/s, decode latency, peak VRAM), and diagnostic forensics.
- **Fine-Grained DeepSeekMoE**: Shared + routed SwiGLU experts supporting both differentiable $1/K_{\text{eff}}$ load-balancing loss (`aux_loss`) and **DeepSeek-V3 loss-free dynamic bias balancing (`bias_free`)** in FP32 without host-device stream syncs.
- **Research-Grade Assurance (`kilat.assurance`)**:
  - `kilat verify`: 6 pre-flight gates (Theory/Init, FP64 `gradcheck` & Causal Blindness, Reference Equivalence, Data & Leakage Forensics, Single-Batch Memorization, and `0.0` Bitwise Checkpoint/DDP Parity) + Stages 7–8 ($\mu\text{P}$ coordinate check & WSDLR scaling law harness).
  - `kilat chaos`: Non-destructive `Saboteur` injecting 8 classic silent ML bugs (`C-01`..`C-08`) in memory to prove 100% gate detection with zero false negatives.
  - `FlightRecorderCallback`: Asynchronous training telemetry (`telemetry.jsonl`) and automatic circuit breaker with forensic crash dumps before corrupting healthy checkpoints.
- **Production Pre-Training Engine (`kilat.engine`)**: Mixed precision (`bf16` / `fp16` with `GradScaler`), JIT Triton kernel fusion (`torch.compile` with $>10\times$ speedup), experiment tracking (`wandb`, `tensorboard`, `mlflow`, `comet_ml`), Warmup-Stable-Decay (`wsdlr`) & cosine schedules, DDP collective evaluation (`all_reduce`), and atomic Rank-0 checkpoints with full RNG state (`python`, `numpy`, `torch`, `cuda`) + SHA-256 manifest verification.
- **Zero-Copy Data, Inference, & Distillation**: Memory-mapped `.bin`/`.npy` ingestion with `--append-eos` boundary isolation (`kilat prepare-data`), zero-copy `KilatDataCollator`, `LayerHybridCache` incremental/chunked decoding, Hugging Face `safetensors` export (`kilat export-hf`), and Knowledge Distillation (`VanillaKDLoss`, `ReverseKDLoss`, `AdaptiveKDLoss`).
- **Interpretability & Behavioral Benchmarking (`tools/`)**: Mechanistic interpretability (Layer Logit Lens, Activation Patching), data leak & N-gram contamination auditor (`tools/audit`), and counterfactual object tracking evaluation harness (`tools/benchmarking`).

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

# 3. Certify Model Architecture (5-Pillar KMCP: MQAR capability, scaling, & efficiency)
kilat certify --models kilat-macro-synergy --pillar all --output-dir ./certify_report

# 4. Train on the offline synthetic config (with optional dot-list overrides)
kilat train --config configs/debug_cpu.yaml \
  --override training.max_steps=20 \
  --override training.output_dir=./checkpoints/quickstart

# 5. Generate text from the saved checkpoint
kilat generate \
  --checkpoint ./checkpoints/quickstart/checkpoint-best \
  --prompt "Kilat" \
  --max-new-tokens 32

# 6. Export checkpoint to Hugging Face format (safetensors / pytorch_model.bin)
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
| [`configs/certification/smoke_test.yaml`](configs/certification/smoke_test.yaml) | Any CPU / Laptop (`< 10s`) | KMCP 5-Pillar Smoke Suite for fast CI/CD validation | `synthetic` in-memory |
| [`configs/certification/frontier_full.yaml`](configs/certification/frontier_full.yaml) | Single / Multi-GPU | KMCP Full Frontier Certification Suite (MQAR, scaling, latency) | `synthetic` + capability sweeps |

To resume interrupted training with **exact bitwise state & RNG parity**, set:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/small_moe/checkpoint-latest"
```

---

## Architecture Overview

### Flagship: Kilat Macro-Synergy (Interleaved Stacking)

Rather than forcing Global Decay and Latent MLA to compete within the same static linear projection, **Macro-Synergy** decouples them into an alternating layer hierarchy:

```text
Layer 0 (Even) : 100% Pure Global Decay Contextualizer
                 ├── 4-head Triton parallel prefix scan (O(N) prefill)
                 ├── Recurrent state update: s_t = λ s_{t-1} + v_t (O(1) decode)
                 ├── KV Cache = EXACTLY 0 BYTES
                 └── Emits enriched causal temporal manifold for upstream layers

Layer 1 (Odd)  : 100% Pure Latent MLA Associative Pointer
                 ├── Low-rank joint KV compression: c^{KV} = W_{DKV} x + Decoupled RoPE
                 ├── Uses enriched context from Layer 0 to guide precise memory lookup
                 └── Dedicates 100% of capacity to exact associative recall
```

#### Empirical Evidence & Capability Certification

Empirical benchmarks run on CUDA hardware demonstrate decisive memory, throughput, and retrieval advantages:

| Evaluation Metric | Standard Transformer | Kilat Concat (50/50) | **Kilat Macro-Synergy** | Relative Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **KV Cache ($N=2048$)** | $1,024\text{ KB}$ | $320\text{ KB}$ | **$160\text{ KB}$** | **$6.4\times$ smaller** vs Ref, **$2.0\times$** vs Base |
| **KV Cache ($N=8192$)** | $4,096\text{ KB}$ | $1,280\text{ KB}$ | **$640\text{ KB}$** | **$6.4\times$ smaller** vs Ref, **$2.0\times$** vs Base |
| **Prefill Throughput ($N=512$)** | $269,000\text{ tok/s}$ | $266,162\text{ tok/s}$ | **$423,808\text{ tok/s}$** | **$+59\%$ faster** prefill throughput |
| **MQAR Accuracy ($N=256$)** | Stagnates | $15.62\%$ | **$96.09\%$** | 🏆 **$+80.47\%$ absolute gain** |
| **Peak VRAM ($N=4096$)** | OOM | Pass | **Pass (No OOM)** | Zero OOM at long context |

*(For backward-compatibility, Kilat also supports single-layer intra-layer splitting via `recall_ratio`, merging paths via learned context gate $W_O^{(g)} y^{(g)} + W_O^{(r)} y^{(r)}$).*

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

---

## Model Architecture Certification (KMCP: `kilat certify`)

In addition to low-level assurance gates, Kilat features the **Kilat Model Architecture Certification Pipeline (KMCP)** (`kilat certify`), standardizing empirical architectural validation across 5 pillars aligned with frontier research (Stanford Zoology ICLR 2024, Meta Lingua, DeepSeek-V3):

```bash
# Certify flagship architecture across all 5 pillars
kilat certify --models kilat-macro-synergy --pillar all --output-dir verify_signoff
```

| Pillar | Name | Scope & Verification Standards |
| :---: | :--- | :--- |
| **P1** | **Mechanical Correctness** | Future-blindness masking invariants, FP64 analytical `gradcheck`, and single-batch memorization ($\mathcal{L}_{\text{CE}} \to 0$). |
| **P2** | **Mechanistic Capability** | Synthetic capability suite: Stanford Zoology Multi-Query Associative Recall (MQAR), leak-free induction heads, and delimited selective copying. |
| **P3** | **Empirical Scaling** | Compute-controlled token budget curves, non-vacuous Markov-chain transition entropy, and scaling exponent evaluation ($L(C) = A C^{-\alpha} + E$). |
| **P4** | **Hardware Efficiency** | Micro-benchmarking prefill throughput (tokens/sec), incremental decode latency (ms/token), and peak VRAM allocation across sequence lengths ($N \in \{512..8192\}$). |
| **P5** | **Forensic Diagnostics** | Position-wise cross-entropy loss tracking, layer-wise logit lens projection, and token decontamination. |

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
├── configs/                 # Tiered YAML configs (debug_cpu, small_dense, small_moe, standard_moe, certification/)
├── docs/                    # Deep-dive architecture, math derivations, & assurance specs
├── experiments/             # Pre-training data artifacts & benchmark recipes
├── src/kilat/
│   ├── cli.py               # Unified `kilat` CLI entrypoint (verify, chaos, certify, train, prep, generate, export)
│   ├── configs/             # MainConfig, KilatConfig, TrainingConfig, DataLoaderConfig, TokenizerConfig
│   ├── models/              # KilatTransformer, Block, RMSNorm, MLA/Decay Attention, DeepSeekMoE, RoPE, Triton
│   ├── engine/              # KilatTrainer, TrainingArguments, Integrations (W&B/TB), Checkpoint, Optimizer
│   ├── data/                # Datasets, zero-copy KilatDataCollator, DataLoaders, Tokenizers, Memmap Preprocessor
│   ├── inference/           # TextGenerator, GenerationMixin, Samplers, LogitProcessors, StoppingCriteria, HF Export
│   ├── distillation/        # VanillaKDLoss, ReverseKDLoss, AdaptiveKDLoss, BaseStudent, BaseTeacher
│   ├── assurance/           # Gates 1–6, Stages 7–8, Chaos Saboteur (C-01..C-08), FlightRecorderCallback
│   ├── benchmarking/        # 5-Pillar KMCP certification harness, synthetic capability tasks, reporters
│   └── utils/               # BaseConfig, BaseModel, validators, parameter breakdown reports
├── tools/                   # Mechanistic interpretability (logit lens, patching), audit & behavioral benchmarks
└── tests/                   # 387 BDD unit, seam, pipeline, assurance, and chaos tests
```

Detailed engineering blueprints are available in [`docs/`](docs/README.md):
- [**`docs/README.md`**](docs/README.md) — Master Architecture & Documentation Index
- [**`docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md`**](docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md) — Module Inventory & API Contracts
- [**`docs/02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md`**](docs/02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md) — Analytical Derivations & Precision Invariants (`M-01`..`M-07`)
- [**`docs/03_CODEBASE_RESTRUCTURE_PLAN.md`**](docs/03_CODEBASE_RESTRUCTURE_PLAN.md) — Package Design, Config Overrides, DDP, & CLI Guide
- [**`docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md`**](docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md) — 8-Stage Verification, `Saboteur`, & `FlightRecorder`
- [**`docs/05_STEP_BY_STEP_EXECUTION_CHECKLIST.md`**](docs/05_STEP_BY_STEP_EXECUTION_CHECKLIST.md) — Release & Pre-Training Acceptance Gates
- [**`docs/06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md`**](docs/06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md) — KMCP 5-Pillar Architecture Certification Protocol & Synthetic Task Specs
- [**`docs/07_MACRO_SYNERGY_AND_FUTURE_ARCHITECTURE_BLUEPRINT.md`**](docs/07_MACRO_SYNERGY_AND_FUTURE_ARCHITECTURE_BLUEPRINT.md) — Macro-Synergy Interleaved Stacking (L0 Decay $\to$ L1 MLA) Empirical Results & Sub-Quadratic Roadmap

---

## Development & Verification Commands

```bash
make check        # Ruff lint + Mypy static type check (0 errors)
make test         # Run full 387-test BDD suite (0 warnings, >= 80% coverage)
make verify       # Run Pre-Training Assurance Gates on configs/debug_cpu.yaml
make chaos        # Run Active Chaos Saboteur suite (C-01..C-08)
make certify      # Run KMCP 5-Pillar Certification on flagship kilat-macro-synergy
make pre-commit   # Run all git pre-commit hooks across the repository
```

---

## Roadmap

- [x] Dense, SwiGLU, and Fine-Grained DeepSeekMoE (`aux_loss` & `bias_free`) architectures
- [x] Flagship Macro-Synergy Interleaved Stacking (L0 Global Decay -> L1 Latent MLA, $6.4\times$ smaller KV-cache, $96.1\%$ MQAR)
- [x] 5-Pillar KMCP Model Architecture Certification Pipeline (`kilat certify`) with Stanford Zoology MQAR suite
- [x] $\mu\text{P}$ Width-Scaled Initialization Scheme (`init_scheme="width_scaled"`) for narrow model numerical stability
- [x] `LayerHybridCache` incremental & chunked-prefill KV-cache generation
- [x] 8-Stage Pre-Training Assurance (`kilat verify`), Active Chaos `Saboteur` (`kilat chaos`), & `FlightRecorder`
- [x] DDP-aware engine, atomic SHA-256 checkpoints (`0.0` bitwise resume parity), & `WSDLR` scheduler
- [x] Knowledge Distillation (`VanillaKDLoss`, `ReverseKDLoss`, `AdaptiveKDLoss`) & Hugging Face export
- [x] `torch.compile` Triton kernel fusion (TorchInductor JIT acceleration: $>10\times$ speedup)
- [x] Multi-backend experiment tracking (TensorBoard, Weights & Biases, MLflow, Comet ML)
- [ ] Next-Gen: Dual-Rank Latent Attention (Capped Global Anchors + Sparse Needle Recall)
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

*Licensed under Apache 2.0. See [LICENSE](LICENSE) for details.*
