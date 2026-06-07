<div align="center">

# ⚡ Kilat : Kernelized Lightweight Attention

**Lightweight Transformer Training & Inference Toolkit**

*Built for researchers who care about what's under the hood.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)

</div>

---

Kilat (*Indonesian: lightning*) is a modular toolkit for training and deploying transformer-based language models — from a single dense baseline to DeepSeek-V2 style Mixture-of-Experts architectures. Designed for researchers who want production-grade training loops without the overhead of a full framework.

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig

model = KilatTransformer(KilatConfig(vocab_size=50_000, n_embd=640, n_layer=8, ffn_mode="dense"))
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
# → 45.2M parameters
```

---

## Why Kilat?

Most training frameworks give you either too much magic (HuggingFace Trainer) or too little structure (raw PyTorch scripts). Kilat sits in between: a clean, hackable codebase with the production features you'd otherwise rebuild yourself.

- **Real training loop** — gradient accumulation, AMP (FP16/BF16/FP32), early stopping, checkpointing, WandB integration
- **Two FFN modes** — dense SwiGLU and DeepSeek-V2 style MoE with shared experts
- **Hybrid attention** — linear global-decay heads + latent MLA heads, fused via learned gate
- **KV-cache inference** — autoregressive generation with temperature / top-k / top-p / repetition penalty
- **Flexible data** — stream from Parquet, JSON/JSONL, or in-memory lists; efficient batch packing for long sequences
- **No framework lock-in** — build configs in code or export them as YAML; checkpoints are plain PyTorch state dicts
- **Preflight checks** — VRAM estimation and end-to-end health checks before real training starts

---

## Quick Start

### Install

```bash
pip install git+https://github.com/Airukua/kilat.git
```

Verify:
```bash
python -c "from arc.model import KilatTransformer; print('✓ Kilat ready')"
```

### Train a model in ~20 lines

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig
from utils.vram_check import check_vram_fit
from utils.health_check import run_health_check
from training.trainer import KilatTrainer
from training.arguments import TrainingArguments
from data.dataset import KilatDataset

config = KilatConfig(vocab_size=50_000, n_embd=640, n_layer=8, n_head=10, ffn_mode="dense")
model  = KilatTransformer(config)

train_dataset = KilatDataset("data/train.parquet", key_name="input_ids")
eval_dataset  = KilatDataset("data/eval.parquet",  key_name="input_ids")

args = TrainingArguments(
    output_dir="./checkpoints",
    training_mode="epochs",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    precision="bf16",
)

# Fail fast if the GPU budget is too small
vram_report = check_vram_fit(
    model,
    args,
    train_dataset=train_dataset,
    data_collator=None,
    raise_on_fail=False,
)
print(vram_report.pretty())

# Optional: smoke-test 1 sample, training, checkpointing, and resume
health_report = run_health_check(
    model,
    train_dataset,
    eval_dataset=eval_dataset,
    args=args,
)
print(health_report.pretty())

KilatTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset).train()
```

### Run inference

```bash
# Single prompt
python -m inference.inference \
  --checkpoint ./checkpoints/best \
  --mode generate \
  --prompt "Pada zaman dahulu" \
  --max_new_tokens 128 --temperature 0.8

# Interactive chat
python -m inference.inference --checkpoint ./checkpoints/best --mode chat

# Batch (prompts.txt → completions.json)
python -m inference.inference --checkpoint ./checkpoints/best \
  --mode batch --input_file prompts.txt --output_file completions.json
```

---

## Architecture

### FFN modes

Kilat supports three FFN modes, switchable via a single config field:

| Mode | Description | When to use |
|------|-------------|-------------|
| `dense` | Standard SwiGLU FFN | Baselines, small models |
| `moe_shared` | DeepSeek-V2 style — shared + routed experts | MoE with stable training dynamics |

---

### KilatAttention

The attention module splits `n_head` heads into two specialised paths that run in
parallel and merge via a learned gate. This hybrid design trades a fraction of
precise-recall capacity for O(N) compute and dramatically reduced KV-cache memory.

```
                     Input x  [B, N, D]
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
  ╔═══════════════════╗           ╔════════════════════╗
  ║   PATH 1          ║           ║   PATH 2           ║
  ║   Global Decay    ║           ║   Latent MLA       ║
  ║   (linear, O(N))  ║           ║   (softmax, O(N²)) ║
  ╚═══════════════════╝           ╚════════════════════╝
          │                                │
          ▼                                ▼
   ┌─────────────┐                 ┌──────────────────┐
   │  v_proj     │                 │  q_a → LN → q_b  │
   │  (values)   │                 │  kv_a → LN → kv_b│
   └──────┬──────┘                 └────────┬─────────┘
          │  [B, N, H_g·Dh]                 │  Q [B, H_r, N, Dh]
          ▼                                 │  K,V from latent cache
   ┌─────────────────┐                      ▼
   │  λ = σ(log_λ)   │             ┌─────────────────────┐
   │  per-head decay │             │  KV-cache in latent │
   └──────┬──────────┘             │  space (B, N, L)    │
          │                        │  4–8× smaller than  │
          ▼                        │  full K,V matrices  │
   ┌──────────────────────┐        └──────────┬──────────┘
   │  Triton causal decay │                   │
   │                      │                   ▼
   │  full seq:           │        ┌─────────────────────┐
   │    Σ λ^(i-j)·V[j]   │        │  SDPA (is_causal)   │
   │    O(N) kernel       │        │  Q·Kᵀ/√Dh → V      │
   │                      │        └──────────┬──────────┘
   │  incremental:        │                   │
   │    λ·state + V_new   │                   │
   │    O(1) per step     │                   │
   └──────────┬───────────┘                   │
              │  out_global                   │  out_recall
              │  [B, N, H_g·Dh]              │  [B, N, H_r·Dh]
              └──────────────┬────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  cat along head dim │
                  │  [B, N, D]          │
                  └──────────┬──────────┘
                             │
              x (residual) ──┤
                             ▼
                  ┌─────────────────────────────┐
                  │  γ_net([x, out_combined])    │
                  │                             │
                  │  Linear → ReLU → Linear → σ │
                  │  gate ∈ (0,1)^D, elem-wise  │
                  └──────────┬──────────────────┘
                             │  out * gate
                             ▼
                  ┌─────────────────────┐
                  │  c_proj             │
                  │  Linear(D → D)      │
                  └──────────┬──────────┘
                             │
                     output [B, N, D]
                   + cache (optional)
```

**Cache structure** during autoregressive generation:

```
past_key_values = (
    global_state,   # (B, H_g, Dh)  — 1 state vector per head, entire history
    latent_kv,      # (B, total_len, latent_dim)  — compressed, not full K,V
)
```

**Head allocation** is controlled by `recall_ratio` in `KilatConfig`:

```
recall_ratio = 0.0  →  all global-decay heads  (fastest, least precise)
recall_ratio = 0.5  →  50/50 split             (default)
recall_ratio = 1.0  →  all latent MLA heads    (most precise)
```

---

### KV-cache memory comparison

For a 1024-token sequence with `n_embd=1024`, `n_recall_heads=8`, `head_dim=128`, `latent_dim=256`:

```
Standard attention KV-cache:  2 × 8 × 128 × 1024  =  2,097,152 floats
Kilat latent KV-cache:              256 × 1024     =    262,144 floats
                                                       ─────────────────
                                                            8× reduction
```

---

## Project Structure

```
kilat/
├── arc/               # Model architecture
│   ├── model.py       # KilatTransformer
│   ├── blocks.py      # Transformer blocks
│   ├── attention.py   # KilatAttention (hybrid global-decay + latent MLA)
│   ├── ffn.py         # Dense / MoE / MoE-shared FFN
│   └── triton_ops.py  # Triton causal decay kernel
├── data/              # Data pipeline
│   ├── dataset.py     # Parquet, JSON, in-memory
│   └── collator.py    # PackedTokenBatchLoader
├── training/          # Training infrastructure
│   ├── trainer.py     # KilatTrainer
│   ├── arguments.py   # TrainingArguments
│   ├── checkpointing.py
│   ├── early_stopping.py
│   └── optim_utils.py
├── inference/         # Inference & CLI
│   ├── inference.py   # CLI entry point
│   ├── generator.py   # KilatGenerator
│   └── chat_session.py
├── utils/
│   ├── config.py      # KilatConfig / TrainingConfig / MainConfig
│   ├── vram_check.py  # GPU memory estimation before training
│   ├── health_check.py# smoke test for train + checkpoint + resume
│   └── sanity_check.py
└── configs/           # Example YAML configs
    ├── small_dense.yaml
    └── moe_standard.yaml
```

---

## Configuration

You can build configs directly in Python, then optionally export to YAML:

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig, TrainingConfig, MainConfig
from training.arguments import TrainingArguments

model_cfg = KilatConfig(
    vocab_size=50_000,
    n_embd=768,
    n_layer=12,
    n_head=12,
    ffn_mode="moe",
    recall_ratio=0.5,   # 50% latent MLA, 50% global decay
    latent_dim=192,     # KV compression dim (default: n_embd // 4)
)

train_cfg = TrainingConfig(
    output_dir="./checkpoints",
    training_mode="steps",
    max_steps=100,
    per_device_train_batch_size=1,
    precision="bf16",
    report_to="none",
)

config = MainConfig(model=model_cfg, training=train_cfg)
config.to_yaml("configs/my_experiment.yaml")  # optional

model = KilatTransformer(config.model)
args = TrainingArguments(**config.training.to_dict())
```

See `configs/` for ready-made examples, or keep everything in Python for Kaggle / Colab workflows.

### Preflight checks

Before launching a real run, you can do:

```python
from utils.vram_check import check_vram_fit
from utils.health_check import run_health_check

report = check_vram_fit(model, args, train_dataset=train_dataset, data_collator=collator)
print(report.pretty())

health = run_health_check(model, train_dataset, eval_dataset=eval_dataset, data_collator=collator)
print(health.pretty())
```

- `check_vram_fit(...)` estimates whether the current model + batch size + sequence length should fit on the available GPU.
- `run_health_check(...)` uses a tiny subset of data to verify that forward/backward, checkpoint save, and checkpoint resume all work.

---

## Roadmap

- [x] Dense / MoE-shared architectures
- [x] Hybrid attention (global decay + latent MLA)
- [x] KV-cache generation
- [x] Mixed precision (FP16, BF16, FP32)
- [x] Parquet + JSON/JSONL streaming
- [x] WandB integration
- [ ] Flash Attention 2 integration
- [ ] Multi-GPU (DDP / FSDP)
- [ ] ONNX / TorchScript export
- [ ] Additional sampling strategies (beam search, contrastive decoding)

## Contributing

PRs are welcome. Useful areas:

- Triton kernel optimizations
- Distributed training (DDP/FSDP)
- Additional sampling strategies
- Training scripts for common public datasets
- Documentation

```bash
git checkout -b feature/your-feature
# make changes
git commit -m "feat: describe what you did"
git push origin feature/your-feature
# open a PR
```

---

## Citation

If Kilat helps your research, please cite:

```bibtex
@software{kilat2026,
  author  = {Abdul Wahid Rukua},
  title   = {Kilat: Kernelized Lightweight Attention},
  year    = {2026},
  url     = {https://github.com/Airukua/kilat}
}
```

---

## Author

**Abdul Wahid Rukua** — AI/ML Engineer & Researcher

Focus: efficient sequence modeling, Indonesian NLP, low-resource language technology

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdul--wahid--rukua-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdul-wahid-rukua/)
[![GitHub](https://img.shields.io/badge/GitHub-Airukua-181717?style=flat-square&logo=github)](https://github.com/Airukua)

---

*Licensed under MIT. See [LICENSE](LICENSE) for details.*
