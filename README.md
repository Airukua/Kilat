<div align="center">

# ⚡ Kilat 

**Kernelized Lightweight Lightweight Transformer Training & Inference Toolkit**

*Built for researchers who care about what's under the hood.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)

<p align="center">
  <img src="images/illustration.png" alt="Kilat illustration" width="720" />
</p>

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
from training.trainer import KilatTrainer
from training.args import TrainingArguments
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader

config = KilatConfig(vocab_size=50_000, n_embd=640, n_layer=8, n_head=10, ffn_mode="dense")
model  = KilatTransformer(config)

train_dataset = PretrainingDataset("data/train.parquet", key_name="input_ids")
eval_dataset  = PretrainingDataset("data/eval.parquet",  key_name="input_ids")
train_loader = build_train_dataloader(
    train_dataset,
    batch_size=32,
    pad_token_id=config.pad_token_id,
)
eval_loader = build_eval_dataloader(
    eval_dataset,
    batch_size=32,
    pad_token_id=config.pad_token_id,
)

args = TrainingArguments(
    output_dir="./checkpoints",
    training_mode="epochs",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    precision="bf16",
)

KilatTrainer(model=model, args=args, train_dataloader=train_loader, eval_dataloader=eval_loader).train()
```

To continue a stopped run, point `resume_from_checkpoint` to the checkpoint
directory you want to restore:

```python
args = TrainingArguments(
    output_dir="./checkpoints",
    resume_from_checkpoint="./checkpoints/checkpoint-best",
    training_mode="epochs",
    num_train_epochs=3,
)
```

When you use `MainConfig.from_yaml(...)`, put the same path under:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/checkpoint-best"
```

That restores model weights, optimizer, scheduler, scaler, and training
counters so training continues from the exact saved state.

### Distillation

Coming soon.

### Data pipeline

The `data/` folder is split into four layers:

- `data/converter.py` turns Parquet shards into flat `.npy` memmaps for fast training.
- `data/dataset.py` provides the dataset primitives:
  - `PretrainingDataset` for random access over memmap, Parquet, JSON, JSONL, or in-memory samples.
  - `StreamingDataset` for large Parquet corpora that should be read sequentially.
  - `PackedDataset` for bin-packing short sequences into fixed-length blocks.
  - `ConcatDataset` for mixing multiple map-style datasets.
- `data/collator.py` pads/truncates batches for causal language modeling.
- `data/dataloader.py` builds train/eval `DataLoader` objects with sensible defaults.

For a quick smoke test, you can use the included dummy Parquet file:

```python
from data.dataset import StreamingDataset

ds = StreamingDataset("data/dummy_test.parquet", pad_token_id=0)
print(ds._has_mask)
print(next(iter(ds)))
```

Recommended data formats:

- Parquet file: `*.parquet` or `*.parq`
- Directory of Parquet shards
- JSON file: `*.json`
- JSONL file: `*.jsonl`
- In-memory `list` of token dicts or token lists

Each sample should contain token IDs under a key such as `input_ids`:

```python
{"input_ids": [101, 2023, 2003, 1037, 3978]}
```

For normal training, pass a `KilatDataCollator`:

```python
from data.collator import KilatDataCollator

collator = KilatDataCollator(
    pad_token_id=0,
    max_length=512,
    ignore_index=-100,
)
```

- It pads variable-length sequences into one batch tensor.
- It truncates long samples to `max_length`.
- It creates `labels` for causal language modeling.

Use `collator=None` only when the dataset already yields ready-to-train batches or when you are using a custom streaming pipeline that performs collation itself.

If you want a higher-level constructor, use the helpers in `data/dataloader.py`:

```python
from data.dataloader import build_train_dataloader
from data.dataset import PretrainingDataset

train_ds = PretrainingDataset("data/train.parquet", key_name="input_ids")
train_loader = build_train_dataloader(train_ds, batch_size=8, pad_token_id=0)
```

For large corpora that start as Parquet, the usual path is:

1. Convert Parquet to memmap with `data.converter.parquet_to_memmap(...)`.
2. Load the resulting `.npy` with `PretrainingDataset`.
3. Use `build_train_dataloader(...)` with `KilatDataCollator`.

### Run inference

Inference reuses the shared tokenizer wrapper from `data/tokenizer.py`, so the
checkpoint and preprocessing pipeline stay aligned.

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
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── arc/                               # Model architecture
│   ├── __init__.py
│   ├── attention.py                   # KilatAttention (hybrid global-decay + latent MLA)
│   ├── blocks.py                      # Transformer blocks
│   ├── ffn.py                         # Dense / MoE / MoE-shared FFN
│   ├── model.py                       # KilatTransformer
│   └── triton_ops.py                  # Triton causal decay kernel
├── configs/
│   ├── moe_standart.yaml              # MoE configuration (standard)
│   └── small_dense.yaml               # Dense baseline config
├── data/                              # Data pipeline
│   ├── __init__.py
│   ├── converter.py                   # Dataset conversion helpers
│   ├── collator.py                    # PackedTokenBatchLoader
│   ├── dataloader.py                  # Train/eval dataloader builders
│   ├── dataset.py                     # Parquet, JSON, in-memory
│   └── tokenizer.py                   # Shared tokenizer wrapper
├── distiliation/                      # Knowledge distillation (coming soon)
│   ├── __init__.py
│   ├── losses.py
│   ├── student.py
│   └── teacher.py
├── experiments/                       # Notebooks and scripts
│   ├── 01_dataset.Ipynb
│   ├── alkitab_text.txt
│   ├── kilat1.0.py
│   └── tinyshakespeare.txt
├── images/
│   └── illustration.png
├── generation/                        # Inference & CLI
│   ├── __init__.py
│   ├── chat_session.py
│   ├── generation_config.py
│   ├── generator.py                   # KilatGenerator
│   ├── inference.py                   # CLI entry point
│   └── model_loader.py
├── training/                          # Training infrastructure
│   ├── __init__.py
│   ├── args.py                        # TrainingArguments
│   ├── callbacks.py
│   ├── integration.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py                     # KilatTrainer
│   └── trainer_utils.py
└── utils/
    ├── __init__.py
    ├── config.py                      # KilatConfig / TrainingConfig / MainConfig
    └── validators.py
```

---

## Configuration

Kilat keeps configuration in four small objects:

- `KilatConfig` for model architecture
- `TokenizerConfig` for tokenization metadata
- `TrainingConfig` for training hyperparameters and runtime settings
- `MainConfig` for bundling the three together into one YAML file

Build them directly in Python, then optionally export to YAML. The tokenizer
section is required and must match the tokenizer used to create the dataset, so
evaluation prompts decode correctly.

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig, TokenizerConfig, TrainingConfig, MainConfig
from training.args import TrainingArguments
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader

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
    scheduler_type="cosine",
    atomic_checkpoint=True,
    precision="bf16",
    report_to="none",
)

tokenizer_cfg = TokenizerConfig(
    tokenizer_type="gpt2",
    tokenizer_name_or_path="gpt2",
    local_files_only=True,
)

config = MainConfig(model=model_cfg, tokenizer=tokenizer_cfg, training=train_cfg)
config.to_yaml("configs/my_experiment.yaml")  # optional

model = KilatTransformer(config.model)
args = TrainingArguments(**config.training.to_dict())

train_dataset = PretrainingDataset("data/train.parquet", key_name="input_ids")
eval_dataset = PretrainingDataset("data/eval.parquet", key_name="input_ids")
train_loader = build_train_dataloader(
    train_dataset,
    batch_size=config.training.per_device_train_batch_size,
    pad_token_id=config.model.pad_token_id,
)
eval_loader = build_eval_dataloader(
    eval_dataset,
    batch_size=config.training.per_device_eval_batch_size,
    pad_token_id=config.model.pad_token_id,
)
```

`TrainingConfig.to_dict()` mirrors `TrainingArguments`, so the same YAML-backed
configuration can be passed directly into the trainer:

```python
trainer = KilatTrainer(model=model, args=args, train_dataloader=train_loader, eval_dataloader=eval_loader)
```

See `configs/` for ready-made examples, or keep everything in Python for Kaggle / Colab workflows.

Minimal YAML shape:

```yaml
model:
  vocab_size: 50000
  n_embd: 768
  n_layer: 12
  n_head: 12
tokenizer:
  tokenizer_type: gpt2
  tokenizer_name_or_path: gpt2
training:
  output_dir: ./checkpoints
  training_mode: epochs
  num_train_epochs: 3
  per_device_train_batch_size: 8
  precision: bf16
```

Notable training fields:

- `resume_from_checkpoint` to continue from a saved run
- `scheduler_type` and `scheduler_kwargs` to choose the LR schedule
- `atomic_checkpoint` to write checkpoints safely via temp-directory rename
- `report_to` to pick logging backends, including `["wandb", "tensorboard"]`
- `metric_for_best_model` and `greater_is_better` to control early stopping and best-checkpoint selection
- `per_device_train_batch_size` and `per_device_eval_batch_size` to size the loaders built by `data.dataloader`

### Tokenizer configuration

Kilat keeps tokenizer handling centralized through `data/tokenizer.KilatTokenizer`
while `TokenizerConfig` still documents how the tokenizer is resolved from config.

Supported decode-time tokenizer styles:

- `tokenizer_type: "auto"` — load any Hugging Face tokenizer via `AutoTokenizer.from_pretrained(...)`
- `tokenizer_type: "sentencepiece"` — load a local SentencePiece model via `tokenizer_model_path`

Use this when your training data was tokenized with:

- a pretrained HF tokenizer like GPT-2, LLaMA, etc.
- your own tokenizer saved locally in HF format
- your own SentencePiece tokenizer

Example for a Hugging Face tokenizer:

```python
tokenizer_cfg = TokenizerConfig(
    tokenizer_type="auto",
    tokenizer_name_or_path="gpt2",
    local_files_only=True,
)
```

Example for a local tokenizer you trained yourself:

```python
tokenizer_cfg = TokenizerConfig(
    tokenizer_type="auto",
    tokenizer_name_or_path="./tokenizers/my_tokenizer",
    local_files_only=True,
)
```

Example for SentencePiece:

```python
tokenizer_cfg = TokenizerConfig(
    tokenizer_type="sentencepiece",
    tokenizer_name_or_path="./tokenizers/my_spm",
    tokenizer_model_path="./tokenizers/my_spm/sp_tokenizer.model",
    local_files_only=True,
)
```

Important: the tokenizer used for decoding eval samples must match the tokenizer
that produced the dataset, otherwise the printed prompt can look like garbled text.

## Roadmap

- [x] Dense / MoE-shared architectures
- [x] Hybrid attention (global decay + latent MLA)
- [x] KV-cache generation
- [x] Mixed precision (FP16, BF16, FP32)
- [x] Parquet + JSON/JSONL streaming
- [x] WandB integration
- [ ] Optimize Gate fused
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
<!-- [![GitHub](https://img.shields.io/badge/GitHub-Airukua-181717?style=flat-square&logo=github)](https://github.com/Airukua) -->

---

*Licensed under MIT. See [LICENSE](LICENSE) for details.*
