
<div align="center">

# ⚡ Kilat 

**Kernelized Lightweight Transformer Training & Inference Toolkit**

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

Kilat (*Indonesian: lightning*) is a modular toolkit for training and deploying transformer-based language models — from a single dense baseline to Mixture-of-Experts architectures. Designed for researchers who want production-grade training loops without the overhead of a full framework.

```python
from arc.model import KilatTransformer
from configs.model_config import KilatConfig

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

Optional reporting backends are not installed by default. If you cloned the
repo locally and want `report_to="wandb"`, `tensorboard`, `mlflow`, or
`comet_ml`, install the extra package set:

```bash
pip install -e ".[reporting]"
```

Verify:
```bash
python -c "from arc.model import KilatTransformer; print('✓ Kilat ready')"
```

### Train a model

```python
from arc.model import KilatTransformer
from configs.main_config import MainConfig
from training.trainer import KilatTrainer
from training.args import TrainingArguments
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader
from data.collator import KilatDataCollator

# Load configuration from YAML (or build programmatically)
config = MainConfig.from_yaml("configs/small_dense.yaml")
model = KilatTransformer(config)

# Build tokenizer automatically from config
tokenizer = config.build_tokenizer()

# Create datasets
train_dataset = PretrainingDataset(
    source=config.dataloader.train_data_path,
    key_name="input_ids",
    chunk_size=config.dataloader.max_seq_length,
)
eval_dataset = PretrainingDataset(
    source=config.dataloader.eval_data_path,
    key_name="input_ids",
    chunk_size=config.dataloader.max_seq_length,
)

# Create collator and dataloaders
collator = KilatDataCollator(
    pad_token_id=config.model.pad_token_id,
    max_length=config.dataloader.max_seq_length,
)

train_loader = build_train_dataloader(
    train_dataset,
    batch_size=config.dataloader.train_batch_size,
    collate_fn=collator,
    num_workers=config.dataloader.num_workers,
)
eval_loader = build_eval_dataloader(
    eval_dataset,
    batch_size=config.dataloader.eval_batch_size,
    collate_fn=collator,
    num_workers=config.dataloader.num_workers,
)

# Training arguments
args = TrainingArguments(**config.training.to_dict())

# Train
trainer = KilatTrainer(
    model=model,
    args=args,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    tokenizer_config=config.tokenizer,
)
trainer.train()
```

To resume a stopped run, set `resume_from_checkpoint` in the YAML config:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/checkpoint-best"
```

The trainer restores model weights, optimizer, scheduler, scaler, callback states, and training counters exactly.

### Run inference

Comming Soon!

---

## Architecture

### FFN modes

| Mode | Description | When to use |
|------|-------------|-------------|
| `dense` | Standard SwiGLU FFN | Baselines, small models |
| `moe` | Mixture-of-Experts with optional shared experts | Scalable training, DeepSeek-V2 style |

Shared experts are controlled by `num_shared_experts` in `KilatConfig` — set to `0` for standard MoE, or `>0` for DeepSeek-V2 style MoE with always-active shared experts.

### KilatAttention

The attention module splits `n_head` heads into two specialised paths that run in parallel and merge via a learned gate. This hybrid design trades a fraction of precise-recall capacity for O(N) compute and dramatically reduced KV-cache memory.

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


**Head allocation** is controlled by `recall_ratio`:

```
recall_ratio = 0.0  →  all global-decay heads  (fastest, least precise)
recall_ratio = 0.5  →  50/50 split             (default)
recall_ratio = 1.0  →  all latent MLA heads    (most precise)
```

**Cache structure** during autoregressive generation:

```
past_key_values = (
    global_state,   # (B, H_g, Dh) — 1 state vector per head
    latent_kv,      # (B, total_len, latent_dim) — compressed
)
```

---

## Project Structure

```
kilat/
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
├── arc/                               # Model architecture
│   ├── attention.py                   # KilatAttention (hybrid)
│   ├── blocks.py                      # Transformer blocks
│   ├── ffn.py                         # Dense / MoE FFN
│   ├── model.py                       # KilatTransformer
│   └── triton_ops.py                  # Triton causal decay kernel
├── configs/                           # Configuration classes
│   ├── __init__.py
│   ├── base.py                        # Base YAML utilities
│   ├── main_config.py                 # MainConfig (aggregator)
│   ├── model_config.py                # KilatConfig
│   ├── tokenizer_config.py            # TokenizerConfig
│   ├── dataloader_config.py           # DataLoaderConfig
│   ├── training_config.py             # TrainingConfig
│   └── sample/                        # Example YAML configs
│       ├── moe_standart.yaml
│       └── small_dense.yaml
├── data/                              # Data pipeline
│   ├── converter.py                   # Parquet → memmap conversion
│   ├── collator.py                    # KilatDataCollator
│   ├── dataloader.py                  # DDP-aware DataLoader builders
│   ├── dataset.py                     # PretrainingDataset, StreamingDataset, etc.
│   └── tokenizer.py                   # Unified tokenizer wrapper
├── training/                          # Training infrastructure
│   ├── args.py                        # TrainingArguments
│   ├── callbacks.py                   # Callback system
│   ├── integration.py                 # Logging integrations
│   ├── optimizer.py                   # AdamW with parameter groups
│   ├── scheduler.py                   # LR schedulers (cosine, linear, etc.)
│   ├── trainer.py                     # KilatTrainer
│   └── trainer_utils.py               # Checkpointing, metrics, helpers
├── pipeline/                          # Conversion utilities
│   └── converter/
│       ├── convert_to_hf.py           # Checkpoint → HuggingFace format
│       └── to_memmap.py               # Parquet → .npy memmap
├── experiments/                       # Notebooks and scripts
├── utils/
│   ├── config.py                      # Legacy config (deprecated)
│   ├── report.py                      # Parameter counting
│   └── validators.py                  # Tensor validation utilities
└── images/
    └── illustration.png
```

---

## Configuration

Kilat uses `MainConfig` as the single source of truth, bundling model, tokenizer, dataloader, and training configurations.

### YAML workflow

```yaml
# configs/my_experiment.yaml
experiment:
  name: "my-experiment"

model:
  vocab_size: 50000
  n_embd: 768
  n_layer: 12
  n_head: 12
  ffn_mode: moe
  num_experts: 8
  active_experts: 2
  recall_ratio: 0.5

tokenizer:
  tokenizer_type: auto
  tokenizer_name_or_path: gpt2
  local_files_only: true

dataloader:
  train_batch_size: 8
  eval_batch_size: 8
  num_workers: 4
  max_seq_length: 1024
  pin_memory: true
  train_data_path: data/train/tokens.npy
  eval_data_path: data/val/tokens.npy

training:
  output_dir: ./checkpoints
  training_mode: epochs
  num_train_epochs: 3
  learning_rate: 5e-5
  precision: bf16
  report_to: "none"
```

Load with:

```python
from configs.main_config import MainConfig

config = MainConfig.from_yaml("configs/my_experiment.yaml")
model = KilatTransformer(config)
tokenizer = config.build_tokenizer()
args = TrainingArguments(**config.training.to_dict())
```

### Tokenizer configuration

Supported tokenizer types:

- `auto`: Load via `AutoTokenizer.from_pretrained()` (HuggingFace Hub or local)
- `sentencepiece`: Load a local SentencePiece model

```yaml
# HuggingFace tokenizer
tokenizer:
  tokenizer_type: auto
  tokenizer_name_or_path: gpt2
  local_files_only: true

# Local tokenizer
tokenizer:
  tokenizer_type: auto
  tokenizer_name_or_path: ./tokenizers/my_tokenizer

# SentencePiece
tokenizer:
  tokenizer_type: sentencepiece
  tokenizer_model_path: ./tokenizers/spm.model
```

`MainConfig.from_yaml` will warn if the tokenizer vocabulary size does not match `model.vocab_size`.

### DataLoader configuration

Key fields for performance tuning:

- `num_workers`: CPU processes for data loading
- `pin_memory`: Faster GPU transfer (requires CUDA)
- `prefetch_factor`: Batches to preload per worker
- `persistent_workers`: Keep workers alive across epochs
- `use_packing`: Enable bin‑packing to eliminate padding waste

### Training configuration

Notable fields:

- `resume_from_checkpoint`: Path to resume from, or `"latest"`
- `scheduler_type`: `cosine`, `linear`, `polynomial`, `inverse_sqrt`, `wsdlr`, `rex`
- `precision`: `fp32`, `fp16`, `bf16`
- `report_to`: `"none"`, `"all"`, or list of backends (`wandb`, `tensorboard`, `mlflow`, `comet_ml`)
- `atomic_checkpoint`: Use atomic rename to prevent corrupted checkpoints
- `metric_for_best_model` / `greater_is_better`: Control early stopping

---

## Data Pipeline

The `data/` folder provides four layers:

1. **`converter.py`** – Converts Parquet shards to flat `.npy` memmaps for fast training
2. **`dataset.py`** – Dataset primitives:
   - `PretrainingDataset`: Random access over memmap, Parquet, JSON, JSONL
   - `StreamingDataset`: Sequential reading for large Parquet corpora
   - `PackedDataset`: Bin‑packing short sequences into fixed blocks
   - `ConcatDataset`: Mix multiple sources with optional weights
3. **`collator.py`** – `KilatDataCollator` for padding/truncation and causal LM labels
4. **`dataloader.py`** – DDP‑aware `build_train_dataloader` / `build_eval_dataloader` helpers

### Example: Convert Parquet to memmap

```python
from data.converter import parquet_to_memmap

parquet_to_memmap(
    input_path="./data/tokenized/",
    output_path="./data/tokens.npy",
    key_name="input_ids",
    verbose=True,
)
```

### Example: Packed dataset (zero padding waste)

```python
from data.dataset import PretrainingDataset, PackedDataset

base = PretrainingDataset("tokens.npy", chunk_size=256)
packed = PackedDataset(base, block_size=1024, eos_token_id=2)
```

### Example: Weighted mixing

```python
from data.dataset import ConcatDataset

web = PretrainingDataset("web.npy", chunk_size=1024)
books = PretrainingDataset("books.npy", chunk_size=1024)
code = PretrainingDataset("code.npy", chunk_size=1024)

mixed = ConcatDataset([web, books, code], weights=[0.7, 0.2, 0.1])
```

---

## Checkpointing

Kilat saves atomic, self-contained checkpoints:

```
checkpoint-best/
├── config.json              # HF model config
├── config.yaml              # Human‑readable config
├── model.safetensors        # Model weights
├── training_state.pt        # Optimizer, scheduler, scaler, state
├── training_args.json       # Training hyperparameters
└── tokenizer_config.json    # Tokenizer metadata
```

To resume:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/checkpoint-best"
```

Or programmatically:

```python
args.resume_from_checkpoint = "./checkpoints/checkpoint-best"
```

---

## Distributed Training

`build_train_dataloader` and `build_eval_dataloader` automatically detect an active `torch.distributed` process group and use `DistributedSampler`. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 experiments/train.py
```

Call `set_dataloader_epoch(loader, epoch)` at the start of each epoch:

```python
from data.dataloader import set_dataloader_epoch

for epoch in range(num_epochs):
    set_dataloader_epoch(train_loader, epoch)
    for batch in train_loader:
        ...
```

> KilatTrainer does **not** initialise `torch.distributed` itself — use `torchrun` or your own launcher.

---

## Converting to HuggingFace Format

```bash
python -m pipeline.converter.convert_to_hf \
  -c ./checkpoints/checkpoint-best \
  -o ./converted_model
```

The converted model can be loaded with:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./converted_model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./converted_model")
```

---

## Roadmap

- [x] Dense / MoE architectures (shared experts via `num_shared_experts`)
- [x] Hybrid attention (global decay + latent MLA)
- [x] KV-cache generation
- [x] Mixed precision (FP16, BF16, FP32)
- [x] Parquet + JSON/JSONL streaming
- [x] Bin-packing (`PackedDataset`) and weighted mixing (`ConcatDataset`)
- [x] DDP-aware `DataLoader` builders with `DistributedSampler`
- [x] WandB / TensorBoard / MLflow / Comet integration
- [x] Unified configuration system (`MainConfig`, YAML)
- [x] Tokenizer saving in checkpoints
- [x] HuggingFace conversion with tokenizer
- [ ] Flash Attention 2 integration
- [ ] Multi-GPU (FSDP) support
- [ ] ONNX / TorchScript export
- [ ] Knowledge distillation (`distilation/`)

---

## Contributing

PRs are welcome. Useful areas:

- Triton kernel optimizations
- FSDP integration
- Additional sampling strategies
- Training scripts for public datasets

```bash
git checkout -b feature/your-feature
git commit -m "feat: describe your change"
git push origin feature/your-feature
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

---

*Licensed under MIT. See [LICENSE](LICENSE) for details.*

