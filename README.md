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

### Distill a student from a teacher

Kilat also includes a separate distillation path under `distiliation/`. In this
setup, the teacher is a frozen model that produces logits, while the student is
the model you actually train.

How to set them up:

1. Load or build a teacher backend with `load_teacher(...)`.
2. Load or build a student backend with `load_student(...)` or `build_student(...)`.
3. Make sure the teacher and student use the same vocabulary size.
4. Choose a distillation loss, such as `vanilla`, `reverse`, or `adaptive`.
5. Run `DistillTrainer` the same way you would run the regular trainer.

The factories are backend-specific:

- `load_teacher("kilat", checkpoint_dir, ...)` loads a frozen Kilat checkpoint.
- `load_teacher("huggingface", model_name_or_path, ...)` loads a Hugging Face model.
- `load_student("kilat", checkpoint_dir, ...)` resumes a Kilat student checkpoint.
- `build_student("kilat", vocab_size=..., n_embd=..., n_layer=..., n_head=...)`
  creates a new smaller Kilat student from config.
- `build_student("huggingface", model_name_or_path=...)` starts from a pretrained
  Hugging Face model.

If you want to train a student from scratch, use the `build_student(...)` path
for Kilat models. For example:

```python
student = build_student(
    "kilat",
    vocab_size=50000,
    n_embd=256,
    n_layer=4,
    n_head=4,
    ffn_mode="dense",
)
```

That creates a fresh student with random initialization, which is the usual
choice when you want a smaller architecture than the teacher. If you are using
Hugging Face as the student backend, you can also start from a pretrained model
or use `HuggingFaceStudent.from_scratch(...)` directly with a model config.

For Kilat-to-Kilat distillation, the student is usually a smaller model with the
same tokenizer and vocabulary as the teacher.

Minimal example:

```python
from distiliation import DistillTrainer, load_teacher, load_student, build_loss
from training.arguments import TrainingArguments
from data.dataset import KilatDataset

teacher = load_teacher("kilat", "./checkpoints/teacher-best", device="cuda")
student = load_student("kilat", "./checkpoints/student-init", device="cuda")

train_dataset = KilatDataset("data/train.parquet", key_name="input_ids")
eval_dataset = KilatDataset("data/eval.parquet", key_name="input_ids")

args = TrainingArguments(
    output_dir="./distill-runs",
    training_mode="epochs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    precision="bf16",
)

loss_fn = build_loss("vanilla", temperature=4.0, alpha=0.5)

trainer = DistillTrainer(
    student=student,
    teacher=teacher,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_fn=loss_fn,
)
trainer.train()
```

If you prefer, you can also pass `loss_name="vanilla"` directly instead of
creating `loss_fn` manually.

Quick notes:
- The teacher stays frozen and only produces logits.
- The student must match the teacher's vocabulary size and output format.
- `DistillTrainer` only needs a `forward()` that returns logits and a
  `vocab_size` property.
- Distillation checkpoints store extra state for losses with learnable
  parameters, such as `adaptive`.

### Data format and collator

`KilatDataset` accepts these training inputs:

- a Parquet file: `*.parquet` or `*.parq`
- a directory of Parquet shards
- a JSON file: `*.json`
- a JSONL file: `*.jsonl`
- an in-memory `list` of dictionaries

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

Use `collator=None` only when you are using streaming datasets that already yield ready-to-train `(input_ids, labels)` batches. In that mode, the trainer skips the normal collator path.

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
│   ├── collator.py                    # PackedTokenBatchLoader
│   ├── dataset.py                     # Parquet, JSON, in-memory
├── distiliation/                      # Knowledge distillation (note: dir name as in ls)
│   ├── __init__.py
│   ├── distill_trainer.py
│   ├── losses.py
│   ├── student.py
│   └── teacher.py
├── experiments/                       # Notebooks and scripts
│   ├── 01_nano_GPT.ipynb
│   ├── 02_demo.ipynb
│   ├── 03_tiny_amq.ipynb
│   ├── MBGkilat01_light.py
│   ├── alkitab_text.txt
│   ├── demo_data.jsonl
│   ├── demo_data.parquet
│   └── tinyshakespeare.txt
├── images/
│   └── illustration.png
├── inference/                         # Inference & CLI
│   ├── __init__.py
│   ├── chat_session.py
│   ├── generation_config.py
│   ├── generator.py                   # KilatGenerator
│   ├── inference.py                   # CLI entry point
│   └── model_loader.py
├── training/                          # Training infrastructure
│   ├── __init__.py
│   ├── arguments.py                   # TrainingArguments
│   ├── checkpointing.py
│   ├── early_stopping.py
│   ├── logging_utils.py
│   ├── optim_utils.py
│   └── trainer.py                     # KilatTrainer
└── utils/
    ├── __init__.py
    ├── callback.py
    ├── config.py                      # KilatConfig / TrainingConfig / MainConfig
    ├── health_check.py                # smoke test for train + checkpoint + resume
    ├── sanity_check.py
    └── vram_check.py                  # empirical GPU memory probing before training
```

---

## Configuration

You can build configs directly in Python, then optionally export to YAML:

The tokenizer section is required and must match the tokenizer used to create
the dataset, so evaluation prompts decode correctly.

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig, TokenizerConfig, TrainingConfig, MainConfig
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

tokenizer_cfg = TokenizerConfig(
    tokenizer_type="gpt2",
    tokenizer_name_or_path="gpt2",
    local_files_only=True,
)

config = MainConfig(model=model_cfg, tokenizer=tokenizer_cfg, training=train_cfg)
config.to_yaml("configs/my_experiment.yaml")  # optional

model = KilatTransformer(config.model)
args = TrainingArguments(**config.training.to_dict())
```

See `configs/` for ready-made examples, or keep everything in Python for Kaggle / Colab workflows.

### Tokenizer configuration

Kilat supports two decode-time tokenizer styles through `TokenizerConfig`:

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

### Preflight checks

Before launching a real run, you can do:

```python
from utils.vram_check import check_vram_fit
from utils.health_check import run_health_check

report = check_vram_fit(model, train_args, train_dataset=train_dataset, data_collator=collator)
print(report.pretty())

health = run_health_check(model, train_dataset, eval_dataset=eval_dataset, data_collator=collator)
print(health.pretty())
```

- `check_vram_fit(...)` probes the real model on actual batches and reports the largest batch size that fits before OOM.
- `run_health_check(...)` uses a tiny subset of data to verify that forward/backward, checkpoint save, and checkpoint resume all work.

---

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
