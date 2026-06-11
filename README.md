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

> 📖 **Full documentation coming soon.**

---

## Install

```bash
pip install git+https://github.com/Airukua/kilat.git
```

For reporting backends (wandb, tensorboard, mlflow, comet_ml):

```bash
pip install -e ".[reporting]"
```

---

## Quick Start

### Train

```python
from arc.model import KilatTransformer
from configs.main_config import MainConfig
from training.trainer import KilatTrainer
from training.args import TrainingArguments
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader
from data.collator import KilatDataCollator

config = MainConfig.from_yaml("configs/small_dense.yaml")
model = KilatTransformer(config)
tokenizer = config.build_tokenizer()

train_dataset = PretrainingDataset(config.dataloader.train_data_path, key_name="input_ids", chunk_size=config.dataloader.max_seq_length)
eval_dataset  = PretrainingDataset(config.dataloader.eval_data_path,  key_name="input_ids", chunk_size=config.dataloader.max_seq_length)

collator     = KilatDataCollator(pad_token_id=config.model.pad_token_id, max_length=config.dataloader.max_seq_length)
train_loader = build_train_dataloader(train_dataset, batch_size=config.dataloader.train_batch_size, collate_fn=collator)
eval_loader  = build_eval_dataloader(eval_dataset,  batch_size=config.dataloader.eval_batch_size,  collate_fn=collator)

trainer = KilatTrainer(
    model=model,
    args=TrainingArguments(**config.training.to_dict()),
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    tokenizer_config=config.tokenizer,
)
trainer.train()
```

### Generate

```python
from arc.model import KilatTransformer
from data.tokenizer import AutoTokenizer
from pipeline.generation.generator import TextGenerator

model     = KilatTransformer.from_pretrained("./checkpoints/my-model")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/my-model")
generator = TextGenerator(model, tokenizer)

text = generator.generate("Once upon a time", max_new_tokens=100)
print(text)
```

---

## Configuration

Kilat uses YAML as its single source of truth:

```yaml
# configs/my_experiment.yaml
model:
  vocab_size: 50000
  n_embd: 768
  n_layer: 12
  n_head: 12
  ffn_mode: moe        # dense | moe
  num_experts: 8
  active_experts: 2
  recall_ratio: 0.5    # 0.0 = all global-decay, 1.0 = all MLA

tokenizer:
  tokenizer_type: auto
  tokenizer_name_or_path: gpt2

dataloader:
  train_batch_size: 8
  max_seq_length: 1024
  train_data_path: data/train/tokens.npy
  eval_data_path:  data/val/tokens.npy

training:
  output_dir: ./checkpoints
  num_train_epochs: 3
  learning_rate: 5e-5
  precision: bf16
  report_to: "none"
```

To resume a run:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/checkpoint-best"
```

---

## Architecture

### FFN Modes

| Mode | Description |
|------|-------------|
| `dense` | Standard SwiGLU FFN |
| `moe` | Mixture-of-Experts (set `num_shared_experts > 0` for DeepSeek-V2 style) |

### Hybrid Attention

Each layer splits heads into two parallel paths merged via a learned gate:

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
              └──────────────┬────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  cat + learned gate │
                  │  γ_net([x, out])    │
                  │  gate ∈ (0,1)^D     │
                  └──────────┬──────────┘
                             │
                     output [B, N, D]
                   + cache (optional)
```

`recall_ratio` controls the head split: `0.0` = all decay heads (fastest), `1.0` = all MLA heads (most precise), `0.5` = default.

---

## Generation Strategies

| Strategy | Key params |
|----------|-----------|
| Greedy | `do_sample=False` |
| Sampling | `do_sample=True, temperature=0.8` |
| Top-K | `top_k=50` |
| Top-P (Nucleus) | `top_p=0.95` |
| Beam Search | `num_beams=4` |
| Contrastive | `sampling_strategy="contrastive", contrastive_penalty=0.5` |

---

## Distributed Training

```bash
torchrun --nproc_per_node=8 experiments/train.py
```

Dataloaders automatically use `DistributedSampler` when a process group is active. Call `set_dataloader_epoch(loader, epoch)` at each epoch boundary.

---

## Convert to HuggingFace

```bash
python -m pipeline.converter.convert_to_hf \
  -c ./checkpoints/checkpoint-best \
  -o ./converted_model
```

---

## Roadmap

- [x] Dense / MoE architectures
- [x] Hybrid attention (global decay + latent MLA)
- [x] KV-cache generation & mixed precision
- [x] DDP-aware DataLoader, bin-packing, weighted mixing
- [x] WandB / TensorBoard / MLflow / Comet integration
- [x] HuggingFace conversion
- [ ] Flash Attention 2
- [ ] FSDP multi-GPU
- [ ] ONNX / TorchScript export
- [ ] Knowledge distillation

---

## Citation

```bibtex
@software{kilat2026,
  author = {Abdul Wahid Rukua},
  title  = {Kilat: Kernelized Lightweight Attention},
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