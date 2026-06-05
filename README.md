# Kilat: Lightweight Transformer Training & Inference Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, modular transformer training and inference toolkit with support for dense and Mixture-of-Experts (MoE) architectures, efficient KV-cache generation, and production-ready training loops.

## Why Kilat?

- **Production-Ready**: Battle-tested training loop with gradient accumulation, AMP, checkpointing, and early stopping.
- **Flexible Architecture**: Dense transformers, standard MoE, or DeepSeek-V2 style MoE with shared experts.
- **Developer-Friendly**: Clear module separation, type hints, and comprehensive configuration system.
- **Efficient Inference**: KV-cache support with multi-mode CLI (generate, chat, batch processing).
- **Scale Your Data**: Parquet streaming, JSON/JSONL support, and in-memory datasets—choose what fits your workflow.

## Features

✅ **Model Architectures**
- Dense transformer (SwiGLU FFN)
- Standard Mixture-of-Experts (MoE)
- DeepSeek-V2 style MoE with shared experts
- Configurable attention (recall ratio, latent projections)

✅ **Training**
- Step-based and epoch-based training modes
- Mixed precision (FP16, BF16, FP32)
- Gradient accumulation & clipping
- Early stopping with patience-based scheduling
- WandB integration for experiment tracking

✅ **Data Handling**
- Parquet files and directories
- JSON/JSONL streaming
- In-memory Python lists
- Efficient batch packing for long sequences

✅ **Inference**
- CLI with three modes: generate, chat, batch
- Autoregressive decoding with KV-cache
- Temperature, top-k, top-p sampling
- Repetition penalty

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (recommended for GPU training)

### Quick Install

```bash
git clone https://github.com/your-username/kilat.git
cd kilat
pip install -e .
```

Or install without editable mode:
```bash
pip install .
```

Verify installation:
```bash
python -c "from arc.model import KilatTransformer; print('✓ Kilat installed')"
```

## Getting Started

### 1. Create a Model

```python
from arc.model import KilatTransformer
from utils.config import KilatConfig

# Dense transformer (8 layers, 640 hidden dim)
config = KilatConfig(
    vocab_size=50000,
    n_embd=640,
    n_layer=8,
    n_head=10,
    ffn_mode="dense",
)
model = KilatTransformer(config)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 2. Prepare Your Data

```python
from data.dataset import KilatDataset
from data.collator import PackedTokenBatchLoader

# Load from Parquet
dataset = KilatDataset(
    "path/to/tokenized_data.parquet",
    key_name="input_ids",
    streaming=False  # Set to True for large datasets
)

# Or from in-memory list
dataset = KilatDataset(
    [{"input_ids": [1, 2, 3, ...]}, ...],
    key_name="input_ids"
)

print(f"Dataset size: {len(dataset):,} samples")
```

### 3. Train Your Model

```python
from training.trainer import KilatTrainer
from training.arguments import TrainingArguments
from data.collator import PackedTokenBatchLoader

args = TrainingArguments(
    output_dir="./checkpoints",
    training_mode="epochs",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    precision="fp16",
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    report_to="wandb",  # Optional: remove for console-only logging
)

trainer = KilatTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Optional
    data_collator=PackedTokenBatchLoader(...),
)

trainer.train()
```

### 4. Run Inference

#### Generate Text (Single Prompt)
```bash
python -m inference.inference \
  --checkpoint /path/to/checkpoint \
  --mode generate \
  --prompt "Once upon a time" \
  --max_new_tokens 128 \
  --temperature 0.8
```

#### Interactive Chat
```bash
python -m inference.inference \
  --checkpoint /path/to/checkpoint \
  --mode chat \
  --system_prompt "You are a helpful assistant."
```

#### Batch Processing
```bash
python -m inference.inference \
  --checkpoint /path/to/checkpoint \
  --mode batch \
  --input_file prompts.txt \
  --output_file completions.json
```

## Project Structure

```
kilat/
├── arc/                      # Core transformer architecture
│   ├── model.py             # KilatTransformer
│   ├── blocks.py            # Transformer blocks
│   ├── attention.py         # Attention mechanisms
│   ├── ffn.py               # Feed-forward networks (dense + MoE)
│   └── triton_ops.py        # Optional Triton kernels
├── data/                     # Data loading and processing
│   ├── dataset.py           # KilatDataset (Parquet, JSON, etc.)
│   ├── collator.py          # Batch collation
│   └── tokens/              # Tokenizer and pre-tokenized data
├── training/                 # Training infrastructure
│   ├── trainer.py           # KilatTrainer
│   ├── arguments.py         # TrainingArguments config
│   ├── checkpointing.py     # Checkpoint save/load
│   ├── early_stopping.py    # Early stopping callback
│   └── optim_utils.py       # Optimizer & scheduler setup
├── inference/               # Inference pipelines
│   ├── inference.py         # CLI entry point
│   ├── generator.py         # KilatGenerator
│   ├── model_loader.py      # Model loading utilities
│   └── chat_session.py      # Interactive chat
├── utils/                    # Utilities
│   ├── config.py            # KilatConfig, TrainingConfig
│   └── sanity_check.py      # Tensor validation
├── configs/                  # Example YAML configs
│   ├── small_dense.yaml     # Small dense config
│   └── moe_standard.yaml    # Standard MoE config
├── pyproject.toml           # Package metadata
├── setup.py                 # Installation script
└── README.md               # This file
```

## Configuration

Models and training are configured via `KilatConfig` and `TrainingArguments`. Export/load as YAML:

```python
from utils.config import KilatConfig

# Create config
config = KilatConfig(vocab_size=50000, n_embd=768, n_layer=12)

# Save as human-readable YAML
config.to_yaml("my_config.yaml")

# Load from YAML
loaded_config = KilatConfig.from_yaml("my_config.yaml")
```

See `configs/` for example YAML files.

## Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **Model Hub Integration**: HuggingFace Transformers
- **Data Processing**: PyArrow, Parquet, JSONL
- **Optimization**: Custom AMP, GradScaler, AdamW
- **Experiment Tracking**: Weights & Biases (optional)
- **Tokenization**: SentencePiece

## Roadmap

- [x] Dense transformer architecture
- [x] Standard MoE implementation
- [x] DeepSeek-V2 style MoE with shared experts
- [x] KV-cache support for efficient generation
- [x] Multi-format data loading (Parquet, JSON, JSONL)
- [ ] Flash Attention integration
- [ ] Multi-GPU distributed training
- [ ] Model quantization & export (ONNX, TorchScript)
- [ ] Additional generation sampling strategies

## Contributing

We welcome contributions! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

**Areas we're looking for help:**
- Example training scripts for common datasets
- Performance optimizations & kernel fusion
- Additional sampling strategies
- Documentation improvements
- Bug reports and fixes

## Citation

If you use Kilat in your research, please cite:

```bibtex
@software{kilat2024,
  author = {Your Name},
  title = {Kilat: Lightweight Transformer Training & Inference Toolkit},
  year = {2024},
  url = {https://github.com/your-username/kilat}
}
```

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE) for details.

## Author

**Your Name** — [GitHub](https://github.com/your-username) | [Email](mailto:your-email@example.com)

---

## FAQ

**Q: Can I use Kilat for production inference?**  
A: Yes. The inference pipeline is optimized for latency with KV-cache support. For high-throughput serving, consider adding batching or integrating with vLLM.

**Q: What's the difference between dense and MoE modes?**  
A: Dense mode uses a single SwiGLU FFN per block. MoE mode routes tokens to a subset of experts. MoE is more parameter-efficient but introduces training complexity (load balancing).

**Q: Can I resume training from a checkpoint?**  
A: Yes. Set `resume_from_checkpoint` in `TrainingArguments` with the checkpoint directory path.

**Q: Does it support distributed training?**  
A: Not yet. This is on the roadmap. For now, single-GPU training is supported.

---

**Questions or Issues?** Open a [GitHub Issue](https://github.com/your-username/kilat/issues) or start a [Discussion](https://github.com/your-username/kilat/discussions).
