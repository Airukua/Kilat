<div align="center">

# ⚡ Kilat

**The Modern Lightweight Sub-Quadratic LLM Framework**  
*Pre-Train, Fine-Tune (SFT), and Deploy Custom Language Models on Consumer GPUs & Google Colab*

[![Open In Colab: Pre-Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_general_pretraining_colab.ipynb)
[![Open In Colab: SFT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_sft_colab.ipynb)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20HuggingFace-tiny--stories--kilat--35M-FFD21E?style=flat-square)](https://huggingface.co/AiRukua/tiny-stories-kilat-35M)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)

<p align="center">
  <img src="images/illustration.png" alt="Kilat illustration" width="720" />
</p>

</div>

---

## 💡 Why Kilat? Train LLMs Without a Supercomputer

Most existing LLM frameworks (like Megatron-LM or standard Transformer implementations) assume you have a cluster of 8x H100 enterprise GPUs. Trying to train or fine-tune models on consumer hardware (like an RTX 3060/4050/4060 laptop GPU, or a free Google Colab T4) usually leads to one frustrating outcome: **`CUDA Out of Memory (OOM)`**.

**Kilat** (*Indonesian: lightning*) is engineered from the ground up for **maximum efficiency and accessibility**. It combines sub-quadratic attention with modern state-of-the-art architectures so developers, indie hackers, and practitioners can build custom domain models, small language models (SLMs), and edge AI agents without enterprise budgets.

### The Lightweight Architectural Advantage: **Macro-Synergy**

Kilat solves the memory bottleneck with its flagship **Macro-Synergy** architecture, which interleaves:
1. **$O(N)$ Global Exponential Decay Contextualizer (Even Layers)**: Uses a hardware-accelerated Triton parallel prefix scan. **Requires EXACTLY 0 BYTES of KV-Cache.**
2. **Latent Multi-Head Latent Attention (MLA) (Odd Layers)**: Low-rank joint key-value compression with Decoupled RoPE, dedicating 100% capacity to associative memory retrieval.

| Metric | Standard Transformer | Kilat Macro-Synergy | Real-World Benefit |
| :--- | :---: | :---: | :--- |
| **KV-Cache Size ($N=2048$)** | $1,024\text{ KB}$ | **$160\text{ KB}$** | 🚀 **$6.4\times$ less VRAM** |
| **KV-Cache Size ($N=8192$)** | $4,096\text{ KB}$ | **$640\text{ KB}$** | 🚀 **$6.4\times$ less VRAM** |
| **Prefill Throughput ($N=512$)** | $269,000\text{ tok/s}$ | **$423,808\text{ tok/s}$** | ⚡ **$+59\%$ faster prefill** |
| **Associative Recall (MQAR)** | Stagnates | **$96.09\%$** | 🎯 **High retrieval accuracy** |
| **Peak VRAM ($N=4096$)** | ❌ OOM | ✅ **Pass (No OOM)** | Run long context on consumer GPUs |

---

## 🚀 1-Click Interactive Google Colab Notebooks

You don't need any local GPU setup to get started. Open and run our end-to-end notebooks directly in Google Colab (compatible with free T4 GPU tier):

| Notebook | Description | Colab Link |
| :--- | :--- | :---: |
| **1. General Pre-Training** | Pre-train a custom Kilat sub-quadratic LLM from scratch, run 6-Gate Assurance, and export to Hugging Face Hub. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_general_pretraining_colab.ipynb) |
| **2. PEFT LoRA Fine-Tuning** | Adapt Kilat with Low-Rank Adaptation (`peft`), verify `PeftModel` across Gates `G0`–`G6`, and merge weights via `merge_and_unload()`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_peft_lora_example.ipynb) |
| **3. SFT & DPO Alignment** | Instruction-tune on conversational datasets with `-100` prompt loss masking, non-reentrant checkpointing, and DPO preference alignment. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_sft_colab.ipynb) |

*Pre-trained model checkpoint available on Hugging Face: [**AiRukua/tiny-stories-kilat-35M**](https://huggingface.co/AiRukua/tiny-stories-kilat-35M).*

---

## 📦 Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/Airukua/Kilat.git

# Or clone for local development / custom training
git clone https://github.com/Airukua/Kilat.git
cd Kilat
pip install -e .
```

*Requirements: Python 3.10+, PyTorch 2.0+ (CUDA recommended for Triton kernel acceleration).*

---

## ⚡ The 4-Step Workflow (Unified CLI)

Kilat provides an all-in-one CLI tool (`kilat`) that covers your entire lifecycle from raw data to a deployable Hugging Face model:

### 1. Pre-Train from Scratch
```bash
# Train on CPU or consumer GPU using lightweight presets
kilat train --config configs/small_dense.yaml \
  --set training.max_steps=1000 \
  --set training.output_dir=./checkpoints/my_base_model
```

### 2. Supervised Fine-Tuning (SFT / Instruction Tuning)
Fine-tune your base model on Alpaca, ChatML, or JSONL instruction datasets with automatic prompt-loss masking:
```bash
kilat sft \
  --model ./checkpoints/my_base_model/checkpoint-best \
  --data ./my_instructions.jsonl \
  --output-dir ./checkpoints/my_sft_model \
  --epochs 3 \
  --batch-size 4 \
  --lr 3e-5
```

### 3. Generate & Test (Interactive Inference)
Generate text using Kilat's optimized incremental `LayerHybridCache`:
```bash
kilat generate \
  --checkpoint ./checkpoints/my_sft_model/checkpoint-best \
  --prompt "Below is an instruction that describes a task.\n\n### Instruction:\nWrite a story about a brave cat.\n\n### Response:\n" \
  --max-new-tokens 64 \
  --do-sample \
  --temperature 0.7
```

### 4. Export to Hugging Face
Convert your trained Kilat checkpoint into standard Hugging Face `model.safetensors` format with tokenizer and configuration:
```bash
kilat export-hf \
  --checkpoint ./checkpoints/my_sft_model/checkpoint-best \
  --output ./hf_export
```

---

## 🐍 Python API Quickstart

### 1. Minimal Pre-Training Script

```python
from kilat import (
    KilatConfig,
    KilatTrainer,
    KilatTransformer,
    TrainingArguments,
    build_dataloaders_from_config,
    MainConfig,
)

# Load lightweight preset (or build custom config in Python)
config = MainConfig.from_yaml("configs/debug_cpu.yaml")
model = KilatTransformer(config)
train_loader, eval_loader = build_dataloaders_from_config(config)

trainer = KilatTrainer(
    model=model,
    args=config.to_training_arguments(),
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=config.build_tokenizer(),
)
trainer.train()
```

### 2. Supervised Fine-Tuning (SFT) in Python

```python
from kilat.data.sft import SFTDataset, SFTDataCollator
from kilat.data.tokenizer import AutoTokenizer
from kilat.engine.sft_trainer import KilatSFTTrainer
from kilat.models.transformer import KilatTransformer
from kilat.engine.args import TrainingArguments

# Load pre-trained base model & tokenizer
model = KilatTransformer.from_pretrained("AiRukua/tiny-stories-kilat-35M")
tokenizer = AutoTokenizer.from_pretrained("AiRukua/tiny-stories-kilat-35M")

# Load instruction dataset (Alpaca or ChatML format)
train_dataset = SFTDataset("instructions.jsonl", tokenizer=tokenizer, max_length=512)
collator = SFTDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

args = TrainingArguments(output_dir="./sft_output", max_steps=100, learning_rate=3e-5)
trainer = KilatSFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)
trainer.train()
```

### 3. Text Generation with `LayerHybridCache`

```python
from kilat.inference.generator import TextGenerator
from kilat.models.transformer import KilatTransformer
from kilat.data.tokenizer import AutoTokenizer

model = KilatTransformer.from_pretrained("./sft_output/checkpoint-best")
tokenizer = AutoTokenizer.from_pretrained("./sft_output/checkpoint-best")

generator = TextGenerator(model, tokenizer)
output = generator.generate("Once upon a time", max_new_tokens=64, temperature=0.7)
print(output)
```

---

## 🎯 Ready-to-Run Hardware Presets (`configs/`)

Choose the preset tailored to your available hardware:

| Preset Config | Recommended Hardware | Parameters | Architecture Highlights |
| :--- | :--- | :---: | :--- |
| [`configs/debug_cpu.yaml`](configs/debug_cpu.yaml) | **Any Laptop CPU** | ~0.5M | Instant smoke test (< 5 seconds), in-memory synthetic data |
| [`configs/small_dense.yaml`](configs/small_dense.yaml) | **Colab T4 / RTX 4050/3060** (6–8 GB VRAM) | ~35M–85M | 12-layer Dense SwiGLU + Macro-Synergy (Decay + MLA) |
| [`configs/small_moe.yaml`](configs/small_moe.yaml) | **RTX 3060 / 4070** (8–16 GB VRAM) | ~120M | 10-layer DeepSeekMoE (8 experts, Top-2, `bias_free` balancing) |
| [`configs/standard_moe.yaml`](configs/standard_moe.yaml) | **Multi-GPU / FSDP** | 500M+ | 24-layer DeepSeekMoE with sequence packing & distributed sync |

---

## 🛡️ Zero-Wasted-Compute Guarantee (Assurance & Verification)

When training LLMs on limited hardware or free cloud tiers, **wasting compute is painful**. In standard training scripts, silent bugs often lurk unnoticed for hours:
- Gradients silently zeroed out or exploding in non-norm layers.
- Data leaking across sequence boundaries without proper causal masks.
- Resuming from checkpoints failing to restore exact RNG seeds and optimizer states.

Kilat solves this with its **Pre-Training Assurance Pipeline**:
```bash
# Run pre-flight verification before spending compute
kilat verify --config configs/small_dense.yaml
```
- **Theory & Init**: Guarantees unbiased initial cross-entropy ($|\mathcal{L}_0 - \ln V| \le 0.15$).
- **FP64 Analytical Gradcheck**: Numerically proves exact mathematical derivatives across custom Triton kernels.
- **Bitwise Parity**: Guarantees `0.0` bitwise checkpoint resumption parity so interrupted runs resume with 100% mathematical fidelity.
- **Active ML Chaos Saboteur (`kilat chaos`)**: Injects simulated ML bugs in-memory to prove the framework catches failures before you start training.

---

## 📁 Repository Structure

```text
Kilat/
├── configs/                 # Ready-to-use YAML configs (laptop, Colab, single-GPU, MoE)
├── notebooks/               # Interactive Colab notebooks (Pre-training & SFT)
├── src/kilat/
│   ├── cli.py               # Unified CLI (train, sft, generate, export-hf, verify, chaos)
│   ├── models/              # KilatTransformer, Macro-Synergy (Decay + Latent MLA), DeepSeekMoE
│   ├── engine/              # KilatTrainer, KilatSFTTrainer, Optimizer, Checkpointer, DDP
│   ├── data/                # SFTDataset, KilatDataCollator, AutoTokenizer, Memmap preprocessor
│   ├── inference/           # TextGenerator, LayerHybridCache, Hugging Face Exporter
│   ├── assurance/           # 6 Pre-flight verification gates, Chaos Saboteur, FlightRecorder
│   └── distillation/        # Knowledge distillation losses (Vanilla, Reverse, Adaptive)
├── tests/                   # 400+ unit, invariant, and integration tests
└── docs/                    # Mathematical derivations, architecture blueprints, and specs
```

---

## 🗺️ Roadmap

- [x] Sub-quadratic Macro-Synergy architecture (Global Decay + Latent MLA)
- [x] $6.4\times$ KV-Cache reduction with $O(N)$ Triton parallel scan
- [x] Supervised Fine-Tuning (SFT) engine with selective prompt masking
- [x] Unified CLI (`kilat train`, `kilat sft`, `kilat generate`, `kilat export-hf`)
- [x] Interactive Google Colab Notebooks (Pre-training & SFT)
- [x] Hugging Face Hub integration & model release ([`AiRukua/tiny-stories-kilat-35M`](https://huggingface.co/AiRukua/tiny-stories-kilat-35M))
- [x] Fine-grained DeepSeekMoE with loss-free dynamic bias balancing (`bias_free`)
- [x] 8-stage pre-training assurance pipeline and active chaos saboteur
- [ ] 4-bit / 8-bit Quantized SFT (QLoRA for Kilat architectures)
- [ ] Direct Preference Optimization (DPO / ORPO) alignment trainer
- [ ] Web-based interactive chat demo (Gradio / Streamlit)
- [ ] ONNX and WebGPU runtime export for in-browser local inference

---

## 📄 License & Citation

Kilat is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

```bibtex
@software{kilat2026,
  author = {Abdul Wahid Rukua},
  title  = {Kilat: The Lightweight Sub-Quadratic LLM Framework},
  year   = {2026},
  url    = {https://github.com/Airukua/Kilat}
}
```

---

## 👤 Author & Community

**Abdul Wahid Rukua** — AI/ML Engineer  
*Focus: Efficient Sequence Modeling, Lightweight LLMs, Indonesian NLP & Low-Resource Language Technologies.*

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdul--wahid--rukua-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdul-wahid-rukua/)
[![GitHub](https://img.shields.io/badge/GitHub-Airukua-181717?style=flat-square&logo=github)](https://github.com/Airukua)
