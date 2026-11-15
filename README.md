<div align="center">

# ⚡ Kilat

**The Modern Lightweight Sub-Quadratic LLM Framework**  
*Pre-Train, Fine-Tune (SFT), and Deploy Custom Language Models on Consumer GPUs & Google Colab*

[![Open In Colab: MoE Pre-Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_moe_pretraining_colab.ipynb)
[![Hugging Face: 150M-Kilat-SMALL-10B](https://img.shields.io/badge/🤗%20HuggingFace-150M--Kilat--SMALL--10B-FFD21E?style=flat-square)](https://huggingface.co/AiRukua/150M-Kilat-SMALL-10B)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Assurance & Chaos](https://img.shields.io/badge/Assurance-16%20Chaos%20Gates%20Passed-brightgreen?style=flat-square)](#-zero-wasted-compute-guarantee-assurance--chaos-engineering)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)

<p align="center">
  <img src="images/illustration.png" alt="Kilat illustration" width="720" />
</p>

</div>

---

## 🌟 What's New in Kilat v2.0?

- 🚀 **Flagship Foundation Model Released**: [**`AiRukua/150M-Kilat-SMALL-10B`**](https://huggingface.co/AiRukua/150M-Kilat-SMALL-10B) — Pre-trained from scratch on **10 Billion tokens of FineWeb-Edu**.
- 🧠 **Macro-Synergy Hybrid Topology**: Combines **Global Exponential Decay** (0-byte KV cache recurrent linear attention) + **Latent MLA** (4x compressed KV cache) + **DeepSeekMoE** (8 routed experts, Top-2 active + 1 shared expert).
- 🛡️ **Active ML Chaos Engineering**: Built-in non-destructive fault injection (`kilat chaos` with 8 core training scenarios `C-01..C-08` + 8 Hugging Face resilience scenarios `HFC-01..HFC-08`).
- ⚡ **Seamless Hugging Face Hub Integration**: Direct inference with `AutoModelForCausalLM.from_pretrained("AiRukua/150M-Kilat-SMALL-10B", trust_remote_code=True)`.

---

## 💡 Why Kilat? Train LLMs Without an Enterprise Supercomputer

Most existing LLM frameworks assume you have a cluster of 8x H100 GPUs. Trying to train or fine-tune models on consumer hardware (such as an RTX 3060/4060 laptop GPU, or a free Google Colab T4) usually leads to **`CUDA Out of Memory (OOM)`**.

**Kilat** (*Indonesian: lightning*) is engineered from the ground up for **maximum efficiency, numerical proof, and hardware accessibility**:
1. **$O(N)$ Global Exponential Decay Contextualizer (Even Layers)**: Hardware-accelerated recurrent linear scan. **Requires EXACTLY 0 BYTES of KV-Cache.**
2. **Latent Multi-Head Latent Attention (MLA) (Odd Layers)**: Low-rank joint key-value compression with Decoupled RoPE, dedicating 100% capacity to associative memory retrieval.
3. **Fine-Grained DeepSeekMoE**: 1 shared expert + 8 routed experts with loss-free dynamic bias balancing, giving 151M parameter capacity while computing only **~59.5M active parameters per token**.

| Metric | Standard Transformer | Kilat Macro-Synergy | Real-World Benefit |
| :--- | :---: | :---: | :--- |
| **KV-Cache Size ($N=2048$)** | $1,024\text{ KB}$ | **$160\text{ KB}$** | 🚀 **$6.4\times$ less VRAM** |
| **KV-Cache Size ($N=8192$)** | $4,096\text{ KB}$ | **$640\text{ KB}$** | 🚀 **$6.4\times$ less VRAM** |
| **Prefill Throughput ($N=512$)** | $269,000\text{ tok/s}$ | **$423,808\text{ tok/s}$** | ⚡ **$+59\%$ faster prefill** |
| **Associative Recall (MQAR)** | Stagnates | **$96.09\%$** | 🎯 **High retrieval accuracy** |
| **Peak VRAM ($N=4096$)** | ❌ OOM | ✅ **Pass (No OOM)** | Run long context on consumer GPUs |

---

## 🚀 Flagship Model: 150M - Kilat SMALL 10B

Our flagship foundation model is available on Hugging Face Hub: [**`AiRukua/150M-Kilat-SMALL-10B`**](https://huggingface.co/AiRukua/150M-Kilat-SMALL-10B).

| Specification | Value |
| :--- | :--- |
| **Total Parameters** | **150,996,480 (~151M)** |
| **Active Parameters / Token** | **59,510,784 (~59.5M)** |
| **Architecture** | Hybrid Global Decay + Latent MLA + DeepSeekMoE |
| **Layers / Heads / Width** | 12 Layers / 6 Heads / 384 Dim (Latent Dim 96) |
| **Experts Configuration** | 8 Routed Experts (Top-2 Active) + 1 Shared Expert |
| **Pre-Training Corpus** | **10 Billion Tokens** ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), single-pass) |
| **Context Window** | 2,048 tokens |
| **License** | Apache 2.0 (Commercial & Research friendly) |

---

## 💻 Quickstart: Generating with the Pre-Trained Model

### 1. Installation
```bash
pip install torch transformers accelerate huggingface_hub
```

### 2. Crucial Guide on Base Model Generation Settings
> ⚠️ **Base Model Note**: `150M-Kilat-SMALL-10B` is a **raw foundation model**, not an instruction/chat tuned assistant. It is designed to continue factual and educational sentences. Without repetition penalties, compact models can fall into n-gram loops.

Always use **tuned sampling parameters** (`temperature=0.75`, `top_p=0.92`, `repetition_penalty=1.18`, `no_repeat_ngram_size=3`):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AiRukua/150M-Kilat-SMALL-10B"

print(f"📥 Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Start with an educational or scientific opening phrase
prompt = "Photosynthesis is the process by which green plants"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Recommended sampling parameters for 150M base models
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.75,
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.18,
    no_repeat_ngram_size=3,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
)

print("\n--- Output ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Output Example:
> *"Photosynthesis is the process by which green plants convert sunlight into energy. The plant can use this energy to produce food, especially carbohydrates, as well as photosynthesis. However, while there are many other methods for converting light energy into chemical energy..."*

---

## 🚀 1-Click Interactive Google Colab Notebooks

You don't need any local GPU setup to get started. Open and run our end-to-end notebooks directly in Google Colab (compatible with free T4 GPU tier):

| Notebook | Description | Colab Link |
| :--- | :--- | :---: |
| **1. MoE Pre-Training (150M 10B)** | Pre-train the 150M DeepSeekMoE + Macro-Synergy architecture from scratch on FineWeb-Edu, resume checkpoints, and export to Hub. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_moe_pretraining_colab.ipynb) |
| **2. General Pre-Training** | Pre-train dense and hybrid sub-quadratic models, run 6-Gate Assurance, and export to Hugging Face Hub. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_general_pretraining_colab.ipynb) |
| **3. PEFT LoRA Fine-Tuning** | Adapt Kilat with Low-Rank Adaptation (`peft`), verify `PeftModel` across Gates `G0`–`G6`, and merge weights via `merge_and_unload()`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_peft_lora_example.ipynb) |
| **4. SFT Alignment** | Supervised fine-tuning with `-100` prompt loss masking and non-reentrant activation checkpointing. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Airukua/Kilat/blob/main/notebooks/kilat_sft_colab.ipynb) |

---

## 📦 Installation for Local Development

```bash
# Clone the repository
git clone https://github.com/Airukua/Kilat.git
cd Kilat

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

*Requirements: Python 3.10+, PyTorch 2.0+ (CUDA recommended for Triton kernel acceleration).*

---

## ⚡ The Unified CLI (`kilat`)

Kilat includes a complete, production-ready command line interface:

### 1. Pre-Train from Scratch
```bash
# Pre-train with DeepSeekMoE preset on single or multi-GPU
kilat train --config configs/small_moe.yaml \
  --set training.max_steps=10000 \
  --set training.output_dir=./checkpoints/my_moe_model
```

### 2. Supervised Fine-Tuning (SFT / Instruction Tuning)
Fine-tune base models on Alpaca, ChatML, or JSONL instruction datasets:
```bash
kilat sft \
  --model ./checkpoints/my_moe_model/checkpoint-best \
  --data ./my_instructions.jsonl \
  --output-dir ./checkpoints/my_sft_model \
  --epochs 3 \
  --batch-size 4 \
  --lr 3e-5
```

### 3. Export to Hugging Face
Convert any Kilat checkpoint into standard Hugging Face `model.safetensors` format with remote code adapters and optimal `generation_config.json`:
```bash
kilat export-hf \
  --checkpoint ./checkpoints/my_moe_model/checkpoint-best \
  --output ./checkpoints/my_moe_model_hf
```

---

## 🛡️ Zero-Wasted-Compute Guarantee: Assurance & Chaos Engineering

In ML systems, errors rarely raise explicit exceptions. Models can run smoothly and report decreasing loss while silently corrupting gradients, learning on leaky data, breaking RNG determinism, or dropping optimizer states.

Kilat solves this with an industrial-grade **Assurance & Active ML Chaos Engineering Pipeline**:

```bash
# 1. Run Pre-Flight Validation Gates (Theory, Gradients, Equivalence, Resumption)
kilat verify --config configs/small_moe.yaml

# 2. Run Active ML Chaos Saboteur (C-01 .. C-08)
kilat chaos -s all

# 3. Run Hugging Face Integration Chaos Suite (HFC-01 .. HFC-08)
kilat chaos -s hf
```

### Injected Chaos Scenarios:
- **`C-01..C-08`**: Silent autograd detachments, causal future leaks, cross-batch tensor reshapes, FP16 overflow storms, decay horizon singularities, MoE router collapse, poisoned batches, and mid-save preemption bit-rot.
- **`HFC-01..HFC-08`**: Hugging Face `DynamicCache` foreign injection resilience, bidirectional schema drift (`num_hidden_layers`, `hidden_size`, `num_attention_heads`), adversarial sampling parameters, asymmetric left-padding, PEFT LoRA gradient isolation, and multithreaded streaming cancellation.

---

## 📁 Repository Structure

```text
Kilat/
├── configs/                 # Production-ready YAML configs (laptop, Colab, MoE, multi-GPU)
├── notebooks/               # Interactive Colab notebooks (MoE 150M, SFT, PEFT LoRA)
├── src/kilat/
│   ├── cli.py               # Unified CLI (train, sft, generate, export-hf, verify, chaos)
│   ├── models/              # KilatTransformer, Macro-Synergy (Decay + Latent MLA), DeepSeekMoE
│   ├── engine/              # KilatTrainer, KilatSFTTrainer, Optimizer, Checkpointer, DDP
│   ├── data/                # SFTDataset, KilatDataCollator, AutoTokenizer, Memmap preprocessor
│   ├── inference/           # TextGenerator, LayerHybridCache, Hugging Face Exporter & Hub Templates
│   ├── integrations/        # Hugging Face ecosystem bridges (AutoModel, PEFT LoRA, TRL)
│   ├── assurance/           # 6 Pre-flight verification gates, FlightRecorder, Active Chaos Saboteur
│   └── distillation/        # Knowledge distillation losses (Vanilla, Reverse, Adaptive)
├── tests/                   # 400+ unit, invariant, and chaos integration tests
└── docs/                    # Mathematical derivations, architecture blueprints, and model cards
```

---

## 🤝 Contributing & Community

Contributions are very welcome! Whether you are implementing new attention kernels, optimizing memory efficiency, improving documentation, or adding chaos scenarios:

- Please read our [**Contributing Guide**](CONTRIBUTING.md) to understand our 5 Golden Invariants, code standards, and verification workflow.
- All participants must adhere to our [**Code of Conduct**](CODE_OF_CONDUCT.md).

---

## 📄 License & Citation

Kilat is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

> Kilat is built to be shared. Anyone is welcome to use, study, modify, and build upon it, including for commercial purposes. You don't owe me payment for using my work, but please respect the effort behind it. Preserve the original attribution, acknowledge the project, and never claim someone else's work as your own.

```bibtex
@misc{kilat2026,
  author = {Abdul Wahid Rukua},
  title  = {Kilat: Kernelized Lightweight Attention Transformer with Hybrid Macro-Synergy and Sparse MoE},
  year   = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Airukua/Kilat}}
}
```

---

## 👤 Author & Community

**Abdul Wahid Rukua** — AI/ML Engineer  
*Focus: Efficient Sequence Modeling, Lightweight LLMs, and Sub-Quadratic Architectures.*

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-AiRukua-FFD21E?style=flat-square)](https://huggingface.co/AiRukua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdul--wahid--rukua-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdul-wahid-rukua/)
[![GitHub](https://img.shields.io/badge/GitHub-Airukua-181717?style=flat-square&logo=github)](https://github.com/Airukua)
