---
license: apache-2.0
language:
  - en
tags:
  - kilat
  - moe
  - deepseek-moe
  - mixture-of-experts
  - mla
  - latent-attention
  - global-decay
  - sub-quadratic
  - linear-attention
  - fineweb-edu
  - base-model
  - pretrained
  - text-generation
datasets:
  - HuggingFaceFW/fineweb-edu
pipeline_tag: text-generation
library_name: transformers
inference: true
widget:
  - text: "Photosynthesis is the process by which green plants"
    example_title: "Biology: Photosynthesis"
  - text: "In computer science, an algorithm is defined as"
    example_title: "Computer Science: Algorithm"
  - text: "The fundamental difference between DNA and RNA is"
    example_title: "Genetics: DNA vs RNA"
  - text: "Newtonian mechanics principles state that an object"
    example_title: "Physics: Mechanics"
---

# ⚡ 150M - Kilat SMALL 10B

<div align="center">

**A 151M-Parameter Sub-Quadratic Hybrid Language Model with Latent MLA & DeepSeekMoE**  
*Pre-trained from scratch on 10 Billion tokens of FineWeb-Edu*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Architecture: Kilat v2.0](https://img.shields.io/badge/Architecture-Kilat%20Macro--Synergy-orange.svg)](#architecture-specifications)
[![Tokens: 10B FineWeb-Edu](https://img.shields.io/badge/Tokens-10B%20FineWeb--Edu-green.svg)](#pre-training-details)
[![Parameters: 151M Total / 59.5M Active](https://img.shields.io/badge/Parameters-151M%20Total%20%7C%2059.5M%20Active-purple.svg)](#model-specifications)

</div>

---

## 📌 Executive Summary

**Kilat SMALL 10B** is a compact, high-efficiency **Base Pre-Trained Language Model** built on the **Kilat v2.0 Macro-Synergy architecture**. It combines:
1. **Sub-Quadratic Global Exponential Decay (Recurrent Linear Attention)** for linear-time sequence processing and constant memory state.
2. **Latent Multi-Head Latent Attention (MLA)** for compressed key-value representations (inspired by DeepSeek-V2).
3. **Fine-Grained Mixture of Experts (DeepSeekMoE)** featuring 8 routed experts (Top-2 active) and 1 dedicated shared expert.

Trained from scratch on **10 Billion tokens** of high-quality educational web text ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), Kilat SMALL 10B delivers dense-model capacity (~151M parameters) while consuming compute equivalent to only **~59.5M active parameters per token**.

> ⚠️ **Important Notice on Base Models**:  
> This checkpoint is a **raw pre-trained foundation model**, **NOT an instruction-tuned or chat assistant**. It does not use conversational templates (`<|user|>`, `<|assistant|>`). It is designed for **continuous text completion** and serves as the ideal foundation for **Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, or **Domain Adaptation**.

---

## 💡 Crucial Guide: Getting Clean Generation from a Base Model

Because this is a compact 150M base language model, **generation output quality heavily depends on sampling settings**:

### ❌ What Causes "Jelly" or Repetitive Generation?
- **Pure Greedy Search (`do_sample=False, repetition_penalty=1.0`)**:  
  Without repetition penalties, compact models fall into recursive n-gram loops (e.g., *"The process of photosynthesis is the process by which..."*).
- **High Unclipped Temperature (`temperature >= 1.0, top_p=1.0`)**:  
  Without top-k/top-p truncation, small models sample low-probability tokens from the long tail, resulting in hallucinated syntax.
- **Prompting Like a Chatbot (`"Question: ... Answer:"` or `"User: ... Assistant:"`)**:  
  Base models have never seen conversational chat markers and will hallucinate dialog transcripts.

### ✅ Optimal Generation Hyperparameters
For coherent, factually grounded educational completions, always use:
- **`temperature`**: `0.70` – `0.75` (optimal balance between creativity and determinism)
- **`top_p` (Nucleus)**: `0.90` – `0.92`
- **`top_k`**: `50` (clips the erratic vocabulary tail)
- **`repetition_penalty`**: `1.15` – `1.20` (eliminates looping phrases)
- **`no_repeat_ngram_size`**: `3` (strictly forbids 3-gram repetitions)

---

## 🚀 Quickstart & Inference

### 1. Python Code Snippet (Transformers)

Ensure you have `transformers >= 4.45.0` installed:

```bash
pip install --upgrade transformers torch accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AiRukua/150M-Kilat-SMALL-10B"

print(f"📥 Loading model from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Base models excel at continuing factual, scientific, and educational sentences
prompts = [
    "Photosynthesis is the process by which green plants",
    "In computer science, an algorithm is defined as",
    "The fundamental difference between DNA and RNA is",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.75,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.18,
        no_repeat_ngram_size=3,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print(f"PROMPT : {prompt}")
    print(f"OUTPUT : {generated_text}")
```

### 2. Sample Output Comparisons

| Prompt | Naive Greedy (`rep_penalty=1.0`) | **Optimized Base Settings (Recommended)** |
| :--- | :--- | :--- |
| *Photosynthesis is the process by which green plants* | *"...convert sunlight into carbohydrates. The process is called photosynthesis. The process of photosynthesis is the process by which plants..."* (Endless loop) | *"convert sunlight into energy. The plant can use this energy to produce food, especially carbohydrates, as well as photosynthesis. However, while there are many other methods for converting light energy into chemical energy..."* |
| *The fundamental difference between DNA and RNA is* | *"...that DNA is DNA and RNA is RNA. The difference between DNA and RNA is that DNA is..."* | *"that the cell has to get information from itself, or its own genome. This means you have to create an mRNA copy of your gene in order to make a protein molecule that can be translated into proteins..."* |

---

## 🏛️ Architecture Specifications

Kilat implements a **Macro-Synergy Hybrid Topology** that decouples long-range semantic retention from local causal recall:

```
[Input Tokens] ──> [Word Embeddings (wte)] ──> [12x Macro-Synergy Blocks] ──> [RMSNorm (ln_f)] ──> [LM Head]
                                                      │
               ┌──────────────────────────────────────┴──────────────────────────────────────┐
               │                                                                             │
      [Odd Layers: Latent MLA]                                                     [Even Layers: Global Decay]
  Compressed Low-Rank Key/Value Projection                                    Recurrent Linear-Time State Matrix
  Decoupled RoPE Rotary Position Embeddings                                   Analytical Exponential Horizon σ(λ)
               │                                                                             │
               └──────────────────────────────────────┬──────────────────────────────────────┘
                                                      │
                                           [DeepSeekMoE Feed-Forward]
                                      1 Shared Expert (Global Knowledge)
                                    8 Routed Experts (Sparse Specialization)
                                          Top-2 Activation + Aux Loss
```

### Hyperparameter Breakdown

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Total Parameters ($\Phi_{\text{total}}$)** | **150,996,480 (~151M)** | Full weights stored in memory / safetensors |
| **Active Parameters ($\Phi_{\text{active}}$)** | **59,510,784 (~59.5M)** | Effective parameters computed per token |
| **Number of Layers ($L$)** | **12** | 6 Global Decay layers + 6 Latent MLA layers |
| **Hidden Dimension ($d_{\text{model}}$)** | **384** | Internal representation width |
| **Attention Heads ($H$)** | **6** | Head dimension $d_{\text{head}} = 64$ |
| **Latent MLA Dimension ($d_{\text{latent}}$)** | **96** | 4x compression ratio for KV state |
| **Decoupled RoPE Dimension** | **32** | Rotary position embeddings on queries & decoupled keys |
| **Total Routed Experts ($N_r$)** | **8** | Fine-grained SwiGLU expert feed-forward networks |
| **Active Routed Experts ($K_r$)** | **2 (Top-2)** | Sparse softmax-gated expert selection |
| **Shared Experts ($N_s$)** | **1** | Always-active dense feed-forward trunk |
| **Expert Intermediate Size** | **1,024** | $8/3 \times d_{\text{model}}$ SwiGLU expansion |
| **Context Length ($S_{\text{max}}$)** | **2,048** | Pre-training sequence window |
| **Vocabulary Size ($V$)** | **50,257** | Byte-level BPE tokenizer (GPT-2 vocabulary) |
| **Weight Tying** | **True** | `lm_head.weight` tied to `wte.weight` |

---

## 🔬 Mathematical Formulation

### 1. Global Exponential Decay Path (Even Layers)
The recurrent decay state $S_t \in \mathbb{R}^{H \times d_k \times d_v}$ evolves causally:
$$S_t = \lambda \odot S_{t-1} + K_t^T V_t, \quad \lambda = \sigma(\text{log\_lambda}) \in (0, 1)$$
$$O_t = \frac{Q_t S_t}{\sum_{\tau=1}^t \lambda^{t-\tau} + \epsilon}$$
This operates in $\mathcal{O}(S)$ linear time and constant $\mathcal{O}(1)$ memory per step during autoregressive decoding.

### 2. Compressed Latent Attention MLA (Odd Layers)
Queries, keys, and values are compressed into low-rank latent spaces:
$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_{\text{latent}}}, \quad K_t^C = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}$$
Queries and keys receive decoupled rotary position embeddings ($K_t^R = \text{RoPE}(W^{KR} h_t)$), allowing long-range position awareness without storing full-rank KV states.

### 3. DeepSeekMoE Sparse Routing
For input token representation $x$:
$$\text{MoE}(x) = \sum_{s=1}^{N_s} \text{FFN}_s^{\text{shared}}(x) + \sum_{i \in \text{Top-}K} g_i(x) \text{FFN}_i^{\text{routed}}(x)$$
where gate weights $g_i(x) = \operatorname{Softmax}(\text{Top-}K(x W_g))$ are balanced across training steps with load-balancing auxiliary loss $\mathcal{L}_{\text{aux}} = \alpha \cdot N_r \sum_{i=1}^{N_r} f_i P_i$.

---

## 📊 Pre-Training Details

- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (educational web dump filtered for high-quality scientific, mathematical, and explanatory prose).
- **Token Volume**: **10,000,000,000 tokens** (10 Billion tokens, single pass, zero duplication).
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay = $0.1$).
- **Learning Rate**: Peak $6.0 \times 10^{-4}$ with Warmup-Stable-Decay schedule.
- **Precision**: Mixed Precision BF16 with FP32 residual scaling and Master Weights.
- **Batch Size**: 512,000 tokens per global step ($B_{\text{batch}} = 256, S = 2048$).
- **Gradient Clipping**: Global L2 norm clipped at $1.0$.

---

## 🎯 Recommended Downstream Roadmap

1. **Supervised Fine-Tuning (SFT)**:  
   Use `trl.SFTTrainer` or Kilat's native `kilat sft` CLI with datasets like `Open-Orca`, `SlimOrca`, or Indonesian instruction corpora (`indo-instruct`).
2. **Continued Pretraining (CPT)**:  
   Expand multilingual knowledge by continuing pre-training on Indonesian Wikipedia, formal Indonesian news, and domain-specific code/STEM data.
3. **PEFT (LoRA)**:  
   Use `get_kilat_lora_config(r=16, target_modules=["gate", "up", "down", "q_proj", "v_proj_global"])` to adapt the model with $< 1\%$ additional parameters.

---

## ⚠️ Limitations & Ethical Considerations

- **Knowledge Boundary**: Knowledge is strictly derived from web crawls up to the pre-training cutoff.
- **Base Model Behavior**: May produce offensive, biased, or untruthful continuations if prompted with leading or controversial prefixes.
- **Hallucinations**: Like all compact language models, factual accuracy must be verified by external retrieval or human review.
- **Not a Chatbot**: Must be aligned via SFT/RLHF before deployment in conversational customer-facing environments.

---

## 📜 License & Citation

This model is licensed under the **Apache License 2.0**.

If you use Kilat or this model in your research or applications, please cite:

```bibtex
@misc{kilat2026,
  author = {AiRukua},
  title = {Kilat: Kernelized Lightweight Attention Transformer with Hybrid Macro-Synergy and Sparse MoE},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub repository},
  howpublished = {\url{https://huggingface.co/AiRukua/150M-Kilat-SMALL-10B}}
}
```
