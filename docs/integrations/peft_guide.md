# Kilat PEFT LoRA Fine-Tuning & Evaluation Guide (Phase P1)

This guide explains how to fine-tune Kilat hybrid models using Hugging Face **PEFT (Parameter-Efficient Fine-Tuning)** with Low-Rank Adaptation (**LoRA**), and how to evaluate exported models using `lm-evaluation-harness`.

---

## 1. Architectural Considerations for Kilat Hybrid Models

Kilat's Macro-Synergy architecture interleaves two distinct sequence-mixing mechanisms:
- **Global Exponential Decay ($O(N)$ Recurrence)**: Uses `v_proj_global` and `o_proj_global` linear projections, paired with 1D recurrent decay rates (`log_lambda`).
- **Low-Rank Latent MLA (Recall)**: Uses `q_proj` (or `q_a_proj`, `q_b_proj`), `kv_a_proj`, `kv_b_proj`, and `o_proj_recall` linear projections.
- **SwiGLU / DeepSeekMoE FFN**: Uses `gate`, `up`, and `down` linear projections.

### Golden Invariant: Never Apply LoRA to `log_lambda`
Recurrent decay logits (`log_lambda`) govern the exponential memory horizon of the causal recurrence. They are 1D vectors (not linear weight matrices) and must **never** be targeted by low-rank adapters. Kilat's canonical helper `get_kilat_lora_config()` automatically enforces this invariant.

---

## 2. Quickstart: PEFT LoRA Fine-Tuning

```python
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from kilat.integrations.huggingface import get_kilat_lora_config, register_kilat

# 1. Register Kilat with Hugging Face AutoClasses (or load from an export-hf folder)
register_kilat()

model_path = "./checkpoints/hf_export"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. Build the verified Kilat LoRA configuration
lora_config = get_kilat_lora_config(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

# 3. Wrap model & inspect trainable parameter ratio
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 4. Save & merge adapter weights after training
peft_model.save_pretrained("./checkpoints/kilat_lora_adapter")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./checkpoints/kilat_merged_hf")
```

### Canonical Target Modules

| Subsystem | Module Names | Description |
| :--- | :--- | :--- |
| **SwiGLU / MoE FFN** | `gate`, `up`, `down` | Feed-forward gating, up-projection, and down-projection |
| **Latent MLA** | `q_proj`, `q_a_proj`, `q_b_proj`, `kv_a_proj`, `kv_b_proj`, `o_proj_recall` | Query and low-rank Key-Value latent projections |
| **Global Decay** | `v_proj_global`, `o_proj_global` | Value and output projections on the $O(N)$ decay path |

---

## 3. Standard Benchmark Evaluation (`lm-evaluation-harness`)

Once a Kilat checkpoint (or merged LoRA checkpoint) has been exported via `kilat export-hf`, evaluate it on standard NLP benchmarks using EleutherAI's `lm-evaluation-harness`:

```bash
# Export native checkpoint to standalone Hugging Face format
kilat export-hf \
    --checkpoint ./checkpoints/checkpoint-final \
    --output ./checkpoints/hf_export

# Run zero-shot evaluation via lm_eval
bash experiments/benchmarks/lm_eval_run.sh ./checkpoints/hf_export
```
