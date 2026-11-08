# Kilat Configuration Presets (`configs/`)

Every experiment in Kilat is driven by a single validated `MainConfig` YAML file combining five sections: `experiment`, `model`, `tokenizer`, `dataloader`, and `training`.

## Available Tiered Presets

| Preset | Target Hardware | Architecture | Dataset Mode | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| [`debug_cpu.yaml`](./debug_cpu.yaml) | Any CPU / Laptop (`< 5s`) | 2-layer Hybrid MLA/Decay + 4-expert MoE (`d=64`) | `synthetic` (offline byte tokenizer) | Instant onboarding, CI smoke tests, and `kilat verify --full` |
| [`small_dense.yaml`](./small_dense.yaml) | 1x GPU (`8–16 GB VRAM`) | 12-layer Dense SwiGLU + Hybrid MLA/Decay + RoPE (`d=768`) | `memmap` (`.npy` / `.bin`) | Dense baseline pre-training & scaling comparisons |
| [`small_moe.yaml`](./small_moe.yaml) | 1x GPU (`8–16 GB VRAM`) | 10-layer DeepSeekMoE (8 routed, Top-2, 1 shared, `bias_free`) | `memmap` (`.npy` / `.bin`) | Single-GPU sparse MoE research & ablation (~460M params) |
| [`standard_moe.yaml`](./standard_moe.yaml) | Multi-Node FSDP / EP (`BF16`) | 24-layer DeepSeekMoE (64 routed, Top-8, 2 shared, `d=2048`) | `memmap` with sequence packing | 54.2B parameter MoE architecture (FSDP / EP recipe target) |

## Running & Overriding Configs via CLI

Any nested field can be overridden at launch time using `--override section.key=value` without duplicating YAML files:

```bash
# 1. Run pre-flight mathematical & system assurance gates
kilat verify --config configs/debug_cpu.yaml --full

# 2. Launch single-GPU MoE training with dot-list overrides
kilat train --config configs/small_moe.yaml \
  --override training.learning_rate=2e-4 \
  --override model.recall_ratio=0.25 \
  --override training.precision=fp16

# 3. Launch multi-GPU DDP training
torchrun --nproc_per_node=8 -m kilat.cli train --config configs/standard_moe.yaml
```

## Key Hyperparameter Reference

- **`model.recall_ratio`** (`0.0` .. `1.0`): Fraction of attention heads allocated to low-rank **Latent MLA** (`1 - recall_ratio` use $O(N)$ **Global Exponential Decay**).
- **`model.ffn_mode`**: `"dense"` (Gated SwiGLU; `"swiglu"` supported as alias) or `"moe"` (`DeepSeekMoE`).
- **`model.router_balance_mode`**: `"aux_loss"` (differentiable $1/K_{\text{eff}}$ load-balancing penalty) or `"bias_free"` (DeepSeek-V3 loss-free dynamic bias updates without gradient interference).
- **`training.scheduler_type`**: `"wsdlr"` (Warmup-Stable-Decay), `"cosine"`, `"cosine_with_min_lr"`, `"linear"`, `"constant"`, or `"polynomial"`.
- **`training.atomic_checkpoint`**: When `true`, writes checkpoints to a temporary directory on Rank 0 alongside a `checkpoint_manifest.json` SHA-256 digest before atomic rename.
