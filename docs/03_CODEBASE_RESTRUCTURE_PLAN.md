# 03 — Modular Codebase Architecture, Configuration System, Data Pipeline, & CLI Reference (`v2.0.0`)

> **Document Purpose**: Reference specification for Kilat's modular package layout, single-source-of-truth configuration system, zero-copy memory-mapped data pipeline, DDP & atomic SHA-256 checkpointing, and unified `kilat` CLI.

---

## 1. Architectural Design Principles From Frontier LLM Codebases

| Reference Codebase | Core Pattern Adopted in Kilat | Engineering Benefit |
| :--- | :--- | :--- |
| **`pytorch/torchtitan`** | Strict separation between pure mathematical model definitions (`kilat.models`) and runtime engine/parallelism (`kilat.engine`), passing a unified `KilatConfig` object into every `Block`. | Eliminates parameter tunneling and guarantees 100% config fidelity across attention and MoE layers. |
| **`facebookresearch/lingua`** | Flat, stub-free package hierarchy combined with non-intrusive runtime telemetry (`FlightRecorderCallback`) outside `nn.Module.forward()`. | Zero 0-byte stub files and zero CUDA stream synchronization stalls during forward passes. |
| **`Lightning-AI/litgpt`** | Unified `kilat` CLI entrypoint (`verify`, `chaos`, `train`, `prepare-data`, `generate`, `export-hf`) paired with tiered `configs/*.yaml` starting from an instant offline CPU config (`debug_cpu.yaml`). | New users can verify and train Kilat out-of-the-box in $< 5$ seconds on any machine. |
| **`allenai/OLMo-core`** | Zero-copy tensor collation in `KilatDataCollator` and deterministic atomic checkpoints with full RNG state + SHA-256 manifest verification. | High dataloader throughput and `0.0` bitwise resumption parity. |
| **`deepseek-ai/DeepSeek-V3`** | FP32 router computation, separation of pure language NLL from MoE auxiliary loss, and loss-free dynamic bias load balancing (`router_balance_mode="bias_free"`). | Prevents auxiliary loss from degrading language modeling perplexity. |

---

## 2. Canonical Package Architecture (`src/kilat/`)

```text
Kilat/
├── configs/
│   ├── README.md                      # Preset guide & CLI override documentation
│   ├── debug_cpu.yaml                 # Tier-0: Offline CPU smoke & assurance config (<5s)
│   ├── small_dense.yaml               # Tier-1: Single-GPU Dense SwiGLU + Hybrid MLA/Decay + RoPE
│   ├── small_moe.yaml                 # Tier-2: Single-GPU DeepSeekMoE (bias_free) + Hybrid MLA
│   └── standard_moe.yaml              # Tier-3: Multi-GPU DDP 64-expert DeepSeekMoE
├── src/kilat/
│   ├── __init__.py                    # Top-level exports (__version__ = "2.0.0")
│   ├── cli.py                         # Unified CLI: `kilat <subcommand>`
│   ├── configs/
│   │   ├── base.py                    # YAML load/dump helpers
│   │   ├── model_config.py            # KilatConfig
│   │   ├── training_config.py         # TrainingConfig (+ .to_training_arguments())
│   │   ├── dataloader_config.py       # DataLoaderConfig
│   │   ├── tokenizer_config.py        # TokenizerConfig
│   │   └── main_config.py             # MainConfig (+ .apply_overrides())
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py             # KilatTransformer, KilatModelOutput
│   │   ├── blocks.py                  # Block(config=KilatConfig), RMSNorm
│   │   ├── attention.py               # KilatAttention, KilatAttentionRoPE, LayerHybridCache
│   │   ├── ffn.py                     # SwiGLU, DeepSeekMoE, FeedForward
│   │   ├── rope.py                    # RotaryPositionalEmbedding, apply_rotary_pos_emb*
│   │   └── kernels/
│   │       ├── __init__.py
│   │       └── flash_decay.py         # FlashDecayFunction, triton_global_decay, _reference_global_decay_fp64
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # KilatTrainer
│   │   ├── args.py                    # TrainingArguments (+ .to_dict() / .from_dict())
│   │   ├── distributed.py             # DistributedContext (rank, world_size, barrier, all_reduce)
│   │   ├── checkpoint.py              # Atomic rank-0 save, full RNG state, SHA-256 manifest, load
│   │   ├── optimizer.py               # create_optimizer (safe ndim < 2 & norm no_decay)
│   │   ├── scheduler.py               # SchedulerType, get_scheduler, WSDLR, REX, etc.
│   │   ├── callbacks.py               # TrainerState, TrainerControl, EarlyStoppingCallback, etc.
│   │   └── integrations.py            # Console, WandB, TensorBoard, MLflow, Comet callbacks
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PretrainingDataset, StreamingDataset, SyntheticTokenDataset
│   │   ├── collator.py                # KilatDataCollator (zero-copy tensor fast path)
│   │   ├── dataloader.py              # build_train_dataloader, build_eval_dataloader, build_dataloaders_from_config
│   │   ├── tokenizer.py               # TokenizerWrapper, AutoTokenizer, ByteLevelDebugTokenizer
│   │   └── preprocess.py              # prepare_memmap_dataset, parquet_to_memmap
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py               # TextGenerator
│   │   ├── generation_mixin.py        # GenerationMixin
│   │   ├── generation_config.py       # GenerationConfig
│   │   ├── beam_search.py             # BeamSearchScorer, BeamHypotheses
│   │   ├── logit_processor.py         # LogitsProcessorList & warpers
│   │   ├── sampler.py                 # Greedy, Multinomial, TopK, TopP, Typical, Contrastive, Mirostat
│   │   ├── stopping_criteria.py       # StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria, etc.
│   │   └── export_hf.py               # convert_checkpoint_to_hf
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── losses.py                  # VanillaKDLoss, ReverseKDLoss, AdaptiveKDLoss
│   │   ├── student.py                 # BaseStudent
│   │   └── teacher.py                 # BaseTeacher
│   ├── assurance/
│   │   ├── __init__.py
│   │   ├── runner.py                  # 8-stage verification orchestrator + report generator
│   │   ├── scaling.py                 # Stage 7 (muP + WSDLR scaling) & Stage 8 readiness sign-off
│   │   ├── flight_recorder.py         # FlightRecorderCallback (telemetry + circuit breaker)
│   │   ├── gates/                     # gate1_theory .. gate6_system
│   │   └── chaos/                     # Saboteur (C-01..C-08) & chaos suite runner
│   └── utils/
│       ├── __init__.py
│       ├── banner.py                  # CLI startup banner
│       ├── base_model.py              # BaseConfig, BaseModel
│       ├── report.py                  # Parameter counting & architecture inspection
│       └── validators.py              # Input & configuration validators
```

---

## 3. Configuration System & CLI Overrides

- **Single Source of Truth (`MainConfig`)**: Loads `experiment`, `model`, `tokenizer`, `dataloader`, and `training` sections from YAML and validates cross-component constraints (such as `model.vocab_size` alignment and head divisibility).
- **Direct Conversion to Runtime Arguments**: `config.to_training_arguments()` constructs a validated `TrainingArguments` instance directly from `MainConfig` without manual field copying.
- **Validated Dot-List Overrides (`MainConfig.apply_overrides`)**:
  ```bash
  kilat train --config configs/small_moe.yaml \
    --override training.learning_rate=1e-4 \
    --override model.router_balance_mode=bias_free
  ```
  Unknown section or field names immediately raise a `KeyError` to prevent silent flag typos.

---

## 4. Zero-Copy Data Pipeline & Memmap Preprocessing

1. **Offline Corpus Tokenization (`kilat prepare-data` / `prepare_memmap_dataset`)**:
   Streams Parquet, JSONL/JSON, or raw TXT files into contiguous `.bin` or `.npy` memory-mapped token arrays alongside a `<output>.meta.json` manifest recording `dtype`, `num_tokens`, `tokenizer`, and the file's SHA-256 checksum.
2. **Zero-Copy Collation (`KilatDataCollator`)**:
   Stacks uniform-length `torch.Tensor` or `np.ndarray` slices directly via `torch.stack` without intermediate Python `.tolist()` conversions, and masks all padding positions (`attention_mask == 0`) with `labels = -100`.
3. **One-Call DataLoader Builder (`build_dataloaders_from_config`)**:
   Automatically constructs `(train_loader, eval_loader)` from `MainConfig` across `synthetic`, `memmap`, `parquet`, and `jsonl` formats with hardware-guarded `pin_memory`.

---

## 5. DDP Hardening & Atomic SHA-256 Checkpointing

- **DDP Collective Safety (`kilat.engine.distributed`)**:
  - `self.state.is_world_process_zero` is initialized from `get_global_rank() == 0`.
  - `_default_eval_loop` performs `all_reduce_sum` on `[sum_nll, valid_tokens, sum_aux]` across all ranks so `eval_loss`, `eval_ppl`, and `EarlyStoppingCallback` decisions are identical on every GPU.
  - `_run_epoch` calls `sampler.set_epoch(epoch)` at every epoch boundary.
- **Bitwise-Exact Atomic Checkpointing (`kilat.engine.checkpoint`)**:
  - Rank 0 writes `pytorch_model.bin`, `config.json`, `training_args.json`, `training_state.pt` (including `python`, `numpy`, `torch`, and `cuda` RNG states), and `checkpoint_manifest.json` (per-file SHA-256 checksums + byte sizes) into a temporary directory `.tmp_checkpoint_*`, then atomically renames it to `checkpoint-<step>` bracketed by `dist.barrier()`.
  - `load_checkpoint` verifies every file against `checkpoint_manifest.json` before loading weights and strips `module.` / `_orig_mod.` wrappers automatically.

---

## 6. Unified `kilat` CLI Subcommands (`src/kilat/cli.py`)

| Subcommand | Example Invocation | Description |
| :--- | :--- | :--- |
| `kilat verify` | `kilat verify --config configs/debug_cpu.yaml --full` | Runs Gates 1–6 (and Stages 7–8 with `--full`) and emits signed `assurance_report.{json,md}`. |
| `kilat chaos` | `kilat chaos --scenarios all` | Injects all 8 non-destructive `Saboteur` ML faults (`C-01`..`C-08`) and verifies 100% detection. |
| `kilat prepare-data` | `kilat prepare-data --input data.jsonl --output train.bin --tokenizer gpt2` | Converts Parquet/JSONL/TXT into a memory-mapped token dataset + `.meta.json`. |
| `kilat train` | `kilat train --config configs/small_moe.yaml --override training.max_steps=1000` | Runs pre-flight Gate 1 + `KilatTrainer` with `FlightRecorderCallback` telemetry. |
| `kilat generate` | `kilat generate --checkpoint ./checkpoints/small_moe/checkpoint-best --prompt "Hello"` | Generates text from a saved checkpoint with configurable sampling strategies. |
| `kilat export-hf` | `kilat export-hf --checkpoint ./checkpoints/small_moe/checkpoint-best --output ./hf_model` | Exports a Kilat checkpoint to Hugging Face format (`safetensors` / `pytorch_model.bin`). |
