# 01 — Agent Anti-Hallucination Contract & Ground-Truth Codebase Map (`v2.0.0`)

> **Document Purpose**: Eliminate false assumptions and hallucination traps when modifying Kilat's PyTorch/LLM codebase.  
> **Hard Rule**: Before editing any file in `src/` or `tests/`, every agent **must** verify file paths, class hierarchies, method signatures, and tensor contracts against this ground-truth map.

---

## 1. Ground-Truth Package & File Inventory (`src/kilat/`)

All legacy directories (`arc/`, `training/`, `pipeline/`, `distiliation/`, `sample_config/`, `experiments/`) have been permanently removed. **Never recreate them or import from them.**

| Canonical File Path | Primary Classes / Functions | Critical Ground-Truth Notes |
| :--- | :--- | :--- |
| `src/kilat/__init__.py` | `KilatTransformer`, `MainConfig`, `KilatConfig`, `KilatTrainer`, `TrainingArguments`, `DistributedContext`, `PretrainingDataset`, `build_train_dataloader`, `build_eval_dataloader`, `build_dataloaders_from_config`, `KilatDataCollator`, `GenerationConfig`, `TextGenerator` | Top-level public API (`__version__ = "2.0.0"`). |
| `src/kilat/cli.py` | `main`, `build_parser` (`verify`, `chaos`, `train`, `prepare-data`, `generate`, `export-hf`) | Unified console script (`kilat = "kilat.cli:main"`). |
| `src/kilat/models/transformer.py` | `KilatTransformer(BaseModel, GenerationMixin)`, `KilatModelOutput` | Computes internal causal shift on `logits[:, :-1, :]` and `labels[:, 1:]`; separates `ce_loss` from `aux_loss` in `model._last_loss_breakdown`; scales residual projections by $1/\sqrt{2L}$. |
| `src/kilat/models/blocks.py` | `RMSNorm(nn.Module)`, `Block(nn.Module)` | `RMSNorm` upcasts variance to `float32` (preserving `float64` for `gradcheck`); `Block` accepts a full `config: KilatConfig` (or legacy kwargs). |
| `src/kilat/models/attention.py` | `KilatAttention`, `KilatAttentionRoPE`, `LayerHybridCache` | Splits heads into `n_global_heads` ($O(N)$ exponential decay) and `n_recall_heads` (low-rank Latent MLA + optional Decoupled RoPE). |
| `src/kilat/models/rope.py` | `RotaryPositionalEmbedding`, `RotaryEmbedding`, `apply_rotary_pos_emb`, `apply_rotary_pos_emb_single` | Rotary positional transformations for standard and decoupled MLA heads. |
| `src/kilat/models/kernels/flash_decay.py` | `HAS_TRITON`, `FlashDecayFunction(torch.autograd.Function)`, `triton_global_decay`, `compute_decay_normalizer_from_logit`, `_reference_global_decay_fp64` | Custom $O(N)$ forward & reverse-scan backward autograd function with $\varphi_1(w)$ Taylor guard as $\lambda \to 1^-$. |
| `src/kilat/models/ffn.py` | `SwiGLU`, `DeepSeekMoE`, `FeedForward` | Supports `num_shared_experts >= 0`, $1/K_{\text{eff}}$ normalized load balance, Top-1 gradient preservation, FP32 router logits, and DeepSeek-V3 `bias_free` routing. |
| `src/kilat/configs/base.py` | `load_yaml_file`, `dump_yaml_file` | YAML I/O helpers only (**not** `BaseConfig`). |
| `src/kilat/utils/base_model.py` | `BaseConfig`, `BaseModel(nn.Module)` | `BaseConfig` and `BaseModel` live here (**not** `transformers.PretrainedConfig`). |
| `src/kilat/configs/model_config.py` | `KilatConfig(BaseConfig)` | Regular Python class inheriting `BaseConfig`. **NOT a `@dataclass`**. |
| `src/kilat/configs/main_config.py` | `MainConfig` | Single-source YAML config container with `.to_dict()`, `.from_dict()`, `.to_training_arguments()`, and `.apply_overrides()`. **NOT a `@dataclass`**. |
| `src/kilat/configs/training_config.py` | `TrainingConfig` | Regular Python class with `.to_training_arguments()`. **NOT a `@dataclass`**. |
| `src/kilat/configs/dataloader_config.py` | `DataLoaderConfig` | Supports `dataset_format: "synthetic" | "memmap" | "parquet" | "jsonl"`. **NOT a `@dataclass`**. |
| `src/kilat/configs/tokenizer_config.py` | `TokenizerConfig` | Supports `auto`, `gpt2`, `sentencepiece`, and offline `custom` (`ByteLevelDebugTokenizer`). **NOT a `@dataclass`**. |
| `src/kilat/data/dataset.py` | `PretrainingDataset`, `StreamingDataset`, `ConcatDataset`, `SyntheticTokenDataset`, `KilatDataset` | Deterministic synthetic dataset (`SyntheticTokenDataset`) + memmap/Parquet/JSONL datasets. |
| `src/kilat/data/collator.py` | `KilatDataCollator` | Zero-copy tensor fast path for uniform sequences; supports explicit `padding_side="right" | "left"` and `-100` pad masking. |
| `src/kilat/data/dataloader.py` | `build_train_dataloader`, `build_eval_dataloader`, `build_dataloaders_from_config` | Hardware-aware `pin_memory` and DDP `DistributedSampler` wiring. |
| `src/kilat/data/tokenizer.py` | `TokenizerWrapper`, `AutoTokenizer`, `ByteLevelDebugTokenizer`, `build_tokenizer`, `build_byte_tokenizer` | Offline byte-level debug tokenizer + HF/SentencePiece wrappers. |
| `src/kilat/data/preprocess.py` | `prepare_memmap_dataset`, `parquet_to_memmap`, `parquet_to_memmap_sharded` | Converts Parquet, JSONL/JSON, or TXT into `.bin`/`.npy` memmaps + `.meta.json` with SHA-256 digest. |
| `src/kilat/engine/args.py` | `TrainingArguments`, `PrecisionMode`, `TrainingMode` | Regular Python class with `.to_dict()` and `.from_dict()`. **NOT a `@dataclass`**. |
| `src/kilat/engine/trainer.py` | `KilatTrainer` | Core training & evaluation engine with exact partial accumulation scaling, pure token-weighted NLL evaluation, and DDP collectives. |
| `src/kilat/engine/distributed.py` | `DistributedContext`, `is_distributed_initialized`, `get_global_rank`, `get_local_rank`, `get_world_size`, `is_main_process`, `barrier`, `all_reduce_sum` | Unified DDP state and collective primitives. |
| `src/kilat/engine/checkpoint.py` | `save_checkpoint`, `load_checkpoint`, `verify_checkpoint_manifest`, `capture_rng_state`, `restore_rng_state`, `compute_total_steps`, `set_seed` | Atomic rank-0 checkpointing with `dist.barrier()`, full RNG states (`python`, `numpy`, `torch`, `cuda`), and `checkpoint_manifest.json` SHA-256 verification. |
| `src/kilat/engine/optimizer.py` | `create_optimizer`, `create_scheduler`, `compute_total_steps` | Excludes all `ndim < 2`, `RMSNorm`, `LayerNorm`, `.bias`, and `log_lambda` parameters from `weight_decay`. |
| `src/kilat/engine/scheduler.py` | `SchedulerType`, `get_scheduler` (`cosine`, `cosine_with_min_lr`, `wsdlr`, `rex`, `linear`, `constant`, `polynomial`, `inverse_sqrt`) | Learning rate schedule implementations. |
| `src/kilat/engine/callbacks.py` | `TrainerState` (`@dataclass`), `TrainerControl` (`@dataclass`), `TrainerCallback`, `CallbackHandler`, `EarlyStoppingCallback`, `LoggingCallback`, `ProgressCallback` | Callback lifecycle primitives. |
| `src/kilat/engine/integrations.py` | `ConsoleCallback`, `WandbCallback`, `TensorBoardCallback`, `MLflowCallback`, `CometCallback`, `get_reporting_integration_callbacks` | External experiment tracking integrations. |
| `src/kilat/inference/` | `generator.py` (`TextGenerator`), `generation_mixin.py` (`GenerationMixin`), `generation_config.py` (`GenerationConfig`), `beam_search.py`, `logit_processor.py`, `sampler.py`, `stopping_criteria.py`, `export_hf.py` (`convert_checkpoint_to_hf`) | Complete inference, sampling, stopping criteria, and Hugging Face export suite. |
| `src/kilat/distillation/` | `losses.py` (`VanillaKDLoss`, `ReverseKDLoss`, `AdaptiveKDLoss`, `DistillLossOutput`), `student.py` (`BaseStudent`), `teacher.py` (`BaseTeacher`) | Knowledge distillation module. |
| `src/kilat/assurance/` | `runner.py`, `scaling.py`, `flight_recorder.py`, `types.py`, `gates/gate1..6_*.py`, `chaos/{saboteur,scenarios,runner}.py` | 8-stage Pre-Training Assurance Gates, Active Chaos `Saboteur`, and runtime `FlightRecorderCallback`. |

---

## 2. The 12 Forbidden Agent Hallucination Traps

1. **False `@dataclass` Assumption on Configs**: `KilatConfig`, `MainConfig`, `TrainingConfig`, `DataLoaderConfig`, `TokenizerConfig`, and `TrainingArguments` are **NOT** `@dataclass` classes. Always use `.to_dict()` and `.from_dict()`, never `dataclasses.asdict()`.
2. **CPU-Only Testing Illusion on `@triton.jit` Kernels**: `triton_global_decay` must always route through `FlashDecayFunction(torch.autograd.Function)` so both CPU and CUDA preserve exact analytical gradients for `V` and `log_lambda`.
3. **Parameter Tunneling & `num_shared_experts == 0`**: Always pass `config: KilatConfig` into `Block` and `FeedForward`. `DeepSeekMoE` must allow `num_shared_experts >= 0` (`validate_non_negative_int`).
4. **Double Causal Label Shifting**: `KilatTransformer.forward` shifts `logits[..., :-1, :]` and `labels[..., 1:]` internally once. `input_ids` and `labels` fed to `model(input_ids, labels=labels)` must have identical aligned shape `(B, S)`. Valid target count is `(labels[:, 1:] != -100).sum()`.
5. **MoE $1/K_{\text{eff}}$ Scaling & Top-1 Router Gradient Death**: Divide expert selection frequency by $K_{\text{eff}} = \text{active\_experts} \times \text{fine\_grained\_factor}$, compute router logits in `float32`, and skip Top-K weight renormalization when $K_{\text{eff}} == 1$.
6. **Host-Device CUDA Stream Syncs Inside `nn.Module.forward()`**: Never call `.item()`, `.all()`, or `.any()` inside `forward()` during normal training (`debug_nan_checks=False`).
7. **Decode Normalization Blowup When `recall_ratio == 0.0`**: Always read past sequence length from `LayerHybridCache.seq_len`, never from `latent_kv.shape[1]` (which is `0` when `n_recall_heads == 0`).
8. **Partial Gradient Accumulation Under-Scaling & `floor` Step Mismatch**: Scale micro-batch loss by the actual window size $m_w = \min(G, B_{\text{total}} - \text{start})$ and compute epoch steps via `math.ceil(len(dataloader) / grad_acc)`.
9. **Perplexity Contaminated by `aux_loss`**: Language perplexity must be computed strictly from token-weighted pure cross-entropy loss (`exp(ce_loss)`), never `exp(ce_loss + aux_loss)`.
10. **DDP Split-Brain & Checkpoint Race Conditions**: Initialize `state.is_world_process_zero = (get_global_rank() == 0)`, `all_reduce` evaluation accumulators `[sum_nll, valid_tokens, sum_aux]`, call `sampler.set_epoch(epoch)`, and bracket Rank-0 atomic saves with `dist.barrier()`.
11. **Recreating Retired Legacy Folders**: Never recreate `src/kilat/arc/`, `src/kilat/training/`, `src/kilat/pipeline/`, `src/kilat/distiliation/`, `sample_config/`, or `experiments/`.
12. **Mocking Core PyTorch Mechanics in Invariant Tests**: Only external network/reporting I/O may be mocked; model forward/backward, optimizers, schedulers, collators, and checkpoints must be tested on real PyTorch tensors.
