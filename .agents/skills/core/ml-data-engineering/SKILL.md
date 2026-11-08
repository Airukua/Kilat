---
name: ml-data-engineering
description: Machine Learning Data Pipeline Engineering for Kilat. Covers dataset abstractions, sequence packing, causal label shifting, loss masking, memmap storage, pin_memory hardware awareness, and data leakage detection.
---

# ML Data Engineering Skill

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Dataset loaders (`PretrainingDataset`, `StreamingDataset`, `KilatDataset`).
- Tokenizer integration, vocabulary mapping, or special token handling (`<|endoftext|>`, `<pad>`, `<s>`).
- Data collation (`KilatDataCollator`), batching, sequence padding, and truncation.
- Sequence packing (concatenating multiple short sequences to maximize context utilization).
- Causal label creation and loss masking.
- Data ingestion pipelines (Parquet to memmap binary format).
- Distributed data sharding and sampling (`DistributedSampler`).

---

## 2. Core Engineering Invariants

### 2.1 Causal Target Alignment & Masking
- **Strict One-Token Shift**: In autoregressive causal language modeling, the input sequence predicts the subsequent token:
  $$\text{input\_ids} = x_{0}, x_{1}, \dots, x_{T-1}$$
  $$\text{labels} = x_{1}, x_{2}, \dots, x_{T}$$
- **Loss Masking**: Any padding tokens, prompt tokens (in instruction tuning), or masked spans must be marked with `ignore_index = -100`:
  ```python
  labels[input_ids == pad_token_id] = -100
  ```
- **Anti-Pattern**: Using `labels = input_ids` without shifting when the model head does not internally shift. In `KilatTransformer`, confirm whether the model shifts internally or expects pre-shifted labels before feeding into `CrossEntropyLoss`.

### 2.2 Sequence Packing & Boundary Leakage
- **Document Boundaries**: When packing multiple documents into a single context window (e.g., 2048 tokens):
  - Insert an EOS token (`<|endoftext|>`) between independent documents.
  - Ensure the attention mechanism or cumulative position IDs prevent attention leakage across distinct document boundaries.
- **Leakage Invariant Test**: Verify that token generation for Document B is mathematically invariant to the contents of preceding Document A.

### 2.3 Hardware-Aware DataLoader Mechanics
- **Pin Memory Guard**:
  ```python
  pin_memory = torch.cuda.is_available() and device.type == "cuda"
  ```
  Setting `pin_memory=True` on CPU-only environments generates PyTorch `UserWarning`s and degrades paging performance.
- **Worker Configuration**:
  - Multi-worker: Use `num_workers = min(4, os.cpu_count())` and `persistent_workers=True` when `num_workers > 0` to prevent worker teardown/respawn overhead per epoch.
  - In-memory datasets: Set `num_workers=0` to avoid unnecessary IPC and memory duplication via fork/spawn.

### 2.4 Memmap Storage Integrity
- When loading large binary token dumps (`np.memmap`):
  - Ensure correct dtype (`np.uint16` for vocab $< 65536$, `np.int64` or `np.int32` for larger vocabs).
  - Assert file size in bytes matches $\text{total\_tokens} \times \text{itemsize}$.
  - Support read-only mode (`mode="r"`) to enable multiple concurrent worker access without disk lockups.

---

## 3. Verification Protocol

1. **Shape Contract**: Dataloader mini-batch must strictly output tensors with:
   - `input_ids.shape == (batch_size, seq_len)`
   - `labels.shape == (batch_size, seq_len)`
   - `input_ids.dtype == torch.long`, `labels.dtype == torch.long`
2. **Boundary Assertion**: Verify that no label tensor contains out-of-vocabulary indices (must be in range $[0, \text{vocab\_size} - 1]$ or equal $-100$).
