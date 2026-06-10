"""
Kilat Transformer Training Script

This script demonstrates a complete training pipeline for the KilatTransformer model
using the KilatTrainer framework. It covers configuration loading, model initialisation,
dataset preparation, DataLoader setup, and training execution.

WHY THIS SCRIPT EXISTS:
    - Provides a production-ready example of how to use all Kilat components together.
    - Serves as a template for training custom KilatTransformer models.
    - Demonstrates best practices for data loading, batching, and training configuration.

ARCHITECTURE OVERVIEW:
    - MainConfig: YAML-based configuration (model hyperparameters, training settings, data paths)
    - KilatConfig: Model-specific configuration derived from MainConfig
    - KilatTransformer: The actual transformer model (attention + feed-forward)
    - PretrainingDataset: Memory-mapped dataset for efficient token loading
    - KilatTrainer: Orchestrates training loop, evaluation, checkpointing, and callbacks

Data Flow:
    Raw tokens (npy memmap) → PretrainingDataset → DataLoader → KilatTrainer → Model → Loss → Backprop

Edge Cases Handled:
    - Different devices (CUDA/CPU) are auto-detected.
    - TrainingArguments excludes batch size fields (already passed via dataloader).
    - Dataset chunk_size matches model's max_seq_length for consistent shapes.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to Python path to enable absolute imports.
# This allows importing modules like 'data.dataset', 'training.trainer', etc.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Kilat components
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader
from data.collator import KilatDataCollator
from training.trainer import KilatTrainer
from training.args import TrainingArguments
from configs.main_config import MainConfig
from arc.model import KilatTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
# Automatically select CUDA if available; CPU fallback for environments without GPU.
# In distributed training, this will be overridden by the launcher.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# 2. Load configuration (MainConfig as single source of truth)
# ---------------------------------------------------------------------------
config = MainConfig.from_yaml('configs/sample/small_dense.yaml')


# ---------------------------------------------------------------------------
# 3. Build tokenizer automatically from config
# ---------------------------------------------------------------------------
print("Building tokenizer...")
tokenizer = config.build_tokenizer()

# Verify tokenizer works (optional)
test_text = "Hello, world!"
test_tokens = tokenizer.encode(test_text)
print(f"Tokenizer test: '{test_text[:20]}...' -> {len(test_tokens)} tokens")


# ---------------------------------------------------------------------------
# 4. Model initialisation
# ---------------------------------------------------------------------------
# Create transformer model with the specified architecture.
# KilatTransformer.__init__ handles MainConfig automatically (extracts .model)
model = KilatTransformer(config)
model.to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


# ---------------------------------------------------------------------------
# 5. Dataset preparation (memory-mapped)
# ---------------------------------------------------------------------------
# PretrainingDataset loads a flat .npy file containing concatenated token IDs.
# It slices the flat array into chunks of `max_seq_length` tokens.
train_dataset = PretrainingDataset(
    source=config.dataloader.train_data_path,
    key_name="input_ids",
    chunk_size=config.dataloader.max_seq_length,
    dtype=np.int32,  # Token IDs fit in 32-bit signed integer
)

eval_dataset = PretrainingDataset(
    source=config.dataloader.eval_data_path,
    key_name="input_ids",
    chunk_size=config.dataloader.max_seq_length,
    dtype=np.int32,
)

print(f"Train samples: {len(train_dataset):,}")
print(f"Eval samples: {len(eval_dataset):,}")


# ---------------------------------------------------------------------------
# 6. DataLoader setup with collator
# ---------------------------------------------------------------------------
# KilatDataCollator handles padding and label masking for causal LM
collator = KilatDataCollator(
    pad_token_id=config.model.pad_token_id,
    max_length=config.dataloader.max_seq_length,
    ignore_index=-100,
)

# build_train_dataloader: shuffles, uses DistributedSampler in DDP, drop_last=True.
# build_eval_dataloader: no shuffle, drop_last=False (preserve all samples).
train_loader = build_train_dataloader(
    train_dataset,
    batch_size=config.dataloader.train_batch_size,
    collate_fn=collator,
    num_workers=config.dataloader.num_workers,
    pin_memory=config.dataloader.pin_memory,
    drop_last=config.dataloader.drop_last,
    seed=config.training.seed,
)

eval_loader = build_eval_dataloader(
    eval_dataset,
    batch_size=config.dataloader.eval_batch_size,
    collate_fn=collator,
    num_workers=config.dataloader.num_workers,
    pin_memory=config.dataloader.pin_memory,
    drop_last=False,  # Keep all validation samples for accurate metrics
    seed=config.training.seed,
)


# ---------------------------------------------------------------------------
# 7. TrainingArguments setup
# ---------------------------------------------------------------------------
# Convert TrainingConfig to dictionary and remove batch size fields
# because they are already passed via DataLoader
training_dict = config.training.to_dict()
training_dict.pop('per_device_train_batch_size', None)
training_dict.pop('per_device_eval_batch_size', None)

# Create TrainingArguments (dataclass) for the trainer.
args = TrainingArguments(**training_dict)


# ---------------------------------------------------------------------------
# 8. Trainer initialisation and execution
# ---------------------------------------------------------------------------
# KilatTrainer orchestrates:
# - Training loop (epochs or steps)
# - Gradient accumulation
# - Mixed precision (autocast + GradScaler)
# - Periodic evaluation and checkpointing
# - Callback dispatching (logging, early stopping, integrations)
# - Model saving (best, final, and periodic checkpoints)
# - Tokenizer saving (for local/custom tokenizers)
trainer = KilatTrainer(
    model=model,
    args=args,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,                    # <-- for checkpoint saving
    tokenizer_config=config.tokenizer,      # <-- for tokenizer metadata
)

# Start training. This method blocks until training finishes (or early stops).
final_state = trainer.train()


# ---------------------------------------------------------------------------
# 9. Training complete
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Best metric ({args.metric_for_best_model}): {final_state.best_metric:.6f}")
print(f"Best checkpoint: {final_state.best_model_checkpoint}")
print(f"Final step: {final_state.global_step}")
print(f"Total elapsed time: {final_state.log_history[-1].get('step', 'N/A')} steps")
print("=" * 60)