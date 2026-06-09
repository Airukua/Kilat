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

import sys
from pathlib import Path

import numpy as np
import torch
from torchinfo import summary

# Add project root to Python path to enable absolute imports.
# This allows importing modules like 'data.dataset', 'training.trainer', etc.
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Kilat components
from data.dataset import PretrainingDataset
from data.dataloader import build_train_dataloader, build_eval_dataloader
from training.trainer import KilatTrainer, TrainingArguments
from utils.config import MainConfig, KilatConfig
from arc.model import KilatTransformer

# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
# Automatically select CUDA if available; CPU fallback for environments without GPU.
# In distributed training, this will be overridden by the launcher.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 2. Load configuration
# ---------------------------------------------------------------------------
# MainConfig reads from YAML (includes model architecture, training hyperparameters,
# dataloader settings, and paths). KilatConfig converts it to the format expected
# by KilatTransformer (e.g., embedding dimensions, number of heads, layers).
#
# WHY separate configs? MainConfig is user-friendly (YAML), KilatConfig is
# optimised for fast internal access (dict/attribute lookup).
main_config = MainConfig.from_yaml('configs/small_dense.yaml')
config = KilatConfig.from_main_config(main_config)

# ---------------------------------------------------------------------------
# 3. Model initialisation
# ---------------------------------------------------------------------------
# Create transformer model with the specified architecture.
model = KilatTransformer(config)
model.to(device)

# Display model architecture summary (depth=3 shows nested module breakdown).
# Useful for verifying parameter count and layer dimensions.
sample_input = torch.randint(0, config.vocab_size, (8, 1024), device=device)
print("\nModel Architecture Summary:")
summary(model, input_data=sample_input, depth=3)

# ---------------------------------------------------------------------------
# 4. Dataset preparation (memory-mapped)
# ---------------------------------------------------------------------------
# PretrainingDataset loads a flat .npy file containing concatenated token IDs.
# It slices the flat array into chunks of `max_seq_length` tokens.
#
# WHY memmap: The file is memory-mapped (zero-copy), allowing datasets larger
# than RAM. The OS page cache loads only the pages needed for current batches.
train = PretrainingDataset(
    source="data/indo/train/tokens.npy",
    key_name="input_ids",
    chunk_size=main_config.dataloader.max_seq_length,  # Matches model's context length
    dtype=np.int32,  # Token IDs fit in 32-bit signed integer
)

val = PretrainingDataset(
    source="data/indo/val/tokens.npy",
    key_name="input_ids",
    chunk_size=main_config.dataloader.max_seq_length,
    dtype=np.int32,
)

# ---------------------------------------------------------------------------
# 5. DataLoader setup
# ---------------------------------------------------------------------------
# build_train_dataloader: shuffles, uses DistributedSampler in DDP, drop_last=True.
# build_eval_dataloader: no shuffle, drop_last=False (preserve all samples).
#
# Both functions automatically handle:
# - Padding to batch's longest sequence (via KilatDataCollator)
# - Multi-GPU distribution (DistributedSampler when torch.distributed is initialised)
# - Worker initialisation for reproducible randomness
train_loader = build_train_dataloader(
    train,
    batch_size=main_config.dataloader.train_batch_size,
    pad_token_id=config.pad_token_id,
    max_length=main_config.dataloader.max_seq_length,
    num_workers=main_config.dataloader.num_workers,
    pin_memory=main_config.dataloader.pin_memory,
    drop_last=main_config.dataloader.drop_last,
)

eval_loader = build_eval_dataloader(
    val,
    batch_size=main_config.dataloader.eval_batch_size,
    pad_token_id=config.pad_token_id,
    max_length=main_config.dataloader.max_seq_length,
    num_workers=main_config.dataloader.num_workers,
    pin_memory=main_config.dataloader.pin_memory,
    drop_last=False,  # Keep all validation samples for accurate metrics
)

# ---------------------------------------------------------------------------
# 6. TrainingArguments setup
# ---------------------------------------------------------------------------
# Convert MainConfig.training dataclass to a dictionary.
# Remove batch size fields because they are already passed via DataLoader.
# This avoids duplicate/conflicting configuration.
training_dict = main_config.training.to_dict()
training_dict.pop('per_device_train_batch_size', None)
training_dict.pop('per_device_eval_batch_size', None)

# Create TrainingArguments (dataclass) for the trainer.
# Fields include: learning_rate, weight_decay, warmup_steps, scheduler_type,
# precision (fp16/bf16), logging/save/eval intervals, early stopping, etc.
args = TrainingArguments(**training_dict)

# ---------------------------------------------------------------------------
# 7. Trainer initialisation and execution
# ---------------------------------------------------------------------------
# KilatTrainer orchestrates:
# - Training loop (epochs or steps)
# - Gradient accumulation
# - Mixed precision (autocast + GradScaler)
# - Periodic evaluation and checkpointing
# - Callback dispatching (logging, early stopping, integrations)
# - Model saving (best, final, and periodic checkpoints)
trainer = KilatTrainer(
    model=model,
    args=args,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Start training. This method blocks until training finishes (or early stops).
# Returns the final TrainerState containing best metrics and checkpoint paths.
final_state = trainer.train()

print("\nTraining completed successfully!")
print(f"Best metric ({args.metric_for_best_model}): {final_state.best_metric:.6f}")
print(f"Best checkpoint: {final_state.best_model_checkpoint}")