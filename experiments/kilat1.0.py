# from __future__ import annotations
# import sys
# from pathlib import Path

# # Add project root to Python path to enable absolute imports across the codebase
# # This allows modules like utils.config to be discovered regardless of execution context
# project_root = Path.cwd().parent
# sys.path.insert(0, str(project_root))

# from utils.config import MainConfig
# from arc.model import KilatTransformer
# from training.trainer import KilatTrainer, TrainingArguments
# from data.collator import KilatDataCollator
# from data.dataset import KilatDataset
# import torch
# from torchinfo import summary
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load model architecture hyperparameters (e.g., n_layers, d_model, n_heads) from YAML
# # small_dense.yaml contains configuration for a compact transformer variant optimized for resource-constrained training
# config = MainConfig.from_yaml("./configs/small_dense.yaml")


# # ============================================================================
# # DATASET INITIALIZATION (STREAMING MODE)
# # ============================================================================
# # WHY: Training on full tokenized datasets (potentially billions of tokens) would cause OOM.
# # SOLUTION: Iterate through parquet file in batches, loading only what's needed for current step.
# # TRADEOFF: Streaming reduces memory at the cost of random access and slightly slower per-epoch iteration.

# # Training dataset - streaming from parquet instead of loading all tokens into memory
# train_parquet_path = "./data/fine-web-edu/train/data.parquet"
# train_dataset = KilatDataset(
#     file_or_data=train_parquet_path, 
#     key_name="input_ids",                    # Column name in parquet storing token sequences
#     streaming=True,                          # ENABLES OOM-FREE TRAINING - never loads full dataset
#     batch_size=8,                            # Small batch due to sequence length 512 (8*512 = 4K tokens/step)
#     sequence_length=512,                     # Must match model's BLOCK_SIZE for causal attention mask
#     pad_token_id=0,                          # PAD token ID; these positions are masked in attention
#     shuffle=True,                            # Shuffle within each read batch, not globally (streaming constraint)
#     drop_last=False,                         # Keep partial batches to avoid data loss
#     seed=42,                                 # Deterministic shuffling for reproducibility
#     read_batch_size=32,                      # Read 32 rows at a time from parquet, then yield in batch_size chunks
# )

# # Validation dataset - same streaming pattern but no shuffling to maintain deterministic evaluation
# val_parquet_path = "./data/fine-web-edu/val/data.parquet"
# val_dataset = KilatDataset(
#     file_or_data=val_parquet_path, 
#     key_name="input_ids",
#     streaming=True,
#     batch_size=8,
#     sequence_length=512,
#     pad_token_id=0,
#     shuffle=False,                           # Disabled for validation to ensure consistent ordering across runs
#     drop_last=False,
#     seed=42,
#     read_batch_size=32,
# )

# # # Test dataset - unseen data for final evaluation post-training
# # test_parquet_path = "./data/tokens/test/tokens/tokenized.parquet"
# # test_dataset = KilatDataset(
# #     file_or_data=test_parquet_path, 
# #     key_name="input_ids",
# #     streaming=True,
# #     batch_size=8,
# #     sequence_length=512,
# #     pad_token_id=0,
# #     shuffle=False,
# #     drop_last=False,
# #     seed=42,
# #     read_batch_size=32,
# # )

# # ============================================================================
# # COLLATION & MODEL INITIALIZATION
# # ============================================================================

# PAD_TOKEN = 0                                # Must match pad_token_id in dataset and tokenizer
# BLOCK_SIZE = 512                             # Model's maximum sequence length (context window)
# IGNORE_INDEX = -100                          # Special value for loss masking; ignored in cross-entropy
# # WHY IGNORE_INDEX = -100? PyTorch's CrossEntropyLoss ignores targets with this value.
# # Used to mask padding tokens and positions we don't want to train (e.g., future tokens in causal LM).

# collate_fn = KilatDataCollator(
#     pad_token_id=PAD_TOKEN,
#     max_length=BLOCK_SIZE,
#     ignore_index=IGNORE_INDEX,
# )

# # Initialize transformer model with architecture defined in small_dense.yaml
# model = KilatTransformer(config.model)
# model = model.to(device)

# # ============================================================================
# # MODEL ARCHITECTURE SUMMARY (DEBUGGING AID)
# # ============================================================================
# # WHY: Verify parameter count, FLOPs, and tensor shapes before expensive training.
# # This catches configuration errors (e.g., wrong hidden size, attention dimension mismatches) early.
# stats = summary(
#     model,
#     input_data={
#         'input_ids': torch.randint(0, 32000, (1, 512), dtype=torch.long, device=device)
#         # 32000 = vocabulary size (assumed from tokenizer); 1x512 = batch_size 1, seq_len 512
#     },
#     device=device, 
#     depth=3,                                 # Show 3 levels of module nesting in the summary
#     col_names=["input_size", "output_size", "num_params", "mult_adds"],
#     verbose=1
# )

# # ============================================================================
# # TRAINER SETUP & EXECUTION
# # ============================================================================

# # Convert training hyperparameters (learning rate, warmup steps, weight decay, etc.) from config to TrainingArguments
# train_args = TrainingArguments(**config.training.to_dict())

# trainer = KilatTrainer(
#     model=model,
#     args=train_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=collate_fn,
#     tokenizer_config=config.tokenizer,
# )

# # Begin autoregressive language model training:
# # Model learns to predict next token given previous context (causal language modeling objective)
# trainer.train()
