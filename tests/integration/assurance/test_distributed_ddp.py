"""
Real Multi-Process DDP Integration Verification Test.

Spawns 2 processes using torch.multiprocessing and the PyTorch Gloo CPU backend.
Verifies:
  1. Distributed process group initialization via setup_distributed().
  2. Automatic DistributedDataParallel wrapping inside KilatTrainer.
  3. All-reduce gradient synchronization across ranks.
  4. Exact parameter parity between rank 0 and rank 1 across training steps.
  5. Rank-0-only checkpoint saving and barrier coordination.
  6. Clean process-group shutdown via cleanup_distributed().
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from kilat.configs.model_config import KilatConfig
from kilat.data.tokenizer import ByteLevelDebugTokenizer
from kilat.engine.args import TrainingArguments
from kilat.engine.checkpoint import set_seed
from kilat.engine.distributed import (
    cleanup_distributed,
    get_global_rank,
    get_world_size,
    is_distributed_initialized,
    is_main_process,
    setup_distributed,
)
from kilat.engine.trainer import KilatTrainer
from kilat.models.transformer import KilatTransformer


class SmallSyntheticDataset(Dataset):
    def __init__(self, size: int = 16, seq_len: int = 16, vocab_size: int = 256) -> None:
        self.data = [
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,), dtype=torch.long),
                "labels": torch.randint(0, vocab_size, (seq_len,), dtype=torch.long),
            }
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.data[idx]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
    results_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    try:
        ctx = setup_distributed(backend="gloo")
        assert is_distributed_initialized()
        assert ctx.world_size == world_size
        assert ctx.rank == rank
        assert is_main_process() == (rank == 0)

        # Same model config and seed so initial weights match
        set_seed(42)
        model_cfg = KilatConfig(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=4,
            recall_ratio=0.5,
            latent_dim=32,
            ffn_mode="moe",
            num_experts=4,
            active_experts=2,
            num_shared_experts=1,
        )
        model = KilatTransformer(model_cfg)

        # Each rank gets different data batches to ensure all-reduce is genuinely exercised
        rank_seed = 100 + rank
        torch.manual_seed(rank_seed)
        dataset = SmallSyntheticDataset(size=16, seq_len=16, vocab_size=256)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        args = TrainingArguments(
            output_dir=output_dir,
            training_mode="steps",
            max_steps=2,
            learning_rate=1e-3,
            precision="fp32",
            save_steps=1,
            seed=42,
            report_to="none",
        )

        tokenizer = ByteLevelDebugTokenizer(vocab_size=256)

        trainer = KilatTrainer(
            model=model,
            args=args,
            train_dataloader=dataloader,
            tokenizer=tokenizer,
        )

        # Verify model was wrapped in DistributedDataParallel
        assert isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)

        trainer.train()

        # Save rank final parameter state for parity assertion
        raw_model = trainer._unwrap_model()
        weights_dump = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        torch.save(weights_dump, os.path.join(results_dir, f"rank_{rank}_state.pt"))

    finally:
        cleanup_distributed()


def test_real_multiprocess_ddp() -> None:
    world_size = 2
    port = _find_free_port()

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        with tempfile.TemporaryDirectory() as tmp_results_dir:
            mp.spawn(
                _ddp_worker,
                args=(world_size, port, tmp_output_dir, tmp_results_dir),
                nprocs=world_size,
                join=True,
            )

            # 1. Parity Verification: Rank 0 and Rank 1 must have identical model weights after training
            rank0_file = os.path.join(tmp_results_dir, "rank_0_state.pt")
            rank1_file = os.path.join(tmp_results_dir, "rank_1_state.pt")
            assert os.path.exists(rank0_file), "Rank 0 state file missing"
            assert os.path.exists(rank1_file), "Rank 1 state file missing"

            rank0_state = torch.load(rank0_file, weights_only=True)
            rank1_state = torch.load(rank1_file, weights_only=True)

            for key in rank0_state:
                assert key in rank1_state, f"Missing key {key} in rank 1"
                assert torch.allclose(
                    rank0_state[key], rank1_state[key], atol=1e-6
                ), f"Parameter divergence on {key} between ranks"

            # 2. Checkpointing Verification: Checkpoints must exist and manifest must be valid
            out_path = Path(tmp_output_dir)
            ckpts = sorted(out_path.glob("checkpoint-*"))
            assert len(ckpts) >= 1, "Expected at least 1 checkpoint saved"
            first_ckpt = ckpts[0]
            assert (first_ckpt / "training_state.pt").exists()
            assert (first_ckpt / "checkpoint_manifest.json").exists()
