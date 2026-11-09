"""
BDD-style tests for tools/analysis/position_loss.py (Per-position validation loss profiler).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from tools.analysis.position_loss import _bucket_label, position_loss
from kilat import KilatConfig, KilatTransformer
from tests.shared.bdd import Scenario, Given, When, Then, And


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Create a hermetic tiny KilatTransformer checkpoint saved to disk."""
    cfg = KilatConfig(
        vocab_size=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        recall_ratio=0.5,
        latent_dim=16,
        max_position_embeddings=64,
        use_rope=True,
        rope_head_dim=4,
    )
    model = KilatTransformer(cfg)
    ckpt_dir = tmp_path / "tiny_ckpt"
    model.save_pretrained(str(ckpt_dir))
    return ckpt_dir


def test_position_loss_bucket_label_formatting():
    with Scenario("Tools: Position Loss Bucket Label Formatting"):
        with Given("lower and upper interval boundaries"):
            with When("formatting bounded and unbounded intervals"):
                b1 = _bucket_label(0, 8)
                b2 = _bucket_label(8, 16)
                b3 = _bucket_label(256, 1_000_000)

                with Then("the formatted interval strings match standard notation"):
                    assert b1 == "0-7"
                    assert b2 == "8-15"
                    assert b3 == "256+"


def test_position_loss_profiling(tiny_checkpoint: Path, tmp_path: Path, device: torch.device):
    with Scenario("Tools: Per-Position Cross-Entropy Loss Profiling"):
        with Given("a tiny KilatTransformer model and synthetic token stream with periodic EOS"):
            seq_len = 16
            eos_id = 100
            raw_tokens = np.array([1, 2, 3, eos_id, 4, 5, 6, 7, 8, eos_id] * 20, dtype=np.int64)
            data_path = tmp_path / "eval_tokens.npy"
            np.save(data_path, raw_tokens)

            with When("position_loss evaluates cross-entropy across document context buckets"):
                res = position_loss(
                    ckpt=str(tiny_checkpoint),
                    data_path=str(data_path),
                    seq_len=seq_len,
                    num_windows=4,
                    batch_size=2,
                    eos_id=eos_id,
                    device=device,
                )

                with Then("the report records the evaluated window count and overall CE"):
                    assert res["windows"] == 4
                    assert "overall_ce" in res
                    assert not np.isnan(res["overall_ce"])

                with And("doc_position and window_position tables contain populated valid buckets"):
                    assert len(res["by_doc_position"]) > 0
                    assert len(res["by_window_position"]) > 0
                    for entry in res["by_doc_position"]:
                        assert entry["tokens"] > 0
                        assert entry["mean_ce"] > 0.0


def test_position_loss_negative_missing_checkpoint(tmp_path: Path, device: torch.device):
    with Scenario("Tools: Position Loss Negative Testing - Missing Checkpoint"):
        with Given("a nonexistent model checkpoint directory"):
            fake_ckpt = tmp_path / "missing_checkpoint_dir"
            fake_data = tmp_path / "data.npy"
            np.save(fake_data, np.array([1, 2, 3, 4], dtype=np.int64))

            with When("position_loss is called with the missing checkpoint"):
                with Then("an exception is raised"):
                    with pytest.raises((FileNotFoundError, OSError, Exception)):
                        position_loss(
                            ckpt=str(fake_ckpt),
                            data_path=str(fake_data),
                            seq_len=2,
                            num_windows=1,
                            batch_size=1,
                            eos_id=0,
                            device=device,
                        )


def test_position_loss_negative_missing_data_file(tiny_checkpoint: Path, tmp_path: Path, device: torch.device):
    with Scenario("Tools: Position Loss Negative Testing - Missing Data File"):
        with Given("a valid checkpoint but nonexistent data file"):
            missing_data = tmp_path / "missing_data.npy"

            with When("position_loss attempts to load the missing data"):
                with Then("a FileNotFoundError is raised"):
                    with pytest.raises(FileNotFoundError):
                        position_loss(
                            ckpt=str(tiny_checkpoint),
                            data_path=str(missing_data),
                            seq_len=16,
                            num_windows=2,
                            batch_size=1,
                            eos_id=0,
                            device=device,
                        )


def test_position_loss_negative_tokens_without_eos(tiny_checkpoint: Path, tmp_path: Path, device: torch.device):
    with Scenario("Tools: Position Loss Negative Testing - Continuous Non-EOS Sequences"):
        with Given("a token sequence that never contains the target EOS token"):
            raw_tokens = np.arange(1, 100, dtype=np.int64)
            data_path = tmp_path / "no_eos_tokens.npy"
            np.save(data_path, raw_tokens)

            with When("position_loss executes across windows"):
                res = position_loss(
                    ckpt=str(tiny_checkpoint),
                    data_path=str(data_path),
                    seq_len=16,
                    num_windows=2,
                    batch_size=1,
                    eos_id=999,
                    device=device,
                )

                with Then("windows are evaluated successfully without crashing"):
                    assert res["windows"] == 2
                    assert len(res["by_doc_position"]) > 0
