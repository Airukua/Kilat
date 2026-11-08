"""
Preset Instantiation and CLI Override Validation Tests.

Ensures all 4 tiered YAML presets (debug_cpu, small_dense, small_moe, standard_moe)
load cleanly, pass validation, instantiate models correctly, and verify that
both `--set` and `--override` work interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kilat.cli import build_parser
from kilat.configs.main_config import MainConfig
from kilat.models.transformer import KilatTransformer


PRESETS_DIR = Path(__file__).resolve().parents[4] / "configs"


@pytest.mark.parametrize(
    "config_name",
    ["debug_cpu.yaml", "small_dense.yaml", "small_moe.yaml", "standard_moe.yaml"],
)
def test_all_preset_yamls_parse_and_validate(config_name: str) -> None:
    yaml_path = PRESETS_DIR / config_name
    assert yaml_path.exists(), f"Preset not found: {yaml_path}"

    cfg = MainConfig.from_yaml(yaml_path)
    assert cfg.experiment_name is not None
    assert cfg.model.vocab_size > 0
    assert cfg.training.seed is not None


def test_small_dense_instantiation_and_forward() -> None:
    yaml_path = PRESETS_DIR / "small_dense.yaml"
    cfg = MainConfig.from_yaml(yaml_path)

    # Instantiate model directly from config
    # Overriding layer count to 2 for fast unit test execution
    cfg.model.n_layer = 2
    model = KilatTransformer(cfg.model)
    model.eval()

    # Verify dense mode uses SwiGLU
    first_block = model.layers[0]
    assert first_block.mlp.mode == "dense"

    # Forward pass with dummy batch
    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
    out = model(input_ids)
    assert out.logits.shape == (2, 16, cfg.model.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_small_moe_instantiation_and_forward() -> None:
    yaml_path = PRESETS_DIR / "small_moe.yaml"
    cfg = MainConfig.from_yaml(yaml_path)

    cfg.model.n_layer = 2
    model = KilatTransformer(cfg.model)
    model.eval()

    first_block = model.layers[0]
    assert first_block.mlp.mode == "moe"

    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
    out = model(input_ids)
    assert out.logits.shape == (2, 16, cfg.model.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_standard_moe_meta_device_instantiation() -> None:
    yaml_path = PRESETS_DIR / "standard_moe.yaml"
    cfg = MainConfig.from_yaml(yaml_path)

    # 54.2B parameter model instantiated on meta device to verify architecture shapes & parameter scale
    with torch.device("meta"):
        model = KilatTransformer(cfg.model)
        total_params = sum(p.numel() for p in model.parameters())

    # 54.2B DeepSeekMoE check (50B < total_params < 60B)
    assert 50_000_000_000 < total_params < 60_000_000_000
    assert len(model.layers) == 24


def test_cli_override_flag_alias() -> None:
    parser = build_parser()

    # Test --override on 'verify'
    args_verify = parser.parse_args(
        ["verify", "--config", str(PRESETS_DIR / "debug_cpu.yaml"), "--override", "training.seed=123"]
    )
    assert args_verify.set == ["training.seed=123"]

    # Test --set on 'verify'
    args_verify_set = parser.parse_args(
        ["verify", "--config", str(PRESETS_DIR / "debug_cpu.yaml"), "--set", "training.seed=456"]
    )
    assert args_verify_set.set == ["training.seed=456"]

    # Test --override on 'train'
    args_train = parser.parse_args(
        ["train", "--config", str(PRESETS_DIR / "debug_cpu.yaml"), "--override", "training.seed=789"]
    )
    assert args_train.set == ["training.seed=789"]

    # Test multiple --override arguments
    args_multi = parser.parse_args(
        [
            "train",
            "--config",
            str(PRESETS_DIR / "debug_cpu.yaml"),
            "--override",
            "training.learning_rate=1e-4",
            "--set",
            "training.seed=999",
        ]
    )
    assert args_multi.set == ["training.learning_rate=1e-4", "training.seed=999"]
