"""
Canonical Public Import & Clean Package Layout Contract Tests.

Proves that:
1. Every canonical package (`kilat`, `kilat.models`, `kilat.engine`, `kilat.inference`,
   `kilat.distillation`, `kilat.data`, `kilat.configs`, `kilat.assurance`, `kilat.utils`)
   exposes identical class/function objects across top-level and subpackage exports.
2. Obsolete legacy directories (`arc/`, `training/`, `pipeline/`, `distiliation/`) have been
   cleanly removed and zero 0-byte stub `.py` files exist anywhere in `src/kilat/`.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.shared.bdd import And, Given, Scenario, Then, When


def test_canonical_public_imports_and_clean_layout_contract() -> None:
    with Scenario("Unit: Canonical Public API Identity and Zero Obsolete Folders"):
        with Given("the top-level kilat package and all canonical subpackages"):
            import kilat
            import kilat.assurance as canon_assurance
            import kilat.configs as canon_configs
            import kilat.data as canon_data
            import kilat.distillation as canon_distil
            import kilat.engine as canon_engine
            import kilat.inference as canon_infer
            import kilat.models as canon_models
            import kilat.utils as canon_utils

            with When("inspecting model architecture exports in kilat and kilat.models"):
                from kilat.models.attention import KilatAttention, KilatAttentionRoPE
                from kilat.models.blocks import Block, RMSNorm
                from kilat.models.ffn import DeepSeekMoE, FeedForward, SwiGLU
                from kilat.models.kernels import FlashDecayFunction, triton_global_decay
                from kilat.models.rope import RotaryPositionalEmbedding, apply_rotary_pos_emb
                from kilat.models.transformer import KilatModelOutput, KilatTransformer

                with Then("all top-level and subpackage model symbols share exact object identity"):
                    assert kilat.KilatTransformer is KilatTransformer
                    assert canon_models.KilatTransformer is KilatTransformer
                    assert canon_models.KilatModelOutput is KilatModelOutput
                    assert canon_models.Block is Block
                    assert canon_models.RMSNorm is RMSNorm
                    assert canon_models.KilatAttention is KilatAttention
                    assert canon_models.KilatAttentionRoPE is KilatAttentionRoPE
                    assert canon_models.DeepSeekMoE is DeepSeekMoE
                    assert canon_models.FeedForward is FeedForward
                    assert canon_models.SwiGLU is SwiGLU
                    assert canon_models.RotaryPositionalEmbedding is RotaryPositionalEmbedding
                    assert apply_rotary_pos_emb is not None
                    assert FlashDecayFunction is not None
                    assert triton_global_decay is not None

            with And("inspecting training engine exports in kilat and kilat.engine"):
                from kilat.engine.args import TrainingArguments
                from kilat.engine.callbacks import EarlyStoppingCallback, TrainerCallback
                from kilat.engine.checkpoint import load_checkpoint, save_checkpoint
                from kilat.engine.distributed import DistributedContext
                from kilat.engine.integrations import ConsoleCallback, get_reporting_integration_callbacks
                from kilat.engine.optimizer import create_optimizer, create_scheduler
                from kilat.engine.scheduler import SchedulerType, get_scheduler
                from kilat.engine.trainer import KilatTrainer

                assert kilat.KilatTrainer is KilatTrainer
                assert kilat.TrainingArguments is TrainingArguments
                assert canon_engine.KilatTrainer is KilatTrainer
                assert canon_engine.TrainingArguments is TrainingArguments
                assert canon_engine.DistributedContext is DistributedContext
                assert canon_engine.EarlyStoppingCallback is EarlyStoppingCallback
                assert canon_engine.TrainerCallback is TrainerCallback
                assert canon_engine.ConsoleCallback is ConsoleCallback
                assert canon_engine.get_reporting_integration_callbacks is get_reporting_integration_callbacks
                assert canon_engine.create_optimizer is create_optimizer
                assert canon_engine.create_scheduler is create_scheduler
                assert canon_engine.SchedulerType is SchedulerType
                assert canon_engine.get_scheduler is get_scheduler
                assert canon_engine.save_checkpoint is save_checkpoint
                assert canon_engine.load_checkpoint is load_checkpoint

                ctx = DistributedContext.current()
                assert ctx.world_size >= 1
                assert ctx.is_main_process is True

            with And("inspecting inference, data, distillation, configs, assurance, and utils exports"):
                from kilat.data.preprocess import parquet_to_memmap, prepare_memmap_dataset
                from kilat.distillation.losses import AdaptiveKDLoss, VanillaKDLoss
                from kilat.distillation.student import BaseStudent
                from kilat.distillation.teacher import BaseTeacher
                from kilat.inference.beam_search import BeamSearchScorer
                from kilat.inference.export_hf import convert_checkpoint_to_hf
                from kilat.inference.generation_config import GenerationConfig
                from kilat.inference.generation_mixin import GenerationMixin
                from kilat.inference.generator import TextGenerator
                from kilat.inference.logit_processor import LogitsProcessorList
                from kilat.inference.sampler import GreedySampler
                from kilat.inference.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

                assert kilat.TextGenerator is TextGenerator
                assert kilat.GenerationConfig is GenerationConfig
                assert canon_infer.TextGenerator is TextGenerator
                assert canon_infer.GenerationConfig is GenerationConfig
                assert canon_infer.GenerationMixin is GenerationMixin
                assert canon_infer.BeamSearchScorer is BeamSearchScorer
                assert canon_infer.LogitsProcessorList is LogitsProcessorList
                assert canon_infer.GreedySampler is GreedySampler
                assert canon_infer.MaxLengthCriteria is MaxLengthCriteria
                assert canon_infer.StoppingCriteriaList is StoppingCriteriaList
                assert canon_infer.convert_checkpoint_to_hf is convert_checkpoint_to_hf
                assert canon_data.parquet_to_memmap is parquet_to_memmap
                assert canon_data.prepare_memmap_dataset is prepare_memmap_dataset
                assert canon_distil.VanillaKDLoss is VanillaKDLoss
                assert canon_distil.AdaptiveKDLoss is AdaptiveKDLoss
                assert canon_distil.BaseStudent is BaseStudent
                assert canon_distil.BaseTeacher is BaseTeacher
                assert canon_configs.MainConfig is kilat.MainConfig
                assert canon_assurance.run_verification_pipeline is not None
                assert canon_utils.print_kilat_banner is not None

            with And("verifying obsolete legacy folders have been completely removed"):
                repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
                src_root = repo_root / "src" / "kilat"
                for obsolete_dir in ("arc", "training", "pipeline", "distiliation"):
                    assert not (src_root / obsolete_dir).exists(), f"Obsolete folder still exists: {obsolete_dir}"
                    with pytest.raises(ModuleNotFoundError):
                        importlib.import_module(f"kilat.{obsolete_dir}")


def test_zero_empty_stub_files_and_prepare_memmap_jsonl(tmp_path: Path) -> None:
    with Scenario("Unit: Zero 0-Byte Stub Files and JSONL Memmap Preprocessing"):
        with Given("the src/kilat source tree"):
            repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
            src_root = repo_root / "src" / "kilat"
            with When("scanning all Python files for 0-byte stubs"):
                empty_files = [
                    str(p.relative_to(src_root))
                    for p in src_root.rglob("*.py")
                    if p.stat().st_size == 0
                ]
                with Then("zero 0-byte Python files exist anywhere in src/kilat"):
                    assert empty_files == []

        with Given("a JSONL corpus with raw text records"):
            from kilat.assurance.gates.gate4_data import _verify_memmap_file
            from kilat.data.preprocess import prepare_memmap_dataset

            jsonl_path = tmp_path / "corpus.jsonl"
            lines = [
                json.dumps({"text": "Kilat hybrid attention"}),
                json.dumps({"text": "DeepSeek MoE verification"}),
            ]
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            out_bin = tmp_path / "train.bin"

            with When("running prepare_memmap_dataset to generate a uint16 .bin + .meta.json"):
                meta = prepare_memmap_dataset(
                    input_path=str(jsonl_path),
                    output_path=str(out_bin),
                    tokenizer="byte",
                    dtype=np.uint16,
                    verbose=False,
                )

                with Then("the binary memmap and .meta.json pass Gate 4 integrity verification"):
                    assert out_bin.is_file()
                    assert (tmp_path / "train.bin.meta.json").is_file()
                    assert int(meta["num_tokens"]) > 0
                    _verify_memmap_file(str(out_bin), dtype_name="uint16")
