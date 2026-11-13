"""
Numerical Parity and Compatibility Test Suite for Kilat Hybrid Architecture.

Enforces Golden Invariant #1: Numerical Proof over Syntax Passing.
Verifies:
1. Adapter independence (zero internal kilat imports in modeling_kilat.py).
2. Complete artifact export (safetensors, auto_map, adapter files, tokenizer bundle).
3. Numerical logit parity: max |logits_native - logits_hf| < 1e-5.
4. Loss equivalence: |loss_native - loss_hf| < 1e-5.
5. Generation token identity across greedy decode steps.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


_CONFIG_JSON = "config.json"


def test_adapter_independence():
    """Verify standalone Hub distribution template is 100% self-contained and never imports kilat.*"""
    template_path = Path("src/kilat/inference/templates/modeling_kilat_hub.py")
    assert template_path.exists(), "modeling_kilat_hub.py does not exist"

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "kilat." not in stripped, f"Found forbidden internal import: {line}"
            if stripped.startswith(("import kilat", "from kilat")):
                pytest.fail(f"Hub distribution artifact must not depend on kilat package: {line}")


def test_intree_adapter_thinness_and_ssot():
    """Verify in-tree Hugging Face adapter adheres to Senior Architecture line-count & SSOT standards."""
    intree_path = Path("src/kilat/integrations/huggingface/modeling_kilat.py")
    assert intree_path.exists(), "in-tree modeling_kilat.py does not exist"

    with open(intree_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Senior Architecture constraint: must be thin adapter under 200 lines
    assert len(lines) < 200, f"In-tree adapter has {len(lines)} lines; expected < 200 lines (Anti-Monolith rule)"

    # Verify canonical inheritance
    from kilat.integrations.huggingface.modeling_kilat import KilatForCausalLM, KilatModel
    from kilat.models.transformer import KilatTransformer

    assert issubclass(KilatForCausalLM, KilatTransformer)
    assert issubclass(KilatModel, KilatTransformer)


def test_auto_map_metadata_injection():
    """Verify inject_auto_map_metadata properly configures config.json"""
    from kilat.inference.export_hf import inject_auto_map_metadata

    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        dummy_config = {"vocab_size": 1000, "n_embd": 64, "n_head": 4, "n_layer": 2}
        with open(dir_path / _CONFIG_JSON, "w", encoding="utf-8") as f:
            json.dump(dummy_config, f)

        inject_auto_map_metadata(dir_path)

        with open(dir_path / _CONFIG_JSON, "r", encoding="utf-8") as f:
            updated = json.load(f)

        assert updated["model_type"] == "kilat"
        assert "KilatForCausalLM" in updated["architectures"]
        assert "auto_map" in updated
        assert updated["auto_map"]["AutoConfig"] == "configuration_kilat.KilatConfig"
        assert updated["auto_map"]["AutoModel"] == "modeling_kilat.KilatModel"
        assert updated["auto_map"]["AutoModelForCausalLM"] == "modeling_kilat.KilatForCausalLM"


def test_export_bundles_all_hybrid_artifacts():
    """Verify convert_checkpoint_to_hf bundles all required files for HF Hub."""
    from kilat.configs.model_config import KilatConfig
    from kilat.inference.export_hf import convert_checkpoint_to_hf
    from kilat.models.transformer import KilatTransformer

    with tempfile.TemporaryDirectory() as ckpt_dir, tempfile.TemporaryDirectory() as out_dir:
        ckpt_path = Path(ckpt_dir)
        out_path = Path(out_dir)

        cfg = KilatConfig(
            vocab_size=128,
            n_embd=64,
            n_head=4,
            n_layer=2,
            recall_pattern="macro_synergy",
        )
        model = KilatTransformer(cfg)
        model.save_pretrained(ckpt_path)

        # Mock tokenizer files in checkpoint
        with open(ckpt_path / "tokenizer_config.json", "w") as f:
            json.dump({"tokenizer_type": "auto", "pad_token_id": 0}, f)

        res = convert_checkpoint_to_hf(
            checkpoint_dir=ckpt_path,
            output_dir=out_path,
            skip_validation=False,
        )
        assert res["success"] is True

        # Check required files
        assert (out_path / "model.safetensors").exists()
        assert (out_path / _CONFIG_JSON).exists()
        assert (out_path / "configuration_kilat.py").exists()
        assert (out_path / "modeling_kilat.py").exists()
        assert (out_path / "tokenizer_config.json").exists()
        assert (out_path / "special_tokens_map.json").exists()

        with open(out_path / _CONFIG_JSON, "r") as f:
            cfg_data = json.load(f)
        assert "auto_map" in cfg_data


def test_numerical_logits_and_loss_parity():
    """Verify numerical equivalence between Kilat Native and Hugging Face loaded model."""
    from transformers import AutoModelForCausalLM

    from kilat.configs.model_config import KilatConfig
    from kilat.inference.export_hf import convert_checkpoint_to_hf
    from kilat.models.transformer import KilatTransformer

    with tempfile.TemporaryDirectory() as ckpt_dir, tempfile.TemporaryDirectory() as out_dir:
        ckpt_path = Path(ckpt_dir)
        out_path = Path(out_dir)

        cfg = KilatConfig(
            vocab_size=256,
            n_embd=64,
            n_head=4,
            n_layer=2,
            recall_pattern="macro_synergy",
            use_rope=True,
            rope_head_dim=8,
        )
        native_model = KilatTransformer(cfg)
        native_model.eval()
        native_model.save_pretrained(ckpt_path)

        # Export to HF format with auto_map and standalone adapter
        convert_checkpoint_to_hf(
            checkpoint_dir=ckpt_path,
            output_dir=out_path,
            skip_validation=True,
        )

        # Load through Hugging Face AutoModelForCausalLM via remote code
        hf_model = AutoModelForCausalLM.from_pretrained(str(out_path), trust_remote_code=True)
        hf_model.eval()

        # Fixed deterministic input
        torch.manual_seed(42)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = input_ids.clone()

        with torch.no_grad():
            native_out = native_model(input_ids, labels=labels)
            hf_out = hf_model(input_ids, labels=labels)

        # 1. Logit parity assertion
        max_logit_diff = torch.max(torch.abs(native_out.logits - hf_out.logits)).item()
        assert max_logit_diff < 1e-5, f"Logit difference {max_logit_diff} exceeds tolerance 1e-5"

        # 2. Loss parity assertion
        loss_diff = abs(native_out.loss.item() - hf_out.loss.item())
        assert loss_diff < 1e-5, f"Loss difference {loss_diff} exceeds tolerance 1e-5"

        # 3. Greedy generation token matching
        prompt = input_ids[:, :4]
        hf_gen = hf_model.generate(prompt, max_new_tokens=8, do_sample=False)

        # Native greedy generation with TextGenerator
        from kilat.inference.generator import TextGenerator

        class DummyTokenizer:
            def decode(self, ids):
                return ""
            def encode(self, text):
                return []

        gen = TextGenerator(native_model, DummyTokenizer())
        # Inspect raw model greedy steps
        curr_ids = prompt.clone()
        past = None
        for _ in range(8):
            step_out = native_model(curr_ids[:, -1:] if past is not None else curr_ids, past_key_values=past, use_cache=True)
            logits = step_out.logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)
            past = step_out.past_key_values

        assert torch.equal(hf_gen, curr_ids), "Greedy generated token sequences do not match"
