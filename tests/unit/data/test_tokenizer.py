"""
Unit tests for Kilat AutoTokenizer and Tokenizer Utilities.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import yaml

from kilat.data.tokenizer import (
    AutoTokenizer,
    _create_tokenizer_config_from_tokenizer,
    get_vocab_size,
    save_tokenizer,
)
from kilat.configs.tokenizer_config import TokenizerConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


class DummyCustomTokenizer:
    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.name_or_path = "gpt2"
        self.pad_token = None
        self.eos_token = "<|endoftext|>"

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "text"


def test_get_vocab_size_and_infer_config():
    with Scenario("Unit: Tokenizer Vocab Size Resolution & Config Inference"):
        with Given("a custom tokenizer instance with vocab_size and name_or_path"):
            tok = DummyCustomTokenizer(vocab_size=1200)

            with When("get_vocab_size is invoked"):
                v_size = get_vocab_size(tok)
                with Then("the reported vocabulary size is 1200"):
                    assert v_size == 1200

            with When("inferring TokenizerConfig from instance"):
                cfg = _create_tokenizer_config_from_tokenizer(tok)
                with Then("the inferred configuration is valid"):
                    assert cfg.tokenizer_name_or_path == "gpt2"

        with Given("a mock SentencePiece tokenizer with GetPieceSize"):
            mock_spm = MagicMock(spec=["GetPieceSize", "Load", "model_file"])
            mock_spm.GetPieceSize.return_value = 32000
            mock_spm.Load.return_value = True
            mock_spm.model_file = "/path/to/spm.model"

            with When("resolving vocab size of SentencePiece tokenizer"):
                size = get_vocab_size(mock_spm)
                with Then("size equals 32000"):
                    assert size == 32000


def test_save_tokenizer_fallback():
    with Scenario("Unit: Tokenizer Serialization and Save Handling"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tok_checkpoint"

            with Given("a custom tokenizer without HF save_pretrained methods"):
                tok = DummyCustomTokenizer()

                with When("save_tokenizer is called"):
                    save_tokenizer(tok, str(out_dir))
                    with Then("the output directory is created"):
                        assert out_dir.exists()
                    with And("tokenizer_config.json metadata is saved"):
                        cfg_file = out_dir / "tokenizer_config.json"
                        assert cfg_file.exists()


def test_autotokenizer_detection_methods():
    with Scenario("Unit: AutoTokenizer Local File Detection and Heuristics"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("an empty checkpoint directory"):
                with When("checking _has_tokenizer_files"):
                    has_files = AutoTokenizer._has_tokenizer_files(tmp_path)
                    with Then("it returns False"):
                        assert has_files is False

            with Given("a checkpoint directory with tokenizer.json"):
                (tmp_path / "tokenizer.json").touch()
                with When("checking _has_tokenizer_files"):
                    has_files_now = AutoTokenizer._has_tokenizer_files(tmp_path)
                    with Then("it returns True"):
                        assert has_files_now is True

            with Given("a custom TokenizerConfig"):
                cfg_custom = TokenizerConfig(
                    tokenizer_type="custom",
                    custom_builder="dummy_builder",
                    custom_module="dummy_mod",
                )
                with When("evaluating _should_use_local_only for custom config"):
                    use_local = AutoTokenizer._should_use_local_only(cfg_custom, tmp_path)
                    with Then("custom tokenizers always default to local"):
                        assert use_local is True

            with Given("a sentencepiece TokenizerConfig pointing to missing model"):
                cfg_spm = TokenizerConfig(
                    tokenizer_type="sentencepiece",
                    tokenizer_model_path=str(tmp_path / "missing.model"),
                )
                with When("evaluating _should_use_local_only"):
                    use_local_spm = AutoTokenizer._should_use_local_only(cfg_spm, tmp_path)
                    with Then("it returns False when model file is absent"):
                        assert use_local_spm is False


def test_autotokenizer_from_pretrained_priorities():
    with Scenario("Unit: AutoTokenizer.from_pretrained Priority Escalation"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a directory containing tokenizer_config.json (Priority 1)"):
                p1_dir = tmp_path / "p1"
                p1_dir.mkdir()
                cfg_dict = {
                    "tokenizer_type": "custom",
                    "custom_builder": "tests.unit.configs.test_tokenizer_config:build_dummy_tok",
                }
                (p1_dir / "tokenizer_config.json").write_text(json.dumps(cfg_dict))

                with When("AutoTokenizer loads from Priority 1 directory"):
                    tok1 = AutoTokenizer.from_pretrained(p1_dir)
                    with Then("the tokenizer is constructed from tokenizer_config.json"):
                        assert tok1.vocab_size == 500

            with Given("a directory containing config.yaml with tokenizer section (Priority 2)"):
                p2_dir = tmp_path / "p2"
                p2_dir.mkdir()
                yaml_data = {
                    "tokenizer": {
                        "tokenizer_type": "custom",
                        "custom_builder": "tests.unit.configs.test_tokenizer_config:build_dummy_tok",
                    }
                }
                (p2_dir / "config.yaml").write_text(yaml.dump(yaml_data))

                with When("AutoTokenizer loads from Priority 2 directory"):
                    tok2 = AutoTokenizer.from_pretrained(p2_dir)
                    with Then("the tokenizer is constructed from config.yaml"):
                        assert tok2.vocab_size == 500

            with Given("environment variable KILAT_TOKENIZER_BUILDER (Priority 5)"):
                p5_dir = tmp_path / "p5"
                p5_dir.mkdir()
                os.environ["KILAT_TOKENIZER_BUILDER"] = "tests.unit.configs.test_tokenizer_config:build_dummy_tok"

                with When("AutoTokenizer loads with env var configured"):
                    try:
                        tok5 = AutoTokenizer.from_pretrained(p5_dir)
                        with Then("tokenizer is constructed using environment builder"):
                            assert tok5.vocab_size == 500
                    finally:
                        del os.environ["KILAT_TOKENIZER_BUILDER"]


def test_autotokenizer_cache_check_and_priority_3():
    from unittest.mock import MagicMock, patch

    with Scenario("Unit: AutoTokenizer Priority 3, Cache Checks, and Fallback"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a directory with config.json defining tokenizer_name_or_path (Priority 3)"):
                p3_dir = tmp_path / "p3"
                p3_dir.mkdir()
                cfg_dict = {"tokenizer_name_or_path": "custom-tok-model"}
                (p3_dir / "config.json").write_text(json.dumps(cfg_dict))

                with When("AutoTokenizer loads from Priority 3"):
                    with patch("kilat.data.tokenizer.TokenizerConfig.build") as mock_build:
                        mock_build.return_value = DummyCustomTokenizer(600)
                        tok3 = AutoTokenizer.from_pretrained(p3_dir)
                        with Then("tokenizer is created through TokenizerConfig.build"):
                            assert tok3.vocab_size == 600

            with Given("mocked HF AutoTokenizer for caching and fallback checks"):
                with When("testing _is_model_cached"):
                    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
                        mock_tok.return_value = MagicMock()
                        cached = AutoTokenizer._is_model_cached("test-model")
                        with Then("_is_model_cached returns True on success"):
                            assert cached is True

                        mock_tok.side_effect = Exception("Offline")
                        not_cached = AutoTokenizer._is_model_cached("test-model")
                        with And("_is_model_cached returns False on error"):
                            assert not_cached is False

                with When("testing _fallback_tokenizer"):
                    with patch("transformers.AutoTokenizer.from_pretrained") as mock_fallback:
                        mock_fallback.return_value = DummyCustomTokenizer(50257)
                        fb_tok = AutoTokenizer._fallback_tokenizer(local_files_only=True)
                        with Then("fallback returns default tokenizer instance"):
                            assert fb_tok.vocab_size == 50257


def test_autotokenizer_hf_format_compatibility():
    from unittest.mock import patch

    with Scenario("Unit: AutoTokenizer HuggingFace Format Compatibility"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a directory containing HF-style tokenizer_config.json with add_prefix_space"):
                hf_config = {
                    "add_prefix_space": False,
                    "backend": "tokenizers",
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "errors": "replace",
                    "is_local": False,
                    "local_files_only": True,
                    "model_max_length": 1024,
                    "pad_token": "<|endoftext|>",
                    "tokenizer_class": "GPT2Tokenizer",
                    "unk_token": "<|endoftext|>",
                }
                (tmp_path / "tokenizer_config.json").write_text(json.dumps(hf_config))
                (tmp_path / "vocab.json").write_text("{}")
                (tmp_path / "merges.txt").write_text("")

                with When("AutoTokenizer loads from directory without throwing TypeError"):
                    with patch("transformers.AutoTokenizer.from_pretrained") as mock_hf:
                        mock_hf.return_value = DummyCustomTokenizer(50257)
                        tok = AutoTokenizer.from_pretrained(tmp_path)
                        with Then("the tokenizer is loaded successfully from Priority 1"):
                            assert tok.vocab_size == 50257
                            mock_hf.assert_called_once()
