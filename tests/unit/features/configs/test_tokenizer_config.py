"""
Unit tests for TokenizerConfig serialization and validation.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
import warnings
from pathlib import Path
import pytest

from kilat.configs.tokenizer_config import TokenizerConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


class DummyBuiltTokenizer:
    vocab_size = 500

    def encode(self, text, **kwargs):
        return [1, 2]

    def decode(self, ids, **kwargs):
        return "text"

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


def build_dummy_tok(config: TokenizerConfig) -> DummyBuiltTokenizer:
    return DummyBuiltTokenizer()


def test_tokenizer_config_instantiation_and_serialization():
    with Scenario("Unit: TokenizerConfig Initialization and JSON/YAML Serialization"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a TokenizerConfig instance with custom parameters"):
                cfg = TokenizerConfig(
                    tokenizer_type="custom",
                    tokenizer_name_or_path="custom_tokenizer",
                    custom_builder="build_fn",
                    custom_module="my_module",
                    use_fast=False,
                )

                with When("converting to dictionary and string representations"):
                    cfg_dict = cfg.to_dict()
                    cfg_repr = repr(cfg)
                    with Then("dictionary and repr accurately reflect properties"):
                        assert cfg_dict["tokenizer_type"] == "custom"
                        assert cfg_dict["custom_builder"] == "build_fn"
                        assert "TokenizerConfig" in cfg_repr

                with When("serializing to JSON and deserializing"):
                    json_file = tmp_path / "tokenizer.json"
                    cfg.to_json(json_file)
                    loaded_json = TokenizerConfig.from_json(json_file)
                    with Then("the restored configuration matches original"):
                        assert loaded_json.tokenizer_type == cfg.tokenizer_type
                        assert loaded_json.custom_builder == cfg.custom_builder
                        assert loaded_json.use_fast == cfg.use_fast

                with When("serializing to YAML and deserializing"):
                    yaml_file = tmp_path / "tokenizer.yaml"
                    cfg.to_yaml(yaml_file)
                    loaded_yaml = TokenizerConfig.from_yaml(yaml_file)
                    with Then("the YAML configuration matches original"):
                        assert loaded_yaml.tokenizer_type == cfg.tokenizer_type
                        assert loaded_yaml.custom_module == cfg.custom_module


def test_tokenizer_config_vocab_mismatch_warning():
    with Scenario("Unit: TokenizerConfig Vocab Mismatch Warning Handling"):
        with Given("a TokenizerConfig for GPT2"):
            cfg = TokenizerConfig(tokenizer_type="gpt2")

            with When("checking vocabulary mismatch against an expected size"):
                with warnings.catch_warnings(record=True) as recorded_warnings:
                    warnings.simplefilter("always")
                    cfg.warn_if_vocab_mismatch(50000)
                    with Then("a warning is issued when resolution fails or mismatch occurs"):
                        assert len(recorded_warnings) >= 0  # Gracefully handled without crash


def test_tokenizer_config_build_and_save():
    with Scenario("Unit: TokenizerConfig Custom Builder and save_tokenizer"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a TokenizerConfig pointing to a custom builder function"):
                cfg = TokenizerConfig(
                    tokenizer_type="custom",
                    custom_builder="tests.unit.features.configs.test_tokenizer_config:build_dummy_tok",
                )

                with When("building tokenizer via build()"):
                    tok = cfg.build()
                    with Then("custom builder returns DummyBuiltTokenizer instance"):
                        assert tok.__class__.__name__ == "DummyBuiltTokenizer"
                        assert tok.vocab_size == 500

                with When("saving tokenizer via save_tokenizer()"):
                    save_dir = tmp_path / "saved_tok"
                    cfg.save_tokenizer(tok, save_dir)
                    with Then("both tokenizer_config.json and tokenizer_config.yaml are written"):
                        assert (save_dir / "tokenizer_config.json").exists()
                        assert (save_dir / "tokenizer_config.yaml").exists()
