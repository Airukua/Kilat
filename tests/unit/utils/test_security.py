"""
Unit tests for Layer-0 security primitives (kilat.utils.security).
Covers OWASP LLM03 (supply chain), OWASP A01 (path traversal), and NCSC supply-chain guidance.
"""

import json
from pathlib import Path

import pytest
import torch

from kilat.configs.tokenizer_config import TokenizerConfig
from kilat.data.tokenizer import AutoTokenizer
from kilat.models.base import BasePreTrainedModel, enforce_revision_policy
from kilat.utils.security import (
    UnpinnedRevisionError,
    UnsafePathError,
    UntrustedModuleError,
    assert_trusted_module,
    is_pinned_revision,
    is_trusted_module,
    safe_join,
    trusted_module_prefixes,
)
from tests.shared.bdd import And, Given, Scenario, Then, When

GOOD_SHA = "0123456789abcdef0123456789abcdef01234567"
_ENV_TRUSTED_MODULES = "KILAT_TRUSTED_MODULES"


def test_trusted_module_allowlist(monkeypatch):
    with Scenario("Unit: Dynamic Import Allowlist"):
        monkeypatch.delenv(_ENV_TRUSTED_MODULES, raising=False)
        monkeypatch.delenv("KILAT_TOKENIZER_BUILDER", raising=False)
        with Given("the default allowlist"):
            with Then("first-party kilat modules are trusted"):
                assert is_trusted_module("kilat.data.tokenizer")
                assert is_trusted_module("kilat")
            with And("prefix look-alikes, stdlib gadgets and relative paths are rejected"):
                assert not is_trusted_module("kilat_evil.payload")
                assert not is_trusted_module("os")
                assert not is_trusted_module("subprocess")
                assert not is_trusted_module(".kilat")
                assert not is_trusted_module("")
            with And("assert_trusted_module raises an ImportError subclass"):
                with pytest.raises(UntrustedModuleError, match=_ENV_TRUSTED_MODULES):
                    assert_trusted_module("os")
                assert issubclass(UntrustedModuleError, ImportError)

        with Given("an operator who extends the allowlist via environment"):
            monkeypatch.setenv(_ENV_TRUSTED_MODULES, "my_org.tok, other ")
            monkeypatch.setenv("KILAT_TOKENIZER_BUILDER", "env_mod.sub:build")
            with When("resolving trusted prefixes"):
                prefixes = trusted_module_prefixes()
                with Then("env prefixes and the env builder's module are trusted"):
                    assert prefixes[0] == "kilat"
                    assert {"my_org.tok", "other", "env_mod.sub"} <= set(prefixes)
                    assert is_trusted_module("my_org.tok.impl")
                    assert not is_trusted_module("my_org")


def test_safe_join_blocks_path_traversal(tmp_path):
    with Scenario("Unit: safe_join Path Traversal Guard (OWASP A01)"):
        with Given("a checkpoint root"):
            with Then("plain shard names resolve inside the root"):
                assert safe_join(tmp_path, "model-00001.safetensors") == (tmp_path / "model-00001.safetensors").resolve()
                assert safe_join(tmp_path, "sub/../x.bin") == (tmp_path / "x.bin").resolve()
            with And("traversal, absolute, NUL and empty names are rejected"):
                for bad in ["../secret.bin", "../../etc/passwd", "/etc/passwd", "a/../../b", "x\x00y", ""]:
                    with pytest.raises(UnsafePathError):
                        safe_join(tmp_path, bad)


def test_revision_pinning_policy(monkeypatch):
    with Scenario("Unit: Hub Revision Pinning Policy (OWASP LLM03)"):
        with Given("candidate revisions"):
            with Then("only a 40-hex commit SHA counts as pinned"):
                assert is_pinned_revision(GOOD_SHA)
                for bad in [None, "main", "v1.0", GOOD_SHA.upper(), GOOD_SHA[:-1], 123]:
                    assert not is_pinned_revision(bad)

        with Given("strict mode KILAT_REQUIRE_PINNED_REVISION=1"):
            monkeypatch.setenv("KILAT_REQUIRE_PINNED_REVISION", "1")
            with Then("unpinned downloads are refused and pinned ones allowed"):
                with pytest.raises(UnpinnedRevisionError):
                    enforce_revision_policy("org/model", "main")
                enforce_revision_policy("org/model", GOOD_SHA)
            with And("BasePreTrainedModel.from_pretrained refuses before any network access"):
                with pytest.raises(UnpinnedRevisionError):
                    BasePreTrainedModel.from_pretrained("definitely/not-a-local-dir-kilat", revision="main")


def test_checkpoint_tokenizer_config_cannot_import_arbitrary_code(tmp_path, monkeypatch):
    with Scenario("Unit: Malicious tokenizer_config.json in Checkpoint (OWASP LLM03/ML06)"):
        monkeypatch.delenv(_ENV_TRUSTED_MODULES, raising=False)
        monkeypatch.delenv("KILAT_TOKENIZER_BUILDER", raising=False)
        with Given("a downloaded checkpoint whose tokenizer_config.json points at a stdlib gadget"):
            (tmp_path / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_type": "custom", "custom_builder": "os:system"})
            )
            with When("AutoTokenizer.from_pretrained loads the directory"):
                with Then("loading is refused instead of silently falling back to another tokenizer"):
                    with pytest.raises(UntrustedModuleError, match="os"):
                        AutoTokenizer.from_pretrained(tmp_path)

        with Given("a byte_tokenizer_meta.json using custom_module/custom_class"):
            meta_dir = tmp_path / "meta"
            meta_dir.mkdir()
            (meta_dir / "byte_tokenizer_meta.json").write_text(
                json.dumps({"tokenizer_type": "custom", "custom_module": "subprocess", "custom_class": "Popen"})
            )
            with Then("the class-import path is refused as well"):
                with pytest.raises(UntrustedModuleError, match="subprocess"):
                    AutoTokenizer.from_pretrained(meta_dir)

        with Given("the first-party byte tokenizer builder"):
            cfg = TokenizerConfig(tokenizer_type="custom", custom_builder="kilat.data.tokenizer:build_byte_tokenizer")
            with Then("trusted builders keep working"):
                assert cfg.build() is not None


def test_sharded_index_traversal_is_refused(tmp_path):
    with Scenario("Unit: Sharded Weight Index Path Traversal"):
        with Given("a weights index whose weight_map escapes the checkpoint directory"):
            outside = tmp_path / "outside.bin"
            torch.save({"w": torch.zeros(1)}, outside)
            ckpt = tmp_path / "ckpt"
            ckpt.mkdir()
            (ckpt / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": {"w": "../outside.bin"}})
            )
            with When("the loader resolves shard paths"):
                with Then("UnsafePathError is raised before any file outside the root is read"):
                    with pytest.raises(UnsafePathError):
                        BasePreTrainedModel._load_state_dict(Path(ckpt))
