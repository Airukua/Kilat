"""
Unit tests for Kilat BasePreTrainedModel and Hugging Face integration utilities.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from kilat.utils.base_model import BasePreTrainedModel, BaseConfig, _shard_state_dict
from kilat.configs.model_config import KilatConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


class DummyModel(BasePreTrainedModel):
    config_class = KilatConfig

    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.wte.weight

    def forward(self, input_ids):
        h = self.wte(input_ids)
        h = self.linear(h)
        logits = self.lm_head(h)
        return logits


def test_base_model_initialization_and_weights():
    with Scenario("Unit: BasePreTrainedModel Initialization & Utilities"):
        with Given("a DummyModel initialized with KilatConfig"):
            config = KilatConfig(vocab_size=100, n_embd=32, n_layer=1, n_head=2, tie_word_embeddings=True)
            model = DummyModel(config)

            with When("checking weight initialization and parameter counts"):
                num_params = model.num_parameters()
                with Then("parameter count is positive"):
                    assert num_params > 0

                with And("get_memory_footprint returns memory in formatted string"):
                    footprint = model.get_memory_footprint()
                    assert isinstance(footprint, str) and len(footprint) > 0

            with When("testing weight tying between embeddings and lm_head"):
                model._tie_weights()
                with Then("the weights are shared in memory"):
                    assert model.lm_head.weight is model.wte.weight


def test_base_model_save_and_load_pretrained():
    with Scenario("Unit: BasePreTrainedModel Save and Load Pretrained"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "dummy_saved"

            with Given("an initialized model instance"):
                config = KilatConfig(vocab_size=100, n_embd=32, n_layer=1, n_head=2, tie_word_embeddings=False)
                model = DummyModel(config)

                with When("save_pretrained is executed"):
                    model.save_pretrained(str(save_path), safe_serialization=True)
                    with Then("config.json and model.safetensors exist on disk"):
                        assert (save_path / "config.json").exists()
                        assert (save_path / "model.safetensors").exists()

                with When("from_pretrained loads the model from disk"):
                    loaded = DummyModel.from_pretrained(str(save_path))
                    with Then("the loaded model parameters match original weights"):
                        for p1, p2 in zip(model.parameters(), loaded.parameters()):
                            assert torch.allclose(p1, p2)


def test_base_config_and_state_sharding():
    with Scenario("Unit: BaseConfig Serialization and State Dict Sharding"):
        with Given("a BaseConfig instance"):
            cfg = BaseConfig()
            with When("converting to dictionary and string"):
                d = cfg.to_dict()
                rep = repr(cfg)
                with Then("dictionary has required token ids"):
                    assert "pad_token_id" in d
                    assert "BaseConfig" in rep

            with When("restoring from dictionary"):
                cfg2 = BaseConfig.from_dict(d)
                with Then("restored instance preserves settings"):
                    assert cfg2.pad_token_id == cfg.pad_token_id

        with Given("a state dict with tensors"):
            sd = {"weight1": torch.randn(10, 10), "weight2": torch.randn(10, 10)}
            with When("sharding state dict with max_shard_bytes"):
                shards = _shard_state_dict(sd, max_shard_bytes=200)
                with Then("sharding returns partitioned shards"):
                    assert len(shards) >= 1
                    assert isinstance(shards[0], dict)

        with Given("a BaseConfig with torch_dtype"):
            with When("saving BaseConfig to directory and loading back"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cfg_dir = Path(tmp_dir) / "config_dir"
                    b_cfg = BaseConfig(torch_dtype="float32", hidden_size=64)
                    b_cfg.save_pretrained(cfg_dir)
                    with Then("config.json is created on disk"):
                        assert (cfg_dir / "config.json").exists()

                    loaded_cfg = BaseConfig.from_pretrained(cfg_dir)
                    with And("loaded config has same attributes"):
                        assert loaded_cfg.hidden_size == 64
                        assert loaded_cfg.dtype == "float32"


def test_base_model_pytorch_bin_and_properties():
    with Scenario("Unit: BasePreTrainedModel PyTorch Bin Serialization & Properties"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "bin_model"

            with Given("an initialized model instance"):
                config = KilatConfig(vocab_size=64, n_embd=16, n_layer=1, n_head=2)
                model = DummyModel(config)

                with When("inspecting device and trainable parameters"):
                    dev = model.device
                    trainable_params = model.num_parameters(only_trainable=True)
                    all_params = model.num_parameters(only_trainable=False)
                    with Then("device and parameter count are accurate"):
                        assert dev == torch.device("cpu")
                        assert trainable_params > 0
                        assert trainable_params == all_params

                with When("saving with safe_serialization=False (pytorch_model.bin)"):
                    model.save_pretrained(str(save_path), safe_serialization=False)
                    with Then("pytorch_model.bin is created"):
                        assert (save_path / "pytorch_model.bin").exists()

                with When("reloading model from pytorch_model.bin"):
                    loaded = DummyModel.from_pretrained(str(save_path))
                    with Then("the weights match original weights"):
                        for p1, p2 in zip(model.parameters(), loaded.parameters()):
                            assert torch.allclose(p1, p2)


class CheckpointableDummyModel(BasePreTrainedModel):
    config_class = KilatConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.block = nn.Linear(config.n_embd, config.n_embd)
        self.block.gradient_checkpointing = False


def test_base_model_sharded_save_and_load():
    with Scenario("Unit: BasePreTrainedModel Sharded Weight Serialization & Deserialization"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            safe_sharded_dir = tmp_path / "safe_sharded"
            bin_sharded_dir = tmp_path / "bin_sharded"

            with Given("a model instance with multiple weight parameters"):
                config = KilatConfig(vocab_size=128, n_embd=32, n_layer=2, n_head=4)
                model = DummyModel(config)

                with When("saving with small max_shard_bytes producing sharded safetensors"):
                    model.save_pretrained(str(safe_sharded_dir), max_shard_bytes=500, safe_serialization=True)
                    with Then("model.safetensors.index.json exists"):
                        assert (safe_sharded_dir / "model.safetensors.index.json").exists()

                with When("loading model from sharded safetensors"):
                    loaded = DummyModel.from_pretrained(str(safe_sharded_dir))
                    with Then("parameters match"):
                        for p1, p2 in zip(model.parameters(), loaded.parameters()):
                            assert torch.allclose(p1, p2)

                with When("saving with small max_shard_bytes producing sharded pytorch_model.bin"):
                    model.save_pretrained(str(bin_sharded_dir), max_shard_bytes=500, safe_serialization=False)
                    with Then("pytorch_model.bin.index.json exists"):
                        assert (bin_sharded_dir / "pytorch_model.bin.index.json").exists()

                with When("loading model from sharded pytorch_model.bin"):
                    loaded_bin = DummyModel.from_pretrained(str(bin_sharded_dir))
                    with Then("bin parameters match"):
                        for p1, p2 in zip(model.parameters(), loaded_bin.parameters()):
                            assert torch.allclose(p1, p2)


def test_base_model_tied_weights_and_checkpointing():
    from unittest.mock import MagicMock, patch

    with Scenario("Unit: BasePreTrainedModel Tied Weights, Gradient Checkpointing, & Tags"):
        with Given("a CheckpointableDummyModel instance"):
            config = KilatConfig(vocab_size=64, n_embd=16, n_layer=1, n_head=2)
            model = CheckpointableDummyModel(config)

            with When("enabling and disabling gradient checkpointing"):
                model.gradient_checkpointing_enable()
                with Then("is_gradient_checkpointing is True"):
                    assert model.is_gradient_checkpointing is True

                model.gradient_checkpointing_disable()
                with And("is_gradient_checkpointing is False"):
                    assert model.is_gradient_checkpointing is False

            with When("enabling with custom gradient checkpointing function"):
                custom_fn = MagicMock()
                model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_fn)
                with Then("block gradient_checkpointing is set to custom function"):
                    assert model.block.gradient_checkpointing is custom_fn

            with When("testing tied weight removal from state dict"):
                dummy_model = DummyModel(config)
                dummy_model._tied_weights_keys = {"lm_head.weight": "wte.weight"}
                state = {"wte.weight": torch.ones(2, 2), "lm_head.weight": torch.ones(2, 2)}
                cleaned = dummy_model._remove_tied_weights(state)
                with Then("redundant key is pruned"):
                    assert "lm_head.weight" not in cleaned
                    assert "wte.weight" in cleaned

            with When("adding model tags"):
                dummy_model.add_model_tags(["language-model", "kilat"])
                dummy_model.add_model_tags("fast")
                with Then("tags list contains all added tags"):
                    assert "language-model" in dummy_model._hub_tags
                    assert "fast" in dummy_model._hub_tags

            with When("pushing model to hub using mock HfApi"):
                with patch("kilat.utils.base_model._hf_hub.HfApi") as mock_hf:
                    mock_api_inst = MagicMock()
                    mock_hf.return_value = mock_api_inst
                    dummy_model.push_to_hub("kilat-org/test-model")
                    with Then("HfApi create_repo and upload_folder are called"):
                        assert mock_api_inst.create_repo.called
                        assert mock_api_inst.upload_folder.called
