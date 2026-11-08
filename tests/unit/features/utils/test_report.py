"""
Unit tests for Kilat Reporting and Parameter Inspection.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch
import torch.nn as nn

from kilat.utils.report import count_parameters, parameter_breakdown, load_model
from kilat.configs.model_config import KilatConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_parameter_counting():
    with Scenario("Unit: Parameter Counting and Breakdown"):
        with Given("a simple model with known parameter counts"):
            model = nn.Sequential(
                nn.Linear(10, 20, bias=True),  # 10*20 + 20 = 220
                nn.Linear(20, 5, bias=False),  # 20*5 = 100
            )

            with When("counting total parameters"):
                total = count_parameters(model, trainable_only=False)
                with Then("the count equals 320"):
                    assert total == 320

            with When("freezing the second layer and counting trainable parameters"):
                model[1].weight.requires_grad = False
                trainable = count_parameters(model, trainable_only=True)
                with Then("the trainable count equals 220"):
                    assert trainable == 220

            with When("analyzing detailed parameter breakdown"):
                rows, shared = parameter_breakdown(model)
                with Then("the breakdown lists each named parameter layer"):
                    assert len(rows) >= 2
                    assert isinstance(shared, list)


def test_load_model_from_default_config():
    with Scenario("Unit: Model Loader Initialization"):
        with Given("no checkpoint or path provided"):
            with When("load_model is called with default fallback"):
                model = load_model()
                with Then("a KilatTransformer instance is created"):
                    assert model is not None
                    assert hasattr(model, "forward")
                    assert hasattr(model, "config")


def test_parameter_breakdown_with_shared_weights_and_loader_branches():
    import tempfile
    from pathlib import Path
    from kilat.models.transformer import KilatTransformer

    with Scenario("Unit: Shared Parameter Breakdown and Loader Edge Cases"):
        with Given("a model with shared parameters"):
            class SharedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    base = torch.randn(4, 4)
                    self.w1 = nn.Parameter(base)
                    self.w2 = nn.Parameter(base)

            m = SharedModel()
            with When("analyzing parameter breakdown on shared weights"):
                rows, shared = parameter_breakdown(m)
                with Then("shared groups are identified"):
                    assert len(shared) == 1
                    assert set(shared[0]) == {"w1", "w2"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("ambiguous or missing paths for load_model"):
                with When("specifying both checkpoint and config_path"):
                    with Then("ValueError is raised"):
                        with pytest.raises(ValueError):
                            load_model(checkpoint="ckpt", config_path="cfg.yaml")

                with When("specifying non-existent checkpoint directory"):
                    with And("FileNotFoundError is raised"):
                        with pytest.raises(FileNotFoundError):
                            load_model(checkpoint=str(tmp_path / "missing_dir"))

                with When("specifying non-existent config path"):
                    with And("FileNotFoundError is raised"):
                        with pytest.raises(FileNotFoundError):
                            load_model(config_path=str(tmp_path / "missing.yaml"))

            with Given("valid saved checkpoint directory and YAML config"):
                cfg = KilatConfig(vocab_size=32, n_embd=16, n_layer=1, n_head=2)
                ckpt_dir = tmp_path / "saved_ckpt"
                yaml_path = tmp_path / "model_config.yaml"
                model_to_save = KilatTransformer(cfg)
                model_to_save.save_pretrained(str(ckpt_dir))
                cfg.to_yaml(yaml_path)

                with When("loading from config_path with YAML"):
                    loaded_cfg_model = load_model(config_path=str(yaml_path))
                    with Then("model is instantiated"):
                        assert loaded_cfg_model is not None

                with When("loading from checkpoint directory"):
                    loaded_ckpt_model = load_model(checkpoint=str(ckpt_dir))
                    with And("model is loaded from checkpoint weights"):
                        assert loaded_ckpt_model is not None
