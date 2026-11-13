"""
Unit tests for Kilat Text Generation: GenerationConfig, StoppingCriteria, and Samplers.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from kilat.models.transformer import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.inference.generation_config import GenerationConfig
from kilat.inference.stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    EosTokenCriteria,
    StoppingCriteriaList,
)
from kilat.inference.sampler import GreedySampler, TopKSampler, TopPSampler
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_generation_config_initialization():
    with Scenario("Unit: Generation Configuration Serialization & Defaults"):
        with Given("default GenerationConfig initialization"):
            cfg = GenerationConfig()
            with When("checking default decoding mode"):
                with Then("do_sample defaults to False (greedy decoding)"):
                    assert not cfg.do_sample
                with And("temperature is 1.0"):
                    assert cfg.temperature == 1.0
                with And("top_k is 50"):
                    assert cfg.top_k == 50

        with Given("custom generation parameters"):
            with When("custom values are specified"):
                custom = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=64)
                with Then("the custom settings are retained"):
                    assert custom.do_sample is True
                    assert custom.temperature == 0.7
                    assert custom.top_p == 0.9
                    assert custom.max_new_tokens == 64


def test_stopping_criteria_evaluation():
    with Scenario("Unit: Generation Stopping Criteria Rules"):
        with Given("a MaxLengthCriteria set to max_length=10"):
            max_len_crit = MaxLengthCriteria(max_length=10)
            with When("evaluating an input_ids tensor with 8 tokens"):
                short_ids = torch.zeros((1, 8), dtype=torch.long)
                stop_short = max_len_crit(short_ids, None)
                with Then("stopping criteria returns False"):
                    assert not stop_short

            with When("evaluating an input_ids tensor with 10 tokens"):
                full_ids = torch.zeros((1, 10), dtype=torch.long)
                stop_full = max_len_crit(full_ids, None)
                with Then("stopping criteria returns True"):
                    assert stop_full

        with Given("an EosTokenCriteria with eos_token_id=2"):
            eos_crit = EosTokenCriteria(eos_token_id=2)
            with When("last token is not EOS (token 5)"):
                non_eos = torch.tensor([[1, 3, 5]])
                assert not eos_crit(non_eos, None)
                with Then("generation continues"):
                    assert True

            with When("last token is EOS (token 2)"):
                with_eos = torch.tensor([[1, 3, 2]])
                with Then("stopping criteria triggers and returns True"):
                    assert eos_crit(with_eos, None)


def test_sampling_strategies():
    with Scenario("Unit: Token Sampling Strategies"):
        with Given("logits with a dominant token probability"):
            # Token 3 has the highest logit
            logits = torch.tensor([[0.1, 0.2, 0.3, 5.0, 0.4]])

            with When("GreedySampler is applied"):
                greedy = GreedySampler()
                sampled_token = greedy(logits)
                if hasattr(sampled_token, "tokens"):
                    token_val = sampled_token.tokens.item()
                elif isinstance(sampled_token, torch.Tensor):
                    token_val = sampled_token.item()
                else:
                    token_val = sampled_token
                with Then("the argmax token (index 3) is selected"):
                    assert token_val == 3

            with When("TopKSampler is applied with top_k=1"):
                topk = TopKSampler(top_k=1)
                sampled_topk = topk(logits)
                topk_val = sampled_topk.tokens.item() if hasattr(sampled_topk, "tokens") else sampled_topk.item()
                with Then("the top-1 token (index 3) is selected deterministically"):
                    assert topk_val == 3


def test_model_generate_mixin_execution():
    with Scenario("Unit: GenerationMixin Greedy and Sampling Execution"):
        with Given("a KilatTransformer model and prompt tokens"):
            cfg = KilatConfig(vocab_size=64, n_embd=32, n_layer=1, n_head=2, max_seq_len=64)
            model = KilatTransformer(cfg)
            input_ids = torch.tensor([[10, 20]])

            with When("checking can_generate"):
                with Then("model reports True"):
                    assert model.can_generate() is True

            with When("generating via greedy decoding (do_sample=False)"):
                gen_greedy = GenerationConfig(
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=0,
                    eos_token_id=2,
                )
                out_greedy = model.generate(input_ids, generation_config=gen_greedy)
                with Then("output length is extended by 3 new tokens"):
                    assert out_greedy.shape == (1, 5)

            with When("generating via sampling (do_sample=True)"):
                gen_sample = GenerationConfig(
                    max_new_tokens=3,
                    do_sample=True,
                    temperature=0.8,
                    top_k=10,
                    pad_token_id=0,
                    eos_token_id=2,
                )
                out_sample = model.generate(input_ids, generation_config=gen_sample)
                with Then("sampling output length is extended by 3 new tokens"):
                    assert out_sample.shape == (1, 5)


def test_generation_config_io_and_tokens_helpers():
    with Scenario("Unit: GenerationConfig Pretrained Save/Load and Max Tokens Calculation"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a configured GenerationConfig"):
                cfg = GenerationConfig(
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

                with When("querying dictionary representation and max new tokens"):
                    d = cfg.to_dict()
                    max_toks = cfg.get_max_new_tokens(input_length=20)
                    rep = repr(cfg)
                    with Then("dictionary and token limit are computed"):
                        assert d["max_new_tokens"] == 50
                        assert max_toks == 50
                        assert "GenerationConfig" in rep

                with When("saving and reloading GenerationConfig via save_pretrained and from_pretrained"):
                    cfg.save_pretrained(str(tmp_path))
                    loaded = GenerationConfig.from_pretrained(str(tmp_path))
                    with Then("reloaded configuration matches original settings"):
                        assert loaded.max_new_tokens == 50
                        assert loaded.temperature == 0.7
                        assert loaded.do_sample is True
