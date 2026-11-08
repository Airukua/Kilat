"""
Unit tests for Kilat Logit Processors and Warpers.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.pipeline.generation.logit_processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_temperature_logits_warper():
    with Scenario("Unit: Temperature Logits Warper Scaling"):
        with Given("raw unscaled logits"):
            warper = TemperatureLogitsWarper(temperature=2.0)
            input_ids = torch.tensor([[1, 2]])
            scores = torch.tensor([[2.0, 4.0, 6.0]])

            with When("temperature scaling is applied"):
                scaled = warper(input_ids, scores)
                with Then("all logits are divided by temperature (2.0)"):
                    expected = scores / 2.0
                    assert torch.allclose(scaled, expected)


def test_top_k_and_top_p_logits_warper():
    with Scenario("Unit: Top-K and Top-P Probability Logits Warpers"):
        with Given("a distribution of logits with clear rankings"):
            input_ids = torch.tensor([[1]])
            scores = torch.tensor([[1.0, 2.0, 3.0, 10.0, 4.0]])

            with When("TopKLogitsWarper with top_k=2 is applied"):
                topk_warper = TopKLogitsWarper(top_k=2)
                warped_k = topk_warper(input_ids, scores.clone())
                with Then("tokens outside the top 2 are masked with -infinity"):
                    # The top 2 values (10.0 and 4.0) should remain finite
                    assert torch.isfinite(warped_k[0, 3])
                    assert torch.isfinite(warped_k[0, 4])
                    assert warped_k[0, 0] == float("-inf")
                    assert warped_k[0, 1] == float("-inf")

            with When("TopPLogitsWarper with top_p=0.9 is applied"):
                topp_warper = TopPLogitsWarper(top_p=0.9)
                warped_p = topp_warper(input_ids, scores.clone())
                with Then("the highest probability token remains unmasked"):
                    assert torch.isfinite(warped_p[0, 3])


def test_repetition_penalty_processor():
    with Scenario("Unit: Repetition Penalty Logits Processor"):
        with Given("an input sequence with repeating tokens [10, 20, 10]"):
            input_ids = torch.tensor([[10, 20, 10]])
            # Positive logit for token 10
            scores = torch.tensor([[0.0] * 30])
            scores[0, 10] = 4.0

            with When("RepetitionPenaltyLogitsProcessor with penalty=2.0 is applied"):
                rep_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
                penalized = rep_proc(input_ids, scores.clone())
                with Then("the logit for previously generated token 10 is discounted"):
                    # For positive logit 4.0, penalty 2.0 divides by 2.0 -> 2.0
                    assert penalized[0, 10] < scores[0, 10]
                    assert abs(penalized[0, 10].item() - 2.0) < 1e-4


def test_min_length_processor():
    with Scenario("Unit: Min Length Logits Processor"):
        with Given("a sequence shorter than min_length=5 and eos_token_id=2"):
            proc = MinLengthLogitsProcessor(min_length=5, eos_token_id=2)
            short_ids = torch.tensor([[10, 20]])  # length 2 < 5
            scores = torch.tensor([[1.0, 1.0, 5.0, 1.0]])

            with When("processor is applied before reaching min_length"):
                processed = proc(short_ids, scores.clone())
                with Then("the EOS token logit is masked with -infinity to prevent premature termination"):
                    assert processed[0, 2] == float("-inf")


def test_forced_tokens_and_ngram_processors():
    with Scenario("Unit: Forced Tokens, Suppression, and No-Repeat N-Gram"):
        with Given("logit scores tensor"):
            scores = torch.zeros(1, 50)

            with When("applying ForcedBOSTokenLogitsProcessor on prompt start"):
                bos_proc = ForcedBOSTokenLogitsProcessor(bos_token_id=1)
                out_bos = bos_proc(torch.tensor([[]]), scores.clone())
                with Then("bos token logit remains 0 while others are masked"):
                    assert out_bos[0, 1] == 0.0
                    assert out_bos[0, 0] == float("-inf")

            with When("applying ForcedEOSTokenLogitsProcessor at max_length"):
                eos_proc = ForcedEOSTokenLogitsProcessor(max_length=4, eos_token_id=2)
                out_eos = eos_proc(torch.tensor([[10, 20, 30]]), scores.clone())
                with Then("eos token is forced"):
                    assert out_eos[0, 2] == 0.0
                    assert out_eos[0, 0] == float("-inf")

            with When("applying SuppressTokensLogitsProcessor"):
                sup_proc = SuppressTokensLogitsProcessor(suppress_tokens=[5, 6])
                out_sup = sup_proc(torch.tensor([[1, 2]]), scores.clone())
                with Then("specified tokens are masked"):
                    assert out_sup[0, 5] == float("-inf")
                    assert out_sup[0, 6] == float("-inf")

            with When("applying NoRepeatNGramLogitsProcessor with size 2"):
                ngram_proc = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=2)
                out_ngram = ngram_proc(torch.tensor([[10, 20, 10]]), scores.clone())
                with Then("previously followed token 20 is forbidden"):
                    assert out_ngram[0, 20] == float("-inf")


def test_extended_logit_processors_and_builders():
    from unittest.mock import MagicMock
    from kilat.pipeline.generation.generation_config import GenerationConfig
    from kilat.pipeline.generation.logit_processor import (
        GrammarConstrainedLogitsProcessor,
        MinNewTokensLengthLogitsProcessor,
        PrefixConstrainedLogitsProcessor,
        get_logits_processor,
        get_logits_warper,
    )

    with Scenario("Unit: Extended Logit Processors, Grammar Constraints, and Builder Functions"):
        with Given("MinNewTokensLengthLogitsProcessor"):
            proc = MinNewTokensLengthLogitsProcessor(
                prompt_length_to_skip=2,
                min_new_tokens=3,
                eos_token_id=2,
            )
            scores = torch.zeros(1, 10)

            with When("new tokens count is below min_new_tokens"):
                input_ids = torch.tensor([[1, 2, 3]])  # 3 - 2 = 1 < 3
                out = proc(input_ids, scores.clone())
                with Then("EOS token is suppressed"):
                    assert out[0, 2] == float("-inf")

            with When("passing invalid arguments to MinNewTokensLengthLogitsProcessor"):
                with Then("ValueErrors are raised"):
                    with pytest.raises(ValueError):
                        MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=-1, min_new_tokens=5, eos_token_id=2)
                    with pytest.raises(ValueError):
                        MinNewTokensLengthLogitsProcessor(prompt_length_to_skip=2, min_new_tokens=-1, eos_token_id=2)

        with Given("RepetitionPenaltyLogitsProcessor with negative logits and validation"):
            rep = RepetitionPenaltyLogitsProcessor(penalty=2.0)
            neg_scores = torch.tensor([[-2.0, 4.0]])
            input_ids = torch.tensor([[0, 1]])

            with When("applying penalty on negative and positive logits"):
                out_rep = rep(input_ids, neg_scores)
                with Then("negative logit becomes more negative and positive becomes smaller"):
                    assert out_rep[0, 0] == -4.0
                    assert out_rep[0, 1] == 2.0

            with When("penalty <= 0"):
                with Then("ValueError is raised"):
                    with pytest.raises(ValueError):
                        RepetitionPenaltyLogitsProcessor(penalty=0.0)

        with Given("NoRepeatNGramLogitsProcessor input length shorter than ngram"):
            short_ngram = NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=4)
            short_ids = torch.tensor([[1, 2]])
            scores = torch.ones(1, 5)
            with When("sequence length is less than n-gram size"):
                res = short_ngram(short_ids, scores)
                with Then("scores remain unaltered"):
                    assert torch.equal(res, scores)

            with When("no_repeat_ngram_size <= 0"):
                with Then("ValueError is raised"):
                    with pytest.raises(ValueError):
                        NoRepeatNGramLogitsProcessor(no_repeat_ngram_size=0)

        with Given("PrefixConstrainedLogitsProcessor"):
            def allowed_fn(batch_id, prefix):
                return [3, 4]
            prefix_proc = PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn=allowed_fn)
            scores = torch.zeros(1, 10)
            input_ids = torch.tensor([[1, 2]])

            with When("prefix constraint filters logits"):
                filtered = prefix_proc(input_ids, scores)
                with Then("only allowed tokens retain their score"):
                    assert filtered[0, 3] == 0.0
                    assert filtered[0, 4] == 0.0
                    assert filtered[0, 0] == float("-inf")

        with Given("GrammarConstrainedLogitsProcessor"):
            mock_grammar = MagicMock()
            mock_grammar.init_stacks.return_value = ["stack"]
            mock_grammar.accept_token_ids.return_value = ["stack_updated"]
            mock_grammar.accept_token_id.return_value = ["stack_next"]
            mock_grammar.batch_filter_vocab.return_value = torch.tensor([[False, True, False, False]])

            grammar_proc = GrammarConstrainedLogitsProcessor(mock_grammar)
            g_scores = torch.zeros(1, 4)
            g_input = torch.tensor([[10, 20]])

            with When("grammar processor is invoked for initial prefix"):
                g_out = grammar_proc(g_input, g_scores)
                with Then("vocab is filtered per grammar mask"):
                    assert g_out[0, 1] == 0.0
                    assert g_out[0, 0] == float("-inf")

            with When("incremental token arrives"):
                g_input2 = torch.tensor([[10, 20, 30]])
                g_out2 = grammar_proc(g_input2, g_scores)
                with Then("incremental grammar acceptance is applied"):
                    assert g_out2[0, 1] == 0.0

        with Given("GenerationConfig instances for warpers and processors builder"):
            with When("get_logits_warper is called with do_sample=True"):
                cfg_warper = GenerationConfig(do_sample=True, temperature=0.7, top_k=20, top_p=0.9)
                warpers = get_logits_warper(cfg_warper)
                with Then("three warpers are registered in list"):
                    assert len(warpers) == 3

            with When("get_logits_warper is called with do_sample=False"):
                cfg_no_sample = GenerationConfig(do_sample=False)
                warpers_empty = get_logits_warper(cfg_no_sample)
                with Then("empty list is returned"):
                    assert len(warpers_empty) == 0

            with When("get_logits_processor is called"):
                cfg_proc = GenerationConfig(
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    min_length=10,
                    eos_token_id=2,
                )
                processors = get_logits_processor(
                    cfg_proc,
                    input_ids_length=5,
                    prefix_allowed_tokens_fn=lambda b, ids: [1],
                )
                with Then("all configured processors are instantiated"):
                    assert len(processors) == 4
