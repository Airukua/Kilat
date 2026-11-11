"""
Unit tests for stopping criteria in Kilat generation pipeline.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import time
import pytest
import torch

from kilat.inference.stopping_criteria import (
    CompositeCriteria,
    CustomCriteria,
    EndStringCriteria,
    EosTokenCriteria,
    IterationLimitCriteria,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MinLengthCriteria,
    StoppingCriteriaList,
    TimeoutCriteria,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_length_and_tokens_criteria():
    with Scenario("Unit: Length and Token Stopping Criteria"):
        with Given("an input sequence with 5 tokens"):
            input_ids = torch.tensor([[10, 20, 30, 40, 50]])

            with When("evaluating MaxLengthCriteria with max_length=5"):
                crit = MaxLengthCriteria(max_length=5)
                done = crit(input_ids)
                with Then("the criterion signals stopping"):
                    assert bool(done.all()) is True

            with When("evaluating MaxLengthCriteria with max_length=10"):
                crit_longer = MaxLengthCriteria(max_length=10)
                not_done = crit_longer(input_ids)
                with Then("the criterion does not stop"):
                    assert bool(not_done.any()) is False

            with When("evaluating MaxNewTokensCriteria"):
                new_tokens_crit = MaxNewTokensCriteria(max_new_tokens=2, input_length=3)
                stop_flag = new_tokens_crit(input_ids)
                with Then("it detects that 2 new tokens have been added"):
                    assert bool(stop_flag.all()) is True

            with When("evaluating MinLengthCriteria"):
                min_crit = MinLengthCriteria(min_length=10)
                # Should not allow stopping before min_length
                assert min_crit(input_ids).item() is False


def test_eos_token_and_timeout_criteria():
    with Scenario("Unit: EOS Token and Timeout Stopping Criteria"):
        with Given("sequences containing EOS tokens"):
            eos_id = 2
            input_ids = torch.tensor([[10, 20, eos_id], [15, 25, 35]])

            with When("evaluating EosTokenCriteria"):
                eos_crit = EosTokenCriteria(eos_token_id=eos_id)
                done_mask = eos_crit(input_ids)
                with Then("it identifies the sequence with EOS token"):
                    assert done_mask[0].item() is True
                    assert done_mask[1].item() is False

            with When("evaluating TimeoutCriteria"):
                timeout_crit = TimeoutCriteria(timeout_seconds=0.01)
                # First call sets start_time
                timeout_crit(input_ids)
                time.sleep(0.02)
                timeout_done = timeout_crit(input_ids)
                with Then("it signals timeout"):
                    assert bool(timeout_done.all()) is True


def test_composite_and_list_criteria():
    with Scenario("Unit: Composite Criteria and StoppingCriteriaList"):
        with Given("multiple stopping conditions"):
            input_ids = torch.tensor([[1, 2, 3]])
            crit1 = IterationLimitCriteria(max_iterations=1)
            crit2 = CustomCriteria(lambda ids, scores=None: torch.tensor([True]))

            with When("combining in a StoppingCriteriaList"):
                c_list = StoppingCriteriaList([crit1, crit2])
                with Then("evaluating the list returns overall stopping decision"):
                    res = c_list(input_ids)
                    assert bool(res.all()) is True

            with When("combining in CompositeCriteria"):
                comp = CompositeCriteria(criteria=[crit1, crit2])
                comp_res = comp(input_ids)
                with Then("composite criteria evaluates successfully"):
                    assert bool(comp_res.all()) is True


def test_end_string_stopping_criteria():
    with Scenario("Unit: EndStringCriteria Delimiter Detection"):
        with Given("a mock tokenizer with string decoding"):
            class MockTokenizer:
                def decode(self, seq, skip_special_tokens=False):
                    return "".join(chr(t) for t in seq)

            tokenizer = MockTokenizer()
            # ASCII for "HELLO<stop>"
            input_stop = torch.tensor([[ord(c) for c in "HELLO<stop>"]])
            input_continue = torch.tensor([[ord(c) for c in "HELLO WORLD"]])

            with When("evaluating EndStringCriteria with end_strings='<stop>'"):
                crit = EndStringCriteria(tokenizer=tokenizer, end_strings="<stop>")
                stop_res = crit(input_stop)
                cont_res = crit(input_continue)

                with Then("stopping is flagged only when delimiter is present"):
                    assert stop_res[0].item() is True
                    assert cont_res[0].item() is False


def test_stopping_criteria_builders_and_convenience():
    from kilat.inference.generation_config import GenerationConfig
    from kilat.inference.stopping_criteria import (
        get_stopping_criteria,
        should_stop,
    )

    with Scenario("Unit: Stopping Criteria Builder Functions and Convenience Checks"):
        with Given("a GenerationConfig and input sequence"):
            cfg = GenerationConfig(
                max_length=10,
                max_new_tokens=5,
                eos_token_id=2,
            )
            input_ids = torch.tensor([[1, 2, 3]])

            with When("building criteria using get_stopping_criteria"):
                def custom_fn(ids, scores=None):
                    return torch.tensor([False])
                crit_list = get_stopping_criteria(
                    cfg,
                    input_ids,
                    custom_stopping_criteria=[custom_fn],
                )
                with Then("criteria are registered in list"):
                    assert len(crit_list) >= 3

            with When("evaluating should_stop convenience function"):
                cfg_done = GenerationConfig(max_length=3, eos_token_id=2)
                stop_status = should_stop(input_ids, cfg_done)
                with Then("stopping decision matches criteria"):
                    assert bool(stop_status.all()) is True

            with When("passing invalid boundary values to criteria"):
                with Then("ValueErrors are properly raised"):
                    with pytest.raises(ValueError):
                        MaxLengthCriteria(max_length=-1)
                    with pytest.raises(ValueError):
                        MinLengthCriteria(min_length=-1)
                    with pytest.raises(ValueError):
                        IterationLimitCriteria(max_iterations=0)
