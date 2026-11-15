from __future__ import annotations

import pytest

from kilat.assurance.chaos import ALL_HF_SCENARIO_IDS, run_hf_chaos_suite
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_hf_chaos_suite_resilience_across_all_8_scenarios() -> None:
    with Scenario("Integration: Hugging Face Chaos Engineering Suite HFC-01..HFC-08 Resilience"):
        with Given("the 8 Hugging Face fault injection scenarios (DynamicCache, schema drift, sampling, masking, PEFT, boundary loss, streaming)"):
            assert len(ALL_HF_SCENARIO_IDS) == 8

        with When("run_hf_chaos_suite('all') executes baseline and all 8 chaos injection scenarios"):
            report = run_hf_chaos_suite("all")

        with Then("clean baseline generation passes without failure"):
            assert report.baseline_passed is True

        with And("all 8 Hugging Face chaos scenarios pass with 100% resilience (0 false negatives)"):
            failed = [s.to_dict() for s in report.scenarios if not s.caught]
            assert report.scenarios_total == 8
            assert report.scenarios_passed == 8, f"Failed scenarios: {failed}"
            assert report.false_negatives == 0
            assert report.passed is True


def test_hf_chaos_suite_rejects_unknown_scenario_id() -> None:
    with Scenario("Integration: HF Chaos Runner Rejects Unknown Scenario Identifier"):
        with Given("an invalid HF chaos scenario name 'HFC-99'"):
            bad_id = "HFC-99"

        with When("run_hf_chaos_suite is invoked with the unknown identifier"):
            with pytest.raises(ValueError, match="Unknown HF chaos scenario IDs"):
                run_hf_chaos_suite(bad_id)

        with Then("a descriptive ValueError is raised immediately"):
            assert True
