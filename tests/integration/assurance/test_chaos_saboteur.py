from __future__ import annotations

import pytest

from kilat.assurance.chaos import ALL_SCENARIO_IDS, run_chaos_suite
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_active_chaos_saboteur_catches_all_8_silent_ml_bugs() -> None:
    with Scenario("Integration: Active Chaos Saboteur C-01..C-08 100% Meta-Detection & Clean Restore"):
        with Given("the non-destructive in-memory Saboteur across all 8 mandatory ML fault scenarios"):
            assert len(ALL_SCENARIO_IDS) == 8

        with When("run_chaos_suite('all') executes baseline -> C-01..C-08 -> post-chaos verification"):
            suite_report = run_chaos_suite("all")

        with Then("clean baseline passes before chaos injection (0 false positives)"):
            assert suite_report.baseline_passed is True

        with And("all 8 silent ML bugs C-01..C-08 are caught with exact AssuranceFailureCodes (0 false negatives)"):
            failed_catches = [s.to_dict() for s in suite_report.scenarios if not s.caught]
            assert suite_report.scenarios_total == 8
            assert suite_report.scenarios_caught == 8, f"Uncaught scenarios: {failed_catches}"
            assert suite_report.false_negatives == 0

        with And("post-chaos model re-verification passes 100%, proving zero residual hook/state mutation"):
            assert suite_report.post_clean_passed is True
            assert suite_report.passed is True


def test_chaos_suite_rejects_unknown_scenario_id() -> None:
    with Scenario("Integration: Chaos Runner Rejects Unknown Scenario Identifier"):
        with Given("an invalid chaos scenario name 'C-99'"):
            bad_id = "C-99"

        with When("run_chaos_suite is invoked with the unknown identifier"):
            with pytest.raises(ValueError, match="Unknown chaos scenario IDs"):
                run_chaos_suite(bad_id)

        with Then("a descriptive ValueError is raised immediately"):
            assert True
