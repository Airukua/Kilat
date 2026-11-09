"""
BDD-style tests for tools/audit/data_auditor.py (Data & Boundary Integrity Auditor).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from tools.audit.data_auditor import (
    audit_eos_boundaries,
    audit_ngram_contamination,
    run_full_data_audit,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_data_auditor_eos_boundaries_present():
    with Scenario("Tools: Data Auditor EOS Boundary Presence & Distribution"):
        with Given("a token array containing 3 distinct stories separated by EOS token 50256"):
            story = [10, 20, 30, 40, 50, 60, 70, 80, 90, 50256]
            tokens = np.array(story * 3, dtype=np.int64)

            with When("audit_eos_boundaries evaluates the token array"):
                report = audit_eos_boundaries(tokens, eos_token_id=50256)

                with Then("the verdict is PASS_EOS_PRESENT with has_eos=True"):
                    assert report["has_eos"] is True
                    assert report["verdict"] == "PASS_EOS_PRESENT"

                with And("the EOS count is exactly 3"):
                    assert report["eos_count"] == 3

                with And("the mean and median story length are 10.0 tokens"):
                    assert report["mean_story_length"] == 10.0
                    assert report["median_story_length"] == 10.0


def test_data_auditor_ngram_clean_partitions():
    with Scenario("Tools: Data Auditor N-Gram Contamination Between Splits"):
        with Given("disjoint train tokens [1..100] and eval tokens [101..200]"):
            train = np.arange(1, 101, dtype=np.int64)
            eval_ = np.arange(101, 201, dtype=np.int64)

            with When("audit_ngram_contamination checks 8-gram and 13-gram overlap"):
                report = audit_ngram_contamination(train, eval_, ngram_sizes=(8, 13))

                with Then("the verdict is CLEAN"):
                    assert report["verdict"] == "CLEAN"

                with And("the contamination rate is 0.0 for both n-gram sizes"):
                    assert report["ngrams"]["8gram"]["contamination_rate"] == 0.0
                    assert report["ngrams"]["13gram"]["contamination_rate"] == 0.0


def test_data_auditor_full_audit_runs_end_to_end(tmp_path: Path):
    with Scenario("Tools: Data Auditor Full End-to-End Pipeline on Disk"):
        with Given("persisted train and eval .npy token files on disk"):
            train_file = tmp_path / "train.npy"
            eval_file = tmp_path / "eval.npy"
            train_data = np.array([1, 2, 3, 50256, 4, 5, 6, 50256] * 10, dtype=np.int64)
            eval_data = np.array([10, 20, 30, 50256, 40, 50, 60, 50256] * 5, dtype=np.int64)
            np.save(train_file, train_data)
            np.save(eval_file, eval_data)

            with When("run_full_data_audit executes across both files"):
                report = run_full_data_audit(train_path=train_file, eval_path=eval_file, eos_token_id=50256)

                with Then("the audit report contains train, eval, and contamination sections"):
                    assert "train_eos_audit" in report
                    assert "eval_eos_audit" in report
                    assert "contamination_audit" in report

                with And("the token counts and verdict are accurately recorded"):
                    assert report["train_tokens_total"] == len(train_data)
                    assert report["eval_tokens_total"] == len(eval_data)
                    assert report["contamination_audit"]["verdict"] == "CLEAN"


def test_data_auditor_negative_zero_eos_boundaries():
    with Scenario("Tools: Data Auditor Negative Testing - Zero EOS Boundaries"):
        with Given("a token sequence containing zero EOS boundary tokens"):
            tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

            with When("audit_eos_boundaries inspects the un-delimited sequence"):
                report = audit_eos_boundaries(tokens, eos_token_id=50256)

                with Then("the report reports has_eos=False with FAIL_ZERO_EOS verdict"):
                    assert report["has_eos"] is False
                    assert report["eos_count"] == 0
                    assert report["verdict"] == "FAIL_ZERO_EOS"

                with And("the warning message explicitly flags missing boundaries"):
                    assert "Zero EOS tokens found" in report["message"]


def test_data_auditor_negative_100_percent_contamination():
    with Scenario("Tools: Data Auditor Negative Testing - 100% Contaminated Splits"):
        with Given("identical token sequences used for both train and eval"):
            repeated = np.tile(np.arange(50, dtype=np.int64), 5)

            with When("audit_ngram_contamination checks for data leakage"):
                report = audit_ngram_contamination(repeated, repeated, ngram_sizes=(8, 13))

                with Then("the verdict is CONTAMINATED"):
                    assert report["verdict"] == "CONTAMINATED"

                with And("the contamination rate exceeds 90%"):
                    assert report["ngrams"]["8gram"]["contamination_rate"] > 0.90
                    assert report["ngrams"]["13gram"]["contamination_rate"] > 0.90


def test_data_auditor_negative_missing_files(tmp_path: Path):
    with Scenario("Tools: Data Auditor Negative Testing - Missing Files Handling"):
        with Given("nonexistent file paths on disk"):
            missing_train = tmp_path / "nonexistent_train.npy"
            valid_eval = tmp_path / "valid_eval.npy"
            np.save(valid_eval, np.array([1, 2, 3], dtype=np.int64))

            with When("auditing a missing train file"):
                with Then("a FileNotFoundError is raised"):
                    with pytest.raises(FileNotFoundError, match="Train data file not found"):
                        run_full_data_audit(train_path=missing_train, eval_path=valid_eval)

            with When("auditing a missing eval file"):
                valid_train = tmp_path / "valid_train.npy"
                np.save(valid_train, np.array([1, 2, 3], dtype=np.int64))
                missing_eval = tmp_path / "nonexistent_eval.npy"

                with Then("a FileNotFoundError is raised"):
                    with pytest.raises(FileNotFoundError, match="Eval data file not found"):
                        run_full_data_audit(train_path=valid_train, eval_path=missing_eval)
