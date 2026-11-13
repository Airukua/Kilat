"""
Unit and regression tests for Kilat Static Code Quality & SonarLint Hygiene Auditor.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from kilat.assurance.quality import (
    QualityFinding,
    audit_codebase,
    audit_python_code,
    audit_python_file,
)
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_sonar_s107_parameter_limit():
    with Scenario("Quality: python:S107 flags functions with parameter count > 13"):
        with Given("Python code defining a function with 14 parameters"):
            code = (
                "def broad_function(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):\n"
                "    return p1\n"
            )
            with When("audit_python_code analyzes the syntax tree"):
                findings = audit_python_code(code, "test.py")
                with Then("a python:S107 finding is recorded"):
                    assert any(f.rule == "python:S107" for f in findings)
                    finding = next(f for f in findings if f.rule == "python:S107")
                    assert "has 14 parameters" in finding.detail

        with Given("Python code with Triton JIT kernel having > 13 parameters"):
            triton_code = (
                "@triton.jit\n"
                "def flash_kernel(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15):\n"
                "    pass\n"
            )
            with When("audit_python_code analyzes the Triton kernel"):
                findings = audit_python_code(triton_code, "kernel.py")
                with Then("Triton JIT kernel is exempted due to flat hardware pointers"):
                    assert not any(f.rule == "python:S107" for f in findings)


def test_sonar_s5713_redundant_exception():
    with Scenario("Quality: python:S5713 flags redundant Exception in except handlers"):
        with Given("Python code catching both ValueError and Exception"):
            code = (
                "try:\n"
                "    pass\n"
                "except (ValueError, Exception):\n"
                "    pass\n"
            )
            with When("audit_python_code analyzes the except handler"):
                findings = audit_python_code(code, "except.py")
                with Then("a python:S5713 finding is recorded"):
                    assert any(f.rule == "python:S5713" for f in findings)


def test_sonar_s3457_static_fstrings():
    with Scenario("Quality: python:S3457 flags static f-strings without replacement fields"):
        with Given("Python code containing a static f-string without variables"):
            code = 'msg = f"Static string with no vars"\n'
            with When("audit_python_code analyzes the string literal"):
                findings = audit_python_code(code, "static.py")
                with Then("a python:S3457 finding is recorded"):
                    assert any(f.rule == "python:S3457" for f in findings)

        with Given("Python code containing format specifiers inside FormattedValue"):
            code = 'val = 12\nmsg = f"Value is {val:<10}"\n'
            with When("audit_python_code analyzes the formatted value"):
                findings = audit_python_code(code, "spec.py")
                with Then("format specifiers are not falsely flagged as static f-strings"):
                    assert len(findings) == 0


def test_sonar_s8513_and_s3358():
    with Scenario("Quality: python:S8513 chained startswith and python:S3358 nested ternaries"):
        with Given("code with chained startswith calls on the same object"):
            code = 'ok = s.startswith("a") or s.startswith("b")\n'
            with When("audit_python_code analyzes the boolean expression"):
                findings = audit_python_code(code, "start.py")
                with Then("a python:S8513 finding is recorded"):
                    assert any(f.rule == "python:S8513" for f in findings)

        with Given("code with nested ternary conditional expressions"):
            code = 'val = 1 if cond1 else (2 if cond2 else 3)\n'
            with When("audit_python_code analyzes the conditional expression"):
                findings = audit_python_code(code, "ternary.py")
                with Then("a python:S3358 finding is recorded"):
                    assert any(f.rule == "python:S3358" for f in findings)


def test_clean_repository_audit():
    with Scenario("Quality: Kilat source code tree has 0 code smell findings"):
        with Given("the kilat package source root"):
            src_root = Path("src/kilat")
            with When("audit_codebase scans the entire source tree"):
                findings, file_count = audit_codebase([src_root])
                with Then("zero code smells or defects are detected"):
                    assert len(findings) == 0, f"Unexpected findings: {findings}"
                    assert file_count > 30
