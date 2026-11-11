"""
Unit tests for Gate 0 (G0_SECURITY_SUPPLY_CHAIN).

Meta-assertion contract: a gate that has never been shown to fail on a known-bad input is untrusted.
Each rule is proven against a planted insecure source file; the real Kilat package must PASS.
"""

import pytest

from kilat.assurance.gates.gate0_security import (
    GATE_ID,
    audit_tokenizer_config,
    run_gate0_security,
    scan_python_sources,
)
from kilat.assurance.types import AssuranceError, AssuranceFailureCode
from kilat.configs.tokenizer_config import TokenizerConfig
from tests.shared.bdd import And, Given, Scenario, Then, When

PLANTED = {
    "SEC-DESER-1": "import torch\nx = torch.load('m.bin')\n",
    "SEC-DESER-2": "import pickle\nx = pickle.loads(b'')\n",
    "SEC-DESER-3": "import yaml\nx = yaml.load(open('c.yaml'))\n",
    "SEC-DESER-4": "import numpy as np\nx = np.load('a.npy', allow_pickle=True)\n",
    "SEC-RCE-1": "from transformers import AutoModel\nm = AutoModel.from_pretrained('x', trust_remote_code=True)\n",
    "SEC-RCE-2": "eval('1+1')\n",
    "SEC-SHELL-1": "import subprocess\nsubprocess.run('ls', shell=True)\n",
}

SAFE_SOURCE = (
    "import torch, yaml\n"
    "a = torch.load('m.bin', weights_only=True)\n"
    "b = yaml.safe_load('x: 1')\n"
    "model = object()\n"
    "def f(m):\n    m.eval()\n"
)


def _plant(root, name, src):
    pkg = root / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / f"{name.replace('-', '_').lower()}.py").write_text(src)
    return pkg


def test_gate0_passes_on_real_kilat_package():
    with Scenario("Assurance: Gate 0 Clean Baseline on Kilat Source"):
        with Given("the installed kilat package and a default config"):
            with When("run_gate0_security executes"):
                res = run_gate0_security()
                with Then("it passes with zero findings over a non-trivial file count"):
                    assert res.gate_id == GATE_ID
                    assert res.status == "PASS"
                    assert res.metrics["findings_count"] == 0
                    assert res.metrics["files_scanned"] > 50
                with And("the frameworks it attests to are recorded in the report"):
                    joined = " ".join(res.metrics["frameworks"])
                    assert "OWASP" in joined and "NCSC" in joined


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_gate0_detects_each_planted_sink(tmp_path, rule):
    with Scenario(f"Assurance: Gate 0 Detects {rule}"):
        with Given(f"a source tree containing a {rule} sink"):
            pkg = _plant(tmp_path, rule, PLANTED[rule])
            with When("the AST scanner runs"):
                findings, n = scan_python_sources(pkg)
                with Then("exactly that rule fires"):
                    assert n == 1
                    assert [f.rule for f in findings] == [rule]
            with When("the full gate runs on that tree"):
                with Then("it raises AssuranceError rather than passing"):
                    with pytest.raises(AssuranceError) as exc:
                        run_gate0_security(source_roots=[pkg])
                    assert exc.value.details["findings_count"] == 1


def test_gate0_has_no_false_positives_on_safe_idioms(tmp_path):
    with Scenario("Assurance: Gate 0 Ignores Safe Idioms"):
        with Given("weights_only torch.load, yaml.safe_load and nn.Module.eval()"):
            pkg = _plant(tmp_path, "safe", SAFE_SOURCE)
            with Then("no findings are reported"):
                findings, _ = scan_python_sources(pkg)
                assert findings == []


def test_gate0_failure_code_priority(tmp_path):
    with Scenario("Assurance: Gate 0 Reports the Most Severe Code"):
        with Given("a tree with both shell and remote-code sinks"):
            pkg = _plant(tmp_path, "a", PLANTED["SEC-SHELL-1"])
            _plant(tmp_path, "b", PLANTED["SEC-RCE-1"])
            with Then("ERR_REMOTE_CODE_EXECUTION wins"):
                with pytest.raises(AssuranceError) as exc:
                    run_gate0_security(source_roots=[pkg])
                assert exc.value.code is AssuranceFailureCode.ERR_REMOTE_CODE_EXECUTION


def test_gate0_refuses_vacuous_pass(tmp_path):
    with Scenario("Assurance: Gate 0 Never Passes on Zero Files"):
        with Given("an empty source root"):
            with Then("the gate fails instead of reporting PASS"):
                with pytest.raises(AssuranceError, match="0 files"):
                    run_gate0_security(source_roots=[tmp_path])


def test_gate0_audits_untrusted_tokenizer_config(monkeypatch):
    with Scenario("Assurance: Gate 0 Config Audit of Tokenizer Imports"):
        monkeypatch.delenv("KILAT_TRUSTED_MODULES", raising=False)
        monkeypatch.delenv("KILAT_TOKENIZER_BUILDER", raising=False)

        class _Cfg:
            tokenizer = TokenizerConfig(tokenizer_type="custom", custom_builder="evil_pkg.hook:run")

        with Given("a MainConfig-like object whose tokenizer builder is outside the allowlist"):
            with Then("the audit flags it and the gate fails with ERR_UNTRUSTED_DYNAMIC_IMPORT"):
                assert [f.rule for f in audit_tokenizer_config(_Cfg.tokenizer)] == ["SEC-IMPORT-1"]
                with pytest.raises(AssuranceError) as exc:
                    run_gate0_security(_Cfg())
                assert exc.value.code is AssuranceFailureCode.ERR_UNTRUSTED_DYNAMIC_IMPORT
        with Given("a first-party builder"):
            ok = TokenizerConfig(tokenizer_type="custom", custom_builder="kilat.data.tokenizer:build_byte_tokenizer")
            with Then("no finding is produced"):
                assert audit_tokenizer_config(ok) == []
