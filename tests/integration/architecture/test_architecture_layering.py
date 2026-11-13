"""
Architecture fitness test: strict downward dependency layering (senior-software-architecture Pillar 1 & 2).

Every intra-package `kilat.*` import (module-level *and* function-local) is checked against the
allowed-dependency matrix below. Pre-existing violations live in KNOWN_LAYERING_DEBT, which is a
ratchet: new violations fail, and fixed debt must be removed from the register (so it cannot regrow).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.shared.bdd import And, Given, Scenario, Then, When

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "kilat"

# Inference modules that are pure decoding primitives (no model/engine knowledge). They sit below
# `models` because KilatTransformer mixes in GenerationMixin.
DECODING_PRIMITIVES = frozenset({
    "inference.generation_mixin",
    "inference.generation_config",
    "inference.logit_processor",
    "inference.sampler",
    "inference.beam_search",
    "inference.stopping_criteria",
})

# Layer -> set of kilat layers it may import. "*" means unrestricted (composition roots).
ALLOWED: dict[str, frozenset[str]] = {
    "utils": frozenset(),
    "configs": frozenset({"utils"}),
    "decoding": frozenset({"utils", "configs"}),
    "models": frozenset({"utils", "configs", "decoding"}),
    "data": frozenset({"utils", "configs"}),
    "engine": frozenset({"utils", "configs", "models", "data", "decoding"}),
    "distillation": frozenset({"utils", "configs", "models", "data", "decoding", "engine"}),
    "inference": frozenset({"utils", "configs", "models", "data", "decoding"}),
    "assurance": frozenset({"utils", "configs", "models", "data", "decoding", "engine", "inference"}),
    "benchmarking": frozenset({"utils", "configs", "models", "data", "decoding", "engine", "inference"}),
    "integrations": frozenset({"utils", "configs", "models", "decoding", "inference"}),
    "cli": frozenset({"*"}),
    "__init__": frozenset({"*"}),
    "__main__": frozenset({"*"}),
}

# Backward-compatibility shims explicitly sanctioned by the Safe Migration Protocol.
SANCTIONED_SHIMS = frozenset({
    ("utils/base_model.py", "models"),
    ("utils/report.py", "models"),
    ("utils/__init__.py", "models"),  # PEP 562 lazy re-export of models.base symbols
})

# Ratchet register of pre-existing violations (file relative to src/kilat, target layer).
# Remove an entry once it is fixed; never add new ones without an architecture review.
KNOWN_LAYERING_DEBT = frozenset({
    # BaseConfig lives in models.base; KilatConfig (Layer 1) subclasses it.
    ("configs/model_config.py", "models"),
    # TrainingConfig.to_training_arguments() adapts to engine.args.TrainingArguments.
    ("configs/training_config.py", "engine"),
})


def _layer_of(dotted: str) -> str:
    """Map 'kilat.x.y' or 'x/y.py'-style module paths to an architectural layer name."""
    parts = dotted.split(".")
    if parts and parts[0] == "kilat":
        parts = parts[1:]
    if not parts:
        return "__init__"
    if len(parts) >= 2 and ".".join(parts[:2]) in DECODING_PRIMITIVES:
        return "decoding"
    return parts[0]


def _collect_violations() -> set[tuple[str, str]]:
    violations: set[tuple[str, str]] = set()
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC_ROOT).as_posix()
        own_module = rel[:-3].replace("/", ".")
        own_layer = _layer_of(own_module)
        if own_module.endswith("__init__") and own_module.count(".") == 0:
            own_layer = "__init__"
        allowed = ALLOWED.get(own_layer, frozenset())
        if "*" in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            for t in targets:
                if not (t == "kilat" or t.startswith("kilat.")):
                    continue
                tgt = _layer_of(t)
                if tgt in {own_layer, "__init__"} and t != "kilat":
                    continue
                if tgt not in allowed and (rel, tgt) not in SANCTIONED_SHIMS:
                    violations.add((rel, tgt))
    return violations


def test_layering_has_no_new_upward_dependencies():
    with Scenario("Architecture: Strict Downward Dependency Layering"):
        with Given("every kilat.* import across src/kilat (module-level and lazy)"):
            violations = _collect_violations()
            with When("compared against the allowed layer matrix and the debt register"):
                new = violations - KNOWN_LAYERING_DEBT
                with Then("no new upward or circular-prone dependency exists"):
                    assert not new, f"New layering violations (fix or justify in review): {sorted(new)}"


def test_layering_debt_register_is_a_ratchet():
    with Scenario("Architecture: Layering Debt Register Only Shrinks"):
        with Given("the current set of layering violations"):
            violations = _collect_violations()
            with When("checking each registered debt entry"):
                stale = KNOWN_LAYERING_DEBT - violations
                with Then("every registered entry still exists (fixed debt must be deregistered)"):
                    assert not stale, f"Remove fixed entries from KNOWN_LAYERING_DEBT: {sorted(stale)}"
                with And("the scan actually inspected the package"):
                    assert len(list(SRC_ROOT.rglob("*.py"))) > 50
