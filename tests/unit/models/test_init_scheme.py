"""
Unit tests for KilatConfig.init_scheme ("gpt2" legacy vs muP-style "width_scaled").

Root cause being guarded: with GPT-2's fixed N(0, 0.02) init at n_embd=64, the token embedding is drowned in the
pre-norm residual stream after layer 0 and pure-attention stacks plateau at 1/num_kv accuracy on MQAR.
Diagnosed by bidirectional transplant ablations (Kilat attention inside a vanilla shell: 99.9%; vanilla attention
inside the Kilat shell: 26%) and a two-way embedding-scale test (see docs/06, section 4).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from kilat.benchmarking.tasks import mqar
from kilat.configs.model_config import KilatConfig
from kilat.models.transformer import KilatTransformer
from tests.shared.bdd import And, Given, Scenario, Then, When

PRESET_PURE_ATTENTION = "kilat-pure-attention"
SCHEME_WIDTH_SCALED = "width_scaled"


def _micro(scheme: str, **kw) -> KilatTransformer:
    params = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=2, max_seq_len=32)
    params.update(kw)
    if scheme == "gpt2":
        params.update(init_scheme="gpt2", tie_word_embeddings=True)
    return KilatTransformer.from_preset(PRESET_PURE_ATTENTION, **params)


def test_init_scheme_config_contract():
    with Scenario("Unit: init_scheme Config Contract & Backward Compatibility"):
        with Given("a KilatConfig built without init_scheme (as in every pre-existing checkpoint)"):
            legacy = KilatConfig()
            with Then("it keeps the legacy GPT-2 scheme and tied embeddings"):
                assert legacy.init_scheme == "gpt2"
                assert legacy.tie_word_embeddings is True
            with And("a config dict lacking the key also deserializes to gpt2"):
                d = legacy.to_dict()
                d.pop("init_scheme")
                assert KilatConfig.from_dict(d).init_scheme == "gpt2"

        with Given("canonical kilat-* presets (all narrow widths)"):
            for name in ["kilat", "kilat-macro-synergy", PRESET_PURE_ATTENTION, "kilat-pure-decay", "kilat-micro", "kilat-mini"]:
                cfg = KilatConfig.from_preset(name)
                with Then(f"{name} uses width_scaled and untied embeddings"):
                    assert cfg.init_scheme == SCHEME_WIDTH_SCALED
                    assert cfg.tie_word_embeddings is False
            with And("init_scheme round-trips through to_dict/from_dict"):
                cfg = KilatConfig.from_preset(PRESET_PURE_ATTENTION)
                assert KilatConfig.from_dict(cfg.to_dict()).init_scheme == SCHEME_WIDTH_SCALED

        with Given("invalid combinations"):
            with Then("width_scaled with tied embeddings is rejected with an explanation"):
                with pytest.raises(ValueError, match="tie_word_embeddings=False"):
                    KilatConfig(init_scheme=SCHEME_WIDTH_SCALED, tie_word_embeddings=True)
            with And("unknown schemes are rejected"):
                with pytest.raises(ValueError, match="init_scheme"):
                    KilatConfig(init_scheme="xavier")


def test_width_scaled_init_statistics():
    with Scenario("Unit: width_scaled Per-Module Initialization Statistics"):
        with Given("a width_scaled macro-synergy model with n_embd=64, n_layer=2"):
            torch.manual_seed(0)
            m = KilatTransformer.from_preset("kilat-macro-synergy", vocab_size=512, d_model=64, n_layers=2, max_seq_len=32)
            depth = math.sqrt(2 * 2)
            with Then("the input embedding is O(1) and untied from the output head"):
                assert m.wte.weight is not m.lm_head.weight
                assert abs(m.wte.weight.std().item() - 1.0) < 0.05
            with And("the output head starts at std 1/n_embd"):
                assert abs(m.lm_head.weight.std().item() - 1 / 64) < 0.002
            with And("hidden Linear layers use 1/sqrt(fan_in), residual projections divided by sqrt(2L)"):
                for name, mod in m.named_modules():
                    if not isinstance(mod, torch.nn.Linear) or mod is m.lm_head:
                        continue
                    expected = mod.in_features ** -0.5
                    if name.endswith((".down", "o_proj_recall", "o_proj_global", "c_proj")):
                        expected /= depth
                    assert abs(mod.weight.std().item() - expected) / expected < 0.15, name


def test_width_scaled_init_loss_satisfies_gate1():
    with Scenario("Unit: width_scaled Keeps Gate 1 Unbiased Initial Loss"):
        for preset in [PRESET_PURE_ATTENTION, "kilat-macro-synergy", "kilat-pure-decay"]:
            with Given(f"an untrained width_scaled {preset} with V=512"):
                torch.manual_seed(0)
                m = KilatTransformer.from_preset(preset, vocab_size=512, d_model=64, n_layers=2, max_seq_len=64)
                x = torch.randint(0, 512, (8, 64))
                with When("measuring cross-entropy at step 0"):
                    with torch.no_grad():
                        logits = m(x).logits
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 512), x[:, 1:].reshape(-1)).item()
                    with Then("|L(0) - ln V| <= 0.15 (Gate 1 tolerance)"):
                        assert abs(loss - math.log(512)) <= 0.15


def _mini_mqar_accuracy(model: KilatTransformer, steps: int = 450, seed: int = 0) -> float:
    cfg = mqar.MQARConfig(seq_len=32, vocab_size=128, num_kv_pairs=4, batch_size=64)
    xt, yt = mqar.generate_dataset(cfg, 8000, seed=seed)
    xe, ye = mqar.generate_dataset(cfg, 256, seed=999 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (64,), generator=g)
        loss = mqar.compute_causal_loss(model(xt[idx]).logits, yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return mqar.compute_causal_accuracy(model(xe).logits, ye)


def test_width_scaled_fixes_pure_attention_mqar_plateau():
    with Scenario("Capability Regression: Pure-Attention MQAR Plateau (root cause guard)"):
        with Given("identical pure-attention models differing only in init scheme"):
            torch.manual_seed(0)
            fixed = _micro(SCHEME_WIDTH_SCALED)
            torch.manual_seed(0)
            legacy = _micro("gpt2")
            with When("both train 450 steps on mini-MQAR (seq 32, 4 KV pairs)"):
                acc_fixed = _mini_mqar_accuracy(fixed)
                acc_legacy = _mini_mqar_accuracy(legacy)
                with Then("width_scaled solves associative recall"):
                    assert acc_fixed >= 0.88, acc_fixed
                with And("the legacy scheme still exhibits the 1/num_kv plateau (proves this test can fail)"):
                    assert acc_legacy <= 0.50, acc_legacy
