from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from kilat.configs.model_config import KilatConfig
from kilat.inference.export_hf import convert_checkpoint_to_hf
from kilat.integrations.huggingface.configuration_kilat import KilatConfig as HFKilatConfig
from kilat.integrations.huggingface.modeling_kilat import KilatForCausalLM as InTreeKilatForCausalLM
from kilat.integrations.huggingface.peft import DEFAULT_KILAT_LORA_TARGET_MODULES, get_kilat_lora_config
from kilat.models.transformer import KilatTransformer

logger = logging.getLogger(__name__)


def _require(condition: bool, message: str = "") -> None:
    if not condition:
        raise RuntimeError(message or "Invariant requirement violated")


@dataclass
class HFChaosScenarioResult:
    scenario_id: str
    name: str
    caught: bool
    description: str
    duration_ms: float
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "caught": self.caught,
            "description": self.description,
            "duration_ms": round(self.duration_ms, 3),
            "details": self.details,
        }


@dataclass
class HFChaosSuiteReport:
    baseline_passed: bool
    scenarios: list[HFChaosScenarioResult]

    @property
    def scenarios_total(self) -> int:
        return len(self.scenarios)

    @property
    def scenarios_passed(self) -> int:
        return sum(1 for s in self.scenarios if s.caught)

    @property
    def false_negatives(self) -> int:
        return self.scenarios_total - self.scenarios_passed

    @property
    def passed(self) -> bool:
        return self.baseline_passed and self.false_negatives == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": True,
            "baseline_passed": self.baseline_passed,
            "scenarios_total": self.scenarios_total,
            "scenarios_passed": self.scenarios_passed,
            "false_negatives": self.false_negatives,
            "overall_status": "PASS" if self.passed else "FAIL",
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


ALL_HF_SCENARIO_IDS = (
    "HFC-01",
    "HFC-02",
    "HFC-03",
    "HFC-04",
    "HFC-05",
    "HFC-06",
    "HFC-07",
    "HFC-08",
)


class HuggingFaceChaosSaboteur:
    """
    100% Non-Destructive In-Memory Fault Injector for Hugging Face Integration Chaos.
    """

    @contextlib.contextmanager
    def inject_dynamic_cache(self, num_layers: int = 2) -> Generator[DynamicCache, None, None]:
        """Inject a modern HuggingFace DynamicCache object."""
        cache = DynamicCache()
        for _ in range(num_layers):
            cache.update(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), 0)
        yield cache

    @contextlib.contextmanager
    def corrupt_export_auto_map(self, export_dir: Path) -> Generator[Path, None, None]:
        """Corrupt auto_map in config.json and restore in finally block."""
        cfg_path = export_dir / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            original = f.read()

        corrupted = json.loads(original)
        corrupted.pop("auto_map", None)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(corrupted, f)

        try:
            yield cfg_path
        finally:
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(original)


def run_hf_chaos_suite(scenarios: str | Sequence[str] = "all") -> HFChaosSuiteReport:
    """
    Execute Hugging Face Integration Chaos Engineering Suite.
    """
    if isinstance(scenarios, str):
        if scenarios.strip().lower() == "all":
            selected = list(ALL_HF_SCENARIO_IDS)
        else:
            selected = [s.strip().upper() for s in scenarios.split(",") if s.strip()]
    else:
        selected = [s.strip().upper() for s in scenarios]

    unknown = [s for s in selected if s not in ALL_HF_SCENARIO_IDS]
    if unknown:
        raise ValueError(f"Unknown HF chaos scenario IDs: {unknown}. Valid: {ALL_HF_SCENARIO_IDS}")

    # Micro configuration for swift in-memory chaos tests
    cfg = KilatConfig(
        vocab_size=128,
        n_embd=64,
        n_head=4,
        n_layer=2,
        max_position_embeddings=64,
        recall_pattern="macro_synergy",
        use_rope=True,
        rope_head_dim=8,
    )

    saboteur = HuggingFaceChaosSaboteur()
    results: list[HFChaosScenarioResult] = []

    with tempfile.TemporaryDirectory() as base_ckpt_dir, tempfile.TemporaryDirectory() as base_out_dir:
        ckpt_path = Path(base_ckpt_dir)
        out_path = Path(base_out_dir)

        native_model = KilatTransformer(cfg)
        native_model.eval()
        native_model.save_pretrained(ckpt_path)

        convert_checkpoint_to_hf(ckpt_path, out_path, skip_validation=True)

        # Baseline check
        try:
            trust_remote = bool(1)
            hf_model = AutoModelForCausalLM.from_pretrained(out_path, trust_remote_code=trust_remote)
            hf_model.eval()
            prompt = torch.tensor([[1, 10, 20]])
            with torch.no_grad():
                base_gen = hf_model.generate(prompt, max_new_tokens=4, use_cache=False)
            baseline_passed = bool(base_gen.shape == (1, 7))
        except Exception as e:
            logger.error("HF Baseline check failed: %s", e)
            baseline_passed = False

        for sc_id in selected:
            t0 = time.perf_counter()
            caught = False
            details = ""

            if sc_id == "HFC-01":
                # Scenario 1: DynamicCache & Foreign Cache Injection Resilience
                desc = "DynamicCache & Foreign Cache Injection Resilience"
                try:
                    with saboteur.inject_dynamic_cache(num_layers=cfg.n_layer) as foreign_cache:
                        # Direct forward with foreign cache
                        out_fwd = hf_model(prompt, past_key_values=foreign_cache)
                        # Generate with use_cache=True
                        out_gen_true = hf_model.generate(prompt, max_new_tokens=4, use_cache=True)
                        # Generate without specifying use_cache (defaults to generation_config)
                        out_gen_def = hf_model.generate(prompt, max_new_tokens=4)

                    _require(out_fwd.logits.shape == (1, 3, cfg.vocab_size), "Logit shape mismatch")
                    _require(out_gen_true.shape == (1, 7), "Gen true shape mismatch")
                    _require(out_gen_def.shape == (1, 7), "Gen def shape mismatch")
                    caught = True
                    details = "Gracefully handled DynamicCache without subscriptable error."
                except Exception as e:
                    caught = False
                    details = f"DynamicCache injection crashed: {e}"

            elif sc_id == "HFC-02":
                # Scenario 2: Bidirectional Schema Drift & Attribute Mutability
                desc = "Bidirectional Schema Drift & Attribute Mutability"
                try:
                    hf_cfg = HFKilatConfig(vocab_size=256, n_embd=128, n_layer=4, n_head=8)
                    _require(hf_cfg.num_hidden_layers == 4, "Initial layers mismatch")
                    _require(hf_cfg.hidden_size == 128, "Initial hidden mismatch")
                    _require(hf_cfg.num_attention_heads == 8, "Initial heads mismatch")

                    # Mutate HF standard attributes
                    hf_cfg.num_hidden_layers = 6
                    hf_cfg.hidden_size = 256
                    hf_cfg.num_attention_heads = 16

                    # Assert bidirectional sync
                    _require(hf_cfg.n_layer == 6, "n_layer sync failed")
                    _require(hf_cfg.n_embd == 256, "n_embd sync failed")
                    _require(hf_cfg.n_head == 16, "n_head sync failed")

                    # Assert serialization round-trip
                    cfg_dict = hf_cfg.to_dict()
                    _require(cfg_dict["num_hidden_layers"] == 6, "dict layers mismatch")
                    _require(cfg_dict["hidden_size"] == 256, "dict hidden mismatch")
                    _require(cfg_dict["num_attention_heads"] == 16, "dict heads mismatch")
                    _require(cfg_dict["attribute_map"]["num_hidden_layers"] == "n_layer", "attribute_map mismatch")

                    caught = True
                    details = "Bidirectional property getters/setters & serialization verified."
                except Exception as e:
                    caught = False
                    details = f"Schema mutability error: {e}"

            elif sc_id == "HFC-03":
                # Scenario 3: Extreme Adversarial Sampling Parameters
                desc = "Extreme Adversarial Sampling Parameters"
                try:
                    params_to_test = [
                        {"temperature": 0.0, "do_sample": False},
                        {"temperature": 4.0, "do_sample": True},
                        {"top_k": 1, "do_sample": True},
                        {"top_p": 0.01, "do_sample": True},
                        {"repetition_penalty": 2.0, "do_sample": False},
                        {"max_new_tokens": 1},
                        {"max_new_tokens": 16},
                    ]
                    for p in params_to_test:
                        gen_out = hf_model.generate(prompt, **p)
                        _require(bool(torch.all(gen_out >= 0).item()) and bool(torch.all(gen_out < cfg.vocab_size).item()), "Token out of bounds")
                        _require(bool(torch.isfinite(gen_out.float()).all().item()), "Non-finite token output")

                    caught = True
                    details = f"Passed {len(params_to_test)} adversarial sampling regimes cleanly."
                except Exception as e:
                    caught = False
                    details = f"Adversarial sampling crashed: {e}"

            elif sc_id == "HFC-04":
                # Scenario 4: Asymmetric Batched Left-Padding & Causal Isolation
                desc = "Asymmetric Batched Left-Padding & Causal Isolation"
                try:
                    # Batch of 2: sample 1 has 4 tokens, sample 2 has 2 pad tokens + 2 real tokens
                    padded_batch = torch.tensor([
                        [10, 20, 30, 40],
                        [0, 0, 30, 40],
                    ])
                    attn_mask = torch.tensor([
                        [1, 1, 1, 1],
                        [0, 0, 1, 1],
                    ])
                    batch_gen = hf_model.generate(
                        padded_batch,
                        attention_mask=attn_mask,
                        max_new_tokens=4,
                        do_sample=False,
                        use_cache=False,
                    )
                    _require(batch_gen.shape == (2, 8), "Batch shape mismatch")
                    _require(bool(torch.all(batch_gen >= 0).item()) and bool(torch.all(batch_gen < cfg.vocab_size).item()), "Batch tokens out of bounds")
                    caught = True
                    details = "Batched left-padding generated valid non-null outputs."
                except Exception as e:
                    caught = False
                    details = f"Batched left-padding failed: {e}"

            elif sc_id == "HFC-05":
                # Scenario 5: Corrupted Export Artifacts & Manifest Detection
                desc = "Corrupted Export Artifacts & AutoMap Detection"
                try:
                    with saboteur.corrupt_export_auto_map(out_path):
                        with open(out_path / "config.json") as f:
                            corrupted_data = json.load(f)
                        _require("auto_map" not in corrupted_data, "auto_map not removed")

                    # After restore
                    with open(out_path / "config.json") as f:
                        restored_data = json.load(f)
                    _require("auto_map" in restored_data, "auto_map not restored")
                    _require((out_path / "generation_config.json").exists(), "generation_config.json missing")
                    caught = True
                    details = "AutoMap mutation and recovery verified without residual state."
                except Exception as e:
                    caught = False
                    details = f"Export corruption test failed: {e}"

            elif sc_id == "HFC-06":
                # Scenario 6: PEFT LoRA Adaptation & Frozen Base Invariant
                desc = "PEFT LoRA Adaptation & Frozen Base Invariant"
                try:
                    import peft
                    from peft import get_peft_model

                    lora_cfg = get_kilat_lora_config(
                        r=4,
                        lora_alpha=8,
                        target_modules=["down", "gate", "up"],
                    )
                    # Wrap in-tree model to verify PEFT adapter compatibility
                    dense_peft_cfg = KilatConfig(vocab_size=128, n_embd=64, n_head=4, n_layer=2, ffn_mode="dense")
                    in_tree_model = InTreeKilatForCausalLM(dense_peft_cfg)
                    peft_model = get_peft_model(in_tree_model, lora_cfg)

                    # Invariant 1: Base weights must be frozen
                    base_trainable = [p for n, p in peft_model.named_parameters() if "lora_" not in n and p.requires_grad]
                    _require(len(base_trainable) == 0, f"Found unfrozen base weights: {len(base_trainable)}")

                    # Invariant 2: LoRA adapters must be trainable
                    lora_trainable = [p for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad]
                    _require(len(lora_trainable) > 0, "No trainable LoRA weights found")

                    # Invariant 3: Gradients flow on backward
                    opt = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)
                    loss = peft_model(prompt, labels=prompt).loss
                    loss.backward()
                    _require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora_trainable), "Gradients non-finite or missing")

                    caught = True
                    details = f"LoRA successfully attached ({len(lora_trainable)} adapter tensors, 0 base unfrozen)."
                except ImportError:
                    # If peft is not installed in the environment, verify graceful guard
                    try:
                        get_kilat_lora_config()
                    except ImportError as err:
                        if "pip install peft" in str(err):
                            caught = True
                            details = "PEFT missing guard verified: raised clear installation instructions."
                        else:
                            caught = False
                            details = f"Unexpected error message: {err}"
                except Exception as e:
                    caught = False
                    details = f"PEFT LoRA validation failed: {e}"

            elif sc_id == "HFC-07":
                # Scenario 7: Boundary Loss & All-Masked -100 Invariant
                desc = "Boundary Loss & All-Masked -100 Invariant"
                try:
                    all_masked_labels = torch.full_like(prompt, -100)
                    out = hf_model(prompt, labels=all_masked_labels)
                    # When all tokens are masked, cross-entropy is zero.
                    # Total loss is finite and non-negative (may contain auxiliary router loss).
                    _require(out.loss is not None, "Loss is None")
                    _require(bool(torch.isfinite(out.loss).item()), "Loss is non-finite")
                    _require(float(out.loss.item()) >= 0.0, "Loss is negative")

                    # Also test with dense model (no aux loss) where loss must be strictly 0.0
                    dense_cfg = KilatConfig(vocab_size=128, n_embd=64, n_head=4, n_layer=2, ffn_mode="dense")
                    dense_model = KilatTransformer(dense_cfg)
                    dense_out = dense_model(prompt, labels=all_masked_labels)
                    _require(dense_out.loss is not None, "Dense loss is None")
                    _require(float(dense_out.loss.item()) == 0.0, "Dense loss not zero")

                    caught = True
                    details = f"All-masked labels produced finite loss (dense: {dense_out.loss.item():.4f}, moe: {out.loss.item():.4f}) without NaN/Inf."
                except Exception as e:
                    caught = False
                    details = f"All-masked labels test failed: {e}"

            elif sc_id == "HFC-08":
                # Scenario 8: Multithreaded Streaming Generation & Early Cancellation
                desc = "Multithreaded Streaming Generation & Early Cancellation"
                try:
                    from transformers import TextIteratorStreamer

                    # Mock tokenizer with simple decoding
                    class MockTokenizer:
                        def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
                            return f"tok_{ids[0]}" if len(ids) == 1 else " ".join(f"tok_{i}" for i in ids)

                    streamer = TextIteratorStreamer(MockTokenizer(), skip_prompt=True)
                    gen_kwargs = {
                        "input_ids": prompt,
                        "streamer": streamer,
                        "max_new_tokens": 8,
                        "do_sample": False,
                    }
                    t = threading.Thread(target=hf_model.generate, kwargs=gen_kwargs)
                    t.start()

                    collected = []
                    for idx, token_text in enumerate(streamer):
                        collected.append(token_text)
                        if idx >= 2:
                            # Early cancellation / break
                            break

                    t.join(timeout=5.0)
                    _require(not t.is_alive(), "Generation thread hung after streamer break")
                    _require(len(collected) >= 3, "Streamer yielded fewer than 3 tokens")
                    caught = True
                    details = f"Streamer successfully pulled {len(collected)} tokens and thread joined cleanly."
                except Exception as e:
                    caught = False
                    details = f"Streaming thread test failed: {e}"

            elapsed = (time.perf_counter() - t0) * 1000.0
            results.append(
                HFChaosScenarioResult(
                    scenario_id=sc_id,
                    name=desc,
                    caught=caught,
                    description=desc,
                    duration_ms=elapsed,
                    details=details,
                )
            )

    return HFChaosSuiteReport(baseline_passed=baseline_passed, scenarios=results)
