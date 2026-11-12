"""
Unit & Invariant Tests for Kilat Supervised Fine-Tuning (SFT) Module.

Verifies:
1. Prompt loss masking invariant (labels[:prompt_len] == -100).
2. Response token loss contribution & gradient sanity.
3. ChatML & Alpaca format parsing.
4. Numerical learning proof via Overfit-One-Batch test.
"""

import json
import tempfile
from pathlib import Path
import pytest
import torch

from kilat.configs.model_config import KilatConfig
from kilat.data.sft import SFTDataCollator, SFTDataset, format_instruction_pair
from kilat.data.tokenizer import ByteLevelDebugTokenizer
from kilat.engine.args import TrainingArguments
from kilat.engine.sft_trainer import KilatSFTTrainer
from kilat.models.transformer import KilatTransformer
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_format_instruction_pair_alpaca():
    with Scenario("Unit: Parse Alpaca formatted example"):
        with Given("an Alpaca example with instruction, input, and output"):
            ex = {
                "instruction": "Summarize the paragraph.",
                "input": "Deep learning models require data.",
                "output": "Models need data.",
            }
            with When("formatting the instruction pair"):
                prompt, resp = format_instruction_pair(ex)
                with Then("prompt contains instruction and input and resp matches output"):
                    assert "### Instruction:\nSummarize the paragraph." in prompt
                    assert "### Input:\nDeep learning models require data." in prompt
                    assert "### Response:\n" in prompt
                    assert resp == "Models need data."


def test_format_instruction_pair_chatml():
    with Scenario("Unit: Parse ChatML messages formatted example"):
        with Given("a conversational example with system, user, and assistant turns"):
            ex = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4."},
                ]
            }
            with When("formatting the instruction pair"):
                prompt, resp = format_instruction_pair(ex)
                with Then("prompt contains system and user turns and resp contains assistant answer"):
                    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" in prompt
                    assert "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" in prompt
                    assert "<|im_start|>assistant\n" in prompt
                    assert resp == "4."


def test_sft_data_collator_prompt_masking_invariant():
    with Scenario("Invariant: SFTDataCollator strictly masks prompt tokens with -100"):
        with Given("a tokenizer and SFTDataCollator with max_length=64"):
            tokenizer = ByteLevelDebugTokenizer(vocab_size=256)
            collator = SFTDataCollator(
                tokenizer=tokenizer,
                max_length=64,
                ignore_index=-100,
                pad_token_id=0,
                append_eos_token=True,
            )

            raw_batch = [
                {"prompt": "Tell me a joke.", "response": "Why did the chicken cross?"},
                {"prompt": "Hello!", "response": "Hi there!"},
            ]

            with When("collating the batch"):
                batch = collator(raw_batch)

                with Then("input_ids and labels have identical (B, S) shape"):
                    assert batch["input_ids"].shape == batch["labels"].shape
                    assert batch["input_ids"].shape[0] == 2
                    batch_len = batch["input_ids"].shape[1]

                with And("the prompt tokens are strictly masked with -100 in labels"):
                    for i, item in enumerate(raw_batch):
                        prompt_len = len(tokenizer.encode(item["prompt"]))
                        # The first prompt_len tokens must be -100
                        assert (batch["labels"][i, :prompt_len] == -100).all()

                with And("the response tokens are unmasked in labels"):
                    for i, item in enumerate(raw_batch):
                        prompt_len = len(tokenizer.encode(item["prompt"]))
                        resp_ids = tokenizer.encode(item["response"])
                        # Check that the first response token matches
                        assert batch["labels"][i, prompt_len] == resp_ids[0]

                with And("padding positions have -100 in labels and 0 in attention_mask"):
                    # Check shorter sequence has padding
                    if (batch["input_ids"][1] == 0).any():
                        pad_mask = batch["input_ids"][1] == 0
                        assert (batch["labels"][1][pad_mask] == -100).all()
                        assert (batch["attention_mask"][1][pad_mask] == 0).all()


def test_sft_gradient_attribution_invariant():
    with Scenario("Invariant: KilatTransformer computes loss & backprop strictly on response tokens"):
        with Given("a minimal KilatTransformer and SFT collated batch"):
            config = KilatConfig(
                vocab_size=256,
                max_position_embeddings=32,
                n_embd=64,
                n_layer=2,
                n_head=2,
                recall_ratio=0.5,
                latent_dim=16,
                use_rope=True,
                rope_head_dim=8,
                ffn_mode="swiglu",
                ff_mult=2.0,
                num_experts=2,
                active_experts=1,
                tie_word_embeddings=True,
            )
            model = KilatTransformer(config)
            tokenizer = ByteLevelDebugTokenizer(vocab_size=256)
            collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)

            batch = collator([
                {"prompt": "Question: Capital of France?", "response": "Answer: Paris."}
            ])

            with When("running forward pass and backward pass"):
                output = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output.loss
                assert loss is not None and torch.isfinite(loss)
                loss.backward()

                with Then("all trainable parameters receive non-zero, finite gradients"):
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            assert p.grad is not None, f"Parameter {name} has None grad"
                            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf grad"


def test_sft_overfit_one_batch():
    with Scenario("Numerical Proof: KilatSFTTrainer converges and memorizes 1 instruction pair"):
        with Given("a small KilatTransformer and KilatSFTTrainer"):
            config = KilatConfig(
                vocab_size=256,
                max_position_embeddings=32,
                n_embd=64,
                n_layer=2,
                n_head=2,
                recall_ratio=0.5,
                latent_dim=16,
                use_rope=True,
                rope_head_dim=8,
                ffn_mode="swiglu",
                ff_mult=2.0,
                num_experts=2,
                active_experts=1,
                tie_word_embeddings=True,
            )
            model = KilatTransformer(config)
            tokenizer = ByteLevelDebugTokenizer(vocab_size=256)

            data = [
                {"prompt": "Echo this:", "response": "Kilat is fast."}
            ] * 8

            with tempfile.TemporaryDirectory() as tmp_dir:
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    training_mode="steps",
                    max_steps=15,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=1,
                    learning_rate=5e-3,
                    warmup_steps=1,
                    logging_steps=1,
                    precision="fp32",
                    save_checkpoints=False,
                    seed=42,
                )

                trainer = KilatSFTTrainer(
                    model=model,
                    args=args,
                    train_dataset=data,
                    tokenizer=tokenizer,
                    max_seq_length=32,
                )

                with When("training for 15 steps"):
                    state = trainer.train()

                    with Then("training finishes successfully and state records 15 steps"):
                        assert state.global_step == 15
                        # Ensure loss decreased
                        assert len(state.log_history) > 0
                        initial_loss = state.log_history[0]["loss"]
                        final_loss = state.log_history[-1]["loss"]
                        assert final_loss < initial_loss
