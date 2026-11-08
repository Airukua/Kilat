"""
Unit tests for Kilat TextGenerator high-level API.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.pipeline.generation.generator import (
    TextGenerator,
    batch_generate,
    quick_generate,
    stream_generate,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


class SimpleTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.vocab_size = 256

    def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = [ord(c) % 256 for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        return "".join(chr(t if 32 <= t < 127 else 65) for t in tokens)

    def __call__(self, text, return_tensors=None, **kwargs):
        t = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


def test_text_generator_execution():
    with Scenario("Unit: High-Level TextGenerator API"):
        with Given("a small KilatTransformer model and simple tokenizer"):
            torch.manual_seed(42)
            config = KilatConfig(
                vocab_size=256,
                n_embd=64,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                ffn_mode="dense",
            )
            model = KilatTransformer(config)
            tokenizer = SimpleTokenizer()

            with When("TextGenerator generates text from a prompt"):
                generator = TextGenerator(model, tokenizer, device="cpu")
                out_text = generator.generate("Hello", max_new_tokens=4, do_sample=False)
                with Then("the generated text is non-empty string"):
                    assert isinstance(out_text, str)
                    assert len(out_text) > 0


def test_text_generator_batch_stream_and_helpers():
    with Scenario("Unit: TextGenerator Batch, Stream, and Functional Helpers"):
        with Given("a configured model and tokenizer"):
            config = KilatConfig(
                vocab_size=256,
                n_embd=32,
                n_layer=1,
                n_head=2,
                max_seq_len=32,
            )
            model = KilatTransformer(config)
            tokenizer = SimpleTokenizer()
            generator = TextGenerator(model, tokenizer, device="cpu")

            with When("generating a batch of prompts"):
                prompts = ["Hello", "World"]
                batch_res = generator.batch_generate(prompts, max_new_tokens=2, do_sample=False)
                with Then("batch output contains responses for each prompt"):
                    assert len(batch_res) == 2
                    assert all(isinstance(t, str) for t in batch_res)

            with When("streaming token generation"):
                stream_tokens = list(generator.stream("Hi", max_new_tokens=3))
                with Then("tokens are iteratively yielded as strings"):
                    assert len(stream_tokens) > 0
                    assert all(isinstance(tok, str) for tok in stream_tokens)

            with When("accessing to_low_level dictionary"):
                low_level = generator.to_low_level()
                with Then("underlying components are exposed"):
                    assert "model" in low_level
                    assert "tokenizer" in low_level

            with When("invoking quick_generate functional helper"):
                quick_res = quick_generate(model, tokenizer, "Prompt", max_new_tokens=2)
                with Then("output string is returned"):
                    assert isinstance(quick_res, str)

            with When("batch generation is executed"):
                batch_out = generator.batch_generate(["Hi", "Test"], max_new_tokens=3, do_sample=False)
                with Then("a list of generated texts matching batch size is returned"):
                    assert len(batch_out) == 2
                    assert all(isinstance(t, str) for t in batch_out)
