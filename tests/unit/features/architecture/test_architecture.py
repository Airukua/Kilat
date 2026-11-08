"""
Unit tests for Kilat Neural Network Architecture.
Covers RMSNorm, SwiGLU, DeepSeekMoE, RotaryPositionalEmbedding, Block, KilatAttention, and KilatTransformer.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from kilat.arc.attention import KilatAttention
from kilat.arc.blocks import Block, RMSNorm
from kilat.arc.ffn import DeepSeekMoE, FeedForward, SwiGLU
from kilat.arc.model import KilatTransformer
from kilat.arc.rope import RotaryPositionalEmbedding, apply_rotary_pos_emb, build_rope_cache, rope
from kilat.arc.triton_ops import _cpu_global_decay, triton_global_decay
from kilat.configs.model_config import KilatConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_rmsnorm_behavior():
    with Scenario("Unit: RMSNorm Normalization Invariant"):
        with Given("an input tensor with varying scale"):
            norm = RMSNorm(normalized_shape=64, eps=1e-6)
            x = torch.randn(2, 8, 64) * 10.0

            with When("RMSNorm forward pass is applied"):
                out = norm(x)
                with Then("the output preserves tensor dimensions (2, 8, 64)"):
                    assert out.shape == (2, 8, 64)
                with And("the root mean square of elements along the last dimension is approximately 1.0"):
                    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
                    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_rope_embedding_and_application():
    with Scenario("Unit: Decoupled Rotary Positional Embeddings (RoPE)"):
        with Given("a RotaryPositionalEmbedding module with head dimension 32"):
            rope_module = RotaryPositionalEmbedding(rope_dim=32, max_seq_len=128, base=10000.0)
            x = torch.randn(2, 16, 32)

            with When("applying RoPE to input representations"):
                out = rope_module(x, seq_offset=0)
                with Then("the rotated tensor maintains original shape [2, 16, 32]"):
                    assert out.shape == x.shape
                with And("the norm of rotated vectors is preserved under unitary rotation"):
                    norm_diff = torch.norm(x, dim=-1) - torch.norm(out, dim=-1)
                    assert torch.allclose(norm_diff, torch.zeros_like(norm_diff), atol=1e-5)

                with When("querying caches and building rope cache manually"):
                    cos_c, sin_c = rope_module.get_caches()
                    cos, sin = build_rope_cache(seq_len=16, rope_dim=32, device=torch.device("cpu"))
                    applied = apply_rotary_pos_emb(x, cos, sin)
                    with Then("cache shapes and manual application match expected dimensions"):
                        assert cos_c.shape[-1] == 32
                        assert sin_c.shape[-1] == 32
                        assert applied.shape[-2:] == (16, 32)

                with When("using standalone rope functional call on 4D tensor"):
                    x_4d = torch.randn(2, 4, 16, 32)
                    out_4d = rope(x_4d, rope_dim=32)
                    with Then("4D tensor is properly rotated"):
                        assert out_4d.shape == (2, 4, 16, 32)


def test_feedforward_swiglu_and_moe():
    with Scenario("Unit: FeedForward SwiGLU and Mixture-of-Experts"):
        with Given("a SwiGLU module with dimension 64"):
            swiglu = SwiGLU(dim=64, ff_mult=8 / 3)
            x = torch.randn(2, 4, 64)
            with When("forward pass is executed through SwiGLU"):
                out = swiglu(x)
                with Then("the output shape matches input dimension (2, 4, 64)"):
                    assert out.shape == (2, 4, 64)

        with Given("a FeedForward block in MoE mode"):
            moe_ffn = FeedForward(
                dim=64,
                mode="moe",
                num_experts=4,
                active_experts=2,
                aux_loss_coef=0.01,
            )
            with When("forward pass is executed on mini-batch"):
                moe_res = moe_ffn(x)
                out_tensor = moe_res[0] if isinstance(moe_res, tuple) else moe_res
                with Then("the output shape is preserved"):
                    assert out_tensor.shape == (2, 4, 64)


def test_transformer_block():
    with Scenario("Unit: Transformer Layer Block"):
        with Given("a Kilat Block in dense configuration"):
            block = Block(
                n_embd=64,
                n_head=4,
                recall_ratio=0.5,
                ffn_mode="dense",
            )
            x = torch.randn(2, 8, 64)

            with When("executing block forward pass without attention mask"):
                out = block(x)
                # out is a tuple of (hidden_states, cache, aux_loss) or hidden_states
                hidden = out[0] if isinstance(out, tuple) else out
                with Then("hidden states match input shape (2, 8, 64)"):
                    assert hidden.shape == (2, 8, 64)
                with And("no NaN values are produced in residual stream"):
                    assert not torch.isnan(hidden).any()


def test_kilat_transformer_model():
    with Scenario("Unit: KilatTransformer Complete Model Forward & Loss"):
        with Given("a small KilatConfig initialized for unit testing"):
            config = KilatConfig(
                vocab_size=1000,
                n_embd=64,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                ffn_mode="dense",
            )
            model = KilatTransformer(config)
            model.eval()

            with When("forwarding token indices through model without labels"):
                input_ids = torch.randint(0, 1000, (2, 16))
                output = model(input_ids=input_ids)
                with Then("logits tensor is produced with shape [batch_size, seq_len, vocab_size]"):
                    assert output.logits.shape == (2, 16, 1000)

            with When("forwarding token indices with ground truth labels"):
                labels = input_ids.clone()
                output_with_loss = model(input_ids=input_ids, labels=labels)
                with Then("a non-negative scalar cross-entropy loss is computed"):
                    assert output_with_loss.loss is not None
                    assert output_with_loss.loss.item() > 0.0
                with And("loss is numerically finite without NaN or Inf"):
                    assert torch.isfinite(output_with_loss.loss)


def test_kilat_transformer_embeddings_and_pretrained():
    with Scenario("Unit: KilatTransformer Embedding Handlers and Pretrained Loading"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "saved_kilat"

            with Given("a KilatTransformer model"):
                config = KilatConfig(vocab_size=100, n_embd=32, n_layer=1, n_head=2, max_seq_len=64)
                model = KilatTransformer(config)

                with When("querying and setting input/output embeddings"):
                    inp_emb = model.get_input_embeddings()
                    out_emb = model.get_output_embeddings()
                    model.set_input_embeddings(inp_emb)
                    model.set_output_embeddings(out_emb)
                    with Then("getter and setter methods execute cleanly"):
                        assert model.get_input_embeddings() is inp_emb
                        assert model.get_output_embeddings() is out_emb

                with When("executing forward pass with use_cache=True"):
                    ids = torch.randint(0, 100, (2, 8))
                    out_cache = model(ids, use_cache=True)
                    with Then("past_key_values cache is returned"):
                        assert out_cache.past_key_values is not None
                        assert len(out_cache.past_key_values) == 1

                with When("saving and reloading model using from_pretrained"):
                    model.save_pretrained(str(save_path))
                    loaded = KilatTransformer.from_pretrained(str(save_path))
                    with Then("reloaded model maintains same configuration"):
                        assert loaded.config.vocab_size == config.vocab_size


def test_triton_ops_cpu_decay():
    with Scenario("Unit: Triton Global Decay CPU Fallback Recurrence"):
        with Given("batch and head value tensors and decay lambda coefficients"):
            b, h, n, d = 2, 4, 16, 8
            v = torch.randn(b, h, n, d)
            lam = torch.tensor([0.4, 0.6, 0.8, 0.95])

            with When("computing exponential decay via triton_global_decay fallback"):
                out = triton_global_decay(lam, v)
                with Then("the output dimensions match input tensor [2, 4, 16, 8]"):
                    assert out.shape == (b, h, n, d)

                with And("computation matches analytical _cpu_global_decay"):
                    expected = _cpu_global_decay(lam, v)
                    assert torch.allclose(out, expected)
