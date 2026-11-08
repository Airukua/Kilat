"""
Unit tests for sampling strategies in Kilat text generation.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.pipeline.generation.sampler import (
    AdaptiveSampler,
    ContrastiveSampler,
    GreedySampler,
    MirostatSampler,
    MultinomialSampler,
    ReturnValue,
    Sampler,
    TopKSampler,
    TopKTopPSampler,
    TopPSampler,
    TypicalSampler,
    get_sampler,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_greedy_and_multinomial_samplers():
    with Scenario("Unit: Greedy & Multinomial Samplers"):
        with Given("a batch of logits with a dominant token"):
            batch_size = 2
            vocab_size = 50
            logits = torch.randn(batch_size, vocab_size)
            logits[:, 10] = 50.0  # Dominant token at index 10

            with When("sampling with GreedySampler"):
                sampler = GreedySampler()
                tokens = sampler(logits)
                with Then("the highest probability token is picked deterministically"):
                    assert tokens.shape == (batch_size, 1)
                    assert torch.all(tokens == 10)

            with When("sampling with MultinomialSampler"):
                multi_sampler = MultinomialSampler(temperature=1.0)
                sampled_tokens = multi_sampler(logits)
                with Then("the sampled tokens have correct batch shape"):
                    assert sampled_tokens.shape == (batch_size, 1)

            with When("requesting probabilities from MultinomialSampler"):
                tokens_probs = multi_sampler(logits, return_probs=True)
                with Then("a tuple of (tokens, probs) is returned"):
                    assert isinstance(tokens_probs, tuple)
                    toks, probs = tokens_probs
                    assert toks.shape == (batch_size, 1)
                    assert probs.shape == (batch_size, 1)


def test_top_k_and_top_p_samplers():
    with Scenario("Unit: Top-K and Top-P Nucleus Samplers"):
        with Given("logits tensor for 3 sequences"):
            logits = torch.randn(3, 100)

            with When("applying TopKSampler with k=5"):
                top_k_sampler = TopKSampler(top_k=5, temperature=0.7)
                out = top_k_sampler(logits)
                with Then("tokens are sampled within valid range"):
                    assert out.shape == (3, 1)
                    assert (out >= 0).all() and (out < 100).all()

            with When("applying TopPSampler with p=0.9"):
                top_p_sampler = TopPSampler(top_p=0.9, temperature=0.8)
                out_p = top_p_sampler(logits)
                with Then("sampled tokens have correct dimensions"):
                    assert out_p.shape == (3, 1)

            with When("applying combined TopKTopPSampler"):
                combined_sampler = TopKTopPSampler(top_k=10, top_p=0.85)
                res = combined_sampler(logits, return_probs=True)
                with Then("tokens and probabilities are returned"):
                    toks, probs = res
                    assert toks.shape == (3, 1)
                    assert probs.shape == (3, 1)


def test_advanced_samplers():
    with Scenario("Unit: Typical, Contrastive, Mirostat & Adaptive Samplers"):
        with Given("a batch of logits and context"):
            logits = torch.randn(2, 64)

            with When("sampling with TypicalSampler"):
                typical_sampler = TypicalSampler(typical_p=0.9)
                tok_typical = typical_sampler(logits)
                with Then("tokens are properly generated"):
                    assert tok_typical.shape == (2, 1)

            with When("sampling with ContrastiveSampler given previous logits"):
                prev_logits = torch.randn(2, 64)
                contrastive = ContrastiveSampler(contrastive_penalty=0.4, top_k=4)
                tok_contrastive = contrastive(logits, prev_logits=prev_logits)
                with Then("contrastive tokens are generated"):
                    assert tok_contrastive.shape == (2, 1)

            with When("sampling with MirostatSampler"):
                mirostat = MirostatSampler(target_perplexity=8.0, tau=2.0)
                tok_mirostat = mirostat(logits)
                with Then("mirostat tokens are generated"):
                    assert tok_mirostat.shape == (2, 1)

            with When("sampling with AdaptiveSampler"):
                adaptive = AdaptiveSampler()
                tok_adaptive = adaptive(logits)
                with Then("adaptive tokens are generated"):
                    assert tok_adaptive.shape == (2, 1)


def test_sampler_factory_and_base_primitives():
    with Scenario("Unit: Sampler Factory Function and Base Primitives"):
        with Given("base Sampler and ReturnValue primitives"):
            with When("instantiating ReturnValue"):
                t = torch.tensor([1, 2])
                p = torch.tensor([0.5, 0.5])
                ret = ReturnValue(tokens=t, probs=p)
                with Then("attributes store tensors"):
                    assert ret.tokens is t
                    assert ret.probs is p

            with When("calling abstract Sampler base class"):
                base_sampler = Sampler()
                with Then("NotImplementedError is raised"):
                    with pytest.raises(NotImplementedError):
                        base_sampler(torch.randn(2, 4))

        with Given("sampling configurations for get_sampler"):
            with When("requesting different strategies"):
                s_greedy = get_sampler(do_sample=False)
                s_topk = get_sampler(sampling_strategy="top_k", top_k=5)
                s_topp = get_sampler(sampling_strategy="top_p", top_p=0.9)
                s_comb = get_sampler(sampling_strategy="top_k_top_p", top_k=5, top_p=0.9)
                s_typ = get_sampler(sampling_strategy="typical", typical_p=0.8)
                s_contr = get_sampler(sampling_strategy="contrastive", contrastive_penalty=0.5)
                s_miro = get_sampler(sampling_strategy="mirostat")
                s_adapt = get_sampler(sampling_strategy="adaptive")
                s_default = get_sampler(do_sample=True)

                with Then("appropriate sampler subclasses are returned"):
                    assert isinstance(s_greedy, GreedySampler)
                    assert isinstance(s_topk, TopKSampler)
                    assert isinstance(s_topp, TopPSampler)
                    assert isinstance(s_comb, TopKTopPSampler)
                    assert isinstance(s_typ, TypicalSampler)
                    assert isinstance(s_contr, ContrastiveSampler)
                    assert isinstance(s_miro, MirostatSampler)
                    assert isinstance(s_adapt, AdaptiveSampler)
                    assert isinstance(s_default, MultinomialSampler)
