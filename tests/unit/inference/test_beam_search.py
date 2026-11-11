"""
Unit tests for Kilat Beam Search Hypotheses, Scorer, and Generation.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.inference.beam_search import BeamSearchScorer, BeamHypotheses
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_beam_hypotheses_bookkeeping():
    with Scenario("Unit: BeamHypotheses Storage & Length Penalty"):
        with Given("a BeamHypotheses container with num_beams=2, length_penalty=1.0"):
            hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=True)

            with When("adding first completed hypothesis"):
                t1 = torch.tensor([1, 2, 3, 4])  # len 4, score -2.0 / 4 = -0.5
                hyps.add(t1, sum_logprobs=-2.0)
                with Then("the hypothesis count is 1"):
                    assert len(hyps) == 1
                    assert not hyps.is_done

            with When("adding second completed hypothesis with higher score"):
                t2 = torch.tensor([1, 5, 2])     # len 3, score -0.6 / 3 = -0.2
                hyps.add(t2, sum_logprobs=-0.6)
                with Then("the container reaches capacity (num_beams=2) and marks is_done as True"):
                    assert len(hyps) == 2
                    assert hyps.is_done

            with When("requesting the best hypothesis"):
                best = hyps.best()
                with Then("the hypothesis with highest normalized score is returned"):
                    assert torch.equal(best, t2)


def test_beam_search_scorer_process():
    with Scenario("Unit: BeamSearchScorer Step Processing"):
        with Given("a BeamSearchScorer with batch_size=1 and num_beams=2"):
            batch_size = 1
            num_beams = 2
            device = torch.device("cpu")

            scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                length_penalty=1.0,
                do_early_stopping=True,
            )
            scorer.eos_token_id = 2

            with When("processing next beam scores and candidate tokens"):
                vocab_size = 5
                input_ids = torch.tensor([[10, 20], [10, 20]])  # (num_beams, seq_len)
                next_scores = torch.randn(batch_size, num_beams * vocab_size)
                next_tokens = torch.randint(0, vocab_size, (batch_size, num_beams * vocab_size))
                next_indices = torch.randint(0, num_beams, (batch_size, num_beams * vocab_size))

                scores_out, tokens_out, indices_out = scorer.process(
                    input_ids=input_ids,
                    next_scores=next_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                )
                with Then("scorer outputs selected scores, tokens, and indices for next step"):
                    assert scores_out.shape == (batch_size, num_beams)
                    assert tokens_out.shape == (batch_size, num_beams)
                    assert indices_out.shape == (batch_size, num_beams)


def test_beam_search_scorer_finalize():
    with Scenario("Unit: BeamSearchScorer Finalization & Hypothesis Selection"):
        with Given("a configured BeamSearchScorer with candidate beams"):
            scorer = BeamSearchScorer(batch_size=1, num_beams=2, device=torch.device("cpu"))
            scorer.eos_token_id = 2

            input_ids = torch.tensor([[10, 20, 2], [10, 30, 2]])
            final_scores = torch.tensor([-0.5, -0.8])
            final_tokens = torch.tensor([2, 2])
            final_indices = torch.tensor([0, 1])

            with When("finalizing the search results"):
                best_sequence = scorer.finalize(
                    input_ids,
                    final_scores,
                    final_tokens,
                    final_indices,
                    max_length=5,
                )
                with Then("the best sequence is extracted and returned as tensor"):
                    assert best_sequence.shape[0] == 1
                    assert best_sequence.shape[1] == 3
