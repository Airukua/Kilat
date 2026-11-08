"""
Seam 4 Integration Test:
Verifies live autoregressive inference, sampling, and stopping criteria enforcement.
Follows standards from ml_integration_testing_structure_md.md (Seam 4).
"""

import torch

from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.pipeline.generation.generation_config import GenerationConfig
from kilat.pipeline.generation.sampler import GreedySampler
from kilat.pipeline.generation.stoping_criteria import MaxLengthCriteria
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_seam4_autoregressive_serving():
    with Scenario("Integration: Autoregressive Inference Serving Pipeline (Seam 4)"):
        with Given("a running KilatTransformer model and deterministic greedy sampler"):
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
            model.eval()

            sampler = GreedySampler()
            stop_crit = MaxLengthCriteria(max_length=12)
            prompt = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

            with When("executing autoregressive decoding until max_length=12 is satisfied"):
                current_ids = prompt.clone()
                while not stop_crit(current_ids, None):
                    with torch.no_grad():
                        outputs = model(input_ids=current_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                        sampled = sampler(next_token_logits)
                        next_token = sampled.tokens if hasattr(sampled, "tokens") else sampled
                        if next_token.dim() == 1:
                            next_token = next_token.unsqueeze(-1)
                        current_ids = torch.cat([current_ids, next_token], dim=-1)

                with Then("the generated sequence terminates exactly at length 12"):
                    assert current_ids.shape == (1, 12)

                with And("the prompt prefix tokens [10, 20, 30, 40] are strictly preserved"):
                    assert torch.equal(current_ids[:, :4], prompt)

                with And("all generated token IDs reside within valid vocabulary bounds [0, 255]"):
                    assert (current_ids >= 0).all()
                    assert (current_ids < 256).all()
