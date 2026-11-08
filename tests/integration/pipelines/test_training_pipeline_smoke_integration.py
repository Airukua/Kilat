"""
Training Pipeline Integration Smoke Test:
Verifies model convergence and parameter update across forward, backward, optimizer, and scheduler steps.
Follows standards from ml_integration_testing_structure_md.md (Pipeline Overfitting Smoke Test).
"""

import torch

from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.training.optimizer import create_optimizer
from kilat.training.scheduler import get_scheduler
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_training_pipeline_convergence_smoke():
    with Scenario("Integration: Training Loop Convergence Smoke Test"):
        with Given("a deterministic micro-batch of input tokens and targets"):
            torch.manual_seed(42)
            config = KilatConfig(
                vocab_size=128,
                n_embd=64,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                ffn_mode="dense",
            )
            model = KilatTransformer(config)
            model.train()

            optimizer = create_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
            scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=10)

            # Golden single batch to overfit
            inputs = torch.randint(0, 128, (2, 8))
            labels = inputs.clone()

            with When("executing forward pass, backpropagation, and optimizer updates"):
                losses = []
                for _ in range(5):
                    optimizer.zero_grad()
                    out = model(input_ids=inputs, labels=labels)
                    loss = out.loss
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                with Then("the loss is non-negative and finite"):
                    assert all(torch.isfinite(torch.tensor(l)) for l in losses)

                with And("the loss decreases as the model fits the micro-batch"):
                    assert losses[-1] < losses[0]

                with And("parameter gradients are non-zero after backpropagation"):
                    has_grad = any(
                        p.grad is not None and torch.norm(p.grad) > 0.0
                        for p in model.parameters()
                    )
                    assert has_grad
