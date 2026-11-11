"""
Mathematical verification unit tests for M-01: Global Decay Autograd & Near-Unity Stability.
Follows Kilat BDD testing standards.
"""

import torch

from kilat.models.attention import KilatAttention
from kilat.models.kernels.flash_decay import (
    FlashDecayFunction,
    _reference_global_decay_fp64,
    compute_decay_normalizer_from_logit,
    triton_global_decay,
)
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_flash_decay_fp64_gradcheck_and_reference_parity():
    with Scenario("Unit: M-01 FlashDecayFunction Analytical Backward & FP64 Gradcheck"):
        with Given("FP64 value tensors and decay rates spanning normal to near-unity regimes"):
            torch.manual_seed(42)
            B, H, N, D = 2, 4, 7, 6
            V = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
            lam = torch.tensor([0.4, 0.9, 0.99999, 1.0 - 1e-7], dtype=torch.float64, requires_grad=True)

            with When("verifying analytical backward against finite-difference torch.autograd.gradcheck"):
                passed = torch.autograd.gradcheck(
                    FlashDecayFunction.apply,
                    (lam, V),
                    eps=1e-6,
                    atol=1e-6,
                    rtol=1e-5,
                )
                with Then("gradcheck passes strictly in float64"):
                    assert passed is True

            with When("comparing forward and backward against the O(N^2) causal matrix FP64 reference"):
                V_fast = V.detach().clone().requires_grad_(True)
                lam_fast = lam.detach().clone().requires_grad_(True)
                V_ref = V.detach().clone().requires_grad_(True)
                lam_ref = lam.detach().clone().requires_grad_(True)

                grad_seed = torch.randn(B, H, N, D, dtype=torch.float64)
                out_fast = triton_global_decay(lam_fast, V_fast)
                out_ref = _reference_global_decay_fp64(lam_ref, V_ref)

                (out_fast * grad_seed).sum().backward()
                (out_ref * grad_seed).sum().backward()

                with Then("forward outputs match within 1e-12"):
                    assert torch.max(torch.abs(out_fast - out_ref)).item() < 1e-12

                with And("analytical V and lambda gradients match reference autograd within 1e-11"):
                    assert V_fast.grad is not None and V_ref.grad is not None
                    assert lam_fast.grad is not None and lam_ref.grad is not None
                    assert torch.max(torch.abs(V_fast.grad - V_ref.grad)).item() < 1e-11
                    assert torch.max(torch.abs(lam_fast.grad - lam_ref.grad)).item() < 1e-11


def test_decay_normalizer_near_unity_and_empty_head_guards():
    with Scenario("Unit: M-01 Decay Normalizer Singularity Guard & Degenerate Head Safety"):
        with Given("extreme logit log_lambda -> +infinity (lambda -> 1.0) at long sequence position"):
            extreme_logit = torch.tensor([25.0, 80.0], dtype=torch.float32)

            with When("evaluating compute_decay_normalizer_from_logit at position t = 99"):
                z_99 = compute_decay_normalizer_from_logit(extreme_logit, 99)
                z_vec = compute_decay_normalizer_from_logit(extreme_logit, torch.tensor([0, 99]))

                with Then("normalizer converges cleanly to t + 1 = 100.0 instead of collapsing to 0"):
                    assert torch.allclose(z_99, torch.tensor([100.0, 100.0]), atol=1e-4)
                    assert torch.allclose(z_vec, torch.tensor([1.0, 100.0]), atol=1e-4)

        with Given("a KilatAttention module and empty-head edge case (recall_ratio=1.0)"):
            attn_hybrid = KilatAttention(n_embd=32, n_head=4, recall_ratio=0.5)
            attn_mla_only = KilatAttention(n_embd=32, n_head=4, recall_ratio=1.0)
            x = torch.randn(2, 6, 32, requires_grad=True)

            with When("backpropagating through both hybrid and pure-MLA attention"):
                loss_hybrid = attn_hybrid(x).pow(2).mean()
                loss_hybrid.backward()

                x_mla = torch.randn(2, 6, 32, requires_grad=True)
                loss_mla = attn_mla_only(x_mla).pow(2).mean()
                loss_mla.backward()

                with Then("hybrid global decay parameters receive finite non-zero gradients"):
                    assert attn_hybrid.log_lambda.grad is not None
                    assert attn_hybrid.v_proj_global.weight.grad is not None
                    assert torch.isfinite(attn_hybrid.log_lambda.grad).all()
                    assert attn_hybrid.log_lambda.grad.abs().sum().item() > 0.0
                    assert attn_hybrid.v_proj_global.weight.grad.abs().sum().item() > 0.0

                with And("zero-global-head configuration executes backward cleanly without error"):
                    assert x_mla.grad is not None
                    assert torch.isfinite(x_mla.grad).all()
