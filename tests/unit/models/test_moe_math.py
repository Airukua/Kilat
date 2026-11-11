"""
Mathematical verification unit tests for M-04 & M-07:
DeepSeekMoE exact 1/K_eff load-balancing, Top-1 router gradient preservation,
num_shared_experts=0 support, naive-loop equivalence, bias-free balancing, and debug_nan_checks.
"""

import pytest
import torch
import torch.nn as nn

from kilat.models.ffn import DeepSeekMoE, FeedForward, SwiGLU
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_moe_load_balance_exact_normalization_and_top1_gradient():
    with Scenario("Unit: M-04 DeepSeekMoE Exact Load-Balance Normalization & Top-1 Router Gradient"):
        with Given("DeepSeekMoE modules with zero-initialized uniform routers across K=1, K=2, and fine-grained=2"):
            x = torch.randn(2, 8, 32)
            for active_k, fg_factor in [(1, 1), (2, 1), (2, 2)]:
                moe = DeepSeekMoE(
                    dim=32,
                    num_routed_experts=4,
                    num_shared_experts=0,
                    active_experts=active_k,
                    fine_grained_factor=fg_factor,
                    aux_loss_coef=0.05,
                    device_balance_coef=0.0,
                )
                nn.init.zeros_(moe.router.weight)

                with When(f"forwarding uniform router with active_experts={active_k}, fine_grained={fg_factor}"):
                    out, aux_loss = moe(x)
                    with Then("expert balance factor sums to 1.0 so aux_loss equals aux_loss_coef exactly"):
                        assert out.shape == (2, 8, 32)
                        assert torch.allclose(aux_loss, torch.tensor(0.05), atol=1e-6)

        with Given("a Top-1 routed DeepSeekMoE with aux_loss_coef=0.0"):
            torch.manual_seed(7)
            moe_top1 = DeepSeekMoE(
                dim=32,
                num_routed_experts=4,
                num_shared_experts=0,
                active_experts=1,
                aux_loss_coef=0.0,
                device_balance_coef=0.0,
            )
            x_in = torch.randn(2, 6, 32, requires_grad=True)

            with When("backpropagating pure task loss without auxiliary loss"):
                out_top1, _ = moe_top1(x_in)
                loss = out_top1.pow(2).mean()
                loss.backward()

                with Then("router.weight receives non-zero gradient (not killed by w_1/w_1 == 1 normalization)"):
                    assert moe_top1.router.weight.grad is not None
                    assert moe_top1.router.weight.grad.abs().sum().item() > 0.0


def test_moe_sorted_dispatch_equivalence_and_bias_free_mode():
    with Scenario("Unit: M-04 & M-07 Token-Sorted Dispatch Equivalence, Bias-Free Mode, & NaN Guard Toggle"):
        with Given("a DeepSeekMoE layer with shared and fine-grained routed experts"):
            torch.manual_seed(123)
            moe = DeepSeekMoE(
                dim=32,
                num_routed_experts=4,
                num_shared_experts=1,
                active_experts=2,
                fine_grained_factor=2,
                aux_loss_coef=0.01,
                device_balance_coef=0.005,
            )
            x_fast = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
            x_ref = x_fast.detach().clone().requires_grad_(True)
            moe_64 = moe.to(torch.float64)

            with When("comparing fast token-sorted dispatch against naive per-token loop reference in FP64"):
                out_fast, aux_fast = moe_64(x_fast)
                out_ref, aux_ref = moe_64._naive_reference_forward(x_ref)
                grad_probe = torch.randn_like(out_fast)
                ((out_fast * grad_probe).sum() + aux_fast).backward()
                assert x_fast.grad is not None
                fast_in_grad = x_fast.grad.clone()
                assert moe_64.router.weight.grad is not None
                fast_router_grad = moe_64.router.weight.grad.clone()

                moe_64.zero_grad(set_to_none=True)
                ((out_ref * grad_probe).sum() + aux_ref).backward()
                assert x_ref.grad is not None
                assert moe_64.router.weight.grad is not None

                with Then("forward outputs, auxiliary losses, and gradients match within 1e-12"):
                    assert torch.max(torch.abs(out_fast - out_ref)).item() < 1e-12
                    assert torch.abs(aux_fast - aux_ref).item() < 1e-12
                    assert torch.max(torch.abs(fast_in_grad - x_ref.grad)).item() < 1e-12
                    assert torch.max(torch.abs(fast_router_grad - moe_64.router.weight.grad)).item() < 1e-12

        with Given("DeepSeek-V3 bias_free router balance mode and debug_nan_checks flag"):
            moe_bf = DeepSeekMoE(
                dim=16,
                num_routed_experts=4,
                num_shared_experts=0,
                active_experts=1,
                router_balance_mode="bias_free",
            )
            with torch.no_grad():
                moe_bf.router.weight.zero_()
                moe_bf.expert_bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))

            with When("forwarding collapsed expert_bias and calling update_expert_bias"):
                _, aux_bf = moe_bf(torch.randn(2, 4, 16))
                before_bias0 = float(moe_bf.expert_bias[0].item())
                before_bias1 = float(moe_bf.expert_bias[1].item())
                moe_bf.update_expert_bias(gamma=0.01)

                with Then("aux_loss is identically 0.0 and dynamic bias penalizes overloaded expert 0"):
                    assert float(aux_bf.item()) == 0.0
                    assert float(moe_bf.expert_bias[0].item()) < before_bias0
                    assert float(moe_bf.expert_bias[1].item()) > before_bias1

            with When("toggling debug_nan_checks on SwiGLU and FeedForward"):
                ff_strict = FeedForward(dim=16, mode="dense", debug_nan_checks=True)
                bad_x = torch.full((1, 2, 16), float("nan"))
                swiglu_fast = SwiGLU(dim=16, debug_nan_checks=False)
                out_nan = swiglu_fast(bad_x)

                with Then("default forward avoids sync exception while debug_nan_checks=True catches NaN"):
                    assert torch.isnan(out_nan).all()
                    with pytest.raises(ValueError):
                        ff_strict(bad_x)
