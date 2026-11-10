"""
Proprietary Verification BDD Tests for Kilat Optimized Triton Kernels (flash_decay).
Universal ML Verification SOP v1.0 - Bab 11: Hardware & Optimized Kernel Verification.
"""
from __future__ import annotations

import pytest
import torch

from kilat.models.kernels.flash_decay import (
    HAS_TRITON,
    FlashDecayFunction,
    _reference_global_decay_fp64,
    compute_decay_normalizer_from_logit,
    triton_global_decay,
)
from tests.shared.bdd import And, Given, Scenario, Then, When


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_triton_flash_decay_cuda_precision_and_parity():
    with Scenario("Proprietary Kernel: Triton Global Decay CUDA FP32 Parity vs FP64 Oracle"):
        with Given("CUDA tensors spanning different head dimensions D and sequence lengths N"):
            device = torch.device("cuda:0")
            for D in (16, 32, 48, 64):
                for N in (8, 64, 128):
                    torch.manual_seed(42 + D + N)
                    B, H = 2, 4
                    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
                    lam = torch.sigmoid(torch.randn(H, device=device, dtype=torch.float32)).requires_grad_(True)

                    with When(f"executing flash_decay_fwd_kernel on CUDA for D={D}, N={N}"):
                        out_triton = triton_global_decay(lam, V)

                    with Then("Triton forward matches the exact FP64 causal matrix within 1e-4 relative error"):
                        V_fp64 = V.detach().cpu().to(torch.float64)
                        lam_fp64 = lam.detach().cpu().to(torch.float64)
                        out_ref = _reference_global_decay_fp64(lam_fp64, V_fp64).to(device=device, dtype=torch.float32)
                        diff_fwd = (out_triton - out_ref).abs().max().item()
                        assert diff_fwd < 1e-4, f"Forward diff {diff_fwd} exceeds 1e-4 for D={D}, N={N}"

                    with When(f"executing flash_decay_bwd_kernel on CUDA for D={D}, N={N}"):
                        grad_seed = torch.randn_like(out_triton)
                        out_triton.backward(grad_seed)
                        grad_v_triton = V.grad.clone()
                        grad_lam_triton = lam.grad.clone()

                        # FP64 autograd oracle
                        V_ref = V_fp64.clone().requires_grad_(True)
                        lam_ref = lam_fp64.clone().requires_grad_(True)
                        out_oracle = _reference_global_decay_fp64(lam_ref, V_ref)
                        out_oracle.backward(grad_seed.detach().cpu().to(torch.float64))

                    with And("Triton analytical gradients match FP64 reference autograd"):
                        err_dv = (grad_v_triton.cpu() - V_ref.grad.float()).abs().max().item()
                        err_dlam = (grad_lam_triton.cpu() - lam_ref.grad.float()).abs().max().item()
                        assert err_dv < 1e-4, f"Grad V diff {err_dv} exceeds 1e-4 for D={D}, N={N}"
                        assert err_dlam < 1e-3, f"Grad lam diff {err_dlam} exceeds 1e-3 for D={D}, N={N}"


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_triton_flash_decay_cuda_extreme_lambda():
    with Scenario("Proprietary Kernel: Triton Global Decay Extreme Lambda Stability"):
        with Given("CUDA decay rates spanning extreme values (0.0001 to 0.99999)"):
            device = torch.device("cuda:0")
            lam_extreme = torch.tensor([0.0001, 0.5, 0.999, 0.99999], device=device, dtype=torch.float32, requires_grad=True)
            V_extreme = torch.randn(2, 4, 128, 32, device=device, dtype=torch.float32, requires_grad=True)

            with When("running forward and backward through Triton kernels at extreme lambda"):
                out = triton_global_decay(lam_extreme, V_extreme)
                loss = out.sum()
                loss.backward()

            with Then("forward and gradient tensors are strictly finite with no NaN or Inf"):
                assert torch.isfinite(out).all(), "Forward has non-finite values"
                assert V_extreme.grad is not None and torch.isfinite(V_extreme.grad).all(), "Grad V has non-finite values"
                assert lam_extreme.grad is not None and torch.isfinite(lam_extreme.grad).all(), "Grad lam has non-finite values"


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_triton_flash_decay_half_precision():
    with Scenario("Proprietary Kernel: Triton Global Decay FP16 and BF16 Stability"):
        device = torch.device("cuda:0")
        for dt, label in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
            with Given(f"CUDA inputs in {label} precision"):
                V = torch.randn(2, 2, 32, 16, device=device, dtype=dt, requires_grad=True)
                lam = torch.tensor([0.7, 0.95], device=device, dtype=torch.float32, requires_grad=True)

                with When(f"forward and backward pass executes in {label}"):
                    out = triton_global_decay(lam, V)
                    out.sum().backward()

                with Then(f"output retains {label} dtype and gradients are finite"):
                    assert out.dtype == dt
                    assert V.grad is not None and torch.isfinite(V.grad).all()
                    assert lam.grad is not None and torch.isfinite(lam.grad).all()
