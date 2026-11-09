from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kilat.utils.validators import (
    validate_choice,
    validate_finite_tensor,
    validate_less_equal,
    validate_non_negative_int,
    validate_positive_int,
    validate_probability,
    validate_tensor_last_dim,
    validate_tensor_rank,
)


class SwiGLU(nn.Module):
    """
    SwiGLU (SiLU-Gated Linear Unit) feed-forward block:
        SwiGLU(x) = down(Dropout(SiLU(gate(x)) * up(x)))
    """

    def __init__(
        self,
        dim: int,
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        debug_nan_checks: bool = False,
    ) -> None:
        super().__init__()
        validate_positive_int("dim", dim)
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        validate_probability("dropout", dropout)

        hidden = max(64, ((int(dim * ff_mult) + 63) // 64) * 64)

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.debug_nan_checks = debug_nan_checks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(f"x must have rank >= 2, got shape {tuple(x.shape)}")
        validate_tensor_last_dim(x, self.down.out_features, "x")

        out = self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))
        if self.debug_nan_checks:
            return validate_finite_tensor(out, "SwiGLU output")
        return out


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V2 / V3 Mixture-of-Experts with optional shared experts, fine-grained
    expert segmentation, FP32 router computation, exact 1/K_eff load-balancing,
    Top-1 router gradient preservation, and optional auxiliary-loss-free bias balancing.
    """

    expert_bias: torch.Tensor

    def __init__(
        self,
        dim: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        active_experts: int = 8,
        fine_grained_factor: int = 1,
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.001,
        device_balance_coef: float = 0.001,
        router_balance_mode: Literal["aux_loss", "bias_free"] = "aux_loss",
        renormalize_topk: bool = True,
        debug_nan_checks: bool = False,
    ) -> None:
        super().__init__()
        validate_positive_int("dim", dim)
        validate_positive_int("num_routed_experts", num_routed_experts)
        validate_non_negative_int("num_shared_experts", num_shared_experts)
        validate_positive_int("active_experts", active_experts)
        validate_positive_int("fine_grained_factor", fine_grained_factor)
        validate_less_equal("active_experts", active_experts, num_routed_experts)
        validate_probability("dropout", dropout)
        validate_choice("router_balance_mode", router_balance_mode, ("aux_loss", "bias_free"))

        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.active_experts = active_experts
        self.fine_grained_factor = fine_grained_factor
        self.total_routed_experts = num_routed_experts * fine_grained_factor
        self.effective_active_experts = active_experts * fine_grained_factor
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef
        self.router_balance_mode = router_balance_mode
        self.renormalize_topk = renormalize_topk
        self.debug_nan_checks = debug_nan_checks

        expert_ff_mult = ff_mult / float(fine_grained_factor)

        self.router = nn.Linear(dim, self.total_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.total_routed_experts, dtype=torch.float32),
            persistent=(router_balance_mode == "bias_free"),
        )

        self.shared_experts = nn.ModuleList(
            [
                SwiGLU(dim, ff_mult=ff_mult, dropout=dropout, debug_nan_checks=debug_nan_checks)
                for _ in range(num_shared_experts)
            ]
        )
        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(dim, ff_mult=expert_ff_mult, dropout=dropout, debug_nan_checks=debug_nan_checks)
                for _ in range(self.total_routed_experts)
            ]
        )

        # Detached async health telemetry read by FlightRecorderCallback only at logging steps
        self._last_router_stats: Optional[Dict[str, torch.Tensor]] = None

    def _naive_reference_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple reference MoE forward pass for Gate 3 equivalence verification.
        """
        B, S, D = x.shape
        router_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_flat = x.reshape(-1, D)
        T = x_flat.shape[0]
        K_eff = self.effective_active_experts
        N_r = self.total_routed_experts

        shared_out = torch.zeros_like(x)
        for s_exp in self.shared_experts:
            shared_out = shared_out + s_exp(x)

        logits = F.linear(x_flat.to(router_dtype), self.router.weight.to(router_dtype))
        probs = F.softmax(logits, dim=-1)
        selection_scores = probs + self.expert_bias.to(router_dtype) if self.router_balance_mode == "bias_free" else probs
        _, topk_idx = torch.topk(selection_scores, K_eff, dim=-1)
        topk_w = probs.gather(dim=-1, index=topk_idx)
        if K_eff > 1 and self.renormalize_topk:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-12)

        routed_out = torch.zeros_like(x_flat)
        for t in range(T):
            for k in range(K_eff):
                e_id = int(topk_idx[t, k].item())
                w_tk = topk_w[t, k].to(x.dtype)
                routed_out[t : t + 1] = routed_out[t : t + 1] + w_tk * self.routed_experts[e_id](x_flat[t : t + 1])

        counts = torch.bincount(topk_idx.reshape(-1), minlength=N_r)
        expert_fraction = counts.to(router_dtype) / float(max(1, T * K_eff))
        prob_avg = probs.mean(dim=0)
        if self.router_balance_mode == "bias_free":
            total_aux = (probs.sum() * 0.0).to(x.dtype)
        else:
            exp_bal = float(N_r) * (expert_fraction * prob_avg).sum()
            dev_bal = (
                torch.var(expert_fraction, dim=0, unbiased=False) * float(N_r)
                if N_r > 1
                else torch.zeros((), device=x.device, dtype=router_dtype)
            )
            total_aux = (self.aux_loss_coef * exp_bal + self.device_balance_coef * dev_bal).to(x.dtype)

        return (shared_out.reshape(-1, D) + routed_out).reshape(B, S, D), total_aux

    @torch.no_grad()
    def update_expert_bias(self, gamma: float = 1e-3) -> None:
        """
        DeepSeek-V3 auxiliary-loss-free dynamic router bias update:
            b_i <- b_i + gamma * sign(1 / N_r - f_i)
        """
        if self.router_balance_mode != "bias_free" or self._last_router_stats is None:
            return
        frac = self._last_router_stats["expert_fraction"].to(self.expert_bias.dtype)
        target = 1.0 / float(self.total_routed_experts)
        self.expert_bias.add_(gamma * torch.sign(target - frac))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.dim, "x")
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (T, D)
        num_tokens = x_flat.shape[0]
        N_r = self.total_routed_experts
        K_eff = self.effective_active_experts

        # 1. Shared experts (supports num_shared_experts == 0 cleanly)
        if self.num_shared_experts > 0:
            shared_output = torch.zeros_like(x)
            for s_expert in self.shared_experts:
                shared_output = shared_output + s_expert(x)
        else:
            shared_output = torch.zeros_like(x)

        # 2. Router computation in FP32 (or FP64 during gradcheck)
        router_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        router_logits = F.linear(x_flat.to(router_dtype), self.router.weight.to(router_dtype))
        router_probs = F.softmax(router_logits, dim=-1)  # (T, N_r)

        if self.router_balance_mode == "bias_free":
            selection_scores = router_probs + self.expert_bias.to(router_dtype)
        else:
            selection_scores = router_probs

        _, topk_indices = torch.topk(selection_scores, K_eff, dim=-1)  # (T, K_eff)
        topk_weights = router_probs.gather(dim=-1, index=topk_indices)  # (T, K_eff)

        # Preserve router gradient when K_eff == 1 (dividing w_1 / w_1 == 1.0 kills gradient!)
        if K_eff > 1 and self.renormalize_topk:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # 3. Token-sorted batched expert dispatch (eliminates per-expert .any() boolean mask syncs)
        flat_expert_ids = topk_indices.reshape(-1)  # (T * K_eff,)
        flat_weights = topk_weights.reshape(-1)  # (T * K_eff,)
        counts = torch.bincount(flat_expert_ids, minlength=N_r)  # (N_r,)

        routed_output = torch.zeros_like(x_flat)
        if num_tokens > 0:
            if torch.compiler.is_compiling():
                # Static-graph friendly branch for torch.compile(fullgraph=True)
                for expert_idx, expert in enumerate(self.routed_experts):
                    match = topk_indices == expert_idx  # (T, K_eff)
                    token_w = (topk_weights * match.to(router_dtype)).sum(dim=-1, keepdim=True).to(x.dtype)
                    routed_output = routed_output + token_w * expert(x_flat)
            else:
                sorted_expert_ids, perm = flat_expert_ids.sort()
                sorted_token_ids = torch.div(perm, K_eff, rounding_mode="floor")
                sorted_weights = flat_weights.index_select(0, perm).to(x.dtype).unsqueeze(-1)
                sorted_inputs = x_flat.index_select(0, sorted_token_ids)

                # Single tiny host transfer of N_r integer counts for the whole layer
                split_sizes = counts.tolist()
                input_chunks = torch.split(sorted_inputs, split_sizes, dim=0)
                out_chunks = [
                    self.routed_experts[idx](chunk)
                    for idx, (chunk, cnt) in enumerate(zip(input_chunks, split_sizes))
                    if cnt > 0
                ]
                if out_chunks:
                    sorted_outputs = torch.cat(out_chunks, dim=0) * sorted_weights
                    routed_output.index_add_(0, sorted_token_ids, sorted_outputs)

        output = (shared_output.reshape(-1, D) + routed_output).reshape(B, S, D)

        # 4. Exact DeepSeek-V2 load-balancing loss (normalized by K_eff so balanced sum == 1.0)
        denom = float(max(1, num_tokens * K_eff))
        expert_fraction = counts.to(router_dtype) / denom  # sums to 1.0!
        router_prob_avg = router_probs.mean(dim=0)  # sums to 1.0!

        with torch.no_grad():
            eps_ent = 1e-12
            norm_denom = math.log(max(2, N_r))
            entropy = -(expert_fraction * (expert_fraction + eps_ent).log()).sum() / norm_denom
            max_load_ratio = expert_fraction.max() * float(N_r)
            self._last_router_stats = {
                "expert_fraction": expert_fraction.detach(),
                "entropy_norm": entropy.detach(),
                "max_load_ratio": max_load_ratio.detach(),
            }

        if self.router_balance_mode == "bias_free":
            total_aux_loss = (router_probs.sum() * 0.0).to(x.dtype)
        else:
            expert_balance_loss = float(N_r) * (expert_fraction * router_prob_avg).sum()
            # Use population variance (unbiased=False) so uniform routing has exact 0.0 device imbalance
            device_balance_loss = (
                torch.var(expert_fraction, dim=0, unbiased=False) * float(N_r)
                if N_r > 1
                else torch.zeros((), device=x.device, dtype=router_dtype)
            )
            total_aux_loss = (
                self.aux_loss_coef * expert_balance_loss + self.device_balance_coef * device_balance_loss
            ).to(x.dtype)

        if self.debug_nan_checks:
            validate_finite_tensor(output, "DeepSeekMoE output")
            validate_finite_tensor(total_aux_loss, "DeepSeekMoE aux_loss")

        return output, total_aux_loss


class FeedForward(nn.Module):
    """
    Unified feed-forward block operating as dense SwiGLU or DeepSeekMoE.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        mode: Literal["dense", "moe", "swiglu"] = "moe",
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        num_experts: int = 64,
        num_shared_experts: int = 2,
        active_experts: int = 8,
        fine_grained_factor: int = 1,
        aux_loss_coef: float = 0.001,
        device_balance_coef: float = 0.001,
        router_balance_mode: Literal["aux_loss", "bias_free"] = "aux_loss",
        debug_nan_checks: bool = False,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = int(config.n_embd)
            mode = config.ffn_mode
            ff_mult = float(getattr(config, "ff_mult", 8 / 3))
            dropout = float(config.ffn_dropout)
            num_experts = int(config.num_experts)
            num_shared_experts = int(getattr(config, "num_shared_experts", 2))
            active_experts = int(config.active_experts)
            fine_grained_factor = int(getattr(config, "fine_grained_factor", 1))
            aux_loss_coef = float(config.aux_loss_coef)
            device_balance_coef = float(getattr(config, "device_balance_coef", 0.0))
            router_balance_mode = getattr(config, "router_balance_mode", "aux_loss")
            debug_nan_checks = bool(getattr(config, "debug_nan_checks", False))

        if dim is None:
            raise ValueError("FeedForward requires either `dim` or `config` to be provided.")
        validate_positive_int("dim", dim)
        if mode == "swiglu":
            mode = "dense"
        validate_choice("mode", mode, ("dense", "moe"))
        validate_probability("dropout", dropout)
        self.mode = mode
        self.debug_nan_checks = debug_nan_checks

        self.layer: nn.Module
        if mode == "dense":
            self.layer = SwiGLU(
                dim,
                ff_mult=ff_mult,
                dropout=dropout,
                debug_nan_checks=debug_nan_checks,
            )
            self.num_experts = 1
            self.active_experts = 1
        else:
            self.layer = DeepSeekMoE(
                dim=dim,
                num_routed_experts=num_experts,
                num_shared_experts=num_shared_experts,
                active_experts=active_experts,
                fine_grained_factor=fine_grained_factor,
                ff_mult=ff_mult,
                dropout=dropout,
                aux_loss_coef=aux_loss_coef,
                device_balance_coef=device_balance_coef,
                router_balance_mode=router_balance_mode,
                debug_nan_checks=debug_nan_checks,
            )
            self.num_experts = num_experts
            self.active_experts = active_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        validate_tensor_rank(x, 3, "x")
        if self.mode == "dense":
            out = self.layer(x)
            if self.debug_nan_checks:
                validate_finite_tensor(out, "FeedForward[dense] output")
            return out, None

        out, aux_loss = self.layer(x)
        if self.debug_nan_checks:
            validate_finite_tensor(out, "FeedForward[moe] output")
            if aux_loss is not None:
                validate_finite_tensor(aux_loss, "FeedForward[moe] aux_loss")
        return out, aux_loss
