from __future__ import annotations

import contextlib
import os
from typing import Any, Generator, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from kilat.models.blocks import RMSNorm
from kilat.models.ffn import DeepSeekMoE


class SimulatedPreemptionError(RuntimeError):
    """Injected mid-save preemption signal for verifying atomic checkpoint safety (C-08)."""


class Saboteur:
    """
    100% Non-Destructive In-Memory Fault Injector for ML Chaos Engineering (`kilat chaos`).

    HARD INVARIANT:
      `Saboteur` NEVER modifies any `.py` source file on disk. Every fault injection operates
      exclusively via `contextlib.contextmanager` on in-memory PyTorch module hooks, bound
      instance callables, or isolated `tmp_path` scratch buffers, restoring 100% clean state
      inside `finally:` blocks.
    """

    @contextlib.contextmanager
    def detach_global_decay_branch(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-01: Silent Autograd Branch Detachment.
        Detaches `out_global` inside attention's Global Decay path while leaving MLA recall
        connected so `loss.backward()` still succeeds without raising an exception.
        """
        patched: list[tuple[Any, Any]] = []
        for mod in model.modules():
            orig_fn = getattr(mod, "_run_global_decay_path", None)
            if callable(orig_fn):

                def _detached_decay(*args: Any, _orig: Any = orig_fn, **kwargs: Any) -> Any:
                    out_g, state_g = _orig(*args, **kwargs)
                    return out_g.detach(), state_g

                setattr(mod, "_run_global_decay_path", _detached_decay)
                patched.append((mod, orig_fn))
        try:
            yield
        finally:
            for mod, orig_fn in patched:
                setattr(mod, "_run_global_decay_path", orig_fn)

    @contextlib.contextmanager
    def inject_causal_future_leak(
        self,
        model: nn.Module,
        leak_scale: float = 1e-3,
    ) -> Generator[None, None, None]:
        """
        C-02: Subtle Causal Future Leak.
        Mixes `leak_scale * x_{t+1}` into position `t` on attention output.
        """
        handles = []
        blocks = getattr(model, "layers", None)
        target_mod = blocks[0].attn if blocks is not None and len(blocks) > 0 else model

        def _leak_hook(_m: nn.Module, _inp: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                h = out[0]
                if isinstance(h, torch.Tensor) and h.ndim == 3 and h.shape[1] > 1:
                    leaked = h.clone()
                    leaked[:, :-1, :] = leaked[:, :-1, :] + leak_scale * h[:, 1:, :]
                    return (leaked, *out[1:])
                return out
            if isinstance(out, torch.Tensor) and out.ndim == 3 and out.shape[1] > 1:
                leaked = out.clone()
                leaked[:, :-1, :] = leaked[:, :-1, :] + leak_scale * out[:, 1:, :]
                return leaked
            return out

        handles.append(target_mod.register_forward_hook(_leak_hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def corrupt_batch_isolation(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-03: Cross-Batch Tensor Reshape Mix.
        Simulates a buggy `.transpose(0, 1).contiguous().view_as(h)` reshape without
        reversing transpose, mixing tokens across batch samples `b=0..B-1`.
        """
        handles = []
        blocks = getattr(model, "layers", None)
        target_mod = blocks[0].attn if blocks is not None and len(blocks) > 0 else model

        def _batch_mix_hook(_m: nn.Module, _inp: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                h = out[0]
                if isinstance(h, torch.Tensor) and h.ndim == 3 and h.shape[0] > 1:
                    mixed = h.clone()
                    # Cross-sample contamination at identical causal timestep t
                    mixed[0] = h[0] + 0.05 * h[1:].mean(dim=0)
                    return (mixed, *out[1:])
                return out
            if isinstance(out, torch.Tensor) and out.ndim == 3 and out.shape[0] > 1:
                mixed = out.clone()
                mixed[0] = out[0] + 0.05 * out[1:].mean(dim=0)
                return mixed
            return out

        handles.append(target_mod.register_forward_hook(_batch_mix_hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def inject_fp16_rmsnorm_overflow(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-04: FP16 >256 Square Overflow in RMSNorm.
        Replaces RMSNorm variance reduction in RAM with un-upcasted native FP16 squaring
        so `512.0 ** 2 -> Inf -> x * rsqrt(Inf) == 0.0` (or NaN).
        """
        patched: list[tuple[RMSNorm, Any]] = []
        for mod in model.modules():
            if isinstance(mod, RMSNorm):
                orig_fwd = mod.forward

                def _naive_fp16_fwd(x: torch.Tensor, _m: RMSNorm = mod) -> torch.Tensor:
                    # Deliberately omit .float() upcast: overflows float16 for |x| >= 256
                    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + _m.eps)
                    return x * rms * _m.weight

                setattr(mod, "forward", _naive_fp16_fwd)
                patched.append((mod, orig_fwd))
        try:
            yield
        finally:
            for mod, orig_fwd in patched:
                setattr(mod, "forward", orig_fwd)

    @contextlib.contextmanager
    def inject_naive_decay_singularity(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-05: Decay Horizon Singularity (`lambda -> 1^-`).
        Injects naive `(1 - lam^(t+1)) / (1 - lam)` normalizer in float32, which produces
        `0.0 / 0.0 = NaN` when `log_lambda = 25.0` (`float32(sigmoid(25.0)) == 1.0`).
        """
        patched: list[tuple[Any, Any]] = []
        for mod in model.modules():
            orig_fn = getattr(mod, "_run_global_decay_path", None)
            if callable(orig_fn) and getattr(mod, "n_global_heads", 0) > 0:

                def _naive_decay_path(
                    x: torch.Tensor,
                    past_global_state: Any,
                    past_seq_len: int,
                    has_cache: bool,
                    use_cache: bool,
                    _m: Any = mod,
                    _orig: Any = orig_fn,
                ) -> Any:
                    if has_cache and past_global_state is not None and _m.log_lambda is not None:
                        B, N, _ = x.shape
                        Dh = _m.head_dim
                        lam = torch.sigmoid(_m.log_lambda.float()).view(1, _m.n_global_heads, 1)
                        V_g = (
                            _m.v_proj_global(x)
                            .view(B, N, _m.n_global_heads, Dh)
                            .transpose(1, 2)
                            .float()
                        )
                        acc = past_global_state.float()
                        outs = []
                        for j in range(N):
                            acc = lam * acc + V_g[:, :, j, :]
                            t_step = past_seq_len + j
                            # Buggy naive geometric sum: 0.0 / 0.0 = NaN when lam == 1.0 in float32!
                            z_naive = (1.0 - lam.pow(t_step + 1)) / (1.0 - lam)
                            outs.append((acc / z_naive).to(x.dtype))
                        out_g = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, N, _m.n_global_heads * Dh)
                        return out_g, (acc if use_cache else None)
                    return _orig(x, past_global_state, past_seq_len, has_cache, use_cache)

                setattr(mod, "_run_global_decay_path", _naive_decay_path)
                patched.append((mod, orig_fn))
        try:
            yield
        finally:
            for mod, orig_fn in patched:
                setattr(mod, "_run_global_decay_path", orig_fn)

    @contextlib.contextmanager
    def collapse_moe_router(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-06a: MoE Single-Expert Router Collapse.
        Injects +50.0 constant logit bias onto Expert 0 inside DeepSeekMoE routing in RAM
        so 100% of tokens collapse onto a single expert (`entropy_norm ~ 0.0`).
        """
        patched: list[tuple[DeepSeekMoE, Any]] = []
        for mod in model.modules():
            if isinstance(mod, DeepSeekMoE):
                orig_fwd = mod.forward

                def _collapsed_fwd(x: torch.Tensor, _m: DeepSeekMoE = mod) -> tuple[torch.Tensor, torch.Tensor]:
                    import math as _math

                    B, S, D = x.shape
                    x_flat = x.reshape(-1, D)
                    N_r = _m.total_routed_experts
                    K_eff = _m.effective_active_experts
                    logits = F.linear(x_flat.float(), _m.router.weight.float())
                    logits = logits.clone()
                    logits[:, 0] = logits[:, 0] + 50.0  # Force 100% routing collapse onto Expert 0
                    probs = F.softmax(logits, dim=-1)
                    _, topk_idx = torch.topk(probs, K_eff, dim=-1)
                    topk_w = probs.gather(dim=-1, index=topk_idx)
                    if K_eff > 1 and _m.renormalize_topk:
                        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-12)

                    counts = torch.bincount(topk_idx.reshape(-1), minlength=N_r)
                    frac = counts.float() / float(max(1, x_flat.shape[0] * K_eff))
                    ent = -(frac * torch.log(frac + 1e-12)).sum() / max(1e-12, _math.log(max(2, N_r)))
                    _m._last_router_stats = {
                        "expert_fraction": frac.detach(),
                        "entropy_norm": ent.detach(),
                        "max_load_ratio": (frac.max() * float(N_r)).detach(),
                    }
                    shared_out = torch.zeros_like(x)
                    for s_exp in _m.shared_experts:
                        shared_out = shared_out + s_exp(x)
                    routed_out = topk_w[:, :1].to(x.dtype) * _m.routed_experts[0](x_flat)
                    return shared_out + routed_out.reshape(B, S, D), x.sum() * 0.0

                setattr(mod, "forward", _collapsed_fwd)
                patched.append((mod, orig_fwd))
        try:
            yield
        finally:
            for mod, orig_fwd in patched:
                setattr(mod, "forward", orig_fwd)

    @contextlib.contextmanager
    def kill_top1_router_grad(self, model: nn.Module) -> Generator[None, None, None]:
        """
        C-06b: Top-1 Router Renormalization Zero-Grad Bug.
        Reproduces the classic `topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-9)` bug
        when K_eff == 1 and `aux_loss_coef == 0.0`, crushing router gradient by ~10^8x!
        """
        patched: list[tuple[DeepSeekMoE, Any, float, float]] = []
        for mod in model.modules():
            if isinstance(mod, DeepSeekMoE):
                orig_fwd = mod.forward
                orig_aux = mod.aux_loss_coef
                orig_dev = mod.device_balance_coef
                mod.aux_loss_coef = 0.0
                mod.device_balance_coef = 0.0

                def _buggy_top1_fwd(x: torch.Tensor, _m: DeepSeekMoE = mod) -> tuple[torch.Tensor, torch.Tensor]:
                    B, S, D = x.shape
                    x_flat = x.reshape(-1, D)
                    logits = F.linear(x_flat.float(), _m.router.weight.float())
                    probs = F.softmax(logits, dim=-1)
                    _, idx = torch.topk(probs, 1, dim=-1)
                    w = probs.gather(dim=-1, index=idx)
                    # Deadly Top-1 bug: dividing scalar positive weight by itself == 1.0 (d/dw ~ 0!)
                    w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
                    shared_out = torch.zeros_like(x)
                    for s_exp in _m.shared_experts:
                        shared_out = shared_out + s_exp(x)
                    routed_out = torch.zeros_like(x_flat)
                    for e_id, exp in enumerate(_m.routed_experts):
                        mask = idx.squeeze(-1) == e_id
                        if bool(mask.any().item()):
                            routed_out[mask] = routed_out[mask] + w_norm[mask].to(x.dtype) * exp(x_flat[mask])
                    return shared_out + routed_out.reshape(B, S, D), x.sum() * 0.0

                setattr(mod, "forward", _buggy_top1_fwd)
                patched.append((mod, orig_fwd, orig_aux, orig_dev))
        try:
            yield
        finally:
            for mod, orig_fwd, orig_aux, orig_dev in patched:
                setattr(mod, "forward", orig_fwd)
                mod.aux_loss_coef = orig_aux
                mod.device_balance_coef = orig_dev

    @contextlib.contextmanager
    def poison_dataloader_batches(
        self,
        dataloader: Iterable[dict[str, torch.Tensor]],
        mode: str,
        vocab_size: int,
    ) -> Generator[list[dict[str, torch.Tensor]], None, None]:
        """
        C-07: Poisoned Data Batch (`mode='all_masked'` or `mode='oov_token'`).
        Produces an in-memory poisoned batch list without touching underlying dataset files.
        """
        poisoned: list[dict[str, torch.Tensor]] = []
        for idx, batch in enumerate(dataloader):
            b_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if idx == 0:
                if mode == "all_masked":
                    b_copy["labels"] = torch.full_like(b_copy["input_ids"], -100)
                elif mode == "oov_token":
                    b_copy["input_ids"][0, 0] = vocab_size + 5
                else:
                    raise ValueError(f"Unknown poison mode: {mode}")
            poisoned.append(b_copy)
            if idx >= 2:
                break
        yield poisoned

    @contextlib.contextmanager
    def corrupt_checkpoint_byte(self, checkpoint_dir: str) -> Generator[str, None, None]:
        """
        C-08: Flip 1 byte in model weights inside `checkpoint_dir` (and restore in `finally:`)
        to prove SHA-256 `checkpoint_manifest.json` catches silent disk bit-rot / truncation.
        """
        candidates = ["model.safetensors", "pytorch_model.bin", "training_state.pt"]
        target_path = None
        for name in candidates:
            p = os.path.join(checkpoint_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 16:
                target_path = p
                break
        if target_path is None:
            raise FileNotFoundError(f"No weight/state file found in {checkpoint_dir} to corrupt.")

        with open(target_path, "rb") as f:
            original_bytes = f.read()

        mutated = bytearray(original_bytes)
        mid = len(mutated) // 2
        mutated[mid] ^= 0xFF
        with open(target_path, "wb") as f:
            f.write(bytes(mutated))

        try:
            yield target_path
        finally:
            with open(target_path, "wb") as f:
                f.write(original_bytes)
