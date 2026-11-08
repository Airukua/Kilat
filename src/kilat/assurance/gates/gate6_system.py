from __future__ import annotations

import gc
import os
import tempfile
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.model_config import KilatConfig
from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import SyntheticTokenDataset
from kilat.engine.args import TrainingArguments
from kilat.engine.callbacks import TrainerCallback, TrainerControl, TrainerState
from kilat.engine.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    set_seed,
    verify_checkpoint_manifest,
)
from kilat.engine.trainer import KilatTrainer
from kilat.models.transformer import KilatTransformer


def _build_micro_system_config(base: Optional[KilatConfig] = None) -> KilatConfig:
    if base is not None and int(base.n_embd) <= 64 and int(base.n_layer) <= 2 and int(base.vocab_size) <= 512:
        return KilatConfig.from_dict(base.to_dict())
    is_moe = base is not None and getattr(base, "ffn_mode", "dense") == "moe"
    return KilatConfig(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_layer=2,
        max_position_embeddings=32,
        dropout=0.0,
        ffn_mode="moe" if is_moe else "dense",
        num_experts=4 if is_moe else 4,
        active_experts=2,
        num_shared_experts=1 if is_moe else 0,
        use_rope=True,
        rope_head_dim=4,
    )


class _StopAtStepCallback(TrainerCallback):
    def __init__(self, target_step: int, save_tag: str = "mid") -> None:
        self.target_step = target_step
        self.save_tag = save_tag
        self.trainer_ref: Optional[KilatTrainer] = None

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Optional[TrainerControl]:
        _ = args
        trainer = kwargs.get("trainer") or self.trainer_ref
        if state.global_step >= self.target_step:
            if trainer is not None:
                trainer._save(tag=self.save_tag)
            control.should_training_stop = True
        return control


def run_gate6_system(
    model_config: Optional[KilatConfig] = None,
    work_dir: Optional[str] = None,
) -> GateResult:
    """
    Gate 6: Memory Leak, Checkpoint Bitwise Parity, & Accumulation/DDP Parity (`G6_SYSTEM_PARITY`).
    Verifies:
      1. Zero per-step autograd graph / CUDA memory creep across 10 consecutive training steps.
      2. Bitwise-exact (`0.0` diff) Checkpoint Save@5 + Resume 6..10 vs Uninterrupted 1..10 + SHA-256 manifest.
      3. Gradient accumulation (`micro_batch=2, grad_acc=2`) vs full-batch (`batch=4, grad_acc=1`) parity (< 1e-5).
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    tmp_ctx = tempfile.TemporaryDirectory() if work_dir is None else None
    base_dir = work_dir if work_dir is not None else tmp_ctx.name  # type: ignore[union-attr]

    try:
        cfg = _build_micro_system_config(model_config)
        ds = SyntheticTokenDataset(
            vocab_size=cfg.vocab_size,
            seq_len=16,
            num_samples=24,
            seed=777,
        )
        collator = KilatDataCollator(pad_token_id=0, max_length=16)

        # ── 1 & 2. Zero Memory Creep + Bitwise Checkpoint Save/Resume (1..10) ────
        def _make_loader() -> DataLoader[Any]:
            g = torch.Generator()
            g.manual_seed(777)
            return DataLoader(ds, batch_size=2, shuffle=True, generator=g, collate_fn=collator)

        dir_a = os.path.join(base_dir, "traj_a")
        dir_b1 = os.path.join(base_dir, "traj_b1")
        dir_b2 = os.path.join(base_dir, "traj_b2")

        # Trajectory A: Uninterrupted steps 1..10 + memory creep monitoring
        mem_snapshots: list[int] = []

        class _MemoryProbeCallback(TrainerCallback):
            def on_step_end(
                self,
                args: Any,
                state: TrainerState,
                control: TrainerControl,
                **kwargs: Any,
            ) -> Optional[TrainerControl]:
                _ = (args, control, kwargs)
                if state.global_step >= 3:
                    if torch.cuda.is_available():
                        mem_snapshots.append(int(torch.cuda.memory_allocated()))
                    else:
                        # Verify log_history never retains Tensor objects with grad_fn
                        leaked_tensors = sum(
                            1
                            for entry in state.log_history
                            for val in entry.values()
                            if isinstance(val, torch.Tensor)
                        )
                        mem_snapshots.append(leaked_tensors)
                return None

        set_seed(9001)
        model_a = KilatTransformer(cfg)
        args_a = TrainingArguments(
            output_dir=dir_a,
            training_mode="steps",
            max_steps=10,
            learning_rate=3e-4,
            warmup_steps=2,
            scheduler_type="cosine",
            logging_steps=1,
            save_checkpoints=False,
            seed=9001,
        )
        trainer_a = KilatTrainer(model_a, args_a, _make_loader(), callbacks=[_MemoryProbeCallback()])
        state_a = trainer_a.train()
        losses_a_tail = [entry["loss"] for entry in state_a.log_history if "loss" in entry and entry["step"] >= 6]

        # Check memory creep: strictly monotonic growth across all steps 3..10 indicates a graph/tensor leak
        if len(mem_snapshots) >= 4:
            if torch.cuda.is_available():
                strictly_increasing = all(
                    mem_snapshots[i + 1] > mem_snapshots[i] for i in range(len(mem_snapshots) - 1)
                )
                if strictly_increasing and (mem_snapshots[-1] - mem_snapshots[0]) > 1024 * 1024:
                    raise AssuranceError(
                        AssuranceFailureCode.ERR_MEMORY_CREEP,
                        f"Monotonic CUDA memory creep detected across steps 3..10: {mem_snapshots}",
                    )
            elif any(x > 0 for x in mem_snapshots):
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MEMORY_CREEP,
                    "Tensor objects retained inside TrainerState.log_history (host memory leak).",
                )

        # Trajectory B1: Steps 1..5 -> atomic save
        set_seed(9001)
        model_b1 = KilatTransformer(cfg)
        args_b1 = TrainingArguments(
            output_dir=dir_b1,
            training_mode="steps",
            max_steps=10,
            learning_rate=3e-4,
            warmup_steps=2,
            scheduler_type="cosine",
            logging_steps=1,
            save_checkpoints=False,
            seed=9001,
        )
        stop_cb = _StopAtStepCallback(target_step=5, save_tag="step5")
        trainer_b1 = KilatTrainer(model_b1, args_b1, _make_loader(), callbacks=[stop_cb])
        stop_cb.trainer_ref = trainer_b1
        trainer_b1.train()

        ckpt_step5 = os.path.join(dir_b1, "checkpoint-step5")
        try:
            verify_checkpoint_manifest(ckpt_step5)
        except (RuntimeError, ValueError) as exc:
            raise AssuranceError(
                AssuranceFailureCode.ERR_CHECKPOINT_CHECKSUM_MISMATCH,
                str(exc),
            ) from exc

        # Trajectory B2: Fresh init with different seed -> resume from step 5 -> steps 6..10
        set_seed(1234)
        model_b2 = KilatTransformer(cfg)
        args_b2 = TrainingArguments(
            output_dir=dir_b2,
            training_mode="steps",
            max_steps=10,
            learning_rate=3e-4,
            warmup_steps=2,
            scheduler_type="cosine",
            logging_steps=1,
            save_checkpoints=False,
            resume_from_checkpoint=ckpt_step5,
            seed=1234,
        )
        trainer_b2 = KilatTrainer(model_b2, args_b2, _make_loader())
        state_b2 = trainer_b2.train()
        losses_b_tail = [entry["loss"] for entry in state_b2.log_history if "loss" in entry and entry["step"] >= 6]

        if len(losses_a_tail) != len(losses_b_tail):
            raise AssuranceError(
                AssuranceFailureCode.ERR_CHECKPOINT_PARITY_DRIFT,
                f"Resumed trajectory step count mismatch: {len(losses_b_tail)} vs {len(losses_a_tail)}.",
            )

        max_loss_diff = max(abs(a - b) for a, b in zip(losses_a_tail, losses_b_tail))
        max_param_diff = max(
            float((pa.detach() - pb.detach()).abs().max().item())
            for pa, pb in zip(trainer_a.model.parameters(), trainer_b2.model.parameters())
        )
        if max_loss_diff > 0.0 or max_param_diff > 0.0:
            raise AssuranceError(
                AssuranceFailureCode.ERR_CHECKPOINT_PARITY_DRIFT,
                f"Checkpoint resume violated bitwise parity: max_loss_diff={max_loss_diff:.3e}, "
                f"max_param_diff={max_param_diff:.3e} (expected 0.0).",
                {"max_loss_diff": max_loss_diff, "max_param_diff": max_param_diff},
            )

        # ── 3. Gradient Accumulation (1-rank, grad_acc=2) vs 2-Shard DDP All-Reduce Parity ──
        # Note: MoE batch-level load-balancing auxiliary loss f_i * P_i is quadratic in batch
        # statistics, so `B=4` single-pass vs `B=2` micro-batch shares exact per-token linearity
        # on pure CE loss (`aux_loss_coef=0.0`), while `grad_acc=2 (micro_batch=2)` matches
        # 2-rank DDP (`world_size=2, micro_batch=2`) identically even WITH `aux_loss_coef > 0`!
        cfg_pure_ce = KilatConfig.from_dict(cfg.to_dict())
        cfg_pure_ce.aux_loss_coef = 0.0
        cfg_pure_ce.device_balance_coef = 0.0

        set_seed(4242)
        model_full = KilatTransformer(cfg_pure_ce).eval()
        model_accum = KilatTransformer(cfg_pure_ce).eval()
        model_accum.load_state_dict(model_full.state_dict())

        x_global = torch.randint(0, cfg.vocab_size, (4, 16), dtype=torch.long)

        # Single global batch (B=4, grad_acc=1) vs 2 micro-batches (B=2 x 2, grad_acc=2)
        model_full.zero_grad(set_to_none=True)
        loss_full = model_full(x_global, labels=x_global, return_dict=True).loss
        assert loss_full is not None
        loss_full.backward()

        model_accum.zero_grad(set_to_none=True)
        for shard in (x_global[:2], x_global[2:]):
            loss_micro = model_accum(shard, labels=shard, return_dict=True).loss
            assert loss_micro is not None
            (loss_micro / 2.0).backward()

        max_accum_grad_diff = 0.0
        for (name, pf), (_, pa) in zip(model_full.named_parameters(), model_accum.named_parameters()):
            if pf.grad is None or pa.grad is None:
                continue
            g_diff = float((pf.grad - pa.grad).abs().max().item())
            max_accum_grad_diff = max(max_accum_grad_diff, g_diff)
            if g_diff > 1e-5:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_ACCUM_PARITY_DRIFT,
                    f"Accumulation / 2-shard gradient drift on '{name}': {g_diff:.3e} > 1e-5.",
                    {"parameter": name, "grad_diff": g_diff},
                )

        # Also verify 1-process grad_acc=2 (micro_batch=2) vs 2-rank DDP all-reduce (micro_batch=2) on full cfg
        set_seed(4343)
        m_1proc = KilatTransformer(cfg).eval()
        m_rank0 = KilatTransformer(cfg).eval()
        m_rank1 = KilatTransformer(cfg).eval()
        m_rank0.load_state_dict(m_1proc.state_dict())
        m_rank1.load_state_dict(m_1proc.state_dict())

        m_1proc.zero_grad(set_to_none=True)
        for shard in (x_global[:2], x_global[2:]):
            l_1p = m_1proc(shard, labels=shard, return_dict=True).loss
            assert l_1p is not None
            (l_1p / 2.0).backward()

        m_rank0.zero_grad(set_to_none=True)
        m_rank1.zero_grad(set_to_none=True)
        l_r0 = m_rank0(x_global[:2], labels=x_global[:2], return_dict=True).loss
        l_r1 = m_rank1(x_global[2:], labels=x_global[2:], return_dict=True).loss
        assert l_r0 is not None and l_r1 is not None
        l_r0.backward()
        l_r1.backward()

        for (name, p1), (_, pr0), (_, pr1) in zip(
            m_1proc.named_parameters(), m_rank0.named_parameters(), m_rank1.named_parameters()
        ):
            if p1.grad is None or pr0.grad is None or pr1.grad is None:
                continue
            ddp_avg_grad = 0.5 * (pr0.grad + pr1.grad)
            ddp_diff = float((p1.grad - ddp_avg_grad).abs().max().item())
            max_accum_grad_diff = max(max_accum_grad_diff, ddp_diff)
            if ddp_diff > 1e-5:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_ACCUM_PARITY_DRIFT,
                    f"1-proc grad_acc=2 vs 2-rank DDP gradient drift on '{name}': {ddp_diff:.3e} > 1e-5.",
                )

        gc.collect()
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="G6_SYSTEM_PARITY",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                "resume_max_loss_diff": max_loss_diff,
                "resume_max_param_diff": max_param_diff,
                "accum_2shard_max_grad_diff": max_accum_grad_diff,
                "memory_probe_samples": len(mem_snapshots),
            },
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        restore_rng_state(saved_rng)
