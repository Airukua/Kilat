"""Unified report generator for Kilat Model Certification Dossier (Markdown + JSON)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt(val: Any, nd: int = 4) -> str:
    if isinstance(val, float):
        return f"{val:.{nd}f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def render_scorecard_md(dossier: Dict[str, Any]) -> str:
    lines = ["## ⚡ Certification Scorecard\n"]
    lines.append("| Pillar | Focus Area | Status |")
    lines.append("|---|---|---|")

    status_map = {
        "p1": ("1. Mechanical Correctness", dossier.get("p1", {}).get("passed")),
        "p2": ("2. Mechanistic Capability", dossier.get("p2", {}).get("passed")),
        "p3": ("3. Empirical Scaling Laws", dossier.get("p3", {}).get("passed")),
        "p4": ("4. Hardware Efficiency", dossier.get("p4", {}).get("passed")),
        "p5": ("5. Mechanistic Diagnostics", dossier.get("p5", {}).get("passed")),
    }

    for p_key, (name, passed) in status_map.items():
        if p_key in dossier:
            badge = "✅ PASSED" if passed else "❌ FAILED"
            lines.append(f"| {name} | Primary Invariants & Gates | {badge} |")

    lines.append("")
    return "\n".join(lines)


def render_p1_md(p1: Dict[str, Any]) -> str:
    lines = ["### Pillar 1: Mechanical & Causal Correctness\n"]
    c = p1["causal_future_blindness"]
    lines.append(f"- **Causal Mask Perturbation**: Max future leak diff = `{_fmt(c['max_past_leak_diff'], 6)}` "
                 f"(tol: `{c['tolerance']}`) -> {'✅ PASS' if c['passed'] else '❌ FAIL'}")

    g = p1["gradient_connectivity"]
    lines.append(f"- **Gradient Connectivity**: Active tensors with grad = `{g['n_active_param_tensors']}` "
                 f"(NaN: `{g['n_nan']}`, Inf: `{g['n_inf']}`, Dead: `{g['n_dead']}`) -> "
                 f"{'✅ PASS' if g['passed'] else '❌ FAIL'}")

    e = p1["initial_entropy"]
    lines.append(f"- **Initial Uniform Entropy**: Loss = `{_fmt(e['l0'])}` vs ideal ln(V) = `{_fmt(e['ideal_ln_v'])}` "
                 f"-> {'✅ PASS' if e['passed'] else '❌ FAIL'}")

    ov = p1["single_batch_overfit"]
    summary = ov["summary"].get("final_loss", {})
    if summary:
        lines.append(f"- **Single-Batch Overfit**: Final Loss = `{_fmt(summary['mean'])} ± {_fmt(summary['std'])}` "
                     f"(n={summary['n']} seeds) -> {'✅ PASS' if ov['passed'] else '❌ FAIL'}")
    lines.append("")
    return "\n".join(lines)


def render_p2_md(p2: Dict[str, Any]) -> str:
    lines = [f"### Pillar 2: Mechanistic Capability — Task: `{p2.get('task', 'mqar')}`\n"]
    lengths = p2.get("context_lengths", [])
    grid = p2.get("grid", {})

    header = "| Context Length (N) | Exact-Match Accuracy | Cross-Entropy Loss |"
    lines.append(header)
    lines.append("|---|---|---|")
    for sl in lengths:
        cell = grid.get(str(sl), {})
        acc_s = cell.get("accuracy", {})
        loss_s = cell.get("loss", {})
        acc_str = f"{_fmt(acc_s.get('mean', 0.0) * 100, 1)}% ± {_fmt(acc_s.get('std', 0.0) * 100, 1)}%" if acc_s else "-"
        loss_str = f"{_fmt(loss_s.get('mean', 0.0), 3)}" if loss_s else "-"
        lines.append(f"| N = {sl} | {acc_str} | {loss_str} |")
    lines.append("")
    return "\n".join(lines)


def render_p3_md(p3: Dict[str, Any]) -> str:
    lines = ["### Pillar 3: Empirical Scaling Laws & Quality\n"]
    lines.append("| Seq Len | Token Budget | Tokens/Sec | Final Loss EMA | Val Perplexity | Scaling Alpha |")
    lines.append("|---|---|---|---|---|---|")
    for r in p3.get("runs", []):
        t = r["train"]
        e = r["eval"]
        lines.append(
            f"| {r['seq_len']} | {_fmt(r['token_budget'])} | {_fmt(t['tokens_per_sec'], 1)} | "
            f"{_fmt(t['final_loss_ema'])} | {_fmt(e['perplexity'], 2)} | {_fmt(r['scaling_alpha'], 3)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_p4_md(p4: Dict[str, Any]) -> str:
    lines = ["### Pillar 4: Hardware Efficiency & Sub-Quadratic Scaling\n"]
    lines.append("| Seq Len | Prefill Tokens/Sec | Peak VRAM (MB) |")
    lines.append("|---|---|---|")
    for entry in p4.get("seq_len_sweep", []):
        vram = entry["vram"].get("peak_vram_mb")
        vram_str = f"{_fmt(vram, 1)} MB" if vram is not None else "CPU (N/A)"
        tps = entry["prefill"]["prefill_tokens_per_sec"]
        lines.append(f"| {entry['seq_len']} | {_fmt(tps, 1)} tok/s | {vram_str} |")

    d = p4.get("decode", {})
    if d:
        lines.append(f"\n- **Step-by-Step Decode**: `{_fmt(d.get('decode_ms_per_token', 0.0), 2)} ms/token` "
                     f"(`{_fmt(d.get('decode_tokens_per_sec', 0.0), 1)} tok/s` @ ctx={d.get('context_len')})\n")

    if "baseline_comparison" in p4:
        lines.append("#### Comparative Speedup vs Baseline Transformer\n")
        lines.append("| Seq Len | Model Prefill tok/s | Baseline Prefill tok/s | Empirical Speedup |")
        lines.append("|---|---|---|---|")
        for comp in p4["baseline_comparison"].get("sweep", []):
            lines.append(
                f"| {comp['seq_len']} | {_fmt(comp['model_prefill_tps'], 1)} | "
                f"{_fmt(comp['baseline_prefill_tps'], 1)} | **{_fmt(comp['speedup'], 2)}x** |"
            )
        lines.append("")
    return "\n".join(lines)


def render_p5_md(p5: Dict[str, Any]) -> str:
    lines = ["### Pillar 5: Mechanistic Diagnostics & Forensic Health\n"]
    pos = p5.get("position_loss", {})
    if pos:
        lines.append(f"- **Position Loss Context Delta**: `{_fmt(pos.get('early_vs_late_delta', 0.0), 4)}` "
                     f"-> {'✅ PASS' if pos.get('passed') else '❌ FAIL'}")
        lines.append("  Context Window Breakdown: " + ", ".join(f"`{k}: {_fmt(v, 3)}`" for k, v in pos.get("bucket_losses", {}).items()))

    ent = p5.get("layer_entropy", {})
    if ent and "layer_entropies" in ent:
        ent_str = " -> ".join(f"`L{item['layer']}: {_fmt(item['entropy'], 2)}`" for item in ent["layer_entropies"])
        lines.append(f"- **Layer Shannon Entropy Progression**: {ent_str}")

    lines.append("")
    return "\n".join(lines)


def write_certification_dossier(
    output_dir: str,
    model_name: str,
    p1: Optional[Dict[str, Any]] = None,
    p2: Optional[Dict[str, Any]] = None,
    p3: Optional[Dict[str, Any]] = None,
    p4: Optional[Dict[str, Any]] = None,
    p5: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate markdown and raw JSON certification dossiers."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# ⚡ Kilat Model Architecture Certification Dossier — `{model_name}`\n",
        f"_Certified At: {timestamp}_\n",
    ]

    dossier: Dict[str, Any] = {}
    if p1:
        dossier["p1"] = p1
    if p2:
        dossier["p2"] = p2
    if p3:
        dossier["p3"] = p3
    if p4:
        dossier["p4"] = p4
    if p5:
        dossier["p5"] = p5

    lines.append(render_scorecard_md(dossier))
    if p1:
        lines.append(render_p1_md(p1))
    if p2:
        lines.append(render_p2_md(p2))
    if p3:
        lines.append(render_p3_md(p3))
    if p4:
        lines.append(render_p4_md(p4))
    if p5:
        lines.append(render_p5_md(p5))

    md_path = out_p / f"{model_name}_certification_dossier.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_p / f"{model_name}_certification_raw.json"
    json_path.write_text(json.dumps(dossier, indent=2, default=str), encoding="utf-8")

    return md_path
