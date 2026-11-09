"""
Kilat Activation Patching (Causal Intervention) Probe.

Based on Mechanistic Interpretability Benchmark (MIB, Mueller et al., ICML 2025)
and Toward Best Practices of Activation Patching (Zhang & Nanda, ICLR 2024).

Implements residual stream patching between Clean and Corrupted inputs to measure
Normalized Direct Effect (NDE) / Logit Diff Recovery across Transformer layers.
Includes negative control positions to rule out non-specific activation disruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from kilat.configs.main_config import MainConfig
from kilat.models.transformer import KilatTransformer


def get_token_id(tokenizer: Any, token_str: str) -> int:
    """Encode a token string and return single token ID."""
    if not token_str:
        raise ValueError("Could not encode token string: ''")
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if not ids:
        # Fallback with leading space if needed
        ids = tokenizer.encode(" " + token_str.lstrip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not encode token string: '{token_str}'")
    return ids[-1]


def forward_with_residual_cache(
    model: KilatTransformer,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run forward pass and return (final_logits, list of residual states after each layer).
    """
    residuals = []
    x = model.wte(input_ids)
    x = model.drop(x)
    residuals.append(x.clone())

    for layer in model.layers:
        out = layer(x)
        x = out[0] if isinstance(out, tuple) else out
        residuals.append(x.clone())

    normed = model.ln_f(x)
    logits = model.lm_head(normed)
    return logits, residuals


def forward_with_patch(
    model: KilatTransformer,
    input_ids: torch.Tensor,
    patch_layer: int,
    patch_tensor: torch.Tensor,
    patch_pos: int,
) -> torch.Tensor:
    """
    Run forward pass on input_ids, replacing activation at patch_layer and patch_pos
    with patch_tensor, then continuing through the rest of the network.
    """
    x = model.wte(input_ids)
    x = model.drop(x)

    # Patch at embedding level if layer == -1
    if patch_layer == -1:
        x = x.clone()
        x[:, patch_pos, :] = patch_tensor[:, patch_pos, :]

    for l_idx, layer in enumerate(model.layers):
        out = layer(x)
        x = out[0] if isinstance(out, tuple) else out

        if l_idx == patch_layer:
            x = x.clone()
            x[:, patch_pos, :] = patch_tensor[:, patch_pos, :]

    normed = model.ln_f(x)
    logits = model.lm_head(normed)
    return logits


def run_activation_patching(
    model_dir: Union[str, Path],
    clean_prompt: str,
    corrupt_prompt: str,
    target_token: str,
    foil_token: str,
    patch_pos: int = -1,
    control_pos: Optional[int] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Execute causal activation patching across all Transformer layers.
    """
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )

    ckpt_p = Path(model_dir)
    model = KilatTransformer.from_pretrained(str(ckpt_p)).to(device).eval()

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_p), local_files_only=True)
    except Exception:
        pass

    if tokenizer is None:
        cfg_yaml = ckpt_p / "config.yaml"
        if cfg_yaml.is_file():
            try:
                cfg = MainConfig.from_yaml(cfg_yaml)
                if hasattr(cfg, "tokenizer") and cfg.tokenizer is not None:
                    tokenizer = cfg.tokenizer.build()
            except Exception:
                pass

    if tokenizer is None:
        raise ValueError(f"Could not load tokenizer from {ckpt_p}")

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(device)

    clean_ids = clean_inputs["input_ids"]
    corrupt_ids = corrupt_inputs["input_ids"]

    # Verify matching sequence lengths for clean & corrupted
    if clean_ids.shape[1] != corrupt_ids.shape[1]:
        raise ValueError(
            f"Prompt lengths must match for precise activation patching. "
            f"Clean: {clean_ids.shape[1]}, Corrupt: {corrupt_ids.shape[1]}"
        )

    target_id = get_token_id(tokenizer, target_token)
    foil_id = get_token_id(tokenizer, foil_token)

    # 1. Baseline Clean Run
    with torch.no_grad():
        clean_logits, clean_residuals = forward_with_residual_cache(model, clean_ids)
        clean_target_logit = clean_logits[0, -1, target_id].item()
        clean_foil_logit = clean_logits[0, -1, foil_id].item()
        ld_clean = clean_target_logit - clean_foil_logit

        # 2. Baseline Corrupt Run
        corrupt_logits, corrupt_residuals = forward_with_residual_cache(model, corrupt_ids)
        corrupt_target_logit = corrupt_logits[0, -1, target_id].item()
        corrupt_foil_logit = corrupt_logits[0, -1, foil_id].item()
        ld_corrupt = corrupt_target_logit - corrupt_foil_logit

    total_diff = ld_clean - ld_corrupt
    if abs(total_diff) < 1e-6:
        total_diff = 1e-6

    # 3. Layer-by-layer Intervention (Patch Clean activation into Corrupt run)
    layer_effects: List[Dict[str, Any]] = []

    num_layers = len(model.layers)
    with torch.no_grad():
        for l_idx in range(num_layers):
            # clean_residuals[0] is embedding, clean_residuals[l_idx+1] is after layer l_idx
            clean_act = clean_residuals[l_idx + 1]

            patched_logits = forward_with_patch(
                model=model,
                input_ids=corrupt_ids,
                patch_layer=l_idx,
                patch_tensor=clean_act,
                patch_pos=patch_pos,
            )

            patched_target = patched_logits[0, -1, target_id].item()
            patched_foil = patched_logits[0, -1, foil_id].item()
            ld_patched = patched_target - patched_foil

            recovery_pct = ((ld_patched - ld_corrupt) / total_diff) * 100.0

            # Control patch at irrelevant position if requested
            ctrl_recovery_pct = None
            if control_pos is not None:
                ctrl_logits = forward_with_patch(
                    model=model,
                    input_ids=corrupt_ids,
                    patch_layer=l_idx,
                    patch_tensor=clean_act,
                    patch_pos=control_pos,
                )
                ctrl_target = ctrl_logits[0, -1, target_id].item()
                ctrl_foil = ctrl_logits[0, -1, foil_id].item()
                ctrl_ld = ctrl_target - ctrl_foil
                ctrl_recovery_pct = round(((ctrl_ld - ld_corrupt) / total_diff) * 100.0, 2)

            layer_effects.append({
                "layer": l_idx,
                "target_logit": round(patched_target, 4),
                "foil_logit": round(patched_foil, 4),
                "logit_diff": round(ld_patched, 4),
                "recovery_pct": round(recovery_pct, 2),
                "control_recovery_pct": ctrl_recovery_pct,
            })

    return {
        "model_checkpoint": str(ckpt_p),
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "target_token": {"token": target_token, "id": target_id},
        "foil_token": {"token": foil_token, "id": foil_id},
        "patch_position": patch_pos,
        "control_position": control_pos,
        "clean_logit_diff": round(ld_clean, 4),
        "corrupt_logit_diff": round(ld_corrupt, 4),
        "total_logit_diff_span": round(total_diff, 4),
        "layer_effects": layer_effects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kilat Causal Activation Patching Probe")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--clean-prompt", type=str, required=True, help="Clean prompt")
    parser.add_argument("--corrupt-prompt", type=str, required=True, help="Corrupted prompt (same length)")
    parser.add_argument("--target", type=str, required=True, help="Clean target token (e.g. ' blue')")
    parser.add_argument("--foil", type=str, required=True, help="Foil counterfactual token (e.g. ' red')")
    parser.add_argument("--patch-pos", type=int, default=-1, help="Token position to patch (default: -1)")
    parser.add_argument("--control-pos", type=int, default=None, help="Negative control token position")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save report as JSON")
    args = parser.parse_args()

    res = run_activation_patching(
        model_dir=args.checkpoint,
        clean_prompt=args.clean_prompt,
        corrupt_prompt=args.corrupt_prompt,
        target_token=args.target,
        foil_token=args.foil,
        patch_pos=args.patch_pos,
        control_pos=args.control_pos,
        device_str=args.device,
    )

    print("=" * 70)
    print("⚡ KILAT CAUSAL ACTIVATION PATCHING REPORT")
    print("=" * 70)
    print(f"Clean Prompt   : {res['clean_prompt']}")
    print(f"Corrupt Prompt : {res['corrupt_prompt']}")
    print(f"Target / Foil  : {repr(res['target_token']['token'])} (id: {res['target_token']['id']}) vs {repr(res['foil_token']['token'])} (id: {res['foil_token']['id']})")
    print(f"Clean Logit Diff   : {res['clean_logit_diff']:+.4f}")
    print(f"Corrupt Logit Diff : {res['corrupt_logit_diff']:+.4f}")
    print(f"Patch Position     : {res['patch_position']} | Control Position: {res['control_position']}")
    print("-" * 70)
    print(f"{'Layer':<7} | {'Logit Diff':<12} | {'Recovery %':<12} | {'Control %':<12} | {'Causal Status'}")
    print("-" * 70)

    for item in res["layer_effects"]:
        rec = item["recovery_pct"]
        ctrl = f"{item['control_recovery_pct']:+.1f}%" if item["control_recovery_pct"] is not None else "N/A"
        if rec > 75.0:
            status = "★ CRITICAL (Primary Driver)"
        elif rec > 40.0:
            status = "◆ ACTIVE (Emergent Circuit)"
        elif rec > 10.0:
            status = "○ WEAK (Sub-threshold)"
        else:
            status = "· INERT (No causal effect)"

        print(f"Layer {item['layer']:<2} | {item['logit_diff']:+10.4f} | {rec:+10.2f}% | {ctrl:<12} | {status}")
    print("=" * 70)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Report written to {out_p}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
