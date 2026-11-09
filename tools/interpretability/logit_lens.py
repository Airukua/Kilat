"""
Kilat Logit Lens Inspector.

Inspects the residual stream across layers (0..L-1):
1. Captures hidden states after each Transformer block.
2. Applies final RMSNorm and unembedding lm_head.
3. Computes intermediate token probabilities, top-k predictions, and entropy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from kilat.configs.main_config import MainConfig
from kilat.models.transformer import KilatTransformer


def run_logit_lens(
    model_dir: Union[str, Path],
    prompt: str,
    top_k: int = 3,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute Logit Lens probing across all layers for a given prompt."""
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )

    ckpt_p = Path(model_dir)
    model = KilatTransformer.from_pretrained(str(ckpt_p)).to(device).eval()

    cfg_yaml = ckpt_p / "config.yaml"
    if cfg_yaml.is_file():
        cfg = MainConfig.from_yaml(cfg_yaml)
        tokenizer = cfg.tokenizer.build()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_p), local_files_only=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    layer_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        x = model.wte(input_ids)
        x = model.drop(x)

        for l_idx, layer in enumerate(model.layers):
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out

            normed = model.ln_f(x)
            logits = model.lm_head(normed)[0, -1, :]  # last token prediction
            probs = torch.softmax(logits, dim=-1)

            top_vals, top_indices = torch.topk(probs, top_k)

            # Compute Shannon entropy
            log_probs = torch.log(probs + 1e-12)
            entropy = float((-probs * log_probs).sum().item())

            top_items = []
            for val, idx in zip(top_vals, top_indices):
                tok_str = tokenizer.decode([idx.item()])
                top_items.append({
                    "token_id": int(idx.item()),
                    "token_str": tok_str,
                    "probability": round(float(val.item()) * 100, 2),
                })

            layer_results.append({
                "layer": l_idx,
                "entropy": round(entropy, 4),
                "top_predictions": top_items,
            })

    return {
        "model_checkpoint": str(ckpt_p),
        "prompt": prompt,
        "device": str(device),
        "layers_analyzed": len(layer_results),
        "layers": layer_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kilat Logit Lens Inspector")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt to probe")
    parser.add_argument("--top-k", type=int, default=3, help="Top predictions to extract per layer")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save report as JSON")
    args = parser.parse_args()

    res = run_logit_lens(
        model_dir=args.checkpoint,
        prompt=args.prompt,
        top_k=args.top_k,
        device_str=args.device,
    )

    print(f"\nPROMPT: {repr(res['prompt'])}\n" + "=" * 65)
    for l_data in res["layers"]:
        preds_str = " | ".join(
            [f"{repr(item['token_str'])}: {item['probability']}%" for item in l_data["top_predictions"]]
        )
        print(f"Layer {l_data['layer']} (H={l_data['entropy']:.2f}) -> {preds_str}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nReport written to: {out_p}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
