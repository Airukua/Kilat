"""
Command‑line interface for KilatTransformer inference.

Supports three modes:
- generate: single prompt completion
- chat: interactive conversation  
- batch: process multiple prompts from a file

All generation parameters can be set via command‑line arguments.

Why three modes? They represent the three most common LLM deployment patterns:
- generate: one-off inference (scriptable, reproducible)
- chat: human-in-the-loop exploration (debugging, demos)
- batch: offline processing (evaluation, data generation)
"""

import argparse
import json
import time
from pathlib import Path
from .generation_config import GenerationConfig
from .generator import KilatGenerator
from .chat_session import ChatSession
from .model_loader import load_model_and_tokenizer
import torch


def main():
    """
    Entry point for the CLI with subcommand-style interface via --mode.
    
    Design decision: Single parser with mode branching rather than true
    subparsers. This simplifies argument sharing across modes (all generation
    params apply to all modes) while keeping the code linear. The trade-off is
    that --prompt is required only for generate mode but appears in help for
    all modes - acceptable for a focused inference tool.
    """
    parser = argparse.ArgumentParser(
        description="KilatTransformer Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m inference.cli --checkpoint ./checkpoints/kilat-base --mode chat
  python -m inference.cli --checkpoint ./checkpoints/kilat-base --mode generate --prompt "Once upon a time" --temperature 0.8
  python -m inference.cli --checkpoint ./checkpoints/kilat-base --mode batch --input_file prompts.txt --output_file out.json
        """,
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint directory (must contain model.safetensors and config.json)")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["generate", "chat", "batch"], default="generate")

    # Generation parameters - shared across all modes for consistency
    # Defaults match GenerationConfig defaults to maintain predictable behavior
    parser.add_argument("--prompt", type=str, default=None, 
                       help="Input prompt (required for generate mode, ignored otherwise)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)      # 0 = disabled
    parser.add_argument("--top_p", type=float, default=1.0)  # 1.0 = disabled
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # Chat-specific: system prompt persists across conversation turns
    parser.add_argument("--system_prompt", type=str, default=None, 
                       help="System prompt for chat mode (sets model behavior globally)")

    # Batch mode: file-based processing for offline workloads
    parser.add_argument("--input_file", type=str, default=None, 
                       help="File with prompts (one per line, empty lines skipped)")
    parser.add_argument("--output_file", type=str, default="completions.json", 
                       help="Output JSON file (contains prompt+completion pairs)")

    # Model loading options
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], 
                       help="Device to run on (auto-detected if not specified)")
    parser.add_argument("--use_yaml_config", action="store_true", 
                       help="Load config from config.yaml instead of model config.json")

    args = parser.parse_args()

    # Device selection: CLI argument takes precedence over auto-detection
    # Passing None to load_model_and_tokenizer triggers internal auto-detection
    device = torch.device(args.device) if args.device else None

    # Load model - this may take 5-30 seconds depending on model size and hardware
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, device=device, use_yaml_config=args.use_yaml_config
    )

    # Generator wraps the model with KV-cache management and sampling logic
    generator = KilatGenerator(model, tokenizer, device=device)

    # Convert CLI args to GenerationConfig dataclass for type safety
    # All parameters have defaults, so partial specification is fine
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    # --------------------------------------------------------------------
    # Mode dispatch
    # --------------------------------------------------------------------
    
    if args.mode == "chat":
        # Interactive REPL - runs until user exits (Ctrl+C or 'exit')
        # System prompt is optional but recommended for consistent behavior
        session = ChatSession(generator, system_prompt=args.system_prompt)
        session.start()

    elif args.mode == "generate":
        # Single generation - most common inference pattern
        if not args.prompt:
            raise ValueError("--prompt is required for generate mode")
        
        print("Generating...")
        start_time = time.time()
        output = generator.generate(args.prompt, gen_config)
        elapsed = time.time() - start_time

        # Format output for readability with clear visual separation
        print("\n" + "=" * 60)
        print("Generated Output")
        print("=" * 60)
        print(output)
        print("=" * 60)

        # Compute tokens/sec for performance monitoring
        # Note: This double-encodes (once for prompt, once for output) which is
        # inefficient but acceptable for CLI reporting (not hot path)
        new_tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(args.prompt))
        if elapsed > 0:
            print(f"\nGenerated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")

    elif args.mode == "batch":
        # Bulk processing - no interactive output, just results to JSON
        if not args.input_file:
            raise ValueError("--input_file is required for batch mode")
        
        # Read prompts, stripping whitespace and skipping empty lines
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(prompts)} prompts...")
        start_time = time.time()
        results = []
        
        # Sequential processing (no batching) - simpler and memory-predictable
        # A future optimization could implement true batching for speed,
        # but that would require padding and mask management.
        for i, prompt in enumerate(prompts, 1):
            completion = generator.generate(prompt, gen_config)
            results.append({"id": i, "prompt": prompt, "completion": completion})
            
            # Progress reporting every 10 prompts to avoid overwhelming output
            if i % 10 == 0:
                print(f"  Processed {i}/{len(prompts)}")

        elapsed = time.time() - start_time
        
        # Write results as structured JSON for downstream processing
        # ensure_ascii=False preserves non-English characters
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nCompleted {len(prompts)} prompts in {elapsed:.2f}s")
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()