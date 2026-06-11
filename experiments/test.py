"""
Test generation with KilatTransformer using custom AutoTokenizer.

This script demonstrates:
1. Loading model from checkpoint with KilatTransformer.from_pretrained()
2. Loading tokenizer from checkpoint with custom AutoTokenizer
3. Testing forward pass
4. Testing text generation with various strategies
"""

import sys
from pathlib import Path
import torch
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from arc.model import KilatTransformer
from data.tokenizer import AutoTokenizer
from pipeline.generation.generator import TextGenerator


def main():
    model_path = "AiRukua/BabyKilat"
    
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    
    model = KilatTransformer.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Config: vocab_size={model.config.vocab_size}, n_embd={model.config.n_embd}")
    print(f"  n_layer={model.config.n_layer}, n_head={model.config.n_head}")
    
    print("\n" + "=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✓ Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    
    if hasattr(tokenizer, "eos_token"):
        print(f"  EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    if hasattr(tokenizer, "pad_token"):
        print(f"  PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    print("\n" + "=" * 60)
    print("Forward Pass Test")
    print("=" * 60)
    
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens (first 10): {input_ids[0][:10].tolist()}")
    
    with torch.no_grad():
        output = model(input_ids)
        print(f"\nOutput logits shape: {output.logits.shape}")
        print(f"Last token logits shape: {output.logits[:, -1, :].shape}")
    
    print("\n✓ Model forward pass successful!")
    
    print("\n" + "=" * 60)
    print("Greedy Generation (Deterministic)")
    print("=" * 60)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
        )
    
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated:\n{output_text}")
    
    print("\n" + "=" * 60)
    print("Sampling Generation (Stochastic)")
    print("=" * 60)
    
    test_prompts = [
        "In the beginning",
        "The future of artificial intelligence",
        "Hello, how are you?",
    ]
    
    for test_prompt in test_prompts:
        print(f"\nPrompt: {test_prompt}")
        print("-" * 40)
        
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {output_text[:200]}...")
    
    print("\n" + "=" * 60)
    print("Using TextGenerator Wrapper")
    print("=" * 60)
    
    generator = TextGenerator(model, tokenizer, device=device)
    
    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")
    print("-" * 40)
    
    text = generator.generate(prompt, max_new_tokens=30, do_sample=False)
    print(f"Greedy: {text[:150]}...")
    
    text = generator.generate(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )
    print(f"Sampling: {text[:150]}...")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()