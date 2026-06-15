import torch
from kilat import KilatTransformer
from kilat.data import AutoTokenizer
from kilat import TextGenerator


def load_model(model_path: str, device: str):
    """
    Load a KilatTransformer model from a pretrained checkpoint and move it to the specified device.

    Why this function:
        - Centralises model loading logic to avoid repetition.
        - Displays diagnostic information about the loaded model (config, parameter count).
        - Sets the model to evaluation mode (disables dropout, etc.) for deterministic inference.

    Args:
        model_path: Path or HuggingFace-style identifier for the pretrained model.
        device: Target device ('cuda', 'cpu', 'mps').

    Returns:
        Loaded model instance (KilatTransformer) in evaluation mode, already moved to device.

    Side Effects:
        - Prints detailed model information to stdout.
        - May download model files if model_path is a remote identifier.

    Edge Cases & Risks:
        - If the model is not found or incompatible, the underlying `from_pretrained` will raise
          an exception (e.g., OSError, KeyError). This function does not catch them; the caller
          should handle.
        - Moving to device may fail if CUDA is not available and device='cuda' is requested.
        - The printed config values (vocab_size, n_embd, etc.) assume the model config has
          those attributes; if not, it will raise AttributeError.

    Performance:
        - Loading may take several seconds depending on model size and disk I/O.
        - Model weights are loaded into CPU memory first, then transferred to device.
    """
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    print(f"Model path: {model_path}")

    model = KilatTransformer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded on {device}")
    print(f"  Config: vocab_size={model.config.vocab_size}, n_embd={model.config.n_embd}")
    print(f"  n_layer={model.config.n_layer}, n_head={model.config.n_head}")
    return model


def load_tokenizer(model_path: str):
    """
    Load the tokenizer associated with a KilatTransformer model.

    Why this function:
        - Decouples tokenizer loading from model loading for clarity.
        - Provides informative output about tokenizer properties (vocab size, special tokens).

    Args:
        model_path: Path or identifier pointing to the tokenizer files (usually same as model).

    Returns:
        Tokenizer instance (subclass of `PreTrainedTokenizer`).

    Side Effects:
        - Prints tokenizer info (vocab size, type, EOS/PAD tokens if present).

    Edge Cases:
        - If the tokenizer does not have `eos_token` or `pad_token` attributes, those lines
          are skipped without error (hasattr check).
        - May raise OSError if tokenizer files are missing.
    """
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
    return tokenizer


def test_forward_pass(model, tokenizer, device, prompt: str):
    """
    Perform a single forward pass to verify the model runs without errors.

    Why this function:
        - Quick smoke test to ensure model and tokenizer are compatible and device is working.
        - Shows input shape and output logits shape for debugging.

    Args:
        model: Loaded KilatTransformer model.
        tokenizer: Corresponding tokenizer.
        device: Current device (e.g., 'cuda' or 'cpu').
        prompt: Input string for the forward pass.

    Returns:
        Token IDs (torch.Tensor) of the encoded prompt, moved to the device.

    Side Effects:
        - Prints input and output shape information.
        - No model parameters are updated (torch.no_grad()).

    Important:
        - This function does not perform generation; it only runs the model once.
        - Useful for measuring baseline latency or checking for shape mismatches.
    """
    print("\n" + "=" * 60)
    print("Forward Pass Test")
    print("=" * 60)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {prompt}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens (first 10): {input_ids[0][:10].tolist()}")

    with torch.no_grad():
        output = model(input_ids)
        print(f"\nOutput logits shape: {output.logits.shape}")
        print(f"Last token logits shape: {output.logits[:, -1, :].shape}")

    print("\n✓ Model forward pass successful!")
    return input_ids


def test_greedy_generation(model, tokenizer, device, input_ids, prompt: str):
    """
    Generate text using greedy decoding (deterministic, picks highest probability token each step).

    Why greedy:
        - Simple and reproducible; useful for testing and baselines.
        - No sampling parameters needed.

    Args:
        model: KilatTransformer model.
        tokenizer: Tokenizer for decoding.
        device: Computation device.
        input_ids: Tokenized prompt (as returned by test_forward_pass).
        prompt: Original prompt string (for display).

    Side Effects:
        - Prints the generated continuation.

    Important:
        - Greedy decoding often leads to repetitive or boring outputs but is fast.
        - `do_sample=False` disables randomness.
    """
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


def test_sampling_generation(model, tokenizer, device, test_prompts):
    """
    Generate text using sampling-based decoding (stochastic, with temperature, top-k, top-p).

    Why sampling:
        - Produces more diverse and creative outputs.
        - Allows control over randomness via temperature and filtering via top-k/top-p.

    Args:
        model: KilatTransformer model.
        tokenizer: Tokenizer for decoding.
        device: Computation device.
        test_prompts: List of prompt strings to generate from.

    Side Effects:
        - Prints each prompt and the generated continuation (truncated to 200 chars).

    Important:
        - Different runs may produce different outputs even with the same parameters.
        - Parameters used: temperature=0.8 (moderate randomness), top_k=50, top_p=0.95,
          repetition_penalty=1.1 (discourages repeating tokens).
    """
    print("\n" + "=" * 60)
    print("Sampling Generation (Stochastic)")
    print("=" * 60)

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


def test_text_generator_wrapper(model, tokenizer, device, prompt: str):
    """
    Test the convenience wrapper TextGenerator that simplifies the generation API.

    Why this function:
        - Demonstrates the higher-level API that hides low-level token handling.
        - Shows both greedy and sampling generation using the same wrapper.

    Args:
        model: KilatTransformer model.
        tokenizer: Tokenizer.
        device: Computation device.
        prompt: Input prompt.

    Side Effects:
        - Prints the generated text for both greedy and sampling configurations.

    Important:
        - TextGenerator is a wrapper that handles tokenization, generation, and decoding
          in one call. It may not expose all advanced parameters, but is simpler for
          common use cases.
    """
    print("\n" + "=" * 60)
    print("Using TextGenerator Wrapper")
    print("=" * 60)

    generator = TextGenerator(model, tokenizer, device=device)

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


def main():
    """
    Main entry point for the KilatTransformer test script.

    Workflow:
        1. Automatically select the best available device (CUDA > CPU).
        2. Load model and tokenizer from the specified pretrained checkpoint.
        3. Run a forward pass to validate.
        4. Test greedy generation.
        5. Test sampling generation on multiple prompts.
        6. Test the TextGenerator convenience wrapper.
        7. Print a summary of model statistics.

    Expected output:
        - Detailed logs of each step.
        - Final summary with model path, device, vocab size, and parameter count.

    Edge Cases:
        - If CUDA is not available, falls back to CPU (gracefully).
        - The model path 'AiRukua/BabyKilat' is a HuggingFace identifier; requires internet
          access for first run (or cached locally).
    """
    model_path = "AiRukua/BabyKilat"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, device)
    tokenizer = load_tokenizer(model_path)

    # Forward pass test
    prompt_greedy = "Once upon a time"
    input_ids = test_forward_pass(model, tokenizer, device, prompt_greedy)

    # Generation tests
    test_greedy_generation(model, tokenizer, device, input_ids, prompt_greedy)

    test_prompts = [
        "In the beginning",
        "The future of artificial intelligence",
        "Hello, how are you?",
    ]
    test_sampling_generation(model, tokenizer, device, test_prompts)

    test_text_generator_wrapper(model, tokenizer, device, prompt_greedy)

    # Final summary
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