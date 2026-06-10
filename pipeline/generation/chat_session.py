"""
Interactive chat session with conversation history management.

This module provides a simple REPL for chatting with the model.
It maintains a rolling conversation history, formats prompts using a
custom markup language, and trims the history when it exceeds a token budget.

Why a custom format? Instead of relying on a model‑specific chat template,
we define a simple, transparent format:
    <|system|> System message </s>
    <|user|> User message </s>
    <|assistant|> Assistant response </s>

This makes it easy to understand what the model sees and to debug.
For production, you would replace this with `tokenizer.apply_chat_template`
if the tokenizer supports it.
"""

from typing import Optional, List, Dict
from .generator import KilatGenerator


class ChatSession:
    """
    Manages a multi‑turn conversation with a KilatGenerator.
    
    Tracks conversation state, enforces token budget, and provides a clean
    interface for interactive chat. The session maintains a sliding window
    of exchanges, discarding oldest turns when the estimated token count
    exceeds max_history_tokens.
    
    Key design decision: Uses character-based token estimation instead of
    actual tokenization to avoid O(n²) cost per generation step. This is
    acceptable because the budget is a soft guideline, not a hard limit.
    """
    
    def __init__(
        self,
        generator: KilatGenerator,
        system_prompt: Optional[str] = None,
        max_history_tokens: int = 2048,
    ):
        self.generator = generator
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        # Stores messages in chronological order as {"role": str, "content": str}
        # Role must be "user" or "assistant" to match the markup format.
        self.conversation_history: List[Dict[str, str]] = []

    def start(self):
        """
        Run the interactive chat loop.
        
        Handles three special commands:
        - 'exit'/'quit': Terminate the session
        - 'clear': Reset conversation history (keeps system prompt if present)
        - Empty input: Ignored (no-op)
        
        Gracefully handles Ctrl+C to avoid traceback noise in the REPL.
        """
        print("\n" + "=" * 60)
        print("KilatTransformer Chat")
        print("=" * 60)
        print("Type 'exit' or 'quit' to stop, 'clear' to reset conversation.")
        print("=" * 60 + "\n")

        try:
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("\nGoodbye!")
                    break
                elif user_input.lower() == "clear":
                    self.conversation_history = []
                    print("\n[Conversation cleared]\n")
                    continue
                elif not user_input:
                    continue

                prompt = self._build_chat_prompt(user_input)
                print("Assistant: ", end="", flush=True)

                # Fixed generation config for chat.
                # Temperature 0.7 balances creativity vs. coherence for conversation.
                # Top-p 0.9 is a typical nucleus sampling setting.
                # Max 512 tokens prevents extremely long rambling responses.
                response = self.generator.generate(
                    prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=512,
                )
                
                # The model generates continuation after "<|assistant|>\n".
                # We slice the prompt prefix to show only the assistant's response.
                # Assumption: generate() returns the full prompt + new tokens.
                response_text = response[len(prompt):].strip()
                print(response_text)
                print()

                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                self._trim_history()
        except KeyboardInterrupt:
            # Silent exit on Ctrl+C - this is expected REPL behavior
            print("\n\nGoodbye!")

    def _build_chat_prompt(self, user_input: str) -> str:
        """
        Construct the full prompt string from conversation history.
        
        Format follows a simple XML-like markup with explicit termination tokens.
        The trailing "<|assistant|>\n" serves as a prefix for model continuation.
        
        Why this structure? The model has been fine-tuned to recognize this
        pattern and generate responses after seeing the assistant marker.
        """
        parts = []
        if self.system_prompt:
            parts.append(f"<|system|>\n{self.system_prompt}</s>")
        for msg in self.conversation_history:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}</s>")
        parts.append(f"<|user|>\n{user_input}</s>")
        parts.append("<|assistant|>\n")   # Model continues from this point
        return "\n".join(parts)

    def _trim_history(self):
        """
        Maintain token budget by discarding oldest conversation turns.
        
        Design rationale for token estimation:
        - Problem: Tokenizing the full history at every generation step
          would add O(n) tokenizer calls per turn, becoming expensive for
          long conversations.
        - Solution: Use character count as a proxy (≈4 chars/token for English).
        - Trade-off: Accuracy is ~20% off target, but that's fine for a
          soft memory limit. We only need to prevent unbounded growth.
        
        Edge cases handled:
        - Removes in pairs (user+assistant) to preserve conversation continuity
        - Single leftover message gets dropped entirely (shouldn't happen with
          proper pairing, but guards against corrupt state)
        - Empty history terminates the loop
        
        Assumption: max_history_tokens is a soft limit. Occasional overshoot
        is acceptable; we just trim aggressively when detected.
        """
        while self.conversation_history:
            total_chars = sum(len(m["content"]) for m in self.conversation_history)
            # 4 chars/token: Empirical average for English prose.
            # This under-counts for non-English languages but works as a heuristic.
            estimated_tokens = total_chars // 4
            
            if estimated_tokens <= self.max_history_tokens:
                break
                
            # Remove the oldest exchange (user+assistant pair)
            # This preserves the conversational structure and ensures
            # we don't leave orphaned messages that break the turn pattern.
            if len(self.conversation_history) >= 2:
                self.conversation_history = self.conversation_history[2:]
            else:
                # Fallback for corrupted state - shouldn't occur with proper usage
                self.conversation_history = []