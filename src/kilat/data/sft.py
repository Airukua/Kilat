"""
Supervised Fine-Tuning (SFT) & Instruction Tuning Data Pipeline for Kilat.

Implements prompt-response formatting, selective prompt loss masking (ignore_index = -100),
conversational ChatML templating, and causal batch collation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from kilat.utils.validators import validate_positive_int


def format_instruction_pair(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (prompt, response) tuple from various common dataset schemas.

    Supported formats:
    1. ChatML / Messages:
       {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    2. Alpaca format:
       {"instruction": "...", "input": "...", "output": "..."}
    3. Direct prompt-response format:
       {"prompt": "...", "response": "..."} or {"input": "...", "output": "..."}
    """
    # 1. Multi-turn or single-turn messages
    if "messages" in example and isinstance(example["messages"], list):
        messages = example["messages"]
        if not messages:
            raise ValueError("Empty 'messages' list in example.")

        # Build prompt from all turns prior to the final assistant message
        prompt_parts: List[str] = []
        response_text = ""

        # Find the last assistant message
        last_asst_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst_idx = i
                break

        if last_asst_idx == -1:
            raise ValueError("No assistant response found in messages.")

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if i == last_asst_idx:
                prompt_parts.append(f"<|im_start|>{role}\n")
                response_text = content
                break
            else:
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        prompt_text = "".join(prompt_parts)
        return prompt_text, response_text

    # 2. Alpaca format
    if "instruction" in example:
        instruction = example["instruction"]
        inp = example.get("input", "").strip()
        out = example.get("output", "")

        if inp:
            prompt_text = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
            )
        else:
            prompt_text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
        return prompt_text, out

    # 3. Direct prompt/response or input/output
    if "prompt" in example and "response" in example:
        return str(example["prompt"]), str(example["response"])

    if "input" in example and "output" in example:
        return str(example["input"]), str(example["output"])

    if "text" in example:
        # Fallback for plain text, treat as response with empty prompt
        return "", str(example["text"])

    raise KeyError(
        f"Unrecognized example schema with keys: {list(example.keys())}. "
        "Expected ('messages'), ('instruction', 'output'), or ('prompt', 'response')."
    )


class SFTDataset(Dataset):
    """
    PyTorch Dataset for Supervised Fine-Tuning.

    Accepts:
    - Path to a JSON or JSONL file
    - In-memory list of dictionaries
    - Hugging Face Dataset instance
    """

    def __init__(
        self,
        data: Union[str, Path, Sequence[Dict[str, Any]], Any],
    ) -> None:
        self.examples: List[Dict[str, Any]] = []

        if isinstance(data, (str, Path)):
            p = Path(data).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"SFT data file not found: {p}")
            if p.suffix in {".jsonl", ".txt"}:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.examples.append(json.loads(line))
            elif p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                    if isinstance(parsed, list):
                        self.examples = parsed
                    elif isinstance(parsed, dict) and "data" in parsed:
                        self.examples = parsed["data"]
                    else:
                        raise ValueError(f"Unexpected JSON root structure in {p}")
            else:
                raise ValueError(f"Unsupported file format: {p.suffix}")
        elif hasattr(data, "__len__") and hasattr(data, "__getitem__"):
            # Works with Python list, Hugging Face Dataset, etc.
            self.examples = data  # type: ignore
        else:
            raise TypeError(f"Unsupported data source type: {type(data)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        prompt, response = format_instruction_pair(item)
        return {"prompt": prompt, "response": response}


class SFTDataCollator:
    """
    Data Collator for Supervised Fine-Tuning with strict prompt loss masking.

    Guarantees:
    1. Prompt tokens are strictly masked with `ignore_index = -100` in `labels`.
    2. Response tokens are preserved in `labels` for next-token loss computation.
    3. Padding tokens are right-padded with `pad_token_id` in `input_ids` and `-100` in `labels`.
    4. Exact tensor shape contract: `(batch_size, seq_len)`.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        ignore_index: int = -100,
        pad_token_id: Optional[int] = None,
        append_eos_token: bool = True,
    ) -> None:
        validate_positive_int("max_length", max_length)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.append_eos_token = append_eos_token

        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.eos_token_id)
        else:
            self.pad_token_id = 0

        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def _tokenize_pair(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        """
        Tokenize prompt and response separately to ensure exact prompt boundary identification.
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Tokenize response
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if self.append_eos_token and self.eos_token_id is not None:
            if not response_ids or response_ids[-1] != self.eos_token_id:
                response_ids.append(self.eos_token_id)

        # Truncation strategy: truncate prompt from the left if the combined sequence exceeds max_length
        total_len = len(prompt_ids) + len(response_ids)
        if total_len > self.max_length:
            overflow = total_len - self.max_length
            if len(prompt_ids) > overflow + 1:
                # Truncate prompt from left to preserve the instruction and response
                prompt_ids = prompt_ids[overflow:]
            else:
                # Prompt is shorter than overflow; truncate response from right
                excess = overflow - (len(prompt_ids) - 1)
                prompt_ids = prompt_ids[-1:] if prompt_ids else []
                response_ids = response_ids[:-excess] if excess < len(response_ids) else response_ids[:1]

        input_ids = prompt_ids + response_ids
        # Labels: -100 for all prompt tokens, actual token IDs for response tokens
        labels = [self.ignore_index] * len(prompt_ids) + list(response_ids)

        return input_ids, labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Collator received an empty batch.")

        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for item in batch:
            if "prompt" in item and "response" in item:
                prompt = item["prompt"]
                response = item["response"]
            else:
                prompt, response = format_instruction_pair(item)

            input_ids, labels = self._tokenize_pair(prompt, response)

            if len(input_ids) == 0:
                # Edge case guard: at least one token
                input_ids = [self.pad_token_id]
                labels = [self.ignore_index]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        # Determine batch max length for dynamic padding (bounded by self.max_length)
        max_batch_len = min(
            self.max_length,
            max(t.size(0) for t in batch_input_ids)
        )

        padded_inputs: List[torch.Tensor] = []
        padded_labels: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        for inp, lbl in zip(batch_input_ids, batch_labels):
            cur_len = inp.size(0)
            if cur_len < max_batch_len:
                pad_len = max_batch_len - cur_len
                # Right padding
                inp_padded = torch.cat([inp, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)], dim=0)
                lbl_padded = torch.cat([lbl, torch.full((pad_len,), self.ignore_index, dtype=torch.long)], dim=0)
                mask = torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
            else:
                inp_padded = inp[:max_batch_len]
                lbl_padded = lbl[:max_batch_len]
                mask = torch.ones(max_batch_len, dtype=torch.long)

            padded_inputs.append(inp_padded)
            padded_labels.append(lbl_padded)
            attention_masks.append(mask)

        return {
            "input_ids": torch.stack(padded_inputs, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
        }
