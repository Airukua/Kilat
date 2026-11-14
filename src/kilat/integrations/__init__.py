"""
Kilat Third-Party Framework Integrations.

Contains thin, modular adapters to external ML ecosystems:
- huggingface: PEFT LoRA, TRL SFT/DPO, AutoModel classes, Dataset ingestion.
- serving: OpenAI-compatible REST API server with real-time SSE token streaming.
"""

from __future__ import annotations

__all__ = ["huggingface", "serving"]
