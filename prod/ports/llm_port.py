"""
ports/llm_port.py
──────────────────────────────────────────────────────────────────────────────
Abstract interface for LLM (large language model) providers.

Current implementation: GeminiLLMAdapter (Vertex AI Gemini)
To swap to OpenAI GPT: write OpenAILLMAdapter implementing this Protocol,
then change ONE line in services/container.py.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Contract for a JSON-generating LLM provider."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying LLM."""
        ...

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str | None:
        """Send a prompt to the LLM and return its JSON response as a string.

        The caller is responsible for parsing the returned string.  The
        adapter should request JSON-mode output from the underlying model
        so the response is guaranteed to be valid JSON.

        Args:
            system_prompt: System-level instruction.
            user_message:  User-turn content.

        Returns:
            Raw JSON string, or None if the call failed.

        Raises:
            LLMError: On unrecoverable API failure.
        """
        ...
