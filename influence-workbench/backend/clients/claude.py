"""Claude API client wrapping anthropic.AsyncAnthropic."""

from __future__ import annotations

import anthropic


class ClaudeClient:
    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def generate_context(self, completion: str, instruction: str | None = None) -> str:
        """Generate a plausible prompt that naturally precedes *completion*."""
        system = (
            "You are a helpful assistant. Your task is to generate a plausible "
            "context or prompt that would naturally precede the given completion text. "
            "The context should read as if it came from a real document, article, or "
            "conversation. Output ONLY the context text, nothing else."
        )

        user_parts = []
        if instruction:
            user_parts.append(f"Instruction: {instruction}\n\n")
        user_parts.append(f"Completion text:\n{completion}")

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": "".join(user_parts)}],
        )
        return response.content[0].text
