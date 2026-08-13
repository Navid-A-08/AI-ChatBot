"""
Thin wrapper around the Anthropic SDK.

Why this exists:
    Nothing else in the codebase should import `anthropic` directly.
    Isolating the SDK call here means:
    - If the SDK's API shape changes, only this file changes.
    - It's mockable in tests without hitting the real API.
    - It's the one seam where a future multi-provider abstraction
      (Phase 7) would slot in — but we are NOT building that
      abstraction now. This class is Claude-specific on purpose.

How it works:
    Takes already-assembled messages (produced by context.py) and a
    system prompt string, sends them to the Messages API, returns the
    text response. It does not know about conversation history,
    windowing, or memory — that's context.py's job. This wrapper only
    knows how to talk to the API.
"""

import logging

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """A minimal wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def send(self, messages: list[dict], system_prompt: str) -> str:
        """
        Send a list of messages to Claude and return the text response.

        Args:
            messages: list of {"role": "user"|"assistant", "content": str},
                      already windowed/assembled by the context layer.
            system_prompt: the system prompt string (loaded via prompts.py).

        Returns:
            The assistant's text response.

        Raises:
            anthropic.APIError (and subclasses): propagated as-is rather
            than swallowed, so the caller (CLI) decides how to surface
            failures to the user. Hiding API errors here would make
            debugging much harder.
        """
        logger.debug("Sending %d messages to %s", len(messages), self._model)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=messages,
        )

        # response.content is a list of content blocks; for a plain text
        # reply (no tool use yet — that's Phase 6), it's a single text block.
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "".join(text_parts)
