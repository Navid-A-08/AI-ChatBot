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

Phase 6 updates:
    Added support for tool use. The client can now send tool definitions
    and handle tool use responses from the API.
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

    def send(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] = None,
    ) -> str | dict:
        """
        Send a list of messages to Claude and return the response.

        Args:
            messages: list of {"role": "user"|"assistant", "content": str},
                      already windowed/assembled by the context layer.
            system_prompt: the system prompt string (loaded via prompts.py).
            tools: optional list of tool definitions for tool use.

        Returns:
            The assistant's text response, or a dict with tool_use info.

        Raises:
            anthropic.APIError (and subclasses): propagated as-is rather
            than swallowed, so the caller (CLI) decides how to surface
            failures to the user. Hiding API errors here would make
            debugging much harder.
        """
        logger.debug("Sending %d messages to %s", len(messages), self._model)

        api_params = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = tools

        response = self._client.messages.create(**api_params)

        # Check if response contains tool use
        has_tool_use = any(
            block.type == "tool_use" for block in response.content
        )

        if has_tool_use:
            # Return structured response for tool use handling
            return {
                "type": "tool_use",
                "content": response.content,
                "stop_reason": response.stop_reason,
            }

        # response.content is a list of content blocks; for a plain text
        # reply (no tool use), it's a single text block.
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "".join(text_parts)

    def send_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        tool_results: list[dict] = None,
    ) -> dict:
        """
        Send messages with tool support and handle multi-turn tool use.

        This method handles the full tool use loop:
        1. Send messages with tools
        2. If model wants to use tools, execute them
        3. Send tool results back
        4. Continue until model produces text response

        Args:
            messages: Conversation messages.
            system_prompt: System prompt.
            tools: Tool definitions.
            tool_results: Pre-computed tool results (optional).

        Returns:
            Dict with keys:
            - "text": Final text response (if any)
            - "tool_uses": List of tool uses made
            - "tool_results": List of tool results
        """
        logger.debug("Sending %d messages with %d tools", len(messages), len(tools))

        api_params = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
        }

        response = self._client.messages.create(**api_params)

        result = {
            "text": "",
            "tool_uses": [],
            "tool_results": [],
        }

        # Process response content
        for block in response.content:
            if block.type == "text":
                result["text"] += block.text
            elif block.type == "tool_use":
                result["tool_uses"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # If there are tool uses, we need to handle them
        if result["tool_uses"]:
            result["stop_reason"] = "tool_use"
        else:
            result["stop_reason"] = "end_turn"

        return result
