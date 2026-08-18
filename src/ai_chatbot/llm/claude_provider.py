"""
Claude LLM provider implementation.

This wraps the Claude API (via anthropic SDK) to conform to the
BaseLLMProvider interface.
"""

import logging

from anthropic import Anthropic

from ai_chatbot.llm.base import BaseLLMProvider, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """
    Claude API provider implementation.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = Anthropic(api_key=config.api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] = None,
    ) -> LLMResponse:
        """
        Send a message to Claude.
        """
        logger.debug("Sending %d messages to Claude (%s)", len(messages), self._config.model)

        api_params = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

        if tools:
            api_params["tools"] = tools

        response = self._client.messages.create(**api_params)

        # Extract text content
        text_parts = [block.text for block in response.content if block.type == "text"]
        text = "".join(text_parts)

        # Extract usage info
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Extract metadata
        metadata = {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "has_tool_use": any(block.type == "tool_use" for block in response.content),
            "content_blocks": [
                {"type": block.type, "text": getattr(block, "text", None)}
                for block in response.content
            ],
        }

        return LLMResponse(
            text=text,
            model=self._config.model,
            usage=usage,
            metadata=metadata,
        )

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        try:
            # Simple check - just verify API key is set
            return bool(self._config.api_key)
        except Exception:
            return False

    def supports_tools(self) -> bool:
        """Claude supports tool use."""
        return True
