"""
OpenAI LLM provider implementation.

This wraps the OpenAI API to conform to the BaseLLMProvider interface.
Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
"""

import logging

from ai_chatbot.llm.base import BaseLLMProvider, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.

    Requires the openai package to be installed:
    pip install openai
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._config.api_key)
        except ImportError:
            logger.warning(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        except Exception:
            logger.exception("Failed to initialize OpenAI client")

    @property
    def provider_name(self) -> str:
        return "openai"

    def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] = None,
    ) -> LLMResponse:
        """
        Send a message to OpenAI.
        """
        if not self._client:
            raise RuntimeError(
                "OpenAI client not initialized. "
                "Install openai package: pip install openai"
            )

        logger.debug("Sending %d messages to OpenAI (%s)", len(messages), self._config.model)

        # Convert messages to OpenAI format
        openai_messages = []

        # Add system prompt as system message
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Add conversation messages
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        api_params = {
            "model": self._config.model,
            "messages": openai_messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }

        # Handle tools if provided
        if tools and self.supports_tools():
            api_params["tools"] = self._convert_tools_to_openai(tools)

        response = self._client.chat.completions.create(**api_params)

        # Extract text content
        text = response.choices[0].message.content or ""

        # Extract usage info
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        # Extract metadata
        metadata = {
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "has_tool_use": response.choices[0].finish_reason == "tool_calls",
        }

        return LLMResponse(
            text=text,
            model=self._config.model,
            usage=usage,
            metadata=metadata,
        )

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            from openai import OpenAI
            return bool(self._config.api_key)
        except ImportError:
            return False

    def supports_tools(self) -> bool:
        """OpenAI supports tool use (function calling)."""
        return True

    def _convert_tools_to_openai(self, tools: list[dict]) -> list[dict]:
        """
        Convert tool definitions from Claude format to OpenAI format.
        """
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools
