"""
NVIDIA NIM LLM provider implementation.

Supports models available through NVIDIA's build.nvidia.com platform,
including Nemotron, Llama, Mixtral, and other models served via NIM.

The NVIDIA NIM API is OpenAI-compatible, so this provider extends
the OpenAI provider with NVIDIA-specific defaults.
"""

import logging

from ai_chatbot.llm.base import BaseLLMProvider, LLMConfig, LLMResponse

logger = logging.getLogger(__name__)

# NVIDIA NIM base URL for hosted inference
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(BaseLLMProvider):
    """
    NVIDIA NIM API provider implementation.

    Uses the OpenAI-compatible endpoint at integrate.api.nvidia.com.

    Requires the openai package to be installed:
    pip install openai
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client pointed at NVIDIA's endpoint."""
        try:
            from openai import OpenAI

            base_url = self._config.extra_params.get("base_url", NVIDIA_NIM_BASE_URL)

            self._client = OpenAI(
                api_key=self._config.api_key,
                base_url=base_url,
            )
            logger.info("Initialized NVIDIA NIM client with base URL: %s", base_url)
        except ImportError:
            logger.warning(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        except Exception:
            logger.exception("Failed to initialize NVIDIA NIM client")

    @property
    def provider_name(self) -> str:
        return "nvidia"

    def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] = None,
    ) -> LLMResponse:
        """
        Send a message to NVIDIA NIM.
        """
        if not self._client:
            raise RuntimeError(
                "NVIDIA NIM client not initialized. "
                "Install openai package: pip install openai"
            )

        logger.debug(
            "Sending %d messages to NVIDIA NIM (%s)",
            len(messages), self._config.model,
        )

        # Convert messages to OpenAI format
        openai_messages = []

        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt,
            })

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

        response = self._client.chat.completions.create(**api_params)

        text = response.choices[0].message.content or ""

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        metadata = {
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "provider": "nvidia_nim",
        }

        return LLMResponse(
            text=text,
            model=self._config.model,
            usage=usage,
            metadata=metadata,
        )

    def is_available(self) -> bool:
        """Check if NVIDIA NIM API is available."""
        try:
            from openai import OpenAI
            return bool(self._config.api_key)
        except ImportError:
            return False
