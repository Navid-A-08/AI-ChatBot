"""
Base abstraction for LLM providers.

Why this exists:
    To support multiple LLM providers (Claude, OpenAI, local models),
    we need a common interface that abstracts away provider-specific
    differences. This allows:
    1. Easy switching between providers for comparison (Phase 8)
    2. Fallback to alternative providers if one fails
    3. A/B testing different models
    4. Cost optimization by using cheaper models for simple tasks

How it works:
    BaseLLMProvider defines the interface that all providers must implement.
    Concrete implementations (ClaudeProvider, OpenAIProvider) handle the
    provider-specific API calls.

Alternatives considered:
    - Using an existing abstraction (LiteLLM): adds a dependency and
      may not support all features we need. Building our own is more
      educational for a portfolio project.
    - Simple function-based approach: harder to maintain state and
      configuration. Class-based is cleaner.

Trade-offs accepted:
    - Some provider-specific features (like Claude's tool use) may not
      have equivalents in other providers. We handle this gracefully
      by returning empty results for unsupported features.
    - Each provider has different rate limits and pricing. This
      abstraction doesn't handle that — it's the caller's responsibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    model: str
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, etc.
    metadata: dict = field(default_factory=dict)  # Provider-specific metadata


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    model: str
    api_key: str = ""
    max_tokens: int = 1024
    temperature: float = 1.0
    extra_params: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement these methods.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return self.__class__.__name__

    @property
    def model(self) -> str:
        """Return the model being used."""
        return self._config.model

    @abstractmethod
    def send_message(
        self,
        messages: list[dict],
        system_prompt: str = "",
        tools: list[dict] = None,
    ) -> LLMResponse:
        """
        Send a message to the LLM and return a response.

        Args:
            messages: List of message dicts with "role" and "content".
            system_prompt: System prompt string.
            tools: Optional tool definitions.

        Returns:
            LLMResponse with the model's output.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is available (API key configured, etc.).

        Returns:
            True if the provider can be used, False otherwise.
        """
        pass

    def get_tool_definitions(self) -> list[dict]:
        """
        Get tool definitions in the format this provider expects.

        Default implementation returns empty list (no tools).
        Override in subclasses that support tools.
        """
        return []

    def supports_tools(self) -> bool:
        """
        Check if this provider supports tool use.

        Default is False. Override in subclasses that support tools.
        """
        return False


class ProviderRegistry:
    """
    Registry for LLM providers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, type[BaseLLMProvider]] = {}

    def register(self, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a provider class."""
        self._providers[name] = provider_class

    def get_provider(self, name: str) -> type[BaseLLMProvider] | None:
        """Get a provider class by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def create_provider(self, name: str, config: LLMConfig) -> BaseLLMProvider | None:
        """
        Create a provider instance.

        Args:
            name: Provider name.
            config: Configuration for the provider.

        Returns:
            Provider instance, or None if not found.
        """
        provider_class = self._providers.get(name)
        if provider_class:
            return provider_class(config)
        return None


# Global provider registry
default_provider_registry = ProviderRegistry()
