"""
LLM provider implementations.

This package contains the base abstraction and concrete implementations
for different LLM providers.
"""

from ai_chatbot.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    ProviderRegistry,
    default_provider_registry,
)

# Register available providers
def _register_default_providers():
    """Register all available LLM providers."""
    from ai_chatbot.llm.claude_provider import ClaudeProvider
    default_provider_registry.register("claude", ClaudeProvider)

    # Try to register OpenAI if available
    try:
        from ai_chatbot.llm.openai_provider import OpenAIProvider
        default_provider_registry.register("openai", OpenAIProvider)
    except ImportError:
        pass

_register_default_providers()

__all__ = [
    "BaseLLMProvider",
    "LLMConfig",
    "LLMResponse",
    "ProviderRegistry",
    "default_provider_registry",
]
