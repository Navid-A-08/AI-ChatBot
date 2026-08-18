import pytest

from ai_chatbot.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    ProviderRegistry,
    default_provider_registry,
)


def test_llm_response_creation():
    """Test creating an LLMResponse."""
    response = LLMResponse(
        text="Hello world",
        model="claude-3-sonnet",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )

    assert response.text == "Hello world"
    assert response.model == "claude-3-sonnet"
    assert response.usage["prompt_tokens"] == 10


def test_llm_config_creation():
    """Test creating an LLMConfig."""
    config = LLMConfig(
        model="claude-3-sonnet",
        api_key="test-key",
        max_tokens=500,
    )

    assert config.model == "claude-3-sonnet"
    assert config.api_key == "test-key"
    assert config.max_tokens == 500


def test_provider_registry():
    """Test ProviderRegistry operations."""
    registry = ProviderRegistry()

    # Create a mock provider class
    class MockProvider(BaseLLMProvider):
        def send_message(self, messages, system_prompt="", tools=None):
            return LLMResponse(text="mock", model="mock")

        def is_available(self):
            return True

    registry.register("mock", MockProvider)

    assert "mock" in registry.list_providers()
    assert registry.get_provider("mock") == MockProvider


def test_default_registry_has_claude():
    """Test that default registry has Claude provider."""
    assert "claude" in default_provider_registry.list_providers()


def test_create_provider():
    """Test creating a provider instance."""
    config = LLMConfig(
        model="claude-3-sonnet",
        api_key="test-key",
    )

    provider = default_provider_registry.create_provider("claude", config)

    assert provider is not None
    assert provider.provider_name == "claude"


def test_create_unknown_provider():
    """Test creating an unknown provider returns None."""
    config = LLMConfig(model="unknown", api_key="test")

    provider = default_provider_registry.create_provider("unknown_provider", config)

    assert provider is None


def test_claude_provider_initialization():
    """Test Claude provider can be initialized."""
    from ai_chatbot.llm.claude_provider import ClaudeProvider

    config = LLMConfig(
        model="claude-3-sonnet",
        api_key="test-key",
    )

    provider = ClaudeProvider(config)

    assert provider.provider_name == "claude"
    assert provider.model == "claude-3-sonnet"
    assert provider.is_available() is True


def test_claude_provider_supports_tools():
    """Test that Claude provider supports tools."""
    from ai_chatbot.llm.claude_provider import ClaudeProvider

    config = LLMConfig(model="claude-3-sonnet", api_key="test")
    provider = ClaudeProvider(config)

    assert provider.supports_tools() is True


def test_openai_provider_initialization():
    """Test OpenAI provider can be initialized (even without package)."""
    from ai_chatbot.llm.openai_provider import OpenAIProvider

    config = LLMConfig(
        model="gpt-4",
        api_key="test-key",
    )

    # This may fail to initialize client if openai not installed
    # but should not raise an exception
    provider = OpenAIProvider(config)

    assert provider.provider_name == "openai"
    assert provider.model == "gpt-4"


def test_openai_provider_availability():
    """Test OpenAI provider availability check."""
    from ai_chatbot.llm.openai_provider import OpenAIProvider

    config = LLMConfig(model="gpt-4", api_key="test")
    provider = OpenAIProvider(config)

    # Should return True if openai is installed and API key is set
    # or False if openai is not installed
    # Either way, should not raise an exception
    assert isinstance(provider.is_available(), bool)
