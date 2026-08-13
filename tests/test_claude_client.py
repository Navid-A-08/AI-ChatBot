"""
Tests for ClaudeClient.

We mock the Anthropic SDK client rather than calling the real API:
- Tests stay fast and free (no network, no API cost).
- Tests are deterministic (not dependent on model behavior).
- We're testing OUR wrapper logic (how we call the SDK, how we parse
  the response), not Anthropic's API itself.
"""

from unittest.mock import MagicMock, patch

from ai_chatbot.llm.claude_client import ClaudeClient


def _make_fake_response(text: str):
    """Build an object shaped like the SDK's response, with one text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.content = [block]
    return response


@patch("ai_chatbot.llm.claude_client.Anthropic")
def test_send_returns_text_from_response(mock_anthropic_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.messages.create.return_value = _make_fake_response("Hello, human.")
    mock_anthropic_cls.return_value = mock_client_instance

    client = ClaudeClient(api_key="fake-key", model="claude-sonnet-4-6")
    result = client.send(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful.",
    )

    assert result == "Hello, human."


@patch("ai_chatbot.llm.claude_client.Anthropic")
def test_send_passes_correct_arguments_to_api(mock_anthropic_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.messages.create.return_value = _make_fake_response("ok")
    mock_anthropic_cls.return_value = mock_client_instance

    client = ClaudeClient(api_key="fake-key", model="claude-sonnet-4-6", max_tokens=500)
    messages = [{"role": "user", "content": "Hi"}]
    client.send(messages=messages, system_prompt="Be concise.")

    mock_client_instance.messages.create.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="Be concise.",
        messages=messages,
    )
