"""
Tests for Settings loading.

What we're verifying:
    1. Settings loads correctly when required env vars are present.
    2. Settings raises a clear validation error when the required
       ANTHROPIC_API_KEY is missing — this is the "fail fast at
       startup" behavior config.py's docstring promises, so it's
       worth pinning down with a test rather than trusting it works.
"""

import pytest
from pydantic import ValidationError

from ai_chatbot.config import Settings


def test_settings_loads_with_required_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    # _env_file=None: ignore any real .env file on disk during this test,
    # so the test only reflects the env vars we explicitly set above.
    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == "test-key-123"
    # Default value should be used when not explicitly set.
    assert settings.anthropic_model == "claude-sonnet-4-6"


def test_settings_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_respects_custom_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    settings = Settings(_env_file=None)

    assert settings.anthropic_model == "claude-opus-4-8"
