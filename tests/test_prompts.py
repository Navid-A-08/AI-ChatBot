import pytest

from ai_chatbot.prompts import load_prompt


def test_load_prompt_reads_system_prompt():
    content = load_prompt("system_prompt")
    assert len(content) > 0
    assert "helpful" in content.lower()


def test_load_prompt_raises_for_missing_file():
    with pytest.raises(FileNotFoundError, match="nonexistent_prompt"):
        load_prompt("nonexistent_prompt")
