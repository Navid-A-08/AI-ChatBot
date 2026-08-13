"""
Prompt loading utilities.

Why this exists:
    System prompts live in prompts/*.md (see prompts/system_prompt.md),
    not hardcoded in Python. This module is the single place that knows
    how to find and read them, so calling code just says
    `load_prompt("system_prompt")` without caring about file paths.

How it works:
    Prompts live in a `prompts/` directory at the project root (sibling
    to `src/`, not inside the package) — this makes the location
    predictable and matches where README.md points people to edit
    prompts, independent of how the package itself is installed/run.

Failure mode:
    If a prompt file is missing, this raises FileNotFoundError with a
    clear message rather than returning an empty string. A silently
    empty system prompt is a much worse bug (the model just behaves
    with defaults, no error, no clue why) than a loud crash at startup.
"""

from pathlib import Path

# Project root = two levels up from this file (src/ai_chatbot/prompts.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt file by name (without extension) from the prompts/ dir.

    Args:
        name: filename without extension, e.g. "system_prompt" for
              prompts/system_prompt.md

    Returns:
        The file's text content, stripped of leading/trailing whitespace.

    Raises:
        FileNotFoundError: if prompts/{name}.md doesn't exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Expected a file at prompts/{name}.md relative to the project root."
        )
    return path.read_text(encoding="utf-8").strip()
