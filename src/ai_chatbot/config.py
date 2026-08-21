"""
Central application configuration.

Why this exists:
    Scattering `os.getenv(...)` calls throughout the codebase means:
    - No single place to see what config the app needs.
    - No validation (a missing API key fails deep inside some API call,
      not at startup with a clear error).
    - No type safety (everything is a string until you remember to cast it).

    pydantic-settings solves all three: it reads from environment variables
    (and a .env file via python-dotenv under the hood), validates types,
    and fails fast at startup if something required is missing.

Alternatives considered:
    - Plain os.getenv() calls: simplest, but no validation, no single
      source of truth, easy to typo an env var name and get None silently.
    - A hand-written dataclass + manual os.getenv parsing: more explicit,
      slightly less "magic" than pydantic-settings, but you re-implement
      validation/type-casting yourself for no real benefit here.
    - A plain .yaml/.toml config file: fine for non-secret config, but
      secrets (API keys) still shouldn't live in a committed file, so
      you'd end up mixing two config sources. Env vars keep it uniform.

Trade-off accepted:
    Adds a dependency (pydantic-settings) for what is a small amount of
    config right now. Justified because config will grow (chunk size,
    top-k, memory limits, etc. in later phases) and validated config
    prevents a whole class of "silent misconfiguration" bugs.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application settings, loaded from environment variables / .env.

    Field names map to env vars of the same name (case-insensitive),
    e.g. `anthropic_api_key` <- ANTHROPIC_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    log_level: str = "INFO"

    # Phase 4: Context ranking and compression
    context_strategy: str = "window"  # "window" or "ranked"
    max_context_tokens: int = 8000
    max_history_turns: int = 6
    rag_top_k: int = 3

    # Phase 7: Multi-provider support
    llm_provider: str = "claude"  # "claude", "openai", or "nvidia"
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    nvidia_api_key: str = ""
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    def validate_provider_api_key(self) -> None:
        """Check that the selected provider has an API key configured."""
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "nvidia" and not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia")


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Cached (via lru_cache) so we parse env vars once, not on every call.
    Validates that the selected provider has its API key configured.
    """
    settings = Settings()
    settings.validate_provider_api_key()
    return settings
