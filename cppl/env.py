"""Environment-driven LangChain model configuration helpers for CPPL."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_ENV_LOADED = False


@dataclass(frozen=True)
class LLMModelConfig:
    """Resolved LangChain chat-model settings loaded from the environment."""

    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def identity(self) -> dict[str, Optional[str]]:
        """Return cache-safe model identity without credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }


def load_dotenv() -> None:
    """Load project-root .env values into os.environ without overwriting env."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def sanitize_env_token(token: str) -> str:
    """Normalize provider names for environment variable prefix matching."""
    return re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").upper()


def resolve_env_value(*env_names: str) -> Optional[str]:
    """Return the first non-empty environment variable among *env_names*."""
    load_dotenv()
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def provider_env_prefixes(provider: Optional[str]) -> list[str]:
    """Return possible env var prefixes for one provider name."""
    if not provider:
        return []

    prefixes: list[str] = []
    for token in provider.split("/"):
        sanitized = sanitize_env_token(token)
        if sanitized and sanitized not in prefixes:
            prefixes.append(sanitized)

    aliases = {
        "QWEN": ["DASHSCOPE"],
        "DASHSCOPE": ["QWEN"],
        "AZURE": ["AZURE_OPENAI"],
        "AZURE_OPENAI": ["AZURE"],
    }
    for prefix in list(prefixes):
        for alias in aliases.get(prefix, []):
            if alias not in prefixes:
                prefixes.append(alias)
    return prefixes


def resolve_provider_specific_env(
    provider: Optional[str], suffix: str
) -> Optional[str]:
    """Return the first provider-specific env var value for one suffix."""
    env_names = [f"{prefix}_{suffix}" for prefix in provider_env_prefixes(provider)]
    if not env_names:
        return None
    return resolve_env_value(*env_names)


def resolve_llm_model_config() -> Optional[LLMModelConfig]:
    """Resolve a provider and model for the LangChain chat-model factory."""
    configured_provider = resolve_env_value("LLM_PROVIDER")
    raw_model = resolve_env_value("LLM_MODEL")
    if raw_model is None:
        raw_model = resolve_provider_specific_env(configured_provider, "MODEL")
    if raw_model is None:
        return None

    model_provider: Optional[str] = None
    model = raw_model
    if "/" in raw_model:
        model_provider, model = raw_model.split("/", 1)
    provider = (configured_provider or model_provider or "openai").lower()
    if configured_provider and model_provider:
        if configured_provider.lower() != model_provider.lower():
            raise ValueError(
                "LLM_PROVIDER conflicts with the provider prefix in LLM_MODEL: "
                f"{configured_provider!r} != {model_provider!r}"
            )

    base_url = resolve_env_value("LLM_BASE_URL")
    if base_url is None:
        base_url = resolve_provider_specific_env(provider, "BASE_URL")
    if base_url is None:
        base_url = resolve_env_value(
            "ANTHROPIC_BASE_URL" if provider == "anthropic" else "OPENAI_BASE_URL"
        )

    api_key = resolve_env_value("LLM_API_KEY")
    if api_key is None:
        api_key = resolve_provider_specific_env(provider, "API_KEY")
    if api_key is None:
        api_key = resolve_env_value(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        )

    return LLMModelConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def resolve_llm_generation_kwargs() -> dict[str, Any]:
    """Resolve optional generation kwargs from environment variables.

    By default CPPL avoids forcing provider-specific sampling params.
    Set these env vars only when you want to override the model defaults.
    """
    kwargs: dict[str, Any] = {}

    temperature = resolve_env_value("LLM_TEMPERATURE")
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    top_p = resolve_env_value("LLM_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = float(top_p)

    reasoning_effort = resolve_env_value("LLM_REASONING_EFFORT")
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    timeout = resolve_env_value("LLM_TIMEOUT")
    if timeout is not None:
        kwargs["timeout"] = float(timeout)

    max_tokens = resolve_env_value("LLM_MAX_TOKENS")
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    return kwargs
