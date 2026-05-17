"""Environment-driven LLM server configuration helpers for CPPL."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

_ENV_LOADED = False


@dataclass(frozen=True)
class LLMServerOverride:
    """Resolved LLM server settings loaded from environment variables."""

    server_name: str
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


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
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def _sanitize_env_token(token: str) -> str:
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
        sanitized = _sanitize_env_token(token)
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


def resolve_provider_specific_env(provider: Optional[str], suffix: str) -> Optional[str]:
    """Return the first provider-specific env var value for one suffix."""
    env_names = [f"{prefix}_{suffix}" for prefix in provider_env_prefixes(provider)]
    if not env_names:
        return None
    return resolve_env_value(*env_names)


def resolve_llm_server_override() -> Optional[LLMServerOverride]:
    """Resolve a generic LLM server override from environment variables."""
    provider = resolve_env_value("LLM_PROVIDER")
    model = resolve_env_value("LLM_MODEL")
    if model is None:
        model = resolve_provider_specific_env(provider, "MODEL")
    if provider and model and "/" not in model:
        model = f"{provider}/{model}"

    base_url = resolve_env_value("LLM_BASE_URL")
    if base_url is None:
        base_url = resolve_provider_specific_env(provider, "BASE_URL")
    if base_url is None:
        base_url = resolve_env_value("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")

    api_key = resolve_env_value("LLM_API_KEY")
    if api_key is None:
        api_key = resolve_provider_specific_env(provider, "API_KEY")
    if api_key is None:
        api_key = resolve_env_value("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

    if model is None:
        return None

    server_name = resolve_env_value("LLM_SERVER_NAME") or "cppl_env"
    return LLMServerOverride(
        server_name=server_name,
        model=model,
        provider=provider,
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

    return kwargs


def apply_server_override(configs: object, override: LLMServerOverride) -> str:
    """Inject the .env-resolved server into APPL config objects.

    Returns the actual target server name that was updated.
    """
    servers = getattr(configs, "servers", None)
    if servers is None:
        servers = {}
        setattr(configs, "servers", servers)
    if not isinstance(servers, dict):
        raise TypeError("APPL configs.servers must be a dictionary")

    default_servers = getattr(configs, "default_servers", None)
    target_name = override.server_name

    if default_servers is None:
        default_servers = SimpleNamespace()
        setattr(configs, "default_servers", default_servers)
    if isinstance(default_servers, dict):
        default_servers["default"] = target_name
    else:
        setattr(default_servers, "default", target_name)

    server_cfg: dict[str, Any] = {"model": override.model}
    if override.provider:
        server_cfg["provider"] = override.provider
    if override.base_url:
        server_cfg["base_url"] = override.base_url
    if override.api_key:
        server_cfg["api_key"] = override.api_key
    servers[target_name] = server_cfg
    return target_name
