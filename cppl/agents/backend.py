"""APPL-backed LLM transport with isolated requests and transport retries."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .models import AgentRuntimeError, ResolvedAgentConfig
from ..env import apply_server_override, resolve_llm_server_override


class LLMBackendError(AgentRuntimeError):
    """The configured LLM transport failed after its retry budget."""


@dataclass(frozen=True)
class BackendResponse:
    text: str
    transport_retries: int


class APPLBackend:
    """Create one fresh APPL message list per semantic candidate request."""

    _configure_lock = threading.Lock()
    _configured = False

    def __init__(self, config: ResolvedAgentConfig) -> None:
        self.config = config
        self._configure()

    @classmethod
    def _configure(cls) -> None:
        with cls._configure_lock:
            if cls._configured:
                return
            try:
                import appl
            except ImportError as exc:
                raise LLMBackendError(
                    "The 'applang' dependency is required for LLM compilation."
                ) from exc

            override = resolve_llm_server_override()
            if override is None:
                raise LLMBackendError(
                    "Missing LLM configuration. Set LLM_MODEL and credentials "
                    "in the project root .env file."
                )
            target_name = apply_server_override(appl.global_vars.configs, override)
            appl.server_manager.close_server(target_name)
            cls._configured = True

    @staticmethod
    def model_identity() -> dict[str, Any]:
        """Return cache-safe model identity without credentials."""
        override = resolve_llm_server_override()
        if override is None:
            return {"model": None, "provider": None, "base_url": None}
        return {
            "model": override.model,
            "provider": override.provider,
            "base_url": override.base_url,
            "server_name": override.server_name,
        }

    @classmethod
    def peek_model_identity(cls) -> dict[str, Any]:
        """Compatibility alias used before lazily constructing the backend."""
        return cls.model_identity()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse:
        from appl import SystemMessage, UserMessage, gen

        messages = [SystemMessage(system_prompt), UserMessage(user_prompt)]
        kwargs = {
            "max_tokens": output_tokens or self.config.output_tokens,
            "timeout": self.config.request_timeout,
            **self.config.generation_kwargs,
        }
        last_error: Exception | None = None
        for retry in range(self.config.transport_retries + 1):
            try:
                response = gen(messages=messages, **kwargs)
                return BackendResponse(str(response), retry)
            except Exception as exc:
                last_error = exc
                if retry >= self.config.transport_retries:
                    break
                time.sleep(min(2 ** retry, 4))
        raise LLMBackendError(
            "LLM request failed after "
            f"{self.config.transport_retries + 1} transport attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error
