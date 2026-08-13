"""LangChain chat-model transport used by CPPL agent graphs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .models import AgentRuntimeError, ResolvedAgentConfig
from ..env import LLMModelConfig, resolve_llm_model_config


class LLMBackendError(AgentRuntimeError):
    """The configured LangChain model failed after its retry budget."""


@dataclass(frozen=True)
class BackendResponse:
    text: str
    transport_retries: int


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface consumed by the LangGraph module workflow."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse: ...


class _RetryCounter(BaseCallbackHandler):
    """Count attempts tagged by LangChain's RunnableRetry wrapper."""

    def __init__(self) -> None:
        self.attempts = 0
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        with self._lock:
            self.attempts += 1

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        with self._lock:
            self.attempts += 1


def create_chat_model(settings: LLMModelConfig) -> BaseChatModel:
    """Create the provider integration selected by environment configuration."""
    common: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "max_retries": 0,
    }
    if settings.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMBackendError(
                "The 'langchain-anthropic' dependency is required for Anthropic."
            ) from exc
        if settings.base_url:
            common["base_url"] = settings.base_url
        return ChatAnthropic(**common)

    if settings.provider != "openai" and not settings.base_url:
        raise LLMBackendError(
            f"Provider {settings.provider!r} is supported through an OpenAI-compatible "
            "endpoint; set LLM_BASE_URL."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMBackendError(
            "The 'langchain-openai' dependency is required for OpenAI-compatible models."
        ) from exc
    if settings.base_url:
        common["base_url"] = settings.base_url
    return ChatOpenAI(**common)


class LangChainBackend:
    """Invoke provider chat models through a LangChain prompt pipeline."""

    _prompt = ChatPromptTemplate.from_messages(
        [("system", "{{{system_prompt}}}"), ("human", "{{{user_prompt}}}")],
        template_format="mustache",
    )

    def __init__(
        self,
        config: ResolvedAgentConfig,
        *,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self.config = config
        self.settings = resolve_llm_model_config()
        if chat_model is None:
            if self.settings is None:
                raise LLMBackendError(
                    "Missing LLM configuration. Set LLM_MODEL and credentials "
                    "in the project root .env file."
                )
            chat_model = create_chat_model(self.settings)
        self.chat_model = chat_model

    @staticmethod
    def model_identity() -> dict[str, Any]:
        settings = resolve_llm_model_config()
        if settings is None:
            return {"model": None, "provider": None, "base_url": None}
        return settings.identity

    @classmethod
    def peek_model_identity(cls) -> dict[str, Any]:
        return cls.model_identity()

    def _chain(self, output_tokens: int | None) -> Runnable:
        kwargs = {
            "max_tokens": output_tokens or self.config.output_tokens,
            "timeout": self.config.request_timeout,
            **self.config.generation_kwargs,
        }
        model = self.chat_model.bind(**kwargs)
        chain: Runnable = self._prompt | model | StrOutputParser()
        return chain.with_retry(
            stop_after_attempt=self.config.transport_retries + 1,
            wait_exponential_jitter=True,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse:
        counter = _RetryCounter()
        try:
            text = self._chain(output_tokens).invoke(
                {"system_prompt": system_prompt, "user_prompt": user_prompt},
                config={"callbacks": [counter]},
            )
        except Exception as exc:
            raise LLMBackendError(
                "LLM request failed after "
                f"{max(counter.attempts, 1)} transport attempt(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMBackendError("LangChain returned an empty text response.")
        return BackendResponse(text=text, transport_retries=max(counter.attempts - 1, 0))
