"""LangChain chat-model transport used by CPPL agent graphs."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
import json
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .models import AgentRuntimeError, ResolvedAgentConfig
from .tools import IRToolSession, ToolSessionError
from ..harness.schema import ModuleBodySchema
from ..env import LLMModelConfig, resolve_llm_model_config


class LLMBackendError(AgentRuntimeError):
    """The configured LangChain model failed after its retry budget."""


@dataclass(frozen=True)
class BackendResponse:
    text: str
    transport_retries: int
    model_turns: int = 1
    extra_input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    used_tools: bool = False
    tool_fallback_reason: str = ""


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
        self._structured_supported: bool | None = None
        self._tools_supported: bool | None = None
        self._capability_lock = threading.Lock()

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        if not value:
            return 0
        ascii_chars = sum(1 for char in value if ord(char) < 128)
        return max(1, (ascii_chars + 1) // 4 + (len(value) - ascii_chars))

    @classmethod
    def _messages_tokens(cls, messages: list[Any]) -> int:
        total = 0
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                total += cls._estimate_tokens(content)
            else:
                total += cls._estimate_tokens(json.dumps(content, default=str))
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                total += cls._estimate_tokens(json.dumps(tool_calls, default=str))
        return total

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            parts: list[str] = []
            for block in message.content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        return str(message.content or "")

    @staticmethod
    def _tool_support_error(exc: Exception) -> bool:
        if isinstance(exc, (NotImplementedError, AttributeError, TypeError)):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "bind_tools",
                "tool calling is not supported",
                "does not support tools",
                "tools are not supported",
                "unsupported tool",
                "unknown field 'tools'",
                "unknown field: tools",
                "invalid tools",
            )
        )

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
        def require_nonempty(text: str) -> str:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("LLM returned an empty text response")
            return text

        chain: Runnable = (
            self._prompt | model | StrOutputParser() | RunnableLambda(require_nonempty)
        )
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

    def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_session: IRToolSession,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse:
        """Run a native chat-model tool loop with an automatic compatibility fallback."""
        if self.config.tool_mode == "off":
            response = self.generate_structured(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
            return replace(response, tool_fallback_reason="tool mode is disabled")
        with self._capability_lock:
            tools_supported = self._tools_supported
        if tools_supported is False:
            if self.config.tool_mode == "required":
                raise LLMBackendError("The configured model does not support tool calling.")
            response = self.generate_structured(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
            return replace(
                response, tool_fallback_reason="model tool calling is unavailable"
            )

        try:
            response = self._generate_with_tools(
                system_prompt,
                user_prompt,
                tool_session,
                output_tokens=output_tokens,
            )
            with self._capability_lock:
                self._tools_supported = True
            return response
        except ToolSessionError as exc:
            if self.config.tool_mode == "required" or tool_session.stats.calls:
                raise
            response = self.generate_structured(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
            return replace(
                response,
                tool_fallback_reason=(
                    "tool session produced no candidate: " + str(exc)[:240]
                ),
            )
        except Exception as exc:
            unsupported = self._tool_support_error(exc)
            if not unsupported or tool_session.stats.calls:
                raise
            with self._capability_lock:
                self._tools_supported = False
            if self.config.tool_mode == "required":
                raise LLMBackendError(
                    f"The configured model does not support tool calling: {exc}"
                ) from exc
            response = self.generate_structured(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
            return replace(
                response,
                tool_fallback_reason=f"tool calling unavailable: {type(exc).__name__}",
            )

    def _generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_session: IRToolSession,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse:
        kwargs = {
            "max_tokens": output_tokens or self.config.output_tokens,
            "timeout": self.config.request_timeout,
            **self.config.generation_kwargs,
        }
        model = self.chat_model.bind_tools(tool_session.tool_definitions).bind(**kwargs)
        runnable = model.with_retry(
            stop_after_attempt=self.config.transport_retries + 1,
            wait_exponential_jitter=True,
        )
        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        initial_input_tokens = self._messages_tokens(messages)
        total_input_tokens = 0
        total_output_tokens = 0
        counter = _RetryCounter()
        model_turns = 0
        last_text = ""

        for _ in range(self.config.max_tool_rounds):
            try:
                tool_session.check_runtime_budget(
                    max(0, total_input_tokens - initial_input_tokens)
                    + total_output_tokens
                )
            except ToolSessionError:
                if not tool_session.candidate:
                    raise
                return BackendResponse(
                    text=json.dumps(
                        tool_session.candidate,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    transport_retries=max(counter.attempts - model_turns, 0),
                    model_turns=model_turns,
                    extra_input_tokens_estimate=max(
                        0, total_input_tokens - initial_input_tokens
                    ),
                    output_tokens_estimate=total_output_tokens,
                    used_tools=tool_session.stats.calls > 0,
                )
            total_input_tokens += self._messages_tokens(messages)
            try:
                response = runnable.invoke(messages, config={"callbacks": [counter]})
            except Exception as exc:
                raise LLMBackendError(
                    "Tool-enabled LLM request failed after "
                    f"{max(counter.attempts, 1)} transport attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            model_turns += 1
            if not isinstance(response, AIMessage):
                raise LLMBackendError(
                    f"Tool-enabled model returned {type(response).__name__}, expected AIMessage."
                )
            messages.append(response)
            total_output_tokens += self._messages_tokens([response])
            last_text = self._message_text(response).strip()
            tool_calls = list(response.tool_calls or [])
            if not tool_calls:
                if last_text:
                    return BackendResponse(
                        text=last_text,
                        transport_retries=max(counter.attempts - model_turns, 0),
                        model_turns=model_turns,
                        extra_input_tokens_estimate=max(
                            0, total_input_tokens - initial_input_tokens
                        ),
                        output_tokens_estimate=total_output_tokens,
                        used_tools=tool_session.stats.calls > 0,
                    )
                if tool_session.stats.calls and tool_session.candidate:
                    return BackendResponse(
                        text=json.dumps(
                            tool_session.candidate,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        transport_retries=max(counter.attempts - model_turns, 0),
                        model_turns=model_turns,
                        extra_input_tokens_estimate=max(
                            0, total_input_tokens - initial_input_tokens
                        ),
                        output_tokens_estimate=total_output_tokens,
                        used_tools=True,
                    )
                finish_reason = response.response_metadata.get("finish_reason")
                reasoning = response.additional_kwargs.get("reasoning_content", "")
                usage = response.usage_metadata or {}
                raise ToolSessionError(
                    "The tool-enabled model returned neither tool calls nor JSON "
                    f"output (finish_reason={finish_reason!r}, "
                    f"reasoning_chars={len(reasoning) if isinstance(reasoning, str) else 0}, "
                    f"output_tokens={usage.get('output_tokens')!r})."
                )

            for call in tool_calls:
                name = call.get("name", "")
                arguments = call.get("args", {})
                result = tool_session.execute(name, arguments)
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            result, ensure_ascii=False, separators=(",", ":")
                        ),
                        tool_call_id=call.get("id") or f"tool-{tool_session.stats.calls}",
                        name=name or None,
                    )
                )
            if tool_session.submitted:
                return BackendResponse(
                    text=json.dumps(
                        tool_session.candidate,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    transport_retries=max(counter.attempts - model_turns, 0),
                    model_turns=model_turns,
                    extra_input_tokens_estimate=max(
                        0, total_input_tokens - initial_input_tokens
                    ),
                    output_tokens_estimate=total_output_tokens,
                    used_tools=True,
                )

            # Tool arguments may contain an entire IR body. Retaining every
            # completed exchange makes multi-round sessions grow quadratically.
            # The candidate itself persists in IRToolSession, so keep the task
            # prompt and only the most recent completed tool exchange.
            latest_exchange_size = 1 + len(tool_calls)
            messages = messages[:2] + messages[-latest_exchange_size:]

        if tool_session.candidate:
            return BackendResponse(
                text=json.dumps(
                    tool_session.candidate,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                transport_retries=max(counter.attempts - model_turns, 0),
                model_turns=model_turns,
                extra_input_tokens_estimate=max(
                    0, total_input_tokens - initial_input_tokens
                ),
                output_tokens_estimate=total_output_tokens,
                used_tools=tool_session.stats.calls > 0,
            )
        raise ToolSessionError(
            f"Tool session exhausted {self.config.max_tool_rounds} model rounds "
            "without a candidate."
        )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_tokens: int | None = None,
    ) -> BackendResponse:
        """Use provider-native structured output when the integration supports it."""
        if self._structured_supported is False:
            return self.generate(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
        counter = _RetryCounter()
        kwargs = {
            "max_tokens": output_tokens or self.config.output_tokens,
            "timeout": self.config.request_timeout,
            **self.config.generation_kwargs,
        }
        try:
            model = self.chat_model.bind(**kwargs).with_structured_output(ModuleBodySchema)
            chain = (self._prompt | model).with_retry(
                stop_after_attempt=self.config.transport_retries + 1,
                wait_exponential_jitter=True,
            )
            value = chain.invoke(
                {"system_prompt": system_prompt, "user_prompt": user_prompt},
                config={"callbacks": [counter]},
            )
            if isinstance(value, ModuleBodySchema):
                payload = value.model_dump(mode="json", exclude_none=True)
            elif isinstance(value, dict) and "root" in value:
                payload = value["root"]
            else:
                payload = value
            self._structured_supported = True
            return BackendResponse(
                text=json.dumps(payload, separators=(",", ":")),
                transport_retries=max(counter.attempts - 1, 0),
            )
        except (NotImplementedError, AttributeError, TypeError):
            self._structured_supported = False
            return self.generate(
                system_prompt, user_prompt, output_tokens=output_tokens
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                marker in message
                for marker in (
                    "invalid schema",
                    "response_format",
                    "json_schema",
                    "structured output is not supported",
                    "does not support structured",
                )
            ):
                self._structured_supported = False
                return self.generate(
                    system_prompt, user_prompt, output_tokens=output_tokens
                )
            raise LLMBackendError(
                "Structured LLM request failed after "
                f"{max(counter.attempts, 1)} transport attempt(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
