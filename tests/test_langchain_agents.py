"""LangChain model-pipeline and LangGraph orchestration tests."""

import json
from typing import ClassVar

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from cppl import In, Out, module
from cppl.agents.backend import LangChainBackend, LLMBackendError, create_chat_model
from cppl.agents.models import AgentConfig
from cppl.agents.tools import IRToolSession, ToolSessionLimits
from cppl.harness.schema import IRSchemaError, validate_ir_body
from cppl.agents.runtime import CompilationCoordinator
from cppl.env import LLMModelConfig


def config(tmp_path, *, retries=0, tool_mode="auto"):
    return AgentConfig(
        cache_enabled=False,
        cache_dir=tmp_path / "cache",
        max_parallelism=2,
        context_window_tokens=8192,
        output_tokens=2048,
        safety_margin_tokens=512,
        transport_retries=retries,
        tool_mode=tool_mode,
    ).resolve()


def test_langchain_backend_runs_prompt_model_parser_pipeline(tmp_path):
    model = FakeListChatModel(responses=['[{"op":"output","args":{"out":"a"}}]'])
    backend = LangChainBackend(config(tmp_path), chat_model=model)

    response = backend.generate("system instructions", '{"module":"M"}')

    assert response.text.startswith("[")
    assert response.transport_retries == 0


def test_langchain_retry_wrapper_reports_retry_count(tmp_path):
    class FlakyModel(FakeListChatModel):
        failures: int = 1

        def _call(self, *args, **kwargs):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("temporary")
            return super()._call(*args, **kwargs)

    backend = LangChainBackend(
        config(tmp_path, retries=1),
        chat_model=FlakyModel(responses=["ok"]),
    )

    response = backend.generate("system", "user")

    assert response.text == "ok"
    assert response.transport_retries == 1


def test_langchain_retry_wrapper_retries_empty_text(tmp_path):
    backend = LangChainBackend(
        config(tmp_path, retries=1),
        chat_model=FakeListChatModel(responses=["", "usable"]),
    )

    response = backend.generate("system", "user")

    assert response.text == "usable"
    assert response.transport_retries == 1


def test_unknown_openai_compatible_provider_requires_base_url():
    with pytest.raises(LLMBackendError, match="LLM_BASE_URL"):
        create_chat_model(
            LLMModelConfig(provider="deepseek", model="deepseek-chat", api_key="x")
        )


def test_coordinator_accepts_real_langchain_chat_model(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    model = FakeListChatModel(
        responses=[json.dumps([{"op": "output", "args": {"out": "a"}}])]
    )

    def factory(resolved):
        return LangChainBackend(resolved, chat_model=model)

    report = CompilationCoordinator(
        AgentConfig(
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=factory,
        model_identity={"provider": "fake", "model": "fake"},
    ).compile([M])

    assert report.success
    assert report.module_reports["M"].attempts == 1
    assert report.module_reports["M"].simulation_attempts == 0
    assert report.module_reports["M"].static_repairs == 0


def test_model_failure_ends_module_graph_with_report(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    class FailingBackend:
        def __init__(self, resolved):
            pass

        def generate(self, system_prompt, user_prompt, *, output_tokens=None):
            raise LLMBackendError("provider unavailable")

    report = CompilationCoordinator(
        AgentConfig(
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FailingBackend,
        model_identity={"provider": "fake", "model": "fake"},
    ).compile([M])

    assert not report.success
    assert report.module_reports["M"].error_category == "LLMBackendError"
    assert "provider unavailable" in report.module_reports["M"].error


def test_ir_schema_rejects_unknown_operation_and_invalid_id():
    with pytest.raises(IRSchemaError, match="Input tag 'binary'"):
        validate_ir_body([{"id": "x", "op": "binary", "args": ["a", "b"]}])
    with pytest.raises(IRSchemaError, match="pattern"):
        validate_ir_body([{"id": "", "op": "constant", "value": 0, "width": 1}])
    with pytest.raises(IRSchemaError, match="pattern"):
        validate_ir_body([{"id": "300", "op": "constant", "value": 0, "width": 1}])
    with pytest.raises(IRSchemaError, match="pattern"):
        validate_ir_body(
            [
                {
                    "id": ["1"],
                    "op": "mem",
                    "width": 8,
                    "depth": 1,
                    "clock": "clk",
                    "reads": [{"addr": "addr", "enable": "enable"}],
                    "writes": [],
                }
            ]
        )


def test_structured_backend_falls_back_when_provider_rejects_schema(tmp_path):
    class SchemaRejectingModel(FakeListChatModel):
        def with_structured_output(self, *args, **kwargs):
            class RejectingRunnable:
                def invoke(self, *args, **kwargs):
                    raise RuntimeError("Invalid schema for response_format")

            return RejectingRunnable()

    model = SchemaRejectingModel(
        responses=['[{"op":"output","args":{"out":"a"}}]']
    )
    backend = LangChainBackend(config(tmp_path), chat_model=model)
    response = backend.generate_structured("system", "user")
    assert response.text.startswith("[")
    assert backend._structured_supported is False


def test_langchain_backend_runs_native_tool_loop(tmp_path):
    body = [{"op": "output", "args": {"out": "a"}}]

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "replace_ir",
                        "args": {"body": body},
                        "id": "replace-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "submit_ir",
                        "args": {},
                        "id": "submit-1",
                        "type": "tool_call",
                    },
                ],
            )
        ]
    )
    backend = LangChainBackend(config(tmp_path), chat_model=model)
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        response = backend.generate_with_tools("system", "user", session)
    finally:
        session.close()

    assert json.loads(response.text) == body
    assert response.used_tools
    assert response.model_turns == 1
    assert session.stats.counts == {"replace_ir": 1, "submit_ir": 1}


def test_tool_loop_compacts_completed_history(tmp_path):
    body = [{"op": "output", "args": {"out": "a"}}]

    class RecordingToolModel(FakeMessagesListChatModel):
        message_counts: ClassVar[list[int]] = []

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            self.message_counts.append(len(messages))
            return super()._generate(messages, *args, **kwargs)

    RecordingToolModel.message_counts = []
    model = RecordingToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "replace_ir",
                        "args": {"body": body},
                        "id": "replace-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_ir",
                        "args": {"pointer": ""},
                        "id": "query-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=json.dumps(body)),
        ]
    )
    backend = LangChainBackend(config(tmp_path), chat_model=model)
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        response = backend.generate_with_tools("system", "user", session)
    finally:
        session.close()

    assert json.loads(response.text) == body
    assert RecordingToolModel.message_counts == [2, 4, 4]


def test_tool_calling_auto_mode_falls_back_to_existing_generation(tmp_path):
    class NoToolModel(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            raise NotImplementedError("tools unsupported")

    backend = LangChainBackend(
        config(tmp_path),
        chat_model=NoToolModel(responses=['[{"op":"output","args":{"out":"a"}}]']),
    )
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        response = backend.generate_with_tools("system", "user", session)
    finally:
        session.close()

    assert response.text.startswith("[")
    assert response.tool_fallback_reason
    assert backend._tools_supported is False


def test_empty_tool_turn_falls_back_immediately_in_auto_mode(tmp_path):
    class EmptyThenDirectModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            raise NotImplementedError("structured output unavailable")

    model = EmptyThenDirectModel(
        responses=[
            AIMessage(content=""),
            AIMessage(content='[{"op":"output","args":{"out":"a"}}]'),
        ]
    )
    backend = LangChainBackend(config(tmp_path), chat_model=model)
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        response = backend.generate_with_tools("system", "user", session)
    finally:
        session.close()

    assert response.text.startswith("[")
    assert "no candidate" in response.tool_fallback_reason


def test_tool_calling_required_mode_rejects_unsupported_model(tmp_path):
    class NoToolModel(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            raise NotImplementedError("tools unsupported")

    backend = LangChainBackend(
        config(tmp_path, tool_mode="required"),
        chat_model=NoToolModel(responses=["unused"]),
    )
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        with pytest.raises(LLMBackendError, match="does not support tool calling"):
            backend.generate_with_tools("system", "user", session)
    finally:
        session.close()


def test_tool_calling_off_mode_skips_tool_binding(tmp_path):
    class ToolBindingMustNotRun(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert not any(
                isinstance(tool, dict) and tool.get("name") == "replace_ir"
                for tool in tools
            )
            raise NotImplementedError("structured output unavailable")

    backend = LangChainBackend(
        config(tmp_path, tool_mode="off"),
        chat_model=ToolBindingMustNotRun(
            responses=['[{"op":"output","args":{"out":"a"}}]']
        ),
    )
    session = IRToolSession(
        initial_candidate=None,
        task_context="module M",
        validate_candidate=lambda candidate, patterns: ({"body": candidate}, 0),
        has_patterns=False,
        remaining_pattern_attempts=1,
        limits=ToolSessionLimits(),
    )
    try:
        response = backend.generate_with_tools("system", "user", session)
    finally:
        session.close()

    assert response.text.startswith("[")
    assert response.tool_fallback_reason == "tool mode is disabled"


def test_agent_tool_configuration_resolves_from_environment(monkeypatch):
    monkeypatch.setenv("CPPL_AGENT_TOOL_MODE", "required")
    monkeypatch.setenv("CPPL_AGENT_MAX_TOOL_ROUNDS", "7")
    monkeypatch.setenv("CPPL_AGENT_TOOL_TIMEOUT", "1.5")
    monkeypatch.setenv("CPPL_AGENT_MAX_TOOL_OUTPUT_CHARS", "2048")

    resolved = AgentConfig().resolve()

    assert resolved.tool_mode == "required"
    assert resolved.max_tool_rounds == 7
    assert resolved.tool_timeout_seconds == 1.5
    assert resolved.max_tool_output_chars == 2048


def test_agent_tool_configuration_rejects_invalid_values():
    with pytest.raises(ValueError, match="tool_mode"):
        AgentConfig(tool_mode="shell").resolve()
    with pytest.raises(ValueError, match="max_tool_rounds"):
        AgentConfig(max_tool_rounds=0).resolve()
