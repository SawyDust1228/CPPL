"""LangChain model-pipeline and LangGraph orchestration tests."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cppl import In, Out, module
from cppl.agents.backend import LangChainBackend, LLMBackendError, create_chat_model
from cppl.agents.models import AgentConfig
from cppl.harness.schema import IRSchemaError, validate_ir_body
from cppl.agents.runtime import CompilationCoordinator
from cppl.env import LLMModelConfig


def config(tmp_path, *, retries=0):
    return AgentConfig(
        cache_enabled=False,
        cache_dir=tmp_path / "cache",
        max_parallelism=2,
        context_window_tokens=8192,
        output_tokens=2048,
        safety_margin_tokens=512,
        transport_retries=retries,
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
