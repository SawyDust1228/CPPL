"""Tests for cppl.compiler: JSON extraction, prompt building, and compile flow."""

import json

import pytest
from langchain_core.exceptions import OutputParserException

from cppl.env import (
    resolve_llm_generation_kwargs,
    resolve_llm_model_config,
)
from cppl.frontend.types import In, Out
from cppl.frontend.module import module, ModuleDef, PortInfo, InstanceCall
from cppl.agents.worker import (
    extract_json_array,
    build_ports_dict,
    build_instance_ops,
    build_stub_modules,
)
from cppl.frontend.compiler import CompileResult
from cppl.frontend.prompt import build_user_prompt, SYSTEM_PROMPT


# -----------------------------------------------------------------------
# JSON extraction
# -----------------------------------------------------------------------


class TestExtractJson:
    def test_plain_array(self):
        text = '[{"id":"x","op":"constant","value":0,"width":8},{"op":"output","args":{"out":"x"}}]'
        arr = extract_json_array(text)
        assert isinstance(arr, list)
        assert len(arr) == 2

    def test_markdown_fenced(self):
        text = '```json\n[{"op":"output","args":{"out":"a"}}]\n```'
        arr = extract_json_array(text)
        assert arr[0]["op"] == "output"

    def test_markdown_no_lang(self):
        text = '```\n[{"op":"output","args":{}}]\n```'
        arr = extract_json_array(text)
        assert len(arr) == 1

    def test_with_surrounding_text(self):
        text = 'Here is the body:\n```json\n[{"op":"output","args":{}}]\n```\nDone.'
        arr = extract_json_array(text)
        assert arr[0]["op"] == "output"

    def test_invalid_json_raises(self):
        with pytest.raises(OutputParserException):
            extract_json_array("this is not json")

    def test_truncated_json_raises(self):
        with pytest.raises(OutputParserException, match="complete JSON array"):
            extract_json_array('[{"op":"output"}')

    def test_incomplete_fence_raises(self):
        with pytest.raises(OutputParserException, match="fence is incomplete"):
            extract_json_array('```json\n[{"op":"output"}]')


# -----------------------------------------------------------------------
# Ports dict builder
# -----------------------------------------------------------------------


class TestPortsDict:
    def test_basic(self):
        @module
        def M(a: In[8]) -> Out[1]:
            """doc."""
            pass

        d = build_ports_dict(M)
        assert d == {
            "a": {"dir": "input", "width": 8},
            "out": {"dir": "output", "width": 1},
        }

    def test_multi_output(self):
        @module
        def M(x: In[2]) -> {"y": Out[4], "z": Out[1]}:
            """doc."""
            pass

        d = build_ports_dict(M)
        assert "y" in d
        assert d["y"]["dir"] == "output"
        assert d["z"]["width"] == 1


# -----------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------


class TestPromptBuilding:
    def test_user_prompt_contains_module_name(self):
        @module
        def Adder(a: In[8], b: In[8]) -> Out[8]:
            """out = a + b."""
            pass

        prompt = build_user_prompt(Adder)
        assert "Adder" in prompt
        assert "a: input, width 8" in prompt
        assert "out = a + b." in prompt

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100
        assert "output" in SYSTEM_PROMPT
        assert "constant" in SYSTEM_PROMPT


# -----------------------------------------------------------------------
# CompileResult dataclass
# -----------------------------------------------------------------------


class TestCompileResult:
    def test_success(self):
        r = CompileResult(module_dict={"name": "X"}, success=True, attempts=1)
        assert r.success
        assert r.error is None

    def test_failure(self):
        r = CompileResult(success=False, error="boom", attempts=3)
        assert not r.success
        assert r.error == "boom"


# -----------------------------------------------------------------------
# Prompt with instances
# -----------------------------------------------------------------------


class TestPromptWithInstances:
    def test_user_prompt_with_instances(self):
        @module
        def Adder8(a: In[8], b: In[8]) -> Out[8]:
            """out equals a plus b."""
            pass

        @module
        def ALU(
            op_code: In[2], op_a: In[8], op_b: In[8]
        ) -> {"res": Out[8], "zero": Out[1]}:
            adder8_out = Adder8(op_a, op_b)
            return f"res = {adder8_out} when op_code is 00. zero = 1 when res is 0."

        prompt = build_user_prompt(ALU)
        assert "Pre-defined instance operations" in prompt
        assert "Instance of 'Adder8'" in prompt
        assert "Adder8(a=op_a, b=op_b)" in prompt
        assert "adder8_out (width 8)" in prompt
        assert "output port 'out'" in prompt
        assert "Do NOT include any instance operations" in prompt

    def test_user_prompt_no_instances(self):
        @module
        def Simple(a: In[8]) -> Out[8]:
            """out = a."""
            pass

        prompt = build_user_prompt(Simple)
        assert "Pre-defined instance operations" not in prompt


# -----------------------------------------------------------------------
# Instance ops and stub generation
# -----------------------------------------------------------------------


class TestInstanceOps:
    def test_instance_ops(self):
        inst = InstanceCall(
            name=None,
            target_name="Adder8",
            target_ports=[
                PortInfo("a", 8, "input"),
                PortInfo("b", 8, "input"),
                PortInfo("out", 8, "output"),
            ],
            input_map={"a": "x", "b": "y"},
            output_ids=["adder8_out"],
        )
        ops = build_instance_ops([inst])
        assert len(ops) == 1
        assert ops[0]["op"] == "instance"
        assert ops[0]["module"] == "Adder8"
        assert ops[0]["id"] == ["adder8_out"]
        assert ops[0]["args"] == {"a": "x", "b": "y"}

    def test_make_stub_dicts(self):
        inst = InstanceCall(
            name=None,
            target_name="Adder8",
            target_ports=[
                PortInfo("a", 8, "input"),
                PortInfo("b", 8, "input"),
                PortInfo("out", 8, "output"),
            ],
            input_map={"a": "x", "b": "y"},
            output_ids=["adder8_out"],
        )
        stubs = build_stub_modules([inst])
        assert len(stubs) == 1
        stub = stubs[0]
        assert stub["name"] == "Adder8"
        assert "a" in stub["ports"]
        assert "out" in stub["ports"]
        assert stub["body"][-1]["op"] == "output"

    def test_make_stub_deduplicates(self):
        inst1 = InstanceCall(
            name=None,
            target_name="Adder8",
            target_ports=[
                PortInfo("a", 8, "input"),
                PortInfo("b", 8, "input"),
                PortInfo("out", 8, "output"),
            ],
            input_map={"a": "x", "b": "y"},
            output_ids=["adder8_out"],
        )
        inst2 = InstanceCall(
            name=None,
            target_name="Adder8",
            target_ports=inst1.target_ports,
            input_map={"a": "p", "b": "q"},
            output_ids=["adder8_out_1"],
        )
        stubs = build_stub_modules([inst1, inst2])
        assert len(stubs) == 1


class TestLLMModelConfig:
    def test_resolve_generic_llm_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4.1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")

        config = resolve_llm_model_config()

        assert config is not None
        assert config.provider == "openai"
        assert config.model == "gpt-4.1"
        assert config.api_key == "test-key"
        assert config.base_url == "https://example.com/v1"

    def test_resolve_provider_prefixes_bare_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "qwen3.6-plus")

        config = resolve_llm_model_config()

        assert config is not None
        assert config.provider == "openai"
        assert config.model == "qwen3.6-plus"

    def test_resolve_provider_specific_env(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        config = resolve_llm_model_config()

        assert config is not None
        assert config.provider == "deepseek"
        assert config.model == "deepseek-chat"
        assert config.api_key == "deepseek-key"
        assert config.base_url == "https://api.deepseek.com"

    def test_model_prefix_conflict_is_rejected(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "openai/gpt-4.1")

        with pytest.raises(ValueError, match="conflicts"):
            resolve_llm_model_config()

    def test_bare_model_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_MODEL", "gpt-4.1")

        config = resolve_llm_model_config()

        assert config is not None
        assert config.provider == "openai"
        assert config.model == "gpt-4.1"


class TestLLMGenerationKwargs:
    def test_default_generation_kwargs_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("LLM_TOP_P", raising=False)
        monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

        assert resolve_llm_generation_kwargs() == {}

    def test_generation_kwargs_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_TEMPERATURE", "1")
        monkeypatch.setenv("LLM_TOP_P", "0.95")
        monkeypatch.setenv("LLM_REASONING_EFFORT", "none")

        assert resolve_llm_generation_kwargs() == {
            "temperature": 1.0,
            "top_p": 0.95,
            "reasoning_effort": "none",
        }
