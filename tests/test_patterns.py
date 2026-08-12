"""Executable @module pattern and compile/repair loop tests."""

import json

import pytest

from cppl import Case, In, Out, Sequence, Step, module
from cppl.agents.backend import BackendResponse
from cppl.agents.models import AgentConfig
from cppl.agents.runtime import CompilationCoordinator
from cppl.agents.cache import ModuleCache
from cppl.frontend.patterns import pattern_as_dict
from cppl.ir.errors import PatternMismatch
from cppl.ir.infer import infer_widths
from cppl.ir.parser import parse_design
from cppl.ir.patterns import run_patterns


class FakeBackend:
    responses: list[str] = []
    prompts: list[dict] = []

    @classmethod
    def peek_model_identity(cls):
        return {"model": "fake"}

    def __init__(self, config):
        self.config = config

    def generate(self, system_prompt, user_prompt, *, output_tokens=None):
        self.__class__.prompts.append(json.loads(user_prompt))
        if not self.__class__.responses:
            raise AssertionError("Fake backend has no response")
        return BackendResponse(self.__class__.responses.pop(0), 0)


@pytest.fixture(autouse=True)
def reset_fake_backend():
    FakeBackend.responses = []
    FakeBackend.prompts = []


def _config(tmp_path):
    return AgentConfig(
        cache_dir=tmp_path / "cache",
        cache_enabled=True,
        max_parallelism=1,
        context_window_tokens=8192,
        output_tokens=2048,
        safety_margin_tokens=512,
    )


class TestPatternDSL:
    def test_module_accepts_cases_and_sequences(self):
        @module(patterns=[
            Case({"a": -1}, {"out": 0x1FF}, name=" all ones "),
            Sequence([
                Step(inputs={"a": 1}),
                Step(outputs={"out": 1}),
            ]),
        ])
        def M(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        assert [pattern.name for pattern in M.patterns] == [
            "all ones", "sequence[1]",
        ]
        assert dict(M.patterns[0].inputs) == {"a": 0xFF}
        assert dict(M.patterns[0].outputs) == {"out": 0xFF}
        assert pattern_as_dict(M.patterns[1])["kind"] == "sequence"

    def test_bare_module_remains_compatible(self):
        @module
        def M(a: In[1]) -> Out[1]:
            """out equals a."""
            pass

        assert M.patterns == ()

    @pytest.mark.parametrize(
        "patterns,match",
        [
            ([Case({"missing": 0}, {"out": 0})], "unknown port"),
            ([Case({"a": 0}, {})], "at least one output"),
            ([Sequence([])], "at least one step"),
            ([Sequence([Step(inputs={"a": 0})])], "at least one output"),
            ([Case({"a": 0}, {"out": 0}, name="x"),
              Case({"a": 1}, {"out": 1}, name="x")], "Duplicate"),
        ],
    )
    def test_invalid_patterns_fail_during_decoration(self, patterns, match):
        with pytest.raises(ValueError, match=match):
            @module(patterns=patterns)
            def M(a: In[1]) -> Out[1]:
                """out equals a."""
                pass


class TestPatternRunner:
    def test_case_and_sequence_run_from_independent_reset_state(self):
        raw = [{
            "name": "Reg",
            "ports": {
                "clk": {"dir": "input", "width": 1},
                "d": {"dir": "input", "width": 8},
                "q": {"dir": "output", "width": 8},
            },
            "body": [
                {"id": "r", "op": "reg", "args": ["d"], "clock": "clk"},
                {"op": "output", "args": {"q": "r"}},
            ],
        }]
        patterns = (
            Sequence([
                Step({"clk": 0, "d": 7}),
                Step({"clk": 1}, {"q": 7}),
            ], name="capture"),
            Case({"clk": 0}, {"q": 0}, name="fresh_state"),
        )
        modules = parse_design(raw)
        assert run_patterns(modules, "Reg", patterns, widths=infer_widths(modules)) == 2

    def test_mismatch_contains_repair_details(self):
        modules = parse_design([{
            "name": "M",
            "ports": {
                "a": {"dir": "input", "width": 8},
                "out": {"dir": "output", "width": 8},
            },
            "body": [{"op": "output", "args": {"out": "a"}}],
        }])
        with pytest.raises(PatternMismatch) as error:
            run_patterns(
                modules,
                "M",
                [Case({"a": 2}, {"out": 3}, name="plus_one")],
            )
        message = str(error.value)
        assert "plus_one" in message
        assert "step 0" in message
        assert "expected={'out': 3}" in message
        assert "actual={'out': 2}" in message


class TestCompilePatternLoop:
    def test_mismatch_is_repaired_and_patterns_are_in_prompt(self, tmp_path):
        @module(patterns=[Case({"a": 2}, {"out": 3}, name="plus_one")])
        def M(a: In[8]) -> Out[8]:
            """out equals a plus one."""
            pass

        FakeBackend.responses = [
            json.dumps([{"op": "output", "args": {"out": "a"}}]),
            json.dumps([
                {"id": "one", "op": "constant", "value": 1, "width": 8},
                {"id": "sum", "op": "add", "args": ["a", "one"]},
                {"op": "output", "args": {"out": "sum"}},
            ]),
        ]
        report = CompilationCoordinator(
            _config(tmp_path),
            backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        ).compile([M], max_retries=2)

        assert report.success
        assert report.module_reports["M"].attempts == 2
        assert report.module_reports["M"].patterns_checked == 1
        assert FakeBackend.prompts[0]["patterns"][0]["name"] == "plus_one"
        assert any(
            "steps share state" in rule for rule in FakeBackend.prompts[0]["rules"]
        )
        repair = FakeBackend.prompts[1]
        assert repair["task"] == "repair_module_body"
        assert repair["diagnostic"]["category"] == "PatternMismatch"
        assert "expected={'out': 3}" in repair["diagnostic"]["message"]

    def test_exhausted_pattern_mismatch_fails_without_cache(self, tmp_path):
        @module(patterns=[Case({"a": 1}, {"out": 2})])
        def M(a: In[8]) -> Out[8]:
            """out equals a plus one."""
            pass

        bad = json.dumps([{"op": "output", "args": {"out": "a"}}])
        FakeBackend.responses = [bad, bad]
        coordinator = CompilationCoordinator(
            _config(tmp_path),
            backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        )
        report = coordinator.compile([M], max_retries=2)
        module_report = report.module_reports["M"]
        assert not report.success
        assert module_report.error_category == "PatternMismatch"
        assert module_report.module_dict is None
        assert not (tmp_path / "cache").exists()

    def test_cache_hit_revalidates_pattern(self, tmp_path, monkeypatch):
        @module(patterns=[Case({"a": 4}, {"out": 4})])
        def M(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        good = json.dumps([{"op": "output", "args": {"out": "a"}}])
        FakeBackend.responses = [good]
        first = CompilationCoordinator(
            _config(tmp_path), backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        ).compile([M])
        assert first.success

        calls = []
        from cppl.agents import worker as worker_module
        original = worker_module.run_patterns

        def recording_runner(*args, **kwargs):
            calls.append(args[1])
            return original(*args, **kwargs)

        monkeypatch.setattr(worker_module, "run_patterns", recording_runner)
        second = CompilationCoordinator(
            _config(tmp_path), backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        ).compile([M])
        assert second.success
        assert second.module_reports["M"].cache_hit
        assert second.module_reports["M"].patterns_checked == 1
        assert calls == ["M"]

    def test_pattern_changes_cache_identity(self, tmp_path):
        @module(patterns=[Case({"a": 1}, {"out": 1})])
        def A(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        @module(patterns=[Case({"a": 2}, {"out": 2})])
        def B(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        # Equalize the symbol name to isolate pattern identity.
        B.name = A.name
        config = _config(tmp_path).resolve()
        cache = ModuleCache(config, {"model": "fake"})
        assert cache.key_for(A) != cache.key_for(B)

    def test_dependency_ir_changes_parent_cache_identity(self, tmp_path):
        @module(patterns=[Case({"a": 1}, {"out": 1})])
        def Parent(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        dependency_a = [{"name": "Child", "body": [{"version": 1}]}]
        dependency_b = [{"name": "Child", "body": [{"version": 2}]}]
        cache = ModuleCache(_config(tmp_path).resolve(), {"model": "fake"})
        assert cache.key_for(Parent, dependency_a) != cache.key_for(
            Parent, dependency_b
        )

    def test_hierarchical_pattern_uses_real_dependency_ir(self, tmp_path):
        @module
        def Child(a: In[8]) -> Out[8]:
            """out equals a plus one."""
            pass

        @module(patterns=[Case({"x": 2}, {"out": 3})])
        def Parent(x: In[8]) -> Out[8]:
            child = Child(x)
            return f"out equals {child}."

        FakeBackend.responses = [
            json.dumps([
                {"id": "one", "op": "constant", "value": 1, "width": 8},
                {"id": "sum", "op": "add", "args": ["a", "one"]},
                {"op": "output", "args": {"out": "sum"}},
            ]),
            json.dumps([{"op": "output", "args": {"out": "child_out"}}]),
        ]
        report = CompilationCoordinator(
            _config(tmp_path), backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        ).compile([Child, Parent])
        assert report.success
        assert report.module_reports["Parent"].patterns_checked == 1

    def test_missing_real_dependency_fails_before_llm(self, tmp_path):
        @module
        def External(a: In[8]) -> Out[8]:
            """out equals a."""
            pass

        @module(patterns=[Case({"x": 1}, {"out": 1})])
        def Parent(x: In[8]) -> Out[8]:
            child = External(x)
            return f"out equals {child}."

        report = CompilationCoordinator(
            _config(tmp_path), backend_factory=FakeBackend,
            model_identity={"model": "fake"},
        ).compile([Parent])
        module_report = report.module_reports["Parent"]
        assert not report.success
        assert module_report.error_category == "PatternDependencyMissing"
        assert "External" in module_report.error
        assert FakeBackend.prompts == []
