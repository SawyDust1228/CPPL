"""Compilation event and terminal UI tests."""

import io
import json
from concurrent.futures import ThreadPoolExecutor

from cppl import Case, Design, In, Out, module
from cppl.agents.backend import BackendResponse
from cppl.agents.models import AgentConfig
from cppl.agents.models import CompilationReport
from cppl.agents.runtime import CompilationCoordinator
from cppl.harness import CompileEvent, CompileEventEmitter, TerminalCompileUI


class FakeBackend:
    responses: list[str] = []

    def __init__(self, config):
        pass

    def generate(self, system_prompt, user_prompt, *, output_tokens=None):
        return BackendResponse(self.__class__.responses.pop(0), 0)


class RecordingObserver:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_terminal_ui_formats_plain_stderr_line():
    stream = io.StringIO()
    ui = TerminalCompileUI(stream, color=False)

    ui.on_event(
        CompileEvent(
            kind="generate",
            elapsed_seconds=1.25,
            module_name="Adder8",
            attempt=1,
            message="attempt 1/3",
        )
    )

    assert stream.getvalue() == (
        "[CPPL    1.25s] GENERATE Adder8 — attempt 1/3\n"
    )
    assert "\x1b[" not in stream.getvalue()


def test_terminal_ui_respects_no_color(monkeypatch):
    class TTYStream(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = TTYStream()
    ui = TerminalCompileUI(stream)
    ui.on_event(CompileEvent(kind="module_success", elapsed_seconds=0.1))

    assert "\x1b[" not in stream.getvalue()


def test_parallel_event_lines_do_not_interleave():
    stream = io.StringIO()
    emitter = CompileEventEmitter(TerminalCompileUI(stream, color=False))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: emitter.emit(
                    "generate",
                    module_name=f"M{index}",
                    message=f"attempt {index}",
                ),
                range(20),
            )
        )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 20
    assert all(line.startswith("[CPPL ") and "GENERATE M" in line for line in lines)


def test_default_compile_logs_to_stderr(tmp_path, capsys):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    FakeBackend.responses = [json.dumps([{"op": "output", "args": {"out": "a"}}])]
    report = CompilationCoordinator(
        AgentConfig(
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FakeBackend,
        model_identity={"model": "fake"},
    ).compile([M])

    captured = capsys.readouterr()
    assert report.success
    assert captured.out == ""
    assert "START" in captured.err
    assert "GENERATE M" in captured.err
    assert "VALIDATE M" in captured.err
    assert "SUCCESS  M" in captured.err
    assert "DONE" in captured.err


def test_logging_can_be_disabled(tmp_path, capsys):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    FakeBackend.responses = [json.dumps([{"op": "output", "args": {"out": "a"}}])]
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FakeBackend,
        model_identity={"model": "fake"},
    ).compile([M])

    captured = capsys.readouterr()
    assert report.success
    assert captured.out == ""
    assert captured.err == ""


def test_logging_can_be_disabled_from_environment(tmp_path, capsys, monkeypatch):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    monkeypatch.setenv("CPPL_AGENT_LOG_ENABLED", "false")
    FakeBackend.responses = [json.dumps([{"op": "output", "args": {"out": "a"}}])]
    report = CompilationCoordinator(
        AgentConfig(
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FakeBackend,
        model_identity={"model": "fake"},
    ).compile([M])

    captured = capsys.readouterr()
    assert report.success
    assert captured.err == ""


def test_custom_observer_receives_repair_events(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    FakeBackend.responses = [
        "not-json",
        json.dumps([{"op": "output", "args": {"out": "a"}}]),
    ]
    observer = RecordingObserver()
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FakeBackend,
        model_identity={"model": "fake"},
        observer=observer,
    ).compile([M], max_retries=2)

    kinds = [event.kind for event in observer.events]
    assert report.success
    assert "repair_scheduled" in kinds
    assert "repair" in kinds
    assert kinds[-1] == "design_success"


def test_tool_backend_reports_calls_and_emits_safe_events(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    class ToolBackend:
        def __init__(self, config):
            pass

        def generate_with_tools(
            self, system_prompt, user_prompt, tool_session, *, output_tokens=None
        ):
            body = [{"op": "output", "args": {"out": "a"}}]
            assert tool_session.execute("replace_ir", {"body": body})["ok"]
            assert tool_session.execute("validate_ir", {})["ok"]
            assert tool_session.execute("submit_ir", {})["ok"]
            return BackendResponse(
                json.dumps(body),
                0,
                model_turns=2,
                used_tools=True,
            )

    observer = RecordingObserver()
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=ToolBackend,
        model_identity={"model": "tool-fake"},
        observer=observer,
    ).compile([M])

    module_report = report.module_reports["M"]
    assert report.success
    assert report.llm_calls == 2
    assert module_report.model_turns == 2
    assert module_report.tool_calls == 3
    assert module_report.tool_failures == 0
    assert module_report.tool_counts == {
        "replace_ir": 1,
        "validate_ir": 1,
        "submit_ir": 1,
    }
    tool_events = [event for event in observer.events if event.kind.startswith("tool_")]
    assert [event.kind for event in tool_events] == [
        "tool_start",
        "tool_finish",
        "tool_start",
        "tool_finish",
        "tool_start",
        "tool_finish",
    ]
    assert all("body" not in event.message for event in tool_events)


def test_tool_pattern_check_is_not_counted_twice_at_publication(tmp_path):
    @module(patterns=[Case({"a": 7}, {"out": 7})])
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    class PatternToolBackend:
        def __init__(self, config):
            pass

        def generate_with_tools(
            self, system_prompt, user_prompt, tool_session, *, output_tokens=None
        ):
            body = [{"op": "output", "args": {"out": "a"}}]
            assert tool_session.execute("replace_ir", {"body": body})["ok"]
            assert tool_session.execute("run_patterns", {})["ok"]
            assert tool_session.execute("submit_ir", {})["ok"]
            return BackendResponse(json.dumps(body), 0, used_tools=True)

    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=PatternToolBackend,
        model_identity={"model": "pattern-tool-fake"},
    ).compile([M], max_retries=1)

    module_report = report.module_reports["M"]
    assert report.success
    assert module_report.simulation_attempts == 1
    assert module_report.patterns_checked == 1
    assert module_report.tool_counts == {
        "replace_ir": 1,
        "run_patterns": 1,
        "submit_ir": 1,
    }


def test_failed_tool_pattern_result_is_reused_across_repairs(tmp_path):
    @module(patterns=[Case({"a": 7}, {"out": 7})])
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    bad = [{"id": "zero", "op": "constant", "value": 0, "width": 8},
           {"op": "output", "args": {"out": "zero"}}]
    good = [{"op": "output", "args": {"out": "a"}}]

    class RepairingPatternToolBackend:
        calls = 0

        def __init__(self, config):
            pass

        def generate_with_tools(
            self, system_prompt, user_prompt, tool_session, *, output_tokens=None
        ):
            self.__class__.calls += 1
            if self.calls == 1:
                assert tool_session.execute("replace_ir", {"body": bad})["ok"]
                mismatch = tool_session.execute("run_patterns", {})
                assert not mismatch["ok"]
                return BackendResponse(json.dumps(bad), 0, used_tools=True)

            cached = tool_session.execute("run_patterns", {})
            assert not cached["ok"]
            assert tool_session.stats.pattern_attempt_hashes == set()
            assert tool_session.execute("replace_ir", {"body": good})["ok"]
            return BackendResponse(json.dumps(good), 0, used_tools=True)

    RepairingPatternToolBackend.calls = 0
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=RepairingPatternToolBackend,
        model_identity={"model": "pattern-tool-repair-fake"},
    ).compile([M], max_retries=2)

    module_report = report.module_reports["M"]
    assert report.success
    assert module_report.simulation_attempts == 2
    assert RepairingPatternToolBackend.calls == 2


def test_repeated_tool_diagnostic_falls_back_to_direct_generation(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    invalid = json.dumps([{"op": "output", "args": {"wrong": "a"}}])
    valid = json.dumps([{"op": "output", "args": {"out": "a"}}])

    class RepeatingToolBackend:
        tool_calls = 0
        direct_calls = 0

        def __init__(self, config):
            pass

        def generate_with_tools(
            self, system_prompt, user_prompt, tool_session, *, output_tokens=None
        ):
            self.__class__.tool_calls += 1
            return BackendResponse(invalid, 0, used_tools=True)

        def generate_structured(
            self, system_prompt, user_prompt, *, output_tokens=None
        ):
            self.__class__.direct_calls += 1
            return BackendResponse(valid, 0)

    RepeatingToolBackend.tool_calls = 0
    RepeatingToolBackend.direct_calls = 0
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
            tool_mode="auto",
        ),
        backend_factory=RepeatingToolBackend,
        model_identity={"model": "repeating-tool-fake"},
    ).compile([M])

    assert report.success
    assert RepeatingToolBackend.tool_calls == 3
    assert RepeatingToolBackend.direct_calls == 1


def test_observer_errors_do_not_break_compilation(tmp_path):
    @module
    def M(a: In[8]) -> Out[8]:
        """out equals a."""
        pass

    class BrokenObserver:
        def on_event(self, event):
            raise RuntimeError("UI failure")

    FakeBackend.responses = [json.dumps([{"op": "output", "args": {"out": "a"}}])]
    report = CompilationCoordinator(
        AgentConfig(
            log_enabled=False,
            cache_enabled=False,
            cache_dir=tmp_path / "cache",
            max_parallelism=1,
            context_window_tokens=8192,
            output_tokens=2048,
            safety_margin_tokens=512,
        ),
        backend_factory=FakeBackend,
        model_identity={"model": "fake"},
        observer=BrokenObserver(),
    ).compile([M])

    assert report.success


def test_design_forwards_constructor_observer(monkeypatch):
    observer = RecordingObserver()
    design = Design(observer=observer)
    sentinel = CompilationReport(
        modules=[],
        module_reports={},
        duration_seconds=0.0,
        llm_calls=0,
        cache_hits=0,
    )
    received = {}

    def fake_compile(mods, max_retries=3, agent_config=None, *, observer=None):
        received["observer"] = observer
        return sentinel

    monkeypatch.setattr("cppl.design.compile_modules_with_report", fake_compile)

    assert design.compile_with_report() is sentinel
    assert received["observer"] is observer
