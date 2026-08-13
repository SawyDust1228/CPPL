"""Compilation event and terminal UI tests."""

import io
import json
from concurrent.futures import ThreadPoolExecutor

from cppl import Design, In, Out, module
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
