"""LangGraph coordinator for the complete CPPL agent system."""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, TypedDict

from langgraph.graph import END, START, StateGraph

from .backend import LLMBackend, LangChainBackend
from .cache import ModuleCache
from .models import (
    AgentConfig,
    CompilationReport,
    ModuleCompileReport,
    ResolvedAgentConfig,
)
from .worker import ModuleAgentGraph, ModuleAgentState, build_stub_modules
from ..frontend.module import ModuleDef
from ..harness import (
    CompileEventEmitter,
    CompileObserver,
    NullCompileObserver,
    TerminalCompileUI,
)
from ..ir.infer import infer_widths
from ..ir.patterns import run_patterns
from ..ir.parser import parse_design
from ..ir.validator import validate_design


class _LazyBackend:
    """Avoid constructing a provider client when all modules hit the cache."""

    def __init__(
        self,
        config: ResolvedAgentConfig,
        factory: Callable[[ResolvedAgentConfig], LLMBackend],
    ) -> None:
        self.config = config
        self.factory = factory
        self._backend: LLMBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> LLMBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                self._backend = self.factory(self.config)
        return self._backend


def build_dependency_map(mods: list[ModuleDef]) -> dict[str, set[str]]:
    known_names = {mod.name for mod in mods}
    return {
        mod.name: {
            inst.target_name
            for inst in mod.instances
            if inst.target_name in known_names
        }
        for mod in mods
    }


def build_dependent_map(dependencies: dict[str, set[str]]) -> dict[str, set[str]]:
    result = {name: set() for name in dependencies}
    for name, required in dependencies.items():
        for dependency in required:
            result[dependency].add(name)
    return result


def find_blocked_modules(
    failed: set[str],
    dependents: dict[str, set[str]],
) -> set[str]:
    blocked: set[str] = set()
    pending = list(failed)
    while pending:
        current = pending.pop()
        for dependent in dependents.get(current, ()):
            if dependent not in blocked and dependent not in failed:
                blocked.add(dependent)
                pending.append(dependent)
    return blocked


def collect_transitive_dependencies(
    name: str,
    dependencies: dict[str, set[str]],
) -> set[str]:
    result: set[str] = set()
    pending = list(dependencies[name])
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(dependencies[dependency])
    return result


def find_missing_pattern_dependencies(
    mod: ModuleDef,
    known_names: set[str],
) -> tuple[str, ...]:
    if not mod.patterns:
        return ()
    missing: set[str] = set()
    visited: set[str] = set()

    def visit(current: ModuleDef) -> None:
        if current.name in visited:
            return
        visited.add(current.name)
        for instance in current.instances:
            if instance.target_name not in known_names:
                missing.add(instance.target_name)
            elif instance.target_mod is not None:
                visit(instance.target_mod)

    visit(mod)
    return tuple(sorted(missing))


class CompilationState(TypedDict, total=False):
    modules: list[ModuleDef]
    by_name: dict[str, ModuleDef]
    dependencies: dict[str, set[str]]
    dependents: dict[str, set[str]]
    pending: set[str]
    completed: set[str]
    failed: set[str]
    reports: dict[str, ModuleCompileReport]
    started: float
    route: str
    result: CompilationReport


class CompilationCoordinator:
    """Compile a module dependency DAG using nested LangGraph workflows."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        backend_factory: Callable[[ResolvedAgentConfig], LLMBackend] | None = None,
        model_identity: dict | None = None,
        observer: CompileObserver | None = None,
    ) -> None:
        self.config = (config or AgentConfig()).resolve()
        self._backend_factory = backend_factory or LangChainBackend
        self._model_identity = model_identity
        self._observer = observer

    def _build_graph(
        self,
        module_graph: ModuleAgentGraph,
        cache: ModuleCache,
        events: CompileEventEmitter,
    ):
        def initialize(state: CompilationState) -> dict:
            modules = state["modules"]
            by_name = {mod.name: mod for mod in modules}
            reports = {
                mod.name: ModuleCompileReport(module_name=mod.name) for mod in modules
            }
            started = time.monotonic()
            events.emit(
                "design_start",
                stage="initialize",
                message=f"compiling {len(modules)} module(s)",
                metadata={"module_count": len(modules)},
            )
            if len(by_name) != len(modules):
                result = CompilationReport(
                    modules=[],
                    module_reports=reports,
                    duration_seconds=time.monotonic() - started,
                    llm_calls=0,
                    cache_hits=0,
                    design_error="Duplicate ModuleDef names are not supported",
                )
                events.emit(
                    "design_failed",
                    stage="initialize",
                    message=result.design_error or "duplicate module names",
                )
                return {"started": started, "reports": reports, "result": result, "route": "done"}
            dependencies = build_dependency_map(modules)
            return {
                "started": started,
                "by_name": by_name,
                "dependencies": dependencies,
                "dependents": build_dependent_map(dependencies),
                "pending": set(by_name),
                "completed": set(),
                "failed": set(),
                "reports": reports,
                "route": "dispatch",
            }

        def dispatch_ready(state: CompilationState) -> dict:
            pending = set(state["pending"])
            completed = set(state["completed"])
            failed = set(state["failed"])
            reports = dict(state["reports"])
            blocked = find_blocked_modules(failed, state["dependents"])
            for name in sorted(pending & blocked):
                reports[name].status = "blocked"
                reports[name].error_category = "DependencyFailure"
                failed_dependencies = sorted(
                    state["dependencies"][name] & (failed | blocked)
                )
                reports[name].error = "Blocked by failed dependency: " + ", ".join(
                    failed_dependencies
                )
                events.emit(
                    "module_blocked",
                    module_name=name,
                    stage="blocked",
                    message=reports[name].error or "blocked by failed dependency",
                )
                pending.remove(name)

            if not pending:
                return {
                    "pending": pending,
                    "completed": completed,
                    "failed": failed,
                    "reports": reports,
                    "route": "finalize",
                }

            ready = sorted(
                name
                for name in pending
                if state["dependencies"][name] <= completed
            )
            if not ready:
                cycle = ", ".join(sorted(pending))
                for name in pending:
                    reports[name].status = "blocked"
                    reports[name].error_category = "DependencyCycle"
                    reports[name].error = (
                        "Module dependency graph contains a cycle involving: " + cycle
                    )
                    events.emit(
                        "module_blocked",
                        module_name=name,
                        stage="blocked",
                        message=reports[name].error or "dependency cycle",
                    )
                return {
                    "pending": set(),
                    "completed": completed,
                    "failed": failed | pending,
                    "reports": reports,
                    "route": "finalize",
                }

            inputs: list[ModuleAgentState] = []
            events.emit(
                "wave_start",
                stage="dispatch",
                message="ready: " + ", ".join(ready),
                metadata={"modules": tuple(ready)},
            )
            for name in ready:
                dependency_names = collect_transitive_dependencies(
                    name, state["dependencies"]
                )
                inputs.append(
                    {
                        "mod": state["by_name"][name],
                        "dependency_modules": [
                            reports[mod.name].module_dict
                            for mod in state["modules"]
                            if mod.name in dependency_names
                            and reports[mod.name].module_dict is not None
                        ],
                        "missing_pattern_dependencies": find_missing_pattern_dependencies(
                            state["by_name"][name], set(state["by_name"])
                        ),
                    }
                )

            results = module_graph.batch(
                inputs,
                config={
                    "max_concurrency": self.config.max_parallelism,
                    "recursion_limit": max(25, module_graph.max_retries * 3 + 10),
                },
            )
            for name, result in zip(ready, results):
                pending.remove(name)
                reports[name] = result["report"]
                if reports[name].success:
                    completed.add(name)
                else:
                    failed.add(name)

            if failed and self.config.fail_fast:
                blocked = find_blocked_modules(failed, state["dependents"])
                for name in sorted(pending):
                    if name in blocked:
                        reports[name].status = "blocked"
                        reports[name].error_category = "DependencyFailure"
                        reports[name].error = "Blocked by a failed module dependency"
                        events.emit(
                            "module_blocked",
                            module_name=name,
                            stage="blocked",
                            message=reports[name].error,
                        )
                    else:
                        reports[name].status = "cancelled"
                        reports[name].error_category = "FailFastCancellation"
                        reports[name].error = "Cancelled after another module failed"
                        events.emit(
                            "module_cancelled",
                            module_name=name,
                            stage="cancelled",
                            message=reports[name].error,
                        )
                pending.clear()
                return {
                    "pending": pending,
                    "completed": completed,
                    "failed": failed,
                    "reports": reports,
                    "route": "finalize",
                }
            return {
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "reports": reports,
                "route": "dispatch" if pending else "finalize",
            }

        def finalize(state: CompilationState) -> dict:
            modules = state["modules"]
            reports = state["reports"]
            by_name = state["by_name"]
            ordered = [
                reports[mod.name].module_dict
                for mod in modules
                if reports[mod.name].success
                and reports[mod.name].module_dict is not None
            ]
            design_error = None
            if all(reports[mod.name].success for mod in modules):
                try:
                    events.emit(
                        "design_validate",
                        stage="final_validation",
                        message="validating complete design",
                    )
                    external_instances = [
                        inst
                        for mod in modules
                        for inst in mod.instances
                        if inst.target_name not in by_name
                    ]
                    parsed = parse_design(build_stub_modules(external_instances) + ordered)
                    validate_design(parsed)
                    widths = infer_widths(parsed)
                    for mod in modules:
                        try:
                            reports[mod.name].patterns_checked = run_patterns(
                                parsed, mod.name, mod.patterns, widths=widths
                            )
                        except Exception as exc:
                            reports[mod.name].status = "failed"
                            reports[mod.name].error_category = type(exc).__name__
                            reports[mod.name].error = str(exc)
                            reports[mod.name].module_dict = None
                            events.emit(
                                "module_failed",
                                module_name=mod.name,
                                stage="final_validation",
                                message=f"{type(exc).__name__}: {str(exc)[:240]}",
                                metadata={"category": type(exc).__name__},
                            )
                            raise
                    for report in reports.values():
                        if (
                            not report.cache_hit
                            and report.cache_key is not None
                            and report.module_dict is not None
                        ):
                            cache.store(report.cache_key, report.module_dict)
                except Exception as exc:
                    design_error = f"{type(exc).__name__}: {exc}"
            else:
                failed_names = [
                    name for name, report in reports.items() if not report.success
                ]
                design_error = "Compilation failed for: " + ", ".join(failed_names)
            result = CompilationReport(
                modules=ordered if design_error is None else [],
                module_reports=reports,
                duration_seconds=time.monotonic() - state["started"],
                llm_calls=sum(
                    report.attempts + report.compression_calls
                    for report in reports.values()
                ),
                cache_hits=sum(1 for report in reports.values() if report.cache_hit),
                design_error=design_error,
            )
            if result.success:
                events.emit(
                    "design_success",
                    stage="complete",
                    message=(
                        f"{len(result.modules)} module(s), {result.llm_calls} LLM call(s), "
                        f"{result.cache_hits} cache hit(s) in {result.duration_seconds:.2f}s"
                    ),
                    metadata={
                        "module_count": len(result.modules),
                        "llm_calls": result.llm_calls,
                        "cache_hits": result.cache_hits,
                    },
                )
            else:
                events.emit(
                    "design_failed",
                    stage="complete",
                    message=result.design_error or "compilation failed",
                    metadata={"failed_modules": tuple(result.failures)},
                )
            return {"reports": reports, "result": result, "route": "done"}

        def route(state: CompilationState) -> str:
            return state["route"]

        graph = StateGraph(CompilationState)
        graph.add_node("initialize", initialize)
        graph.add_node("dispatch_ready_modules", dispatch_ready)
        graph.add_node("validate_complete_design", finalize)
        graph.add_edge(START, "initialize")
        graph.add_conditional_edges(
            "initialize", route, {"dispatch": "dispatch_ready_modules", "done": END}
        )
        graph.add_conditional_edges(
            "dispatch_ready_modules",
            route,
            {
                "dispatch": "dispatch_ready_modules",
                "finalize": "validate_complete_design",
            },
        )
        graph.add_edge("validate_complete_design", END)
        return graph.compile()

    def compile(
        self,
        mods: Iterable[ModuleDef],
        *,
        max_retries: int = 3,
    ) -> CompilationReport:
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        modules = list(mods)
        identity = self._model_identity
        if identity is None:
            identity = LangChainBackend.peek_model_identity()
        cache = ModuleCache(self.config, identity)
        observer = self._observer
        if observer is None:
            observer = TerminalCompileUI() if self.config.log_enabled else NullCompileObserver()
        events = CompileEventEmitter(observer)
        backend = _LazyBackend(self.config, self._backend_factory)
        module_graph = ModuleAgentGraph(
            self.config,
            backend.get,
            cache,
            max_retries,
            events,
        )
        graph = self._build_graph(module_graph, cache, events)
        try:
            final_state = graph.invoke(
                {"modules": modules},
                config={
                    "max_concurrency": self.config.max_parallelism,
                    "recursion_limit": max(25, len(modules) * 3 + 10),
                },
            )
        except Exception as exc:
            events.emit(
                "design_failed",
                stage="runtime",
                message=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
            raise
        return final_state["result"]
