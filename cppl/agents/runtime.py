"""LangGraph coordinator for the complete CPPL agent system."""

from __future__ import annotations

import threading
import time
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Iterable, TypedDict

from langgraph.graph import END, START, StateGraph

from .backend import LLMBackend, LangChainBackend
from .cache import ModuleCache
from .checkpoint import RunCheckpointStore
from .models import (
    AgentConfig,
    CompiledModuleArtifact,
    CompileOptions,
    CompilationReport,
    ModuleCompileReport,
    ResolvedAgentConfig,
)
from .worker import ModuleAgentGraph, ModuleAgentState, build_stub_modules
from .hierarchy import (
    HierarchyPlan,
    build_hierarchy_plan,
    collect_blocked_modules,
    collect_dependency_closure,
)
from ..frontend.module import ModuleDef
from ..harness import (
    CompileEventEmitter,
    CompileObserver,
    NullCompileObserver,
    TerminalCompileUI,
)
from ..ir.infer import infer_widths
from ..ir.errors import issues_from_errors
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
    hierarchy: HierarchyPlan
    artifacts: dict[str, CompiledModuleArtifact]
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
        checkpoints: RunCheckpointStore,
        run_key: str,
        resume: bool,
    ):
        def initialize(state: CompilationState) -> dict:
            modules = state["modules"]
            started = time.monotonic()
            events.emit(
                "design_start",
                stage="initialize",
                message=f"compiling {len(modules)} module(s)",
                metadata={"module_count": len(modules)},
            )
            try:
                hierarchy = build_hierarchy_plan(modules)
            except ValueError as exc:
                reports = {
                    mod.name: ModuleCompileReport(
                        module_name=mod.name,
                        status="blocked",
                        error_category="HierarchyError",
                        error=str(exc),
                        failure_origin="hierarchy",
                    )
                    for mod in modules
                }
                result = CompilationReport(
                    modules=[],
                    module_reports=reports,
                    duration_seconds=time.monotonic() - started,
                    llm_calls=0,
                    cache_hits=0,
                    design_error=str(exc),
                    run_id=run_key,
                )
                events.emit(
                    "design_failed",
                    stage="initialize",
                    message=str(exc),
                )
                return {
                    "started": started,
                    "reports": reports,
                    "result": result,
                    "route": "done",
                }
            reports = {
                module.name: ModuleCompileReport(
                    module_name=module.name,
                    hierarchy_depth=hierarchy.depths[module.name],
                    direct_dependencies=sorted(hierarchy.dependencies[module.name]),
                )
                for module in modules
            }
            return {
                "started": started,
                "hierarchy": hierarchy,
                "artifacts": {},
                "reports": reports,
                "route": "compile",
            }

        def compile_hierarchy(state: CompilationState) -> dict:
            hierarchy = state["hierarchy"]
            reports = dict(state["reports"])
            artifacts = dict(state.get("artifacts", {}))
            pending = set(hierarchy.by_name)
            failed: set[str] = set()
            running: dict[Future, tuple[str, ModuleAgentState, list[CompiledModuleArtifact]]] = {}

            def build_input(name: str) -> tuple[ModuleAgentState, list[CompiledModuleArtifact]]:
                direct_artifacts = [
                    artifacts[dependency]
                    for dependency in sorted(hierarchy.dependencies[name])
                ]
                closure = collect_dependency_closure(name, hierarchy)
                validation_modules = [
                    artifacts[module.name].module_dict
                    for module in hierarchy.modules
                    if module.name in closure
                ]
                direct_identities = [
                    {
                        "name": artifact.module_name,
                        "artifact_hash": artifact.semantic_hash,
                        "interface_hash": artifact.interface_hash,
                    }
                    for artifact in direct_artifacts
                ]
                return (
                    {
                        "mod": hierarchy.by_name[name],
                        "dependency_modules": validation_modules,
                        "cache_dependency_modules": direct_identities,
                        "missing_pattern_dependencies": find_missing_pattern_dependencies(
                            hierarchy.by_name[name], set(hierarchy.by_name)
                        ),
                        "hierarchy_depth": hierarchy.depths[name],
                        "direct_dependencies": tuple(
                            sorted(hierarchy.dependencies[name])
                        ),
                        "dependency_hashes": {
                            artifact.module_name: artifact.semantic_hash
                            for artifact in direct_artifacts
                        },
                    },
                    direct_artifacts,
                )

            def compile_one(module_input: ModuleAgentState) -> ModuleAgentState:
                name = module_input["mod"].name
                semantic_key = cache.key_for(
                    module_input["mod"],
                    module_input.get("cache_dependency_modules"),
                )
                saved = checkpoints.load(run_key, name, semantic_key) if resume else None
                if saved and saved.get("status") == "success":
                    cached_module = cache.load(
                        semantic_key,
                        lambda value: module_graph._validate_candidate(
                            module_input, value
                        ),
                    )
                    if cached_module is not None:
                        return {
                            "report": ModuleCompileReport(
                                module_name=name,
                                status="success",
                                cache_hit=True,
                                resumed=True,
                                cache_key=semantic_key,
                                module_dict=cached_module,
                                patterns_checked=int(saved.get("patterns_checked", 0)),
                                hierarchy_depth=module_input.get("hierarchy_depth", 0),
                                direct_dependencies=list(
                                    module_input.get("direct_dependencies", ())
                                ),
                                dependency_hashes=dict(
                                    module_input.get("dependency_hashes", {})
                                ),
                                validation_stage="published",
                            )
                        }
                return module_graph.invoke(
                    module_input,
                    config={
                        # Static repair calls are bounded by time/token budgets rather
                        # than max_retries, so the graph recursion ceiling must not
                        # become an accidental syntax-retry limit.
                        "recursion_limit": 10000
                    },
                )

            with ThreadPoolExecutor(max_workers=self.config.max_parallelism) as pool:
                while pending or running:
                    blocked = collect_blocked_modules(failed, hierarchy)
                    for name in sorted(pending & blocked):
                        report = reports[name]
                        report.status = "blocked"
                        report.error_category = "DependencyFailure"
                        report.error = "Blocked by failed dependency: " + ", ".join(
                            sorted(hierarchy.dependencies[name] & (failed | blocked))
                        )
                        report.failure_origin = "dependency"
                        report.validation_stage = "blocked"
                        pending.remove(name)
                        events.emit(
                            "module_blocked",
                            module_name=name,
                            stage="blocked",
                            message=report.error,
                        )

                    if failed and self.config.fail_fast:
                        for name in sorted(pending):
                            report = reports[name]
                            if name in blocked:
                                report.status = "blocked"
                                report.error_category = "DependencyFailure"
                                report.error = "Blocked by a failed module dependency"
                                report.failure_origin = "dependency"
                                event_kind = "module_blocked"
                            else:
                                report.status = "cancelled"
                                report.error_category = "FailFastCancellation"
                                report.error = "Cancelled after another module failed"
                                event_kind = "module_cancelled"
                            report.validation_stage = report.status
                            events.emit(
                                event_kind,
                                module_name=name,
                                stage=report.status,
                                message=report.error,
                            )
                        pending.clear()

                    available = self.config.max_parallelism - len(running)
                    if available > 0 and not (failed and self.config.fail_fast):
                        ready = sorted(
                            (
                                name
                                for name in pending
                                if hierarchy.dependencies[name] <= set(artifacts)
                            ),
                            key=lambda name: (hierarchy.depths[name], name),
                        )[:available]
                        if ready:
                            events.emit(
                                "wave_start",
                                stage="dispatch",
                                message="ready: " + ", ".join(ready),
                                metadata={"modules": tuple(ready)},
                            )
                        for name in ready:
                            module_input, direct_artifacts = build_input(name)
                            pending.remove(name)
                            future = pool.submit(compile_one, module_input)
                            running[future] = (name, module_input, direct_artifacts)

                    if not running:
                        if pending:
                            for name in sorted(pending):
                                report = reports[name]
                                report.status = "blocked"
                                report.error_category = "HierarchySchedulingError"
                                report.error = "No compilable dependency-ready module remains"
                                report.failure_origin = "hierarchy"
                                report.validation_stage = "blocked"
                                failed.add(name)
                            pending.clear()
                        continue

                    done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                    for future in done:
                        name, module_input, direct_artifacts = running.pop(future)
                        try:
                            result = future.result()
                            report = result["report"]
                        except Exception as exc:
                            report = reports[name]
                            report.status = "failed"
                            report.error_category = type(exc).__name__
                            report.error = str(exc)
                            report.failure_origin = "runtime"
                            report.validation_stage = "failed"
                        reports[name] = report
                        if report.success and report.module_dict is not None:
                            artifact = cache.artifact_for(
                                report.module_dict,
                                direct_artifacts,
                                patterns_checked=report.patterns_checked,
                            )
                            artifacts[name] = artifact
                            report.artifact_hash = artifact.semantic_hash
                            report.interface_hash = artifact.interface_hash
                        else:
                            failed.add(name)
                        if report.cache_key is not None:
                            checkpoints.save(
                                run_key,
                                name,
                                report.cache_key,
                                report.status,
                                {
                                    "status": report.status,
                                    "patterns_checked": report.patterns_checked,
                                    "attempts": report.attempts,
                                    "simulation_attempts": report.simulation_attempts,
                                    "static_repairs": report.static_repairs,
                                    "model_turns": report.model_turns,
                                    "tool_calls": report.tool_calls,
                                    "tool_failures": report.tool_failures,
                                    "tool_counts": report.tool_counts,
                                    "tool_fallback_reason": report.tool_fallback_reason,
                                    "artifact_hash": report.artifact_hash,
                                    "dependency_hashes": report.dependency_hashes,
                                    "error_category": report.error_category,
                                    "diagnostics": report.diagnostics,
                                    "failure_fingerprints": report.failure_fingerprints,
                                },
                            )
            return {
                "artifacts": artifacts,
                "reports": reports,
                "route": "finalize",
            }

        def finalize(state: CompilationState) -> dict:
            modules = state["modules"]
            reports = state["reports"]
            hierarchy = state["hierarchy"]
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
                        if inst.target_name not in hierarchy.by_name
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
                            reports[mod.name].diagnostics = [
                                issue.as_dict()
                                for issue in issues_from_errors([exc])
                            ]
                            reports[mod.name].module_dict = None
                            reports[mod.name].failure_origin = "hierarchy"
                            reports[mod.name].validation_stage = "final_design_validation"
                            events.emit(
                                "module_failed",
                                module_name=mod.name,
                                stage="final_validation",
                                message=f"{type(exc).__name__}: {str(exc)[:240]}",
                                metadata={"category": type(exc).__name__},
                            )
                            raise
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
                    report.model_turns
                    for report in reports.values()
                ),
                cache_hits=sum(1 for report in reports.values() if report.cache_hit),
                design_error=design_error,
                run_id=run_key,
                resumed_modules=sum(1 for report in reports.values() if report.resumed),
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
        graph.add_node("compile_hierarchy", compile_hierarchy)
        graph.add_node("validate_complete_design", finalize)
        graph.add_edge(START, "initialize")
        graph.add_conditional_edges(
            "initialize", route, {"compile": "compile_hierarchy", "done": END}
        )
        graph.add_edge("compile_hierarchy", "validate_complete_design")
        graph.add_edge("validate_complete_design", END)
        return graph.compile()

    def compile(
        self,
        mods: Iterable[ModuleDef],
        *,
        max_retries: int = 3,
        options: CompileOptions | None = None,
    ) -> CompilationReport:
        options = options or CompileOptions()
        if (
            options.max_simulation_attempts is not None
            and options.max_semantic_attempts is not None
            and options.max_simulation_attempts != options.max_semantic_attempts
        ):
            raise ValueError(
                "max_simulation_attempts and legacy max_semantic_attempts disagree"
            )
        configured_simulation_attempts = (
            options.max_simulation_attempts
            if options.max_simulation_attempts is not None
            else options.max_semantic_attempts
        )
        if configured_simulation_attempts is not None:
            max_retries = configured_simulation_attempts
        if max_retries <= 0:
            raise ValueError("max_retries/max_simulation_attempts must be positive")
        if options.module_deadline_seconds is not None:
            object.__setattr__(
                self.config, "module_deadline_seconds", options.module_deadline_seconds
            )
        modules = list(mods)
        identity = self._model_identity
        if identity is None:
            identity = LangChainBackend.peek_model_identity()
        cache = ModuleCache(self.config, identity)
        design_payload = {
            "modules": [cache.build_module_contract(mod) for mod in modules],
            "model": identity,
        }
        design_hash = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        run_key = f"{options.run_id or 'design'}:{design_hash}"
        checkpoints = RunCheckpointStore(
            self.config.checkpoint_path, self.config.checkpoint_enabled
        )
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
        graph = self._build_graph(
            module_graph,
            cache,
            events,
            checkpoints,
            run_key,
            options.resume,
        )
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
