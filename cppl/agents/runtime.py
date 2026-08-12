"""DAG scheduler and compilation coordinator for CPPL module agents."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from .backend import APPLBackend
from .cache import ModuleCache
from .models import (
    AgentConfig,
    CompilationReport,
    ModuleCompileReport,
    ResolvedAgentConfig,
)
from .worker import ModuleWorker, _make_stub_dicts
from ..frontend.module import ModuleDef
from ..ir.infer import infer_widths
from ..ir.parser import parse_design
from ..ir.validator import validate_design


class _LazyBackend:
    """Avoid importing/configuring APPL when every module is a cache hit."""

    def __init__(
        self,
        config: ResolvedAgentConfig,
        factory: Callable[[ResolvedAgentConfig], APPLBackend] = APPLBackend,
    ) -> None:
        self.config = config
        self.factory = factory
        self._backend: APPLBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> APPLBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                self._backend = self.factory(self.config)
        return self._backend


def _dependencies(mods: list[ModuleDef]) -> dict[str, set[str]]:
    known_names = {mod.name for mod in mods}
    dependencies: dict[str, set[str]] = {}
    for mod in mods:
        dependencies[mod.name] = {
            inst.target_name
            for inst in mod.instances
            if inst.target_name in known_names
        }
    return dependencies


def _dependents(dependencies: dict[str, set[str]]) -> dict[str, set[str]]:
    result = {name: set() for name in dependencies}
    for name, required in dependencies.items():
        for dependency in required:
            result[dependency].add(name)
    return result


def _blocked_by_failure(
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


class CompilationCoordinator:
    """Compile a module DAG using isolated workers and bounded wave parallelism."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        backend_factory: Callable[[ResolvedAgentConfig], APPLBackend] | None = None,
        model_identity: dict | None = None,
    ) -> None:
        self.config = (config or AgentConfig()).resolve()
        self._backend_factory = backend_factory or APPLBackend
        self._model_identity = model_identity

    def compile(
        self,
        mods: Iterable[ModuleDef],
        *,
        max_retries: int = 3,
    ) -> CompilationReport:
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        started = time.monotonic()
        modules = list(mods)
        by_name = {mod.name: mod for mod in modules}
        reports = {
            mod.name: ModuleCompileReport(module_name=mod.name)
            for mod in modules
        }
        if len(by_name) != len(modules):
            duplicate_error = "Duplicate ModuleDef names are not supported"
            return CompilationReport(
                modules=[],
                module_reports=reports,
                duration_seconds=time.monotonic() - started,
                llm_calls=0,
                cache_hits=0,
                design_error=duplicate_error,
            )

        dependencies = _dependencies(modules)
        dependents = _dependents(dependencies)
        model_identity = self._model_identity
        if model_identity is None:
            model_identity = APPLBackend.peek_model_identity()
        cache = ModuleCache(self.config, model_identity)
        backend = _LazyBackend(self.config, self._backend_factory)

        pending = set(by_name)
        completed: set[str] = set()
        failed: set[str] = set()

        while pending:
            blocked = _blocked_by_failure(failed, dependents)
            for name in sorted(pending & blocked):
                reports[name].status = "blocked"
                reports[name].error_category = "DependencyFailure"
                failed_dependencies = sorted(dependencies[name] & (failed | blocked))
                reports[name].error = (
                    "Blocked by failed dependency: " + ", ".join(failed_dependencies)
                )
                pending.remove(name)

            if not pending:
                break

            ready = sorted(
                name
                for name in pending
                if dependencies[name] <= completed
            )
            if not ready:
                cycle = ", ".join(sorted(pending))
                for name in pending:
                    reports[name].status = "blocked"
                    reports[name].error_category = "DependencyCycle"
                    reports[name].error = (
                        "Module dependency graph contains a cycle involving: " + cycle
                    )
                pending.clear()
                break

            workers = {
                name: ModuleWorker(
                    by_name[name],
                    self.config,
                    backend.get,
                    cache,
                    max_retries,
                )
                for name in ready
            }
            wave_reports: dict[str, ModuleCompileReport] = {}
            with ThreadPoolExecutor(
                max_workers=min(self.config.max_parallelism, len(ready)),
                thread_name_prefix="cppl-agent",
            ) as executor:
                futures = {
                    executor.submit(worker.run): name
                    for name, worker in workers.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        wave_reports[name] = future.result()
                    except Exception as exc:
                        wave_reports[name] = ModuleCompileReport(
                            module_name=name,
                            status="failed",
                            error_category=type(exc).__name__,
                            error=str(exc),
                        )

            for name in ready:
                pending.remove(name)
                reports[name] = wave_reports[name]
                if reports[name].success:
                    completed.add(name)
                else:
                    failed.add(name)

            if failed and self.config.fail_fast:
                blocked = _blocked_by_failure(failed, dependents)
                for name in sorted(pending):
                    if name in blocked:
                        reports[name].status = "blocked"
                        reports[name].error_category = "DependencyFailure"
                        reports[name].error = "Blocked by a failed module dependency"
                    else:
                        reports[name].status = "cancelled"
                        reports[name].error_category = "FailFastCancellation"
                        reports[name].error = "Cancelled after another module failed"
                pending.clear()

        ordered_module_dicts = [
            reports[mod.name].module_dict
            for mod in modules
            if reports[mod.name].success and reports[mod.name].module_dict is not None
        ]
        design_error = None
        if all(reports[mod.name].success for mod in modules):
            try:
                external_instances = [
                    inst
                    for mod in modules
                    for inst in mod.instances
                    if inst.target_name not in by_name
                ]
                external_stubs = _make_stub_dicts(external_instances)
                parsed = parse_design(external_stubs + ordered_module_dicts)
                validate_design(parsed)
                infer_widths(parsed)
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
                name for name, report in reports.items()
                if not report.success
            ]
            design_error = "Compilation failed for: " + ", ".join(failed_names)

        return CompilationReport(
            modules=ordered_module_dicts if design_error is None else [],
            module_reports=reports,
            duration_seconds=time.monotonic() - started,
            llm_calls=sum(
                report.attempts + report.compression_calls
                for report in reports.values()
            ),
            cache_hits=sum(1 for report in reports.values() if report.cache_hit),
            design_error=design_error,
        )
