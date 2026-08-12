"""Compatibility frontend for the isolated CPPL agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..agents.models import (
    AgentConfig,
    CompilationReport,
    ModuleCompileReport,
)
from ..agents.runtime import CompilationCoordinator
from ..ir.errors import CircuitPPLError
from .module import ModuleDef


class CompilationError(CircuitPPLError):
    """Raised when agent-backed module compilation cannot complete."""


@dataclass
class CompileResult:
    """Outcome of compiling a single CPPL module."""

    module_dict: Optional[dict] = None
    success: bool = False
    error: Optional[str] = None
    attempts: int = 0
    report: Optional[ModuleCompileReport] = None


class CompilerSession:
    """Compatibility wrapper; every module now receives an isolated context."""

    def __init__(self, agent_config: AgentConfig | None = None) -> None:
        self.agent_config = agent_config
        self.last_report: CompilationReport | None = None

    def compile(self, mod: ModuleDef, max_retries: int = 3) -> dict:
        report = CompilationCoordinator(self.agent_config).compile(
            [mod],
            max_retries=max_retries,
        )
        self.last_report = report
        module_report = report.module_reports[mod.name]
        if not report.success or module_report.module_dict is None:
            raise CompilationError(
                module_report.error
                or report.design_error
                or "unknown compilation error"
            )
        return module_report.module_dict


def result_from_report(report: ModuleCompileReport) -> CompileResult:
    return CompileResult(
        module_dict=report.module_dict,
        success=report.success,
        error=report.error,
        attempts=report.attempts,
        report=report,
    )


def compile_module(
    mod: ModuleDef,
    max_retries: int = 3,
    agent_config: AgentConfig | None = None,
) -> CompileResult:
    """Compile one module using a bounded, isolated Generator/Repair worker."""
    try:
        report = CompilationCoordinator(agent_config).compile(
            [mod],
            max_retries=max_retries,
        )
        return result_from_report(report.module_reports[mod.name])
    except Exception as exc:
        return CompileResult(
            success=False,
            error=str(exc),
            attempts=0,
        )


def compile_modules_with_report(
    mods: List[ModuleDef],
    max_retries: int = 3,
    agent_config: AgentConfig | None = None,
) -> CompilationReport:
    """Compile a module DAG and return detailed non-sensitive diagnostics."""
    return CompilationCoordinator(agent_config).compile(
        mods,
        max_retries=max_retries,
    )


def compile_modules(
    mods: List[ModuleDef],
    max_retries: int = 3,
    agent_config: AgentConfig | None = None,
) -> List[CompileResult]:
    """Compatibility API returning one result per input module."""
    try:
        compilation = compile_modules_with_report(
            mods,
            max_retries=max_retries,
            agent_config=agent_config,
        )
        return [
            result_from_report(compilation.module_reports[mod.name]) for mod in mods
        ]
    except Exception as exc:
        return [CompileResult(success=False, error=str(exc), attempts=0) for _ in mods]
