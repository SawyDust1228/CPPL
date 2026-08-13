"""CPPL — a Python DSL frontend for hardware description via LLM compilation."""

from .frontend.types import Clock, In, Out
from .frontend.module import module, ModuleDef, PortInfo
from .frontend.patterns import Case, MemoryFixture, Sequence, Step
from .design import Design
from .frontend.compiler import compile_module, CompileResult, CompilationError
from .agents import (
    AgentConfig,
    CompileOptions,
    CompiledModuleArtifact,
    CompilationReport,
    ModuleCompileReport,
)
from .harness import CompileEvent, CompileObserver, TerminalCompileUI
from .ir.interpreter import Interpreter
from .ir.errors import PatternDependencyMissing, PatternMismatch, SimulationError

__all__ = [
    "In",
    "Out",
    "Clock",
    "module",
    "ModuleDef",
    "PortInfo",
    "Case",
    "MemoryFixture",
    "Sequence",
    "Step",
    "Design",
    "compile_module",
    "CompileResult",
    "CompilationError",
    "AgentConfig",
    "CompileOptions",
    "CompilationReport",
    "CompiledModuleArtifact",
    "ModuleCompileReport",
    "CompileEvent",
    "CompileObserver",
    "TerminalCompileUI",
    "Interpreter",
    "SimulationError",
    "PatternMismatch",
    "PatternDependencyMissing",
]
