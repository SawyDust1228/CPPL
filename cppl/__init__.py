"""CPPL — a Python DSL frontend for hardware description via LLM compilation."""

from .frontend.types import Clock, In, Out
from .frontend.module import module, ModuleDef, PortInfo
from .frontend.patterns import Case, Sequence, Step
from .design import Design
from .frontend.compiler import compile_module, CompileResult, CompilationError
from .agents import AgentConfig, CompilationReport, ModuleCompileReport
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
    "Sequence",
    "Step",
    "Design",
    "compile_module",
    "CompileResult",
    "CompilationError",
    "AgentConfig",
    "CompilationReport",
    "ModuleCompileReport",
    "CompileEvent",
    "CompileObserver",
    "TerminalCompileUI",
    "Interpreter",
    "SimulationError",
    "PatternMismatch",
    "PatternDependencyMissing",
]
