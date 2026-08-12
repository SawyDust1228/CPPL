"""CPPL — a Python DSL frontend for hardware description via LLM compilation."""

from .frontend.types import Clock, In, Out
from .frontend.module import module, ModuleDef, PortInfo
from .design import Design
from .frontend.compiler import compile_module, CompileResult, CompilationError
from .agents import AgentConfig, CompilationReport, ModuleCompileReport

__all__ = [
    "In",
    "Out",
    "Clock",
    "module",
    "ModuleDef",
    "PortInfo",
    "Design",
    "compile_module",
    "CompileResult",
    "CompilationError",
    "AgentConfig",
    "CompilationReport",
    "ModuleCompileReport",
]
