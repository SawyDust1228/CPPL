"""CPPL — a Python DSL frontend for hardware description via LLM compilation."""

from .frontend.types import In, Out
from .frontend.module import module, ModuleDef, PortInfo
from .design import Design
from .frontend.compiler import compile_module, CompileResult, CompilationError

__all__ = [
    "In",
    "Out",
    "module",
    "ModuleDef",
    "PortInfo",
    "Design",
    "compile_module",
    "CompileResult",
    "CompilationError",
]
