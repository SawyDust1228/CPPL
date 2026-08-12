"""Frontend: DSL definition and LLM compilation."""

from .types import Clock, In, Out
from .module import module, ModuleDef, PortInfo, InstanceCall
from .patterns import Case, Sequence, Step
from .compiler import (
    CompilationError,
    CompileResult,
    compile_module,
    compile_modules_with_report,
)

__all__ = [
    "Clock", "In", "Out", "module", "ModuleDef", "PortInfo", "InstanceCall",
    "Case", "Sequence", "Step", "CompilationError", "CompileResult",
    "compile_module", "compile_modules_with_report",
]
