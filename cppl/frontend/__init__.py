"""Frontend: DSL definition and LLM compilation."""

from .types import Clock, In, Out
from .module import module, ModuleDef, PortInfo, InstanceCall
from .compiler import compile_module, CompileResult, CompilationError
