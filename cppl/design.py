"""Design orchestrator: collects modules and drives compilation + codegen."""

from __future__ import annotations

import json
from typing import List, Optional

from .ir.parser import parse_design
from .ir.validator import validate_design
from .ir.infer import infer_widths
from .codegen.circt import generate_mlir, generate_verilog

from .frontend.compiler import CompilationError, compile_module
from .frontend.module import ModuleDef


class Design:
    """Collects :class:`ModuleDef` objects and compiles them to hardware."""

    def __init__(self) -> None:
        self._modules: List[ModuleDef] = []
        self._compiled: Optional[List[dict]] = None

    def add(self, mod: ModuleDef) -> "Design":
        """Append a module and its transitive sub-modules.

        Recursively discovers all modules referenced via instance calls
        and adds them in dependency-first order.  Returns *self* for chaining.
        """
        seen = {m.name for m in self._modules}
        self._add_recursive(mod, seen)
        self._compiled = None
        return self

    def _add_recursive(self, mod: ModuleDef, seen: set) -> None:
        """Walk instance calls depth-first, adding dependencies before *mod*."""
        if mod.name in seen:
            return
        seen.add(mod.name)
        for inst in mod.instances:
            if inst.target_mod is not None:
                self._add_recursive(inst.target_mod, seen)
        self._modules.append(mod)

    def compile(self, max_retries: int = 3) -> List[dict]:
        """Compile every module via the LLM and return validated JSON-IR dicts."""
        if self._compiled is not None:
            return self._compiled

        results: List[dict] = []
        for mod in self._modules:
            result = compile_module(mod, max_retries=max_retries)
            if not result.success:
                raise CompilationError(
                    f"Compilation of module '{mod.name}' failed: {result.error}"
                )
            results.append(result.module_dict)

        self._compiled = results
        return results

    def to_json(self, max_retries: int = 3) -> str:
        """Return the compiled design as a pretty-printed JSON string."""
        return json.dumps(self.compile(max_retries=max_retries), indent=2)

    def _ir_pipeline(self, max_retries: int = 3):
        """Run parse → validate → infer and return (modules, widths)."""
        modules_json = self.compile(max_retries=max_retries)
        modules = parse_design(json.dumps(modules_json))
        validate_design(modules)
        widths = infer_widths(modules)
        return modules, widths

    def to_mlir(self, top: Optional[str] = None, max_retries: int = 3) -> str:
        """Compile and generate MLIR text."""
        modules, widths = self._ir_pipeline(max_retries=max_retries)
        return generate_mlir(modules, widths, top=top)

    def to_verilog(
        self,
        top: Optional[str] = None,
        max_retries: int = 3,
        optimize: bool = True,
    ) -> str:
        """Compile and generate Verilog text."""
        modules, widths = self._ir_pipeline(max_retries=max_retries)
        return generate_verilog(modules, widths, top=top, optimize=optimize)
