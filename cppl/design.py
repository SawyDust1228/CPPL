"""Design orchestrator: collects modules and drives compilation + codegen."""

from __future__ import annotations

import json
from typing import List, Optional

from .ir.parser import parse_design
from .ir.validator import validate_design
from .ir.infer import infer_widths

from .agents.models import AgentConfig, CompilationReport
from .frontend.compiler import CompilationError, compile_modules_with_report
from .frontend.module import ModuleDef


class Design:
    """Collects :class:`ModuleDef` objects and compiles them to hardware."""

    def __init__(self, agent_config: AgentConfig | None = None) -> None:
        self._modules: List[ModuleDef] = []
        self._compiled: Optional[List[dict]] = None
        self.agent_config = agent_config
        self.last_compile_report: CompilationReport | None = None

    def add(self, *mods: ModuleDef) -> "Design":
        """Append modules in dependency-first instance-tree order.

        Recursively discovers all modules referenced via instance calls
        and adds them before the module that instantiates them.  Calling
        ``add(top)`` therefore compiles leaves first and the top module last.
        Multiple roots are accepted and are processed left to right.
        """
        seen = {m.name for m in self._modules}
        for mod in mods:
            self._add_recursive(mod, seen)
        self._compiled = None
        self.last_compile_report = None
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

        report = self.compile_with_report(max_retries=max_retries)
        if not report.success:
            raise CompilationError(report.design_error or "unknown compilation error")
        return report.modules

    def compile_with_report(self, max_retries: int = 3) -> CompilationReport:
        """Compile the design and return detailed agent/runtime metadata."""
        if self._compiled is not None and self.last_compile_report is not None:
            return self.last_compile_report

        report = compile_modules_with_report(
            self._modules,
            max_retries=max_retries,
            agent_config=self.agent_config,
        )
        self.last_compile_report = report
        if report.success:
            self._compiled = report.modules
        return report

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
        from .codegen.circt import generate_mlir

        modules, widths = self._ir_pipeline(max_retries=max_retries)
        return generate_mlir(modules, widths, top=top)

    def to_verilog(
        self,
        top: Optional[str] = None,
        max_retries: int = 3,
        optimize: bool = True,
    ) -> str:
        """Compile and generate Verilog text."""
        from .codegen.circt import generate_verilog

        modules, widths = self._ir_pipeline(max_retries=max_retries)
        return generate_verilog(modules, widths, top=top, optimize=optimize)
