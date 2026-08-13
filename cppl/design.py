"""Design orchestrator: collects modules and drives compilation + codegen."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from .ir.parser import parse_design
from .ir.validator import validate_design
from .ir.infer import infer_widths

from .agents.models import AgentConfig, CompilationReport
from .frontend.compiler import CompilationError, compile_modules_with_report
from .frontend.module import ModuleDef
from .harness import CompileObserver

if TYPE_CHECKING:
    from .ir.interpreter import Interpreter


class Design:
    """Collects :class:`ModuleDef` objects and compiles them to hardware."""

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        *,
        observer: CompileObserver | None = None,
    ) -> None:
        self._modules: List[ModuleDef] = []
        self._compiled: Optional[List[dict]] = None
        self.agent_config = agent_config
        self.observer = observer
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
            self.add_recursive(mod, seen)
        self._compiled = None
        self.last_compile_report = None
        return self

    def add_recursive(self, mod: ModuleDef, seen: set) -> None:
        """Walk instance calls depth-first, adding dependencies before *mod*."""
        if mod.name in seen:
            return
        seen.add(mod.name)
        for inst in mod.instances:
            if inst.target_mod is not None:
                self.add_recursive(inst.target_mod, seen)
        self._modules.append(mod)

    def compile(
        self,
        max_retries: int = 3,
        *,
        observer: CompileObserver | None = None,
    ) -> List[dict]:
        """Compile every module via the LLM and return validated JSON-IR dicts."""
        if self._compiled is not None:
            return self._compiled

        report = self.compile_with_report(
            max_retries=max_retries,
            observer=observer,
        )
        if not report.success:
            raise CompilationError(report.design_error or "unknown compilation error")
        return report.modules

    def compile_with_report(
        self,
        max_retries: int = 3,
        *,
        observer: CompileObserver | None = None,
    ) -> CompilationReport:
        """Compile the design and return detailed agent/runtime metadata."""
        if self._compiled is not None and self.last_compile_report is not None:
            return self.last_compile_report

        report = compile_modules_with_report(
            self._modules,
            max_retries=max_retries,
            agent_config=self.agent_config,
            observer=observer if observer is not None else self.observer,
        )
        self.last_compile_report = report
        if report.success:
            self._compiled = report.modules
        return report

    def to_json(
        self,
        max_retries: int = 3,
        *,
        observer: CompileObserver | None = None,
    ) -> str:
        """Return the compiled design as a pretty-printed JSON string."""
        return json.dumps(
            self.compile(max_retries=max_retries, observer=observer), indent=2
        )

    def run_ir_pipeline(
        self,
        max_retries: int = 5,
        *,
        observer: CompileObserver | None = None,
    ):
        """Run parse → validate → infer and return (modules, widths)."""
        modules_json = self.compile(max_retries=max_retries, observer=observer)
        modules = parse_design(json.dumps(modules_json))
        validate_design(modules)
        widths = infer_widths(modules)
        return modules, widths

    def to_mlir(
        self,
        top: Optional[str] = None,
        max_retries: int = 3,
        *,
        observer: CompileObserver | None = None,
    ) -> str:
        """Compile and generate MLIR text."""
        from .codegen.circt import generate_mlir

        pipeline_kwargs = {"max_retries": max_retries}
        if observer is not None:
            pipeline_kwargs["observer"] = observer
        modules, widths = self.run_ir_pipeline(**pipeline_kwargs)
        return generate_mlir(modules, widths, top=top)

    def to_verilog(
        self,
        top: Optional[str] = None,
        max_retries: int = 3,
        optimize: bool = True,
        *,
        observer: CompileObserver | None = None,
    ) -> str:
        """Compile and generate Verilog text."""
        from .codegen.circt import generate_verilog

        pipeline_kwargs = {"max_retries": max_retries}
        if observer is not None:
            pipeline_kwargs["observer"] = observer
        modules, widths = self.run_ir_pipeline(**pipeline_kwargs)
        return generate_verilog(modules, widths, top=top, optimize=optimize)

    def interpreter(
        self,
        top: Optional[str] = None,
        max_retries: int = 3,
        *,
        observer: CompileObserver | None = None,
    ) -> "Interpreter":
        """Compile the design and return a stateful IR interpreter."""
        from .ir.interpreter import Interpreter

        pipeline_kwargs = {"max_retries": max_retries}
        if observer is not None:
            pipeline_kwargs["observer"] = observer
        modules, widths = self.run_ir_pipeline(**pipeline_kwargs)
        return Interpreter(modules, widths=widths, top=top)
