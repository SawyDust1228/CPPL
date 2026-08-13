"""LangGraph workflow for generating and repairing one CPPL module."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.exceptions import OutputParserException

from .backend import LLMBackend
from .cache import ModuleCache
from .context import (
    JSON_OUTPUT_PARSER,
    build_agent_context,
    build_requirement_compression_context,
    estimate_tokens,
    requirement_chunks,
)
from .models import (
    ContextBudgetError,
    DiagnosticPacket,
    ModuleCompileReport,
    ResolvedAgentConfig,
)
from ..harness import CompileEventEmitter
from ..frontend.const_analysis import normalize_constants
from ..frontend.module import InstanceCall, ModuleDef
from ..ir.errors import CircuitPPLError
from ..ir.infer import infer_widths
from ..ir.patterns import run_patterns
from ..ir.parser import parse_design
from ..ir.validator import validate_design


def extract_json_array(text: str) -> list:
    """Parse one JSON array using LangChain's JSON output parser."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    elif stripped.startswith("```"):
        raise OutputParserException(
            "Markdown JSON fence is incomplete; response was likely truncated.",
            llm_output=text,
        )
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise OutputParserException(
            "The response must contain one complete JSON array.",
            llm_output=text,
        )
    try:
        json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OutputParserException(
            f"Invalid JSON array: {exc}", llm_output=text
        ) from exc
    value = JSON_OUTPUT_PARSER.parse(stripped)
    if not isinstance(value, list):
        raise ValueError("response root must be a JSON array")
    return value


def build_ports_dict(mod: ModuleDef) -> dict:
    ports: dict = {}
    for port in mod.ports:
        entry = {"dir": port.direction, "width": port.width}
        if port.kind != "bits":
            entry["type"] = port.kind
        ports[port.name] = entry
    return ports


def build_instance_ops(instances: list[InstanceCall]) -> list:
    operations = []
    for inst in instances:
        operation = {
            "id": inst.output_ids,
            "op": "instance",
            "module": inst.target_name,
            "args": inst.input_map,
        }
        if inst.name:
            operation["name"] = inst.name
        operations.append(operation)
    return operations


def split_instances_by_placement(
    mod: ModuleDef,
) -> tuple[list[InstanceCall], list[InstanceCall]]:
    known = {port.name for port in mod.ports if port.direction == "input"}
    preplaced: list[InstanceCall] = []
    deferred: list[InstanceCall] = []
    for inst in mod.instances:
        if all(reference in known for reference in inst.input_map.values()):
            preplaced.append(inst)
            known.update(inst.output_ids)
        else:
            deferred.append(inst)
    return preplaced, deferred


def contains_instance_op(body: list, inst: InstanceCall) -> bool:
    return any(
        isinstance(operation, dict)
        and operation.get("op") == "instance"
        and operation.get("module") == inst.target_name
        and operation.get("id") == inst.output_ids
        and operation.get("args") == inst.input_map
        and (not inst.name or operation.get("name") == inst.name)
        for operation in body
    )


def build_stub_modules(instances: list[InstanceCall]) -> list[dict]:
    stubs: dict[str, dict] = {}
    for inst in instances:
        if inst.target_name in stubs:
            continue
        ports: dict = {}
        body: list = []
        output_args: dict = {}
        for port in inst.target_ports:
            entry = {"dir": port.direction, "width": port.width}
            if port.kind != "bits":
                entry["type"] = port.kind
            ports[port.name] = entry
            if port.direction == "output":
                constant_id = f"_stub_{port.name}"
                body.append(
                    {
                        "id": constant_id,
                        "op": "constant",
                        "value": 0,
                        "width": port.width,
                    }
                )
                output_args[port.name] = constant_id
        body.append({"op": "output", "args": output_args})
        stubs[inst.target_name] = {
            "name": inst.target_name,
            "ports": ports,
            "body": body,
        }
    return list(stubs.values())


def diagnostic_from_exception(exc: Exception) -> DiagnosticPacket:
    message = str(exc)
    location_match = re.search(r"(Module '[^']+'(?: body\[\d+\])?)", message)
    identifiers = tuple(
        dict.fromkeys(re.findall(r"'([A-Za-z_][A-Za-z0-9_$]*)'", message))
    )
    return DiagnosticPacket(
        category=type(exc).__name__,
        message=message[:1200],
        location=location_match.group(1) if location_match else "",
        related_ids=identifiers[:12],
    )


class ModuleAgentState(TypedDict, total=False):
    mod: ModuleDef
    dependency_modules: list[dict]
    missing_pattern_dependencies: tuple[str, ...]
    report: ModuleCompileReport
    started: float
    cache_key: str
    preplaced: list[InstanceCall]
    deferred: list[InstanceCall]
    ports: dict
    description_override: Optional[str]
    candidate: Optional[list]
    diagnostic: Optional[DiagnosticPacket]
    system_prompt: str
    user_prompt: str
    response_text: str
    route: str


class ModuleAgentGraph:
    """Compiled LangGraph state machine for one module compilation."""

    def __init__(
        self,
        config: ResolvedAgentConfig,
        backend_factory: Callable[[], LLMBackend],
        cache: ModuleCache,
        max_retries: int,
        events: CompileEventEmitter,
    ) -> None:
        self.config = config
        self.backend_factory = backend_factory
        self.cache = cache
        self.max_retries = max_retries
        self.events = events
        self.graph = self._build_graph().compile()

    def _validate_candidate(
        self,
        state: ModuleAgentState,
        module_dict: dict,
    ) -> int:
        dependency_modules = state.get("dependency_modules", [])
        real_names = {module["name"] for module in dependency_modules}
        stubs = [
            stub
            for stub in build_stub_modules(state["mod"].instances)
            if stub["name"] not in real_names
        ]
        modules = parse_design(stubs + dependency_modules + [module_dict])
        validate_design(modules)
        widths = infer_widths(modules)
        return run_patterns(
            modules,
            state["mod"].name,
            state["mod"].patterns,
            widths=widths,
        )

    def _materialize(self, state: ModuleAgentState, body: list) -> tuple[dict, int]:
        missing = [
            inst
            for inst in state["deferred"]
            if not contains_instance_op(body, inst)
        ]
        if missing:
            names = ", ".join(inst.target_name for inst in missing)
            raise ValueError(f"Missing required instance operation(s): {names}")
        full_body = normalize_constants(
            build_instance_ops(state["preplaced"]) + body,
            state["ports"],
            state["mod"].instances,
        )
        module_dict = {
            "name": state["mod"].name,
            "ports": state["ports"],
            "body": full_body,
        }
        return module_dict, self._validate_candidate(state, module_dict)

    def _initialize(self, state: ModuleAgentState) -> dict:
        mod = state["mod"]
        report = ModuleCompileReport(module_name=mod.name, status="cache_lookup")
        preplaced, deferred = split_instances_by_placement(mod)
        return {
            "started": time.monotonic(),
            "report": report,
            "preplaced": preplaced,
            "deferred": deferred,
            "ports": build_ports_dict(mod),
            "candidate": None,
            "diagnostic": None,
            "description_override": None,
        }

    def _lookup_cache(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        module_name = state["mod"].name
        self.events.emit(
            "cache_check",
            module_name=module_name,
            stage="cache_lookup",
            message="checking validated module cache",
        )
        if state.get("missing_pattern_dependencies"):
            report.status = "failed"
            report.error_category = "PatternDependencyMissing"
            report.error = (
                f"Module '{state['mod'].name}' patterns require real dependency IR for: "
                + ", ".join(state["missing_pattern_dependencies"])
            )
            report.duration_seconds = time.monotonic() - state["started"]
            self.events.emit(
                "module_failed",
                module_name=module_name,
                stage="cache_lookup",
                message=report.error or "missing pattern dependency",
                metadata={"category": report.error_category},
            )
            return {"report": report, "route": "done"}

        key = self.cache.key_for(state["mod"], state.get("dependency_modules"))
        report.cache_key = key
        patterns_checked = 0

        def validate(module_dict: dict) -> None:
            nonlocal patterns_checked
            patterns_checked = self._validate_candidate(state, module_dict)

        cached = self.cache.load(key, validate)
        if cached is not None:
            report.status = "success"
            report.cache_hit = True
            report.module_dict = cached
            report.patterns_checked = patterns_checked
            report.duration_seconds = time.monotonic() - state["started"]
            self.events.emit(
                "cache_hit",
                module_name=module_name,
                stage="cache_lookup",
                message="reused validated JSON-IR",
            )
            self.events.emit(
                "module_success",
                module_name=module_name,
                stage="complete",
                message=f"cache hit in {report.duration_seconds:.2f}s",
                metadata={"cache_hit": True},
            )
            return {"cache_key": key, "report": report, "route": "done"}
        self.events.emit(
            "cache_miss",
            module_name=module_name,
            stage="cache_lookup",
            message="generation required",
        )
        return {"cache_key": key, "report": report, "route": "build"}

    def _build_context(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        try:
            context = build_agent_context(
                state["mod"],
                state["preplaced"],
                state["deferred"],
                self.config,
                candidate=state.get("candidate"),
                diagnostic=state.get("diagnostic"),
                description_override=state.get("description_override"),
            )
        except ContextBudgetError as exc:
            if state.get("description_override") is not None:
                return self._failure(state, exc)
            report.status = "compressing"
            self.events.emit(
                "compress",
                module_name=state["mod"].name,
                stage="compressing",
                message="requirements exceed context budget",
            )
            return {"report": report, "route": "compress"}
        report.input_tokens_estimate += context.input_tokens_estimate
        report.compression_events.extend(context.compression_events)
        return {
            "report": report,
            "system_prompt": context.system_prompt,
            "user_prompt": context.user_prompt,
            "route": "generate",
        }

    def _compress(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        compressed: list[tuple[str, list[str]]] = []
        try:
            for source, text in requirement_chunks(state["mod"].docstring, self.config):
                context = build_requirement_compression_context(source, text, self.config)
                response = self.backend_factory().generate(
                    context.system_prompt,
                    context.user_prompt,
                    output_tokens=min(self.config.output_tokens, 2000),
                )
                report.compression_calls += 1
                report.transport_retries += response.transport_retries
                report.input_tokens_estimate += context.input_tokens_estimate
                report.output_tokens_estimate += estimate_tokens(response.text)
                payload = JSON_OUTPUT_PARSER.parse(response.text)
                if not isinstance(payload, dict) or payload.get("sources") != [source]:
                    raise ContextBudgetError(
                        f"Requirement compression did not preserve source {source}."
                    )
                requirements = payload.get("requirements")
                if (
                    not isinstance(requirements, list)
                    or not requirements
                    or any(not isinstance(item, str) or not item.strip() for item in requirements)
                ):
                    raise ContextBudgetError(
                        f"Requirement compression for {source} returned no usable requirements."
                    )
                compressed.append((source, [item.strip() for item in requirements]))
        except Exception as exc:
            return self._failure(state, exc)

        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for source, requirements in compressed:
            for requirement in requirements:
                key = (source, requirement)
                if key not in seen:
                    lines.append(f"[{source}] {requirement}")
                    seen.add(key)
        if not lines:
            return self._failure(
                state,
                ContextBudgetError(
                    f"Module '{state['mod'].name}' requirements could not be compressed."
                ),
            )
        report.compression_events.append("requirements_compressed_with_source_coverage")
        return {
            "description_override": "\n".join(lines),
            "report": report,
            "route": "build",
        }

    def _generate(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        attempt = report.attempts + 1
        report.attempts = attempt
        report.status = "generating" if attempt == 1 else "repairing"
        event_kind = "generate" if attempt == 1 else "repair"
        message = f"attempt {attempt}/{self.max_retries}"
        diagnostic = state.get("diagnostic")
        if diagnostic is not None and attempt > 1:
            message += f" after {diagnostic.category}"
        self.events.emit(
            event_kind,
            module_name=state["mod"].name,
            stage=report.status,
            attempt=attempt,
            message=message,
            metadata={"max_attempts": self.max_retries},
        )
        try:
            response = self.backend_factory().generate(
                state["system_prompt"], state["user_prompt"]
            )
        except Exception as exc:
            return self._failure(state, exc)
        report.transport_retries += response.transport_retries
        report.output_tokens_estimate += estimate_tokens(response.text)
        report.status = "validating"
        self.events.emit(
            "validate",
            module_name=state["mod"].name,
            stage="validating",
            attempt=attempt,
            message="checking JSON-IR and executable patterns",
        )
        return {"response_text": response.text, "report": report, "route": "validate"}

    def _validate(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        candidate: list | None = None
        try:
            candidate = extract_json_array(state["response_text"])
            module_dict, patterns_checked = self._materialize(state, candidate)
        except (OutputParserException, ValueError, CircuitPPLError) as exc:
            diagnostic = diagnostic_from_exception(exc)
            if report.attempts >= self.max_retries:
                return self._failure(state, exc, candidate=candidate or [])
            self.events.emit(
                "repair_scheduled",
                module_name=state["mod"].name,
                stage="repairing",
                attempt=report.attempts + 1,
                message=f"{type(exc).__name__}: {str(exc)[:180]}",
                metadata={"category": type(exc).__name__},
            )
            return {
                "candidate": candidate or [],
                "diagnostic": diagnostic,
                "report": report,
                "route": "repair",
            }
        report.status = "success"
        report.module_dict = module_dict
        report.patterns_checked = patterns_checked
        report.compression_events = list(dict.fromkeys(report.compression_events))
        report.duration_seconds = time.monotonic() - state["started"]
        self.events.emit(
            "module_success",
            module_name=state["mod"].name,
            stage="complete",
            attempt=report.attempts,
            message=f"validated in {report.duration_seconds:.2f}s",
            metadata={
                "attempts": report.attempts,
                "patterns_checked": report.patterns_checked,
            },
        )
        return {"candidate": candidate, "report": report, "route": "done"}

    def _failure(
        self,
        state: ModuleAgentState,
        exc: Exception,
        *,
        candidate: list | None = None,
    ) -> dict:
        report = state["report"]
        report.status = "failed"
        report.error_category = type(exc).__name__
        report.error = str(exc)
        report.module_dict = None
        report.compression_events = list(dict.fromkeys(report.compression_events))
        report.duration_seconds = time.monotonic() - state["started"]
        self.events.emit(
            "module_failed",
            module_name=state["mod"].name,
            stage=report.status,
            attempt=report.attempts or None,
            message=f"{type(exc).__name__}: {str(exc)[:240]}",
            metadata={"category": type(exc).__name__},
        )
        update: dict = {"report": report, "route": "done"}
        if candidate is not None:
            update["candidate"] = candidate
        return update

    @staticmethod
    def _route(state: ModuleAgentState) -> str:
        return state["route"]

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ModuleAgentState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("lookup_cache", self._lookup_cache)
        graph.add_node("build_context", self._build_context)
        graph.add_node("compress_requirements", self._compress)
        graph.add_node("invoke_generation_chain", self._generate)
        graph.add_node("parse_and_validate", self._validate)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "lookup_cache")
        graph.add_conditional_edges(
            "lookup_cache", self._route, {"build": "build_context", "done": END}
        )
        graph.add_conditional_edges(
            "build_context",
            self._route,
            {"compress": "compress_requirements", "generate": "invoke_generation_chain", "done": END},
        )
        graph.add_conditional_edges(
            "compress_requirements",
            self._route,
            {"build": "build_context", "done": END},
        )
        graph.add_conditional_edges(
            "invoke_generation_chain",
            self._route,
            {"validate": "parse_and_validate", "done": END},
        )
        graph.add_conditional_edges(
            "parse_and_validate", self._route, {"repair": "build_context", "done": END}
        )
        return graph

    def invoke(self, state: ModuleAgentState, config: dict | None = None) -> ModuleAgentState:
        return self.graph.invoke(state, config=config)

    def batch(
        self,
        states: list[ModuleAgentState],
        config: dict | None = None,
    ) -> list[ModuleAgentState]:
        return self.graph.batch(states, config=config)
