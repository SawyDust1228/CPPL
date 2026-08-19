"""LangGraph workflow for generating and repairing one CPPL module."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
import time
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.exceptions import OutputParserException

from .backend import BackendResponse, LLMBackend
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
    GenerationBudgetError,
    ModuleCompileReport,
    ResolvedAgentConfig,
)
from .tools import (
    IRToolSession,
    ToolSessionError,
    ToolSessionLimits,
    candidate_hash,
)
from ..harness import CompileEventEmitter
from ..frontend.const_analysis import normalize_constants
from ..frontend.module import InstanceCall, ModuleDef
from ..ir.errors import (
    CircuitPPLError,
    DiagnosticIssue,
    PatternMismatch,
    ValidationError,
    issues_from_errors,
)
from ..ir.infer import infer_widths
from ..ir.patterns import run_patterns
from ..ir.parser import parse_design
from ..ir.validator import validate_design
from ..harness.schema import IRSchemaError, validate_ir_body


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


def diagnostic_from_exception(
    exc: Exception, candidate: list | None = None
) -> DiagnosticPacket:
    message = str(exc)
    structured_issues = tuple(issues_from_errors([exc]))
    location_match = re.search(r"(Module '[^']+'(?: body\[\d+\])?)", message)
    identifiers = tuple(
        dict.fromkeys(re.findall(r"'([A-Za-z_][A-Za-z0-9_$]*)'", message))
    )
    width_values = [int(value) for value in re.findall(r"(?:width|bit)[^0-9]*(\d+)", message)]
    fingerprint_payload = {
        "category": type(exc).__name__,
        "message": re.sub(r"\d+\.\d+s", "<time>", message),
        "candidate": candidate or [],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return DiagnosticPacket(
        category=type(exc).__name__,
        message=message[:4000],
        location=(
            structured_issues[0].location
            if structured_issues and structured_issues[0].location
            else location_match.group(1) if location_match else ""
        ),
        related_ids=identifiers[:12],
        details={
            "widths": width_values[:6],
            "issue_count": len(structured_issues),
            "issues": [issue.as_dict() for issue in structured_issues[:24]],
        },
        fingerprint=fingerprint,
    )


class ModuleAgentState(TypedDict, total=False):
    mod: ModuleDef
    dependency_modules: list[dict]
    cache_dependency_modules: list[dict]
    missing_pattern_dependencies: tuple[str, ...]
    hierarchy_depth: int
    direct_dependencies: tuple[str, ...]
    dependency_hashes: dict[str, str]
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
    tool_pattern_hashes: tuple[str, ...]
    pattern_results: dict[str, tuple[bool, int, str]]
    force_direct_generation: bool
    route: str


def invoke_generation_backend(
    backend: LLMBackend,
    system_prompt: str,
    user_prompt: str,
    *,
    output_tokens: int | None = None,
    tool_session: IRToolSession | None = None,
    tool_mode: str = "auto",
):
    tool_enabled = getattr(backend, "generate_with_tools", None)
    fallback_reason = ""
    if tool_session is not None and tool_mode != "off":
        if callable(tool_enabled):
            return tool_enabled(
                system_prompt,
                user_prompt,
                tool_session,
                output_tokens=output_tokens,
            )
        if tool_mode == "required":
            raise ToolSessionError(
                "The configured LLM backend does not implement tool calling."
            )
        fallback_reason = "backend tool calling is unavailable"
    structured = getattr(backend, "generate_structured", None)
    if callable(structured):
        response = structured(
            system_prompt, user_prompt, output_tokens=output_tokens
        )
    else:
        response = backend.generate(
            system_prompt, user_prompt, output_tokens=output_tokens
        )
    if fallback_reason and isinstance(response, BackendResponse):
        return replace(response, tool_fallback_reason=fallback_reason)
    return response


class ModuleAgentGraph:
    """Compiled LangGraph state machine for one module compilation."""

    def __init__(
        self,
        config: ResolvedAgentConfig,
        backend_factory: Callable[[], LLMBackend],
        cache: ModuleCache,
        max_simulation_attempts: int,
        events: CompileEventEmitter,
    ) -> None:
        self.config = config
        self.backend_factory = backend_factory
        self.cache = cache
        self.max_simulation_attempts = max_simulation_attempts
        self.events = events
        self.graph = self._build_graph().compile()

    def _validate_candidate(
        self,
        state: ModuleAgentState,
        module_dict: dict,
        *,
        run_patterns_enabled: bool = True,
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
        if not run_patterns_enabled:
            return 0
        return run_patterns(
            modules,
            state["mod"].name,
            state["mod"].patterns,
            widths=widths,
        )

    def _materialize(
        self,
        state: ModuleAgentState,
        body: list,
        *,
        run_patterns_enabled: bool = True,
    ) -> tuple[dict, int]:
        missing = [
            inst
            for inst in state["deferred"]
            if not contains_instance_op(body, inst)
        ]
        if missing:
            raise ValidationError(
                [
                    DiagnosticIssue(
                        code="instance.required_operation_missing",
                        location=f"Module '{state['mod'].name}' generated body",
                        message=f"missing required instance operation for '{inst.target_name}'",
                        expected={
                            "module": inst.target_name,
                            "name": inst.name,
                            "id": list(inst.output_ids),
                            "args": dict(inst.input_map),
                        },
                        actual="not emitted",
                        hint="Emit this must_emit instance exactly once after all of its inputs exist.",
                    )
                    for inst in missing
                ],
                summary=(
                    f"Module '{state['mod'].name}' is missing "
                    f"{len(missing)} required instance operation(s)"
                ),
            )
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
        return module_dict, self._validate_candidate(
            state,
            module_dict,
            run_patterns_enabled=run_patterns_enabled,
        )

    def _initialize(self, state: ModuleAgentState) -> dict:
        mod = state["mod"]
        report = ModuleCompileReport(
            module_name=mod.name,
            status="cache_lookup",
            hierarchy_depth=state.get("hierarchy_depth", 0),
            direct_dependencies=list(state.get("direct_dependencies", ())),
            dependency_hashes=dict(state.get("dependency_hashes", {})),
            validation_stage="cache_lookup",
        )
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
            "tool_pattern_hashes": (),
            "pattern_results": {},
            "force_direct_generation": False,
        }

    def _check_budget(self, state: ModuleAgentState) -> None:
        report = state["report"]
        elapsed = time.monotonic() - state["started"]
        tokens = report.input_tokens_estimate + report.output_tokens_estimate
        report.budget_seconds = elapsed
        report.budget_tokens = tokens
        if elapsed >= self.config.module_deadline_seconds:
            raise GenerationBudgetError(
                f"Module '{state['mod'].name}' exceeded its "
                f"{self.config.module_deadline_seconds:.0f}s generation deadline"
            )
        if tokens >= self.config.max_module_tokens:
            raise GenerationBudgetError(
                f"Module '{state['mod'].name}' exceeded its "
                f"{self.config.max_module_tokens} token generation budget"
            )

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
            report.failure_origin = "hierarchy"
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

        key = self.cache.key_for(
            state["mod"], state.get("cache_dependency_modules")
        )
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
        report.validation_stage = "context_built"
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
                self._check_budget(state)
                context = build_requirement_compression_context(source, text, self.config)
                response = self.backend_factory().generate(
                    context.system_prompt,
                    context.user_prompt,
                    output_tokens=min(self.config.output_tokens, 2000),
                )
                report.compression_calls += 1
                report.transport_retries += response.transport_retries
                report.model_turns += response.model_turns
                report.input_tokens_estimate += context.input_tokens_estimate
                report.input_tokens_estimate += response.extra_input_tokens_estimate
                report.output_tokens_estimate += (
                    response.output_tokens_estimate or estimate_tokens(response.text)
                )
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
        try:
            self._check_budget(state)
        except GenerationBudgetError as exc:
            return self._failure(state, exc)
        generation_call = report.attempts + 1
        report.attempts = generation_call
        report.status = "generating" if generation_call == 1 else "repairing"
        report.validation_stage = report.status
        event_kind = "generate" if generation_call == 1 else "repair"
        next_simulation_attempt = min(
            report.simulation_attempts + 1, self.max_simulation_attempts
        )
        message = (
            f"generation call {generation_call}; next simulation attempt "
            f"{next_simulation_attempt}/{self.max_simulation_attempts}"
        )
        diagnostic = state.get("diagnostic")
        if diagnostic is not None and generation_call > 1:
            message += f" after {diagnostic.category}"
        self.events.emit(
            event_kind,
            module_name=state["mod"].name,
            stage=report.status,
            attempt=generation_call,
            message=message,
            metadata={
                "generation_call": generation_call,
                "simulation_attempts": report.simulation_attempts,
                "max_simulation_attempts": self.max_simulation_attempts,
                "static_repairs": report.static_repairs,
            },
        )
        tool_calls = 0
        tool_failures = 0
        tool_counts: dict[str, int] = {}
        tool_pattern_hashes: tuple[str, ...] = ()
        pattern_results: dict[str, tuple[bool, int, str]] = dict(
            state.get("pattern_results", {})
        )

        def merge_tool_stats() -> None:
            report.tool_calls += tool_calls
            report.tool_failures += tool_failures
            for name, count in tool_counts.items():
                report.tool_counts[name] = report.tool_counts.get(name, 0) + count
            report.simulation_attempts += len(tool_pattern_hashes)

        try:
            def validate_tool_candidate(body: list[dict], run_patterns_now: bool):
                validated = validate_ir_body(body)
                return self._materialize(
                    state,
                    validated,
                    run_patterns_enabled=run_patterns_now,
                )

            def emit_tool_event(
                kind: str,
                tool_name: str,
                elapsed: float,
                ok: bool,
                metadata: dict,
            ) -> None:
                self.events.emit(
                    kind,
                    module_name=state["mod"].name,
                    stage="tool_session",
                    attempt=generation_call,
                    message=(
                        f"{tool_name}"
                        if kind == "tool_start"
                        else f"{tool_name} in {elapsed:.3f}s"
                    ),
                    metadata={
                        "tool": tool_name,
                        "duration_seconds": elapsed,
                        "success": ok,
                        **metadata,
                    },
                )

            with IRToolSession(
                initial_candidate=state.get("candidate"),
                task_context=(
                    state["system_prompt"] + "\n\n" + state["user_prompt"]
                ),
                validate_candidate=validate_tool_candidate,
                has_patterns=bool(state["mod"].patterns),
                remaining_pattern_attempts=max(
                    0, self.max_simulation_attempts - report.simulation_attempts
                ),
                pattern_results=pattern_results,
                limits=ToolSessionLimits(
                    timeout_seconds=self.config.tool_timeout_seconds,
                    max_output_chars=self.config.max_tool_output_chars,
                    deadline_monotonic=(
                        state["started"] + self.config.module_deadline_seconds
                    ),
                    remaining_model_tokens=max(
                        1,
                        self.config.max_module_tokens
                        - report.input_tokens_estimate
                        - report.output_tokens_estimate,
                    ),
                ),
                event_callback=emit_tool_event,
            ) as tool_session:
                try:
                    response = invoke_generation_backend(
                        self.backend_factory(),
                        state["system_prompt"],
                        state["user_prompt"],
                        tool_session=tool_session,
                        tool_mode=(
                            "off"
                            if state.get("force_direct_generation")
                            and self.config.tool_mode == "auto"
                            else self.config.tool_mode
                        ),
                    )
                finally:
                    tool_calls = tool_session.stats.calls
                    tool_failures = tool_session.stats.failures
                    tool_counts = dict(tool_session.stats.counts)
                    tool_pattern_hashes = tuple(
                        tool_session.stats.pattern_attempt_hashes
                    )
                    pattern_results = tool_session.pattern_results
        except Exception as exc:
            merge_tool_stats()
            return self._failure(state, exc)
        report.transport_retries += response.transport_retries
        report.model_turns += response.model_turns
        report.input_tokens_estimate += response.extra_input_tokens_estimate
        report.output_tokens_estimate += (
            response.output_tokens_estimate or estimate_tokens(response.text)
        )
        merge_tool_stats()
        if response.tool_fallback_reason:
            report.tool_fallback_reason = response.tool_fallback_reason
        try:
            self._check_budget(state)
        except GenerationBudgetError as exc:
            return self._failure(state, exc)
        report.status = "validating"
        report.validation_stage = "candidate_validation"
        self.events.emit(
            "validate",
            module_name=state["mod"].name,
            stage="validating",
            attempt=generation_call,
            message="checking JSON-IR and executable patterns",
        )
        return {
            "response_text": response.text,
            "tool_pattern_hashes": tool_pattern_hashes,
            "pattern_results": pattern_results,
            "force_direct_generation": False,
            "report": report,
            "route": "validate",
        }

    def _validate(self, state: ModuleAgentState) -> dict:
        report = state["report"]
        candidate: list | None = None
        candidate_was_pattern_checked = False
        pattern_results = dict(state.get("pattern_results", {}))
        try:
            candidate = extract_json_array(state["response_text"])
            candidate = validate_ir_body(candidate)
            digest = candidate_hash(candidate)
            cached_pattern_result = pattern_results.get(digest)
            candidate_was_pattern_checked = cached_pattern_result is not None
            if cached_pattern_result is None:
                module_dict, patterns_checked = self._materialize(state, candidate)
            elif cached_pattern_result[0]:
                module_dict, _ = self._materialize(
                    state, candidate, run_patterns_enabled=False
                )
                patterns_checked = cached_pattern_result[1]
            else:
                raise PatternMismatch(cached_pattern_result[2])
        except (OutputParserException, IRSchemaError, ValueError, CircuitPPLError) as exc:
            simulation_failure = isinstance(exc, PatternMismatch)
            if simulation_failure:
                if not candidate_was_pattern_checked:
                    report.simulation_attempts += 1
            else:
                report.static_repairs += 1
            diagnostic = diagnostic_from_exception(exc, candidate)
            report.failure_fingerprints.append(diagnostic.fingerprint)
            repeated_diagnostic = (
                len(report.failure_fingerprints) >= 3
                and len(set(report.failure_fingerprints[-3:])) == 1
            )
            report.failure_origin = (
                "hierarchy" if state.get("direct_dependencies") else "module"
            )
            if (
                simulation_failure
                and report.simulation_attempts >= self.max_simulation_attempts
            ):
                return self._failure(state, exc, candidate=candidate or [])
            next_simulation_attempt = min(
                report.simulation_attempts + 1, self.max_simulation_attempts
            )
            self.events.emit(
                "repair_scheduled",
                module_name=state["mod"].name,
                stage="repairing",
                attempt=report.attempts + 1,
                message=f"{type(exc).__name__}: {str(exc)[:180]}",
                metadata={
                    "category": type(exc).__name__,
                    "consumed_simulation_attempt": simulation_failure,
                    "simulation_attempts": report.simulation_attempts,
                    "next_simulation_attempt": next_simulation_attempt,
                    "max_simulation_attempts": self.max_simulation_attempts,
                    "static_repairs": report.static_repairs,
                    "force_direct_generation": repeated_diagnostic,
                },
            )
            if repeated_diagnostic and self.config.tool_mode == "auto":
                self.events.emit(
                    "tool_fallback",
                    module_name=state["mod"].name,
                    stage="repairing",
                    attempt=report.attempts + 1,
                    message=(
                        "same diagnostic repeated three times; next generation "
                        "will bypass tools"
                    ),
                )
            return {
                "candidate": candidate or [],
                "diagnostic": diagnostic,
                "pattern_results": pattern_results,
                "force_direct_generation": (
                    repeated_diagnostic and self.config.tool_mode == "auto"
                ),
                "report": report,
                "route": "repair",
            }
        if state["mod"].patterns and not candidate_was_pattern_checked:
            report.simulation_attempts += 1
        report.status = "success"
        report.validation_stage = "published"
        report.module_dict = module_dict
        report.patterns_checked = patterns_checked
        if report.cache_key is not None and not report.cache_hit:
            self.cache.store(report.cache_key, module_dict)
            report.cache_committed = True
        report.compression_events = list(dict.fromkeys(report.compression_events))
        report.duration_seconds = time.monotonic() - state["started"]
        report.budget_seconds = report.duration_seconds
        report.budget_tokens = (
            report.input_tokens_estimate + report.output_tokens_estimate
        )
        self.events.emit(
            "module_success",
            module_name=state["mod"].name,
            stage="complete",
            attempt=report.attempts,
            message=f"validated in {report.duration_seconds:.2f}s",
            metadata={
                "generation_calls": report.attempts,
                "simulation_attempts": report.simulation_attempts,
                "static_repairs": report.static_repairs,
                "patterns_checked": report.patterns_checked,
                "model_turns": report.model_turns,
                "tool_calls": report.tool_calls,
                "tool_failures": report.tool_failures,
            },
        )
        return {
            "candidate": candidate,
            "pattern_results": pattern_results,
            "report": report,
            "route": "done",
        }

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
        report.diagnostics = [
            issue.as_dict() for issue in issues_from_errors([exc])
        ]
        report.module_dict = None
        report.compression_events = list(dict.fromkeys(report.compression_events))
        report.duration_seconds = time.monotonic() - state["started"]
        report.budget_seconds = report.duration_seconds
        report.budget_tokens = (
            report.input_tokens_estimate + report.output_tokens_estimate
        )
        self.events.emit(
            "module_failed",
            module_name=state["mod"].name,
            stage=report.status,
            attempt=report.attempts or None,
            message=f"{type(exc).__name__}: {str(exc)[:240]}",
            metadata={
                "category": type(exc).__name__,
                "model_turns": report.model_turns,
                "tool_calls": report.tool_calls,
                "tool_failures": report.tool_failures,
            },
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
