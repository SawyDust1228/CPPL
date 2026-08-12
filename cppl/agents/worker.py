"""Isolated Generator/Repair worker for one CPPL module."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

from .backend import APPLBackend
from .cache import ModuleCache
from .context import (
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
from ..frontend.const_analysis import normalize_constants
from ..frontend.module import InstanceCall, ModuleDef
from ..ir.errors import CircuitPPLError
from ..ir.infer import infer_widths
from ..ir.parser import parse_design
from ..ir.validator import validate_design


def _extract_json_array(text: str) -> list:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("```"):
        raise json.JSONDecodeError(
            "markdown JSON fence is incomplete; response was likely truncated",
            text,
            0,
        )
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("response root must be a JSON array")
    return value


def _ports_dict(mod: ModuleDef) -> dict:
    ports: dict = {}
    for port in mod.ports:
        entry = {"dir": port.direction, "width": port.width}
        if port.kind != "bits":
            entry["type"] = port.kind
        ports[port.name] = entry
    return ports


def _instance_ops(instances: list[InstanceCall]) -> list:
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


def _split_instances(mod: ModuleDef) -> tuple[list[InstanceCall], list[InstanceCall]]:
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


def _has_instance_op(body: list, inst: InstanceCall) -> bool:
    return any(
        isinstance(operation, dict)
        and operation.get("op") == "instance"
        and operation.get("module") == inst.target_name
        and operation.get("id") == inst.output_ids
        and operation.get("args") == inst.input_map
        and (not inst.name or operation.get("name") == inst.name)
        for operation in body
    )


def _make_stub_dicts(instances: list[InstanceCall]) -> list[dict]:
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
                body.append({
                    "id": constant_id,
                    "op": "constant",
                    "value": 0,
                    "width": port.width,
                })
                output_args[port.name] = constant_id
        body.append({"op": "output", "args": output_args})
        stubs[inst.target_name] = {
            "name": inst.target_name,
            "ports": ports,
            "body": body,
        }
    return list(stubs.values())


def _diagnostic(exc: Exception) -> DiagnosticPacket:
    message = str(exc)
    location_match = re.search(r"(Module '[^']+'(?: body\[\d+\])?)", message)
    identifiers = tuple(dict.fromkeys(re.findall(r"'([A-Za-z_][A-Za-z0-9_$]*)'", message)))
    return DiagnosticPacket(
        category=type(exc).__name__,
        message=message[:1200],
        location=location_match.group(1) if location_match else "",
        related_ids=identifiers[:12],
    )


class ModuleWorker:
    def __init__(
        self,
        mod: ModuleDef,
        config: ResolvedAgentConfig,
        backend_factory: Callable[[], APPLBackend],
        cache: ModuleCache,
        max_retries: int,
    ) -> None:
        self.mod = mod
        self.config = config
        self.backend_factory = backend_factory
        self.cache = cache
        self.max_retries = max_retries
        self.ports = _ports_dict(mod)
        self.preplaced, self.deferred = _split_instances(mod)
        self.preplaced_ops = _instance_ops(self.preplaced)
        self.stub_dicts = _make_stub_dicts(mod.instances)

    def _validate(self, module_dict: dict) -> None:
        design = self.stub_dicts + [module_dict]
        modules = parse_design(design)
        validate_design(modules)
        infer_widths(modules)

    def _materialize(self, llm_body: list) -> dict:
        missing = [
            inst for inst in self.deferred
            if not _has_instance_op(llm_body, inst)
        ]
        if missing:
            names = ", ".join(inst.target_name for inst in missing)
            raise ValueError(f"Missing required instance operation(s): {names}")
        full_body = normalize_constants(
            self.preplaced_ops + llm_body,
            self.ports,
            self.mod.instances,
        )
        module_dict = {
            "name": self.mod.name,
            "ports": self.ports,
            "body": full_body,
        }
        self._validate(module_dict)
        return module_dict

    def _compress_requirements(self, report: ModuleCompileReport) -> str:
        compressed: list[tuple[str, list[str]]] = []
        for source, text in requirement_chunks(self.mod.docstring, self.config):
            context = build_requirement_compression_context(
                source,
                text,
                self.config,
            )
            response = self.backend_factory().generate(
                context.system_prompt,
                context.user_prompt,
                output_tokens=min(self.config.output_tokens, 2000),
            )
            report.compression_calls += 1
            report.transport_retries += response.transport_retries
            report.input_tokens_estimate += context.input_tokens_estimate
            report.output_tokens_estimate += estimate_tokens(response.text)
            try:
                payload = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise ContextBudgetError(
                    f"Requirement compression for {source} did not return JSON: {exc}"
                ) from exc
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

        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for source, requirements in compressed:
            for requirement in requirements:
                key = (source, requirement)
                if key not in seen:
                    lines.append(f"[{source}] {requirement}")
                    seen.add(key)
        if not lines:
            raise ContextBudgetError(
                f"Module '{self.mod.name}' requirements could not be compressed."
            )
        report.compression_events.append("requirements_compressed_with_source_coverage")
        return "\n".join(lines)

    def run(self) -> ModuleCompileReport:
        started = time.monotonic()
        report = ModuleCompileReport(module_name=self.mod.name, status="cache_lookup")
        cache_key = self.cache.key_for(self.mod)
        report.cache_key = cache_key
        cached = self.cache.load(cache_key, self._validate)
        if cached is not None:
            report.status = "success"
            report.cache_hit = True
            report.module_dict = cached
            report.duration_seconds = time.monotonic() - started
            return report

        candidate: Optional[list] = None
        diagnostic: Optional[DiagnosticPacket] = None
        description_override: Optional[str] = None
        try:
            for attempt in range(1, self.max_retries + 1):
                report.status = "generating" if attempt == 1 else "repairing"
                try:
                    context = build_agent_context(
                        self.mod,
                        self.preplaced,
                        self.deferred,
                        self.config,
                        candidate=candidate,
                        diagnostic=diagnostic,
                        description_override=description_override,
                    )
                except ContextBudgetError:
                    if description_override is not None:
                        raise
                    report.status = "compressing"
                    description_override = self._compress_requirements(report)
                    context = build_agent_context(
                        self.mod,
                        self.preplaced,
                        self.deferred,
                        self.config,
                        candidate=candidate,
                        diagnostic=diagnostic,
                        description_override=description_override,
                    )
                report.input_tokens_estimate += context.input_tokens_estimate
                report.compression_events.extend(context.compression_events)
                response = self.backend_factory().generate(
                    context.system_prompt,
                    context.user_prompt,
                )
                report.attempts = attempt
                report.transport_retries += response.transport_retries
                report.output_tokens_estimate += estimate_tokens(response.text)
                report.status = "validating"
                try:
                    candidate = _extract_json_array(response.text)
                    module_dict = self._materialize(candidate)
                except (json.JSONDecodeError, ValueError, CircuitPPLError) as exc:
                    diagnostic = _diagnostic(exc)
                    if candidate is None:
                        candidate = []
                    if attempt >= self.max_retries:
                        raise
                    continue

                report.status = "success"
                report.module_dict = module_dict
                report.compression_events = list(dict.fromkeys(report.compression_events))
                report.duration_seconds = time.monotonic() - started
                return report
        except Exception as exc:
            report.status = "failed"
            report.error_category = type(exc).__name__
            report.error = str(exc)
            report.compression_events = list(dict.fromkeys(report.compression_events))
            report.duration_seconds = time.monotonic() - started
            return report

        report.status = "failed"
        report.error_category = "AgentRuntimeError"
        report.error = "Module worker stopped without a result"
        report.duration_seconds = time.monotonic() - started
        return report
