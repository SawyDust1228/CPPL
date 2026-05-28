"""Core LLM-based compiler for CPPL modules using the appl framework."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..env import (
    apply_server_override,
    resolve_llm_generation_kwargs,
    resolve_llm_server_override,
)
from ..ir.errors import CircuitPPLError
from ..ir.parser import parse_design
from ..ir.validator import validate_design
from ..ir.infer import infer_widths

from .module import InstanceCall, ModuleDef
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .const_analysis import normalize_constants


_APPL_SERVER_CONFIGURED = False


def _configure_appl_server() -> None:
    """Apply .env-provided server overrides to APPL after import-time init."""
    import appl

    global _APPL_SERVER_CONFIGURED
    if _APPL_SERVER_CONFIGURED:
        return

    override = resolve_llm_server_override()
    if override is None:
        raise RuntimeError(
            "Missing LLM configuration. Set LLM_MODEL and credentials in the "
            "project root .env file."
        )

    configs = appl.global_vars.configs
    target_name = apply_server_override(configs, override)
    appl.server_manager.close_server(target_name)
    _APPL_SERVER_CONFIGURED = True


@dataclass
class CompileResult:
    """Outcome of compiling a single CPPL module."""
    module_dict: Optional[dict] = None
    success: bool = False
    error: Optional[str] = None
    attempts: int = 0


class CompilerSession:
    """Stateful LLM compiler session for one design compilation."""

    def __init__(self) -> None:
        from appl import SystemMessage

        _configure_appl_server()
        self.messages = [SystemMessage(SYSTEM_PROMPT)]

    def compile(self, mod: ModuleDef, max_retries: int = 3) -> dict:
        return _compile_one_in_context(
            mod,
            max_retries=max_retries,
            messages=self.messages,
        )


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from an LLM response, stripping markdown fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        raise json.JSONDecodeError(
            "markdown JSON fence is incomplete; response was likely truncated",
            text,
            0,
        )
    return json.loads(text)


def _ports_dict(mod: ModuleDef) -> dict:
    """Convert ModuleDef ports to the JSON-IR ports object."""
    d: dict = {}
    for p in mod.ports:
        entry = {"dir": p.direction, "width": p.width}
        if p.kind != "bits":
            entry["type"] = p.kind
        d[p.name] = entry
    return d


def _instance_ops(instances: List[InstanceCall]) -> list:
    """Generate JSON-IR instance operations from captured InstanceCalls."""
    ops = []
    for inst in instances:
        op = {
            "id": inst.output_ids,
            "op": "instance",
            "module": inst.target_name,
            "args": inst.input_map,
        }
        if inst.name:
            op["name"] = inst.name
        ops.append(op)
    return ops


def _split_instances(mod: ModuleDef) -> tuple[List[InstanceCall], List[InstanceCall]]:
    known = {p.name for p in mod.ports if p.direction == "input"}
    preplaced: List[InstanceCall] = []
    deferred: List[InstanceCall] = []

    for inst in mod.instances:
        if all(ref in known for ref in inst.input_map.values()):
            preplaced.append(inst)
            known.update(inst.output_ids)
        else:
            deferred.append(inst)

    return preplaced, deferred


def _has_instance_op(body: list, inst: InstanceCall) -> bool:
    for op in body:
        if not isinstance(op, dict):
            continue
        if op.get("op") != "instance":
            continue
        if op.get("module") != inst.target_name:
            continue
        if op.get("id") != inst.output_ids:
            continue
        if op.get("args") != inst.input_map:
            continue
        if inst.name and op.get("name") != inst.name:
            continue
        return True
    return False


def _make_stub_dicts(instances: List[InstanceCall]) -> List[dict]:
    """Create minimal valid module dicts for each unique instance target.

    Each stub has matching ports and a body that produces constant outputs,
    allowing the IR validator to accept instance ops referencing these modules.
    """
    seen: Dict[str, dict] = {}
    for inst in instances:
        if inst.target_name in seen:
            continue

        ports: dict = {}
        body: list = []

        for p in inst.target_ports:
            entry = {"dir": p.direction, "width": p.width}
            if p.kind != "bits":
                entry["type"] = p.kind
            ports[p.name] = entry

        output_args: dict = {}
        output_ports = [
            p for p in inst.target_ports if p.direction == "output"
        ]
        for p in output_ports:
            const_id = f"_stub_{p.name}"
            body.append({
                "id": const_id,
                "op": "constant",
                "value": 0,
                "width": p.width,
            })
            output_args[p.name] = const_id

        body.append({"op": "output", "args": output_args})

        seen[inst.target_name] = {
            "name": inst.target_name,
            "ports": ports,
            "body": body,
        }

    return list(seen.values())


def _compile_one_in_context(
    mod: ModuleDef,
    max_retries: int = 3,
    *,
    messages: list,
) -> dict:
    """Compile a single ModuleDef to a validated JSON-IR module dict.

    Uses the appl LLM framework with a feedback loop: if the generated body
    fails validation, the error is fed back and the LLM retries.
    """
    ports = _ports_dict(mod)
    last_error: Optional[str] = None

    preplaced_instances, deferred_instances = _split_instances(mod)
    from appl import AIMessage, UserMessage, gen

    messages.append(UserMessage(build_user_prompt(
        mod,
        preplaced_instances=preplaced_instances,
        deferred_instances=deferred_instances,
    )))

    inst_ops = _instance_ops(preplaced_instances)
    stub_dicts = _make_stub_dicts(mod.instances) if mod.instances else []
    gen_kwargs = {
        "max_tokens": 12000,
        "timeout": 90,
        **resolve_llm_generation_kwargs(),
    }

    for _ in range(max_retries):
        response = gen(messages=messages, **gen_kwargs)
        response_text = str(response)
        messages.append(AIMessage(response_text))

        try:
            llm_body = _extract_json_array(response_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc}\nRaw response:\n{response_text}"
            messages.append(UserMessage(
                f"Your response was not valid JSON. It may have included "
                f"reasoning text or been truncated.\n"
                f"Error: {exc}\n"
                f"Please output ONLY one compact JSON array for the body. "
                f"The first character must be '[' and the last character must be ']'. "
                f"Do not use markdown fences, comments, or explanatory text."
            ))
            continue

        missing_instances = [
            inst for inst in deferred_instances
            if not _has_instance_op(llm_body, inst)
        ]
        if missing_instances:
            missing = ", ".join(inst.target_name for inst in missing_instances)
            last_error = f"Missing required instance operation(s): {missing}"
            messages.append(UserMessage(
                f"{last_error}.\n"
                f"Include each required instance op exactly as listed, after "
                f"its input value IDs are defined."
            ))
            continue

        full_body = normalize_constants(
            inst_ops + llm_body,
            ports,
            mod.instances,
        )
        module_dict = {"name": mod.name, "ports": ports, "body": full_body}

        try:
            design = stub_dicts + [module_dict]
            modules = parse_design(json.dumps(design))
            validate_design(modules)
            infer_widths(modules)
            return module_dict
        except CircuitPPLError as exc:
            last_error = str(exc)
            messages.append(UserMessage(
                f"The generated JSON-IR body failed validation:\n{exc}\n"
                f"Please fix the body and regenerate. Output ONLY the corrected JSON array."
            ))
            continue

    raise CompilationError(
        f"Failed to compile module '{mod.name}' after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


class CompilationError(CircuitPPLError):
    """Raised when LLM-based compilation fails after all retries."""


def compile_module(mod: ModuleDef, max_retries: int = 3) -> CompileResult:
    """Compile a single :class:`ModuleDef` and return a :class:`CompileResult`."""
    try:
        module_dict = CompilerSession().compile(mod, max_retries=max_retries)
        return CompileResult(
            module_dict=module_dict,
            success=True,
            attempts=max_retries,  # we don't track exact attempt inside ppl
        )
    except (CompilationError, CircuitPPLError) as exc:
        return CompileResult(
            success=False,
            error=str(exc),
            attempts=max_retries,
        )


def compile_modules(
    mods: List[ModuleDef],
    max_retries: int = 3,
) -> List[CompileResult]:
    """Compile several modules in one LLM session.

    The system prompt is sent once for the whole batch. Each module still gets
    its own user prompt and retry feedback.
    """
    try:
        session = CompilerSession()
        module_dicts = [
            session.compile(mod, max_retries=max_retries)
            for mod in mods
        ]
    except (CompilationError, CircuitPPLError) as exc:
        return [
            CompileResult(success=False, error=str(exc), attempts=max_retries)
        ]

    return [
        CompileResult(
            module_dict=module_dict,
            success=True,
            attempts=max_retries,
        )
        for module_dict in module_dicts
    ]
