"""Semantic validation: SSA scoping, port coverage, instance checks."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .errors import (
    CycleError,
    DiagnosticIssue,
    SSAError,
    ValidationError,
    issues_from_errors,
)
from .identifiers import is_valid_ssa_id
from .models import (
    ConstantOp,
    ExtractOp,
    InstanceOp,
    MemOp,
    Module,
    Operation,
    OutputOp,
    PortDir,
    RegOp,
)


def validate_design(modules: List[Module], *, strict: bool = True) -> None:
    """Run all semantic validations on a parsed design.

    When *strict* is True (the default), additional static-analysis checks
    are run: combinational cycle detection.  These catch common LLM-generated
    errors and feed back into the retry loop.
    """
    errors: list[ValidationError] = []
    try:
        module_map = check_module_names(modules)
    except ValidationError as exc:
        errors.append(exc)
        module_map = {mod.name: mod for mod in modules}

    valid_for_cycles: set[str] = set()
    for mod in modules:
        before = len(errors)
        for check in (check_ports, check_ssa_and_terminator, check_output_coverage):
            try:
                check(mod)
            except ValidationError as exc:
                errors.append(exc)
        if len(errors) == before:
            valid_for_cycles.add(mod.name)
    # Cross-module checks (instances) require the full module map
    for mod in modules:
        try:
            check_instances(mod, module_map)
        except ValidationError as exc:
            errors.append(exc)
            valid_for_cycles.discard(mod.name)
    # Static analysis checks (strict mode)
    if strict:
        dependency_cache: Dict[str, Dict[str, Set[str]]] = {}
        for mod in modules:
            if mod.name not in valid_for_cycles:
                continue
            try:
                check_combinational_cycles(mod, module_map, dependency_cache)
            except CycleError as exc:
                errors.append(exc)

    if errors:
        issues = issues_from_errors(errors)
        if len(errors) == 1:
            raise type(errors[0])(
                issues,
                summary=f"Design validation found {len(issues)} error(s)",
            )
        raise ValidationError(
            issues,
            summary=f"Design validation found {len(issues)} error(s) across all modules",
        )


def check_module_names(modules: List[Module]) -> Dict[str, Module]:
    seen: Dict[str, Module] = {}
    issues: list[DiagnosticIssue] = []
    for index, mod in enumerate(modules):
        if mod.name in seen:
            issues.append(
                DiagnosticIssue(
                    code="module.duplicate_name",
                    location=f"module[{index}]",
                    message=f"Duplicate module name '{mod.name}'",
                    actual=mod.name,
                    hint="Give every user-defined module a unique name.",
                )
            )
        seen[mod.name] = mod
    if issues:
        raise ValidationError(issues, summary="Duplicate module definitions found")
    return seen


def check_ports(mod: Module) -> None:
    issues: list[DiagnosticIssue] = []
    if not mod.ports:
        issues.append(
            DiagnosticIssue(
                code="port.empty",
                location=f"Module '{mod.name}' ports",
                message="module must have at least one port",
                expected="one or more input/output ports",
                actual=[],
                hint="Declare the module interface in the Python @module signature.",
            )
        )
    for pname, pdef in mod.ports.items():
        loc = f"Module '{mod.name}' port '{pname}'"
        if not is_valid_ssa_id(pname):
            issues.append(
                DiagnosticIssue(
                    code="port.invalid_identifier",
                    location=loc,
                    message="invalid port/SSA identifier",
                    expected="a meaningful identifier beginning with a letter or underscore",
                    actual=pname,
                    hint="Rename the port using letters, digits, '_', or '$'; do not use a numeric-only name.",
                )
            )
        if pdef.width <= 0:
            issues.append(
                DiagnosticIssue(
                    code="port.invalid_width",
                    location=loc,
                    message="port width must be positive",
                    expected="integer >= 1",
                    actual=pdef.width,
                    hint="Set the port width to the intended positive bit width.",
                )
            )
        if pdef.type not in ("bits", "clock"):
            issues.append(
                DiagnosticIssue(
                    code="port.invalid_type",
                    location=loc,
                    message="port has an unsupported type",
                    expected=["bits", "clock"],
                    actual=pdef.type,
                    hint="Use 'bits' for data or 'clock' for a clock input.",
                )
            )
        if pdef.type == "clock":
            if pdef.dir != PortDir.INPUT:
                issues.append(
                    DiagnosticIssue(
                        code="port.clock_direction",
                        location=loc,
                        message="clock port must be an input",
                        expected="input",
                        actual=pdef.dir.value,
                        hint="Change the clock port direction to input.",
                    )
                )
            if pdef.width != 1:
                issues.append(
                    DiagnosticIssue(
                        code="port.clock_width",
                        location=loc,
                        message="clock port width must be 1",
                        expected=1,
                        actual=pdef.width,
                        hint="Declare clock signals as one bit.",
                    )
                )
    if issues:
        raise ValidationError(
            issues, summary=f"Module '{mod.name}' has {len(issues)} port error(s)"
        )


def check_ssa_and_terminator(mod: Module) -> None:
    ctx = f"Module '{mod.name}'"

    if not mod.body:
        raise ValidationError(
            [
                DiagnosticIssue(
                    code="body.empty",
                    location=ctx,
                    message="body must not be empty",
                    expected="operations ending in exactly one output operation",
                    actual=[],
                    hint="Generate the module logic followed by the final output operation.",
                )
            ],
            summary=f"{ctx} has an invalid body",
        )

    structural_issues: list[DiagnosticIssue] = []
    ssa_errors: list[SSAError] = []

    # Check no output ops appear before the last position
    for i, op in enumerate(mod.body[:-1]):
        if isinstance(op, OutputOp):
            structural_issues.append(
                DiagnosticIssue(
                    code="body.early_output",
                    location=f"{ctx} body[{i}]",
                    message="'output' must only appear as the last operation",
                    expected=len(mod.body) - 1,
                    actual=i,
                    hint="Remove this early output and keep one final output operation at the end.",
                )
            )

    # Check terminator is last
    last = mod.body[-1]
    if not isinstance(last, OutputOp):
        structural_issues.append(
            DiagnosticIssue(
                code="body.missing_terminator",
                location=f"{ctx} body[{len(mod.body) - 1}]",
                message="last operation must be 'output'",
                expected="output",
                actual=last.op,
                hint="Append one final output operation whose args map every declared output port.",
            )
        )

    # Build symbol table: start with input ports
    defined: Set[str] = set()
    for pname, pdef in mod.ports.items():
        if pdef.dir == PortDir.INPUT:
            defined.add(pname)

    # Two-pass approach: first define all IDs, then check references.
    # This allows forward references (use before definition) which is
    # natural for hardware where all values in a module are defined
    # simultaneously (combinational logic has no ordering).

    # Pass 1: collect all defined IDs
    for i, op in enumerate(mod.body):
        loc = f"{ctx} body[{i}]"
        try:
            define_ids(op, defined, loc)
        except SSAError as exc:
            ssa_errors.append(exc)

    # Pass 2: check all references are to defined values
    for i, op in enumerate(mod.body):
        loc = f"{ctx} body[{i}]"
        try:
            check_arguments_defined(op, defined, loc)
        except SSAError as exc:
            ssa_errors.append(exc)

    issues = structural_issues + issues_from_errors(ssa_errors)
    if issues:
        error_type = ValidationError if structural_issues else SSAError
        raise error_type(
            issues,
            summary=f"{ctx} body validation found {len(issues)} error(s)",
        )


def check_arguments_defined(op: Operation, defined: Set[str], loc: str) -> None:
    """Check that all argument references are to already-defined values."""
    issues: list[DiagnosticIssue] = []

    def undefined(reference: str, role: str) -> None:
        if reference not in defined:
            issues.append(
                DiagnosticIssue(
                    code="ssa.undefined_reference",
                    location=loc,
                    message=f"{role} references undefined value '{reference}'",
                    expected="an input port or an SSA id defined by an operation",
                    actual=reference,
                    hint="Define this value or replace it with the intended existing SSA id.",
                )
            )

    if isinstance(op, OutputOp):
        for port, ref in op.args.items():
            undefined(ref, f"output port '{port}'")
    elif isinstance(op, InstanceOp):
        for port, ref in op.args.items():
            undefined(ref, f"instance arg '{port}'")
    elif isinstance(op, ConstantOp):
        pass  # no args to check
    elif isinstance(op, RegOp):
        for arg in op.args:
            undefined(arg, "register data input")
        undefined(op.clock, "clock")
        if op.reset:
            undefined(op.reset, "reset")
        if op.enable:
            undefined(op.enable, "enable")
    elif isinstance(op, MemOp):
        undefined(op.clock, "clock")
        if op.reset:
            undefined(op.reset, "reset")
        for i, (addr, enable) in enumerate(op.reads):
            undefined(addr, f"reads[{i}].addr")
            undefined(enable, f"reads[{i}].enable")
        for i, (addr, data, enable) in enumerate(op.writes):
            undefined(addr, f"writes[{i}].addr")
            undefined(data, f"writes[{i}].data")
            undefined(enable, f"writes[{i}].enable")
    else:
        # UnaryOp, BinaryOp, VariadicOp, ExtractOp — all have args as list[str]
        for arg in op.args:
            undefined(arg, f"'{op.op}' operand")

    if issues:
        raise SSAError(issues, summary=f"{loc} has {len(issues)} undefined reference(s)")


def define_ids(op: Operation, defined: Set[str], loc: str) -> None:
    """Register new value IDs, checking for shadowing."""
    issues: list[DiagnosticIssue] = []
    if isinstance(op, OutputOp):
        return  # output doesn't define new values
    elif isinstance(op, (InstanceOp, MemOp)):
        ids = op.id
    else:
        ids = [op.id]

    for id_ in ids:
        if not is_valid_ssa_id(id_):
            issues.append(
                DiagnosticIssue(
                    code="ssa.invalid_identifier",
                    location=loc,
                    message=f"invalid SSA identifier '{id_}'",
                    expected="a meaningful string beginning with a letter or underscore",
                    actual=id_,
                    hint="Rename it descriptively, for example 'byte_select' or 'next_pc'.",
                )
            )
            continue
        if id_ in defined:
            issues.append(
                DiagnosticIssue(
                    code="ssa.shadowing",
                    location=loc,
                    message=f"value '{id_}' is already defined (shadowing)",
                    actual=id_,
                    hint="Assign a unique descriptive SSA id to this result.",
                )
            )
            continue
        defined.add(id_)

    if issues:
        raise SSAError(issues, summary=f"{loc} defines invalid SSA value(s)")


def check_output_coverage(mod: Module) -> None:
    """OutputOp args keys must exactly match the module's output port set."""
    ctx = f"Module '{mod.name}'"
    output_ports: Set[str] = {
        name for name, pdef in mod.ports.items() if pdef.dir == PortDir.OUTPUT
    }
    if not mod.body or not isinstance(mod.body[-1], OutputOp):
        return  # check_ssa_and_terminator reports the structural error.
    last = mod.body[-1]

    provided = set(last.args.keys())
    missing = output_ports - provided
    extra = provided - output_ports

    if missing or extra:
        expected = sorted(output_ports)
        actual = sorted(provided)
        mismatch_parts = []
        if missing:
            mismatch_parts.append(f"output op missing ports: {sorted(missing)}")
        if extra:
            mismatch_parts.append(f"output op has extra ports: {sorted(extra)}")
        raise ValidationError(
            [
                DiagnosticIssue(
                    code="output.port_keys",
                    location=f"{ctx} body[{len(mod.body) - 1}]",
                    message=(
                        "; ".join(mismatch_parts)
                        + f"; expected exact output args keys {expected}, got {actual}"
                    ),
                    expected=expected,
                    actual=actual,
                    hint=(
                        "Use exactly the declared output port names as args keys, "
                        "preserving spelling and capitalization. No automatic renaming is applied."
                    ),
                )
            ],
            summary=f"{ctx} final output port mapping is invalid",
        )


def check_instances(mod: Module, module_map: Dict[str, Module]) -> None:
    """Validate InstanceOp references: module exists, ports match."""
    ctx = f"Module '{mod.name}'"
    instance_names: Set[str] = set()
    issues: list[DiagnosticIssue] = []
    for i, op in enumerate(mod.body):
        if not isinstance(op, InstanceOp):
            continue
        loc = f"{ctx} body[{i}]"
        if op.name:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", op.name):
                issues.append(
                    DiagnosticIssue(
                        code="instance.invalid_name",
                        location=loc,
                        message=f"invalid instance name '{op.name}'",
                        expected="an identifier beginning with a letter or underscore",
                        actual=op.name,
                        hint="Rename the instance using letters, digits, '_', or '$'.",
                    )
                )
            if op.name in instance_names:
                issues.append(
                    DiagnosticIssue(
                        code="instance.duplicate_name",
                        location=loc,
                        message=f"duplicate instance name '{op.name}'",
                        actual=op.name,
                        hint="Give each instance in the module a unique name.",
                    )
                )
            instance_names.add(op.name)

        target = module_map.get(op.module)
        if target is None:
            issues.append(
                DiagnosticIssue(
                    code="instance.unknown_module",
                    location=loc,
                    message=f"instance references unknown module '{op.module}'",
                    expected=sorted(module_map),
                    actual=op.module,
                    hint="Use the exact name of a declared child module.",
                )
            )
            continue

        # Check input port matching
        target_inputs = {
            name for name, pdef in target.ports.items() if pdef.dir == PortDir.INPUT
        }
        provided_inputs = set(op.args.keys())
        if provided_inputs != target_inputs:
            missing = target_inputs - provided_inputs
            extra = provided_inputs - target_inputs
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            issues.append(
                DiagnosticIssue(
                    code="instance.input_ports",
                    location=loc,
                    message=f"instance input ports mismatch ({', '.join(parts)})",
                    expected=sorted(target_inputs),
                    actual=sorted(provided_inputs),
                    hint=f"Map every input port of module '{op.module}' exactly once.",
                )
            )

        # Check output count matches id list length
        target_outputs = [
            name for name, pdef in target.ports.items() if pdef.dir == PortDir.OUTPUT
        ]
        if len(op.id) != len(target_outputs):
            issues.append(
                DiagnosticIssue(
                    code="instance.output_count",
                    location=loc,
                    message=(
                        f"instance has {len(op.id)} output id(s) but module "
                        f"'{op.module}' has {len(target_outputs)} output port(s)"
                    ),
                    expected={"count": len(target_outputs), "ports": target_outputs},
                    actual={"count": len(op.id), "ids": list(op.id)},
                    hint="Provide one meaningful SSA id per child output, in declaration order.",
                )
            )

    if issues:
        raise ValidationError(
            issues,
            summary=f"Module '{mod.name}' instance validation found {len(issues)} error(s)",
        )


def module_output_input_dependencies(
    mod: Module,
    module_map: Dict[str, Module],
    cache: Dict[str, Dict[str, Set[str]]],
    active: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Return output-port dependencies on this module's input ports."""
    if mod.name in cache:
        return cache[mod.name]
    if active is None:
        active = set()
    if mod.name in active:
        return {}

    active.add(mod.name)
    deps: Dict[str, Set[str]] = {}
    input_ports = {
        name for name, pdef in mod.ports.items() if pdef.dir == PortDir.INPUT
    }
    for name in input_ports:
        deps[name] = {name}

    for op in mod.body:
        if isinstance(op, OutputOp):
            continue

        if isinstance(op, ConstantOp):
            deps[op.id] = set()

        elif isinstance(op, RegOp):
            # Sequential boundary: register output has no same-cycle deps.
            deps[op.id] = set()

        elif isinstance(op, MemOp):
            for idx, (addr, enable) in enumerate(op.reads):
                out_id = op.id[idx]
                deps[out_id] = set(deps.get(addr, set())) | set(deps.get(enable, set()))
            # Write ports are sequential and do not affect read outputs here.

        elif isinstance(op, InstanceOp):
            target = module_map.get(op.module)
            child_deps = (
                module_output_input_dependencies(target, module_map, cache, active)
                if target is not None
                else {}
            )
            target_outputs = []
            if target is not None:
                target_outputs = [
                    name
                    for name, pdef in target.ports.items()
                    if pdef.dir == PortDir.OUTPUT
                ]
            for out_id, child_out in zip(op.id, target_outputs):
                out_deps: Set[str] = set()
                for child_input in child_deps.get(child_out, set()):
                    parent_ref = op.args.get(child_input)
                    if parent_ref is not None:
                        out_deps.update(deps.get(parent_ref, set()))
                deps[out_id] = out_deps

        else:
            out_id = op.id
            out_deps: Set[str] = set()
            for arg in op.args:
                out_deps.update(deps.get(arg, set()))
            deps[out_id] = out_deps

    last = mod.body[-1]
    assert isinstance(last, OutputOp)
    result = {port: set(deps.get(ref, set())) for port, ref in last.args.items()}
    active.remove(mod.name)
    cache[mod.name] = result
    return result


def build_combinational_graph(
    mod: Module,
    module_map: Dict[str, Module],
    dependency_cache: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, List[str]]:
    """Build a combinational dependency graph (adjacency list: dep -> [dependents]).

    Edges represent combinational (same-cycle) dataflow: an edge from A to B
    means "B combinationally depends on A".  Sequential elements (RegOp,
    MemOp write ports) do NOT create edges because they break timing paths.
    """
    graph: Dict[str, List[str]] = {}

    # Ensure every defined ID has an entry (including leaf nodes)
    for pname, pdef in mod.ports.items():
        if pdef.dir == PortDir.INPUT:
            graph.setdefault(pname, [])

    for op in mod.body:
        if isinstance(op, OutputOp):
            continue

        if isinstance(op, ConstantOp):
            graph.setdefault(op.id, [])

        elif isinstance(op, RegOp):
            # Sequential boundary — no combinational edges
            graph.setdefault(op.id, [])

        elif isinstance(op, MemOp):
            # Read ports: combinational from addr+enable to output
            for idx, (addr, enable) in enumerate(op.reads):
                out_id = op.id[idx]
                graph.setdefault(out_id, [])
                graph.setdefault(addr, [])
                graph.setdefault(enable, [])
                graph[addr].append(out_id)
                graph[enable].append(out_id)
            # Write ports are sequential — no edges

        elif isinstance(op, InstanceOp):
            target = module_map.get(op.module)
            child_deps = (
                module_output_input_dependencies(target, module_map, dependency_cache)
                if target is not None
                else {}
            )
            target_outputs = []
            if target is not None:
                target_outputs = [
                    name
                    for name, pdef in target.ports.items()
                    if pdef.dir == PortDir.OUTPUT
                ]
            for out_id in op.id:
                graph.setdefault(out_id, [])
            for out_id, child_out in zip(op.id, target_outputs):
                for child_input in child_deps.get(child_out, set()):
                    arg_ref = op.args.get(child_input)
                    if arg_ref is None:
                        continue
                    graph.setdefault(arg_ref, [])
                    graph[arg_ref].append(out_id)

        else:
            # Combinational ops: UnaryOp, BinaryOp, VariadicOp, MuxOp, CastOp, ExtractOp
            out_id = op.id
            graph.setdefault(out_id, [])
            for arg in op.args:
                graph.setdefault(arg, [])
                graph[arg].append(out_id)

    return graph


# DFS 3-color constants
_WHITE, _GRAY, _BLACK = 0, 1, 2


def check_combinational_cycles(
    mod: Module,
    module_map: Dict[str, Module],
    dependency_cache: Dict[str, Dict[str, Set[str]]],
) -> None:
    """Detect combinational cycles using DFS with 3-color marking."""
    graph = build_combinational_graph(mod, module_map, dependency_cache)

    color: Dict[str, int] = {node: _WHITE for node in graph}
    parent: Dict[str, str] = {}

    def visit_dependency(node: str) -> None:
        color[node] = _GRAY
        for neighbor in graph.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == _GRAY:
                # Back edge — reconstruct cycle path
                cycle = [neighbor, node]
                cur = node
                while cur != neighbor:
                    cur = parent.get(cur, neighbor)
                    if cur == neighbor:
                        break
                    cycle.append(cur)
                cycle.reverse()
                raise CycleError(
                    f"Module '{mod.name}': combinational cycle: "
                    f"{' -> '.join(cycle)}"
                )
            if color[neighbor] == _WHITE:
                parent[neighbor] = node
                visit_dependency(neighbor)
        color[node] = _BLACK

    for node in list(graph.keys()):
        if color[node] == _WHITE:
            visit_dependency(node)
