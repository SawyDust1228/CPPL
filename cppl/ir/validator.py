"""Semantic validation: SSA scoping, port coverage, instance checks."""

from __future__ import annotations

from typing import Dict, List, Set

from .errors import CycleError, SSAError, ValidationError
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
    module_map = _check_module_names(modules)
    for mod in modules:
        _check_ports(mod)
        _check_ssa_and_terminator(mod)
        _check_output_coverage(mod)
    # Cross-module checks (instances) require the full module map
    for mod in modules:
        _check_instances(mod, module_map)
    # Static analysis checks (strict mode)
    if strict:
        for mod in modules:
            _check_combinational_cycles(mod)


def _check_module_names(modules: List[Module]) -> Dict[str, Module]:
    seen: Dict[str, Module] = {}
    for mod in modules:
        if mod.name in seen:
            raise ValidationError(f"Duplicate module name '{mod.name}'")
        seen[mod.name] = mod
    return seen


def _check_ports(mod: Module) -> None:
    if not mod.ports:
        raise ValidationError(f"Module '{mod.name}': must have at least one port")
    for pname, pdef in mod.ports.items():
        if pdef.width <= 0:
            raise ValidationError(
                f"Module '{mod.name}': port '{pname}' width must be positive"
            )


def _check_ssa_and_terminator(mod: Module) -> None:
    ctx = f"Module '{mod.name}'"

    if not mod.body:
        raise ValidationError(f"{ctx}: body must not be empty")

    # Check no output ops appear before the last position
    for i, op in enumerate(mod.body[:-1]):
        if isinstance(op, OutputOp):
            raise ValidationError(
                f"{ctx} body[{i}]: 'output' must only appear as the last operation"
            )

    # Check terminator is last
    last = mod.body[-1]
    if not isinstance(last, OutputOp):
        raise ValidationError(f"{ctx}: last operation must be 'output'")

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
        _define_ids(op, defined, loc)

    # Pass 2: check all references are to defined values
    for i, op in enumerate(mod.body):
        loc = f"{ctx} body[{i}]"
        _check_args_defined(op, defined, loc)


def _check_args_defined(op: Operation, defined: Set[str], loc: str) -> None:
    """Check that all argument references are to already-defined values."""
    if isinstance(op, OutputOp):
        for port, ref in op.args.items():
            if ref not in defined:
                raise SSAError(f"{loc}: output port '{port}' references undefined value '{ref}'")
    elif isinstance(op, InstanceOp):
        for port, ref in op.args.items():
            if ref not in defined:
                raise SSAError(f"{loc}: instance arg '{port}' references undefined value '{ref}'")
    elif isinstance(op, ConstantOp):
        pass  # no args to check
    elif isinstance(op, RegOp):
        for arg in op.args:
            if arg not in defined:
                raise SSAError(f"{loc}: references undefined value '{arg}'")
        if op.clock not in defined:
            raise SSAError(f"{loc}: clock references undefined value '{op.clock}'")
        if op.reset and op.reset not in defined:
            raise SSAError(f"{loc}: reset references undefined value '{op.reset}'")
        if op.enable and op.enable not in defined:
            raise SSAError(f"{loc}: enable references undefined value '{op.enable}'")
    elif isinstance(op, MemOp):
        if op.clock not in defined:
            raise SSAError(f"{loc}: clock references undefined value '{op.clock}'")
        if op.reset not in defined:
            raise SSAError(f"{loc}: reset references undefined value '{op.reset}'")
        for i, (addr, enable) in enumerate(op.reads):
            if addr not in defined:
                raise SSAError(f"{loc}: reads[{i}].addr references undefined value '{addr}'")
            if enable not in defined:
                raise SSAError(f"{loc}: reads[{i}].enable references undefined value '{enable}'")
        for i, (addr, data, enable) in enumerate(op.writes):
            if addr not in defined:
                raise SSAError(f"{loc}: writes[{i}].addr references undefined value '{addr}'")
            if data not in defined:
                raise SSAError(f"{loc}: writes[{i}].data references undefined value '{data}'")
            if enable not in defined:
                raise SSAError(f"{loc}: writes[{i}].enable references undefined value '{enable}'")
    else:
        # UnaryOp, BinaryOp, VariadicOp, ExtractOp — all have args as list[str]
        for arg in op.args:
            if arg not in defined:
                raise SSAError(f"{loc}: references undefined value '{arg}'")


def _define_ids(op: Operation, defined: Set[str], loc: str) -> None:
    """Register new value IDs, checking for shadowing."""
    if isinstance(op, OutputOp):
        return  # output doesn't define new values
    elif isinstance(op, (InstanceOp, MemOp)):
        for id_ in op.id:
            if id_ in defined:
                raise SSAError(f"{loc}: value '{id_}' is already defined (shadowing)")
            defined.add(id_)
    else:
        id_ = op.id
        if id_ in defined:
            raise SSAError(f"{loc}: value '{id_}' is already defined (shadowing)")
        defined.add(id_)


def _check_output_coverage(mod: Module) -> None:
    """OutputOp args keys must exactly match the module's output port set."""
    ctx = f"Module '{mod.name}'"
    output_ports: Set[str] = {
        name for name, pdef in mod.ports.items() if pdef.dir == PortDir.OUTPUT
    }
    last = mod.body[-1]
    assert isinstance(last, OutputOp)

    provided = set(last.args.keys())
    missing = output_ports - provided
    extra = provided - output_ports

    if missing:
        raise ValidationError(
            f"{ctx}: output op missing ports: {sorted(missing)}"
        )
    if extra:
        raise ValidationError(
            f"{ctx}: output op has extra ports: {sorted(extra)}"
        )


def _check_instances(mod: Module, module_map: Dict[str, Module]) -> None:
    """Validate InstanceOp references: module exists, ports match."""
    ctx = f"Module '{mod.name}'"
    for i, op in enumerate(mod.body):
        if not isinstance(op, InstanceOp):
            continue
        loc = f"{ctx} body[{i}]"
        target = module_map.get(op.module)
        if target is None:
            raise ValidationError(
                f"{loc}: instance references unknown module '{op.module}'"
            )

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
            raise ValidationError(
                f"{loc}: instance input ports mismatch ({', '.join(parts)})"
            )

        # Check output count matches id list length
        target_outputs = [
            name for name, pdef in target.ports.items() if pdef.dir == PortDir.OUTPUT
        ]
        if len(op.id) != len(target_outputs):
            raise ValidationError(
                f"{loc}: instance has {len(op.id)} output id(s) "
                f"but module '{op.module}' has {len(target_outputs)} output port(s)"
            )


def _build_comb_graph(mod: Module) -> Dict[str, List[str]]:
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
            # Conservative: every input feeds every output
            for out_id in op.id:
                graph.setdefault(out_id, [])
            for arg_ref in op.args.values():
                graph.setdefault(arg_ref, [])
                for out_id in op.id:
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


def _check_combinational_cycles(mod: Module) -> None:
    """Detect combinational cycles using DFS with 3-color marking."""
    graph = _build_comb_graph(mod)

    color: Dict[str, int] = {node: _WHITE for node in graph}
    parent: Dict[str, str] = {}

    def _dfs(node: str) -> None:
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
                _dfs(neighbor)
        color[node] = _BLACK

    for node in list(graph.keys()):
        if color[node] == _WHITE:
            _dfs(node)
